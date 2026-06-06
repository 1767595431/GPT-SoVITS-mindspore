# modified from https://github.com/yangdongchao/SoundStorm/blob/master/soundstorm/s1/AR/models/t2s_model.py
# reference: https://github.com/lifeiteng/vall-e
#import torch
from tqdm import tqdm

from AR.models.utils import make_pad_mask
from AR.models.utils import (
    topk_sampling,
    sample,
    logits_to_probs,
    multinomial_sample_one_no_sync,
    dpo_loss,
    make_reject_y, 
    get_batch_logps
)
from AR.modules.embedding import SinePositionalEmbedding
from AR.modules.embedding import TokenEmbedding
import time
import mindspore as ms
from mindspore.nn import LayerNorm
from AR.modules.transformer import TransformerEncoder
from AR.modules.transformer import TransformerEncoderLayer
from mindspore import nn,ops,Parameter
#from torchmetrics.classification import MulticlassAccuracy

default_config = {
    "embedding_dim": 512,
    "hidden_dim": 512,
    "num_head": 8,
    "num_layers": 12,
    "num_codebook": 8,
    "p_dropout": 0.0,
    "vocab_size": 1024 + 1,
    "phoneme_vocab_size": 512,
    "EOS": 1024,
}


class Text2SemanticDecoder(nn.Cell):
    def __init__(self, config, norm_first=False, top_k=3):
        super(Text2SemanticDecoder, self).__init__()
        self.model_dim = config["model"]["hidden_dim"]
        self.embedding_dim = config["model"]["embedding_dim"]
        self.num_head = config["model"]["head"]
        self.num_layers = config["model"]["n_layer"]
        self.norm_first = norm_first
        self.vocab_size = config["model"]["vocab_size"]
        self.phoneme_vocab_size = config["model"]["phoneme_vocab_size"]
        self.p_dropout = config["model"]["dropout"]
        self.EOS = config["model"]["EOS"]
        self.norm_first = norm_first
        assert self.EOS == self.vocab_size - 1
        # should be same as num of kmeans bin
        # assert self.EOS == 1024
        self.bert_proj = nn.Dense(1024, self.embedding_dim)
        self.ar_text_embedding = TokenEmbedding(
            self.embedding_dim, self.phoneme_vocab_size, self.p_dropout
        )
        self.ar_text_position = SinePositionalEmbedding(
            self.embedding_dim, dropout=0.1, scale=False, alpha=True
        )
        self.ar_audio_embedding = TokenEmbedding(
            self.embedding_dim, self.vocab_size, self.p_dropout
        )
        self.ar_audio_position = SinePositionalEmbedding(
            self.embedding_dim, dropout=0.1, scale=False, alpha=True
        )

        self.h = TransformerEncoder(
            TransformerEncoderLayer(
                d_model=self.model_dim,
                nhead=self.num_head,
                dim_feedforward=self.model_dim * 4,
                dropout=0.1,
                batch_first=True,
                norm_first=norm_first,
            ),
            num_layers=self.num_layers,
            norm=LayerNorm(self.model_dim) if norm_first else None,
        )

        self.ar_predict_layer = nn.Dense(self.model_dim, self.vocab_size, has_bias=False)
        self.loss_fct = nn.CrossEntropyLoss(reduction="sum")

        """
        self.ar_accuracy_metric = MulticlassAccuracy(
            self.vocab_size,
            top_k=top_k,
            average="micro",
            multidim_average="global",
            ignore_index=self.EOS,
        )"""

    def make_input_data(self, x, x_lens, y, y_lens, bert_feature):
        x = self.ar_text_embedding(x)
        x = x + self.bert_proj(bert_feature.swapaxes(1, 2))
        x = self.ar_text_position(x)
        x_mask = make_pad_mask(x_lens)

        y_mask = make_pad_mask(y_lens)  # bool, shape [B, Ty]
        y_mask_int = y_mask.type(ms.int64)
        codes = y.type(ms.int64) * (1 - y_mask_int)

        # Training
        # AR Decoder
        y, targets = self.pad_y_eos(codes, y_mask_int, eos_id=self.EOS)
        x_len = x_lens.max()
        y_len = y_lens.max()
        y_emb = self.ar_audio_embedding(y)
        y_pos = self.ar_audio_position(y_emb)

        # padding mask（逐位置是否可见）仍保持 bool
        xy_padding_mask = ops.concat([x_mask, y_mask], axis=1)  # [B, Tx+Ty]
        ar_xy_padding_mask = xy_padding_mask  # bool

        # —— 注意力 mask：用 float 构造 + pad，最后转回 bool ——
        # x 的自注意：全 0（可见），右侧 pad y_len 列为 1（屏蔽 x->y）
        x_attn_mask_f32 = ops.zeros((x_len, x_len), dtype=ms.float32)
        x_attn_mask_pad = ops.pad(
            x_attn_mask_f32,
            (0, y_len),
            value=1.0,
        )

        # y 的上三角：上三角为 1（屏蔽未来），左侧 pad x_len 列为 0（与 x 拼接对齐）
        y_attn_mask_f32 = ops.triu(
            ops.ones((y_len, y_len), dtype=ms.float32),
            diagonal=1,
        )
        y_attn_mask_pad = ops.pad(
            y_attn_mask_f32,
            (x_len, 0),
            value=0.0,
        )

        xy_attn_mask = ops.concat([x_attn_mask_pad, y_attn_mask_pad], axis=0)  # float32
        xy_attn_mask = xy_attn_mask > 0.5  # 转回 bool

        # 将 padding mask broadcast 到多头，再并到总的 attn mask 里
        bsz, src_len = x.shape[0], x_len + y_len
        _xy_padding_mask = (
            ar_xy_padding_mask.view(bsz, 1, 1, src_len)
            .broadcast_to((-1, self.num_head, -1, -1))
            .reshape(bsz * self.num_head, 1, src_len)
        )  # bool

        # 任何一个为 True 都视为屏蔽
        xy_attn_mask = xy_attn_mask.logical_or(_xy_padding_mask)  # bool

        # gradio 代码里后续期望一个 float 的 -inf/0 mask
        new_attn_mask = ops.zeros_like(xy_attn_mask, dtype=x.dtype)  # float
        new_attn_mask.masked_fill_(xy_attn_mask, float("-inf"))
        xy_attn_mask = new_attn_mask  # float

        # x 和完整的 y 一次性输入模型
        xy_pos = ops.concat([x, y_pos], axis=1)

        return xy_pos, xy_attn_mask, targets

        x = self.ar_text_embedding(x)
        x = x + self.bert_proj(bert_feature.swapaxes(1, 2))
        x = self.ar_text_position(x)
        x_mask = make_pad_mask(x_lens)

        y_mask = make_pad_mask(y_lens)
        y_mask_int = y_mask.type(ms.int64)
        codes = y.type(ms.int64) * (1 - y_mask_int)

        # Training
        # AR Decoder
        y, targets = self.pad_y_eos(codes, y_mask_int, eos_id=self.EOS)
        x_len = x_lens.max()
        y_len = y_lens.max()
        y_emb = self.ar_audio_embedding(y)
        y_pos = self.ar_audio_position(y_emb)

        xy_padding_mask = ops.concat([x_mask, y_mask], axis=1)

        ar_xy_padding_mask = xy_padding_mask

        x_attn_mask = ops.pad(
            ops.zeros((x_len, x_len), dtype=ms.bool_),
            (0, y_len),
            value=True,
        )
        
        y_attn_mask = ops.pad(
            ops.triu(
                ops.ones((y_len, y_len), dtype=ms.bool_),
                diagonal=1,
            ),
            (x_len, 0),
            value=False,
        )

        xy_attn_mask = ops.concat([x_attn_mask, y_attn_mask], axis=0)
        bsz, src_len = x.shape[0], x_len + y_len
        _xy_padding_mask = (
            ar_xy_padding_mask.view(bsz, 1, 1, src_len)
            .broadcast_to((-1, self.num_head, -1, -1))
            .reshape(bsz * self.num_head, 1, src_len)
        )
        xy_attn_mask = xy_attn_mask.logical_or(_xy_padding_mask)
        new_attn_mask = ops.zeros_like(xy_attn_mask, dtype=x.dtype)
        new_attn_mask.masked_fill_(xy_attn_mask, float("-inf"))
        xy_attn_mask = new_attn_mask
        # x 和完整的 y 一次性输入模型
        xy_pos = ops.concat([x, y_pos], axis=1)

        return xy_pos, xy_attn_mask, targets

    def construct(self, x, x_lens, y, y_lens, bert_feature):
        """
        x: phoneme_ids
        y: semantic_ids
        """

        reject_y, reject_y_lens = make_reject_y(y, y_lens)

        xy_pos, xy_attn_mask, targets = self.make_input_data(x, x_lens, y, y_lens, bert_feature)

        xy_dec, _ = self.h(
            (xy_pos, None),
            mask=xy_attn_mask,
        )
        x_len = x_lens.max()
        logits = self.ar_predict_layer(xy_dec[:, x_len:])

        ###### DPO #############
        reject_xy_pos, reject_xy_attn_mask, reject_targets = self.make_input_data(x, x_lens, reject_y, reject_y_lens, bert_feature)

        reject_xy_dec, _ = self.h(
            (reject_xy_pos, None),
            mask=reject_xy_attn_mask,
        )
        x_len = x_lens.max()
        reject_logits = self.ar_predict_layer(reject_xy_dec[:, x_len:])

        # loss
        # from feiteng: 每次 duration 越多, 梯度更新也应该更多, 所以用 sum

        loss_1 = ops.cross_entropy(logits.permute(0, 2, 1), targets, reduction="sum")
        #acc = self.ar_accuracy_metric(logits.permute(0, 2, 1).detach(), targets).item()等我找到可以用再更新吧
        acc=1

        A_logits, R_logits = get_batch_logps(logits, reject_logits, targets, reject_targets)
        loss_2, _, _ = dpo_loss(A_logits, R_logits, 0, 0, 0.2, reference_free=True)
        
        loss = loss_1 + loss_2

        return loss, acc

    def construct_old(self, x, x_lens, y, y_lens, bert_feature):
        """
        x: phoneme_ids
        y: semantic_ids
        """
        x = self.ar_text_embedding(x)
        x = x + self.bert_proj(bert_feature.swapaxes(1, 2))
        x = self.ar_text_position(x)
        x_mask = make_pad_mask(x_lens)

        y_mask = make_pad_mask(y_lens)
        y_mask_int = y_mask.type(ms.int64)
        codes = y.type(ms.int64) * (1 - y_mask_int)

        # Training
        # AR Decoder
        y, targets = self.pad_y_eos(codes, y_mask_int, eos_id=self.EOS)
        x_len = x_lens.max()
        y_len = y_lens.max()
        y_emb = self.ar_audio_embedding(y)
        y_pos = self.ar_audio_position(y_emb)

        xy_padding_mask = ops.concat([x_mask, y_mask], axis=1)
        ar_xy_padding_mask = xy_padding_mask

        x_attn_mask = ops.pad(
            ops.zeros((x_len, x_len), dtype=ms.bool_, ),
            (0, y_len),
            value=True,
        )
        y_attn_mask = ops.pad(
            ops.triu(
                ops.ones((y_len, y_len), dtype=ms.bool_),
                diagonal=1,
            ),
            (x_len, 0),
            value=False,
        )
        xy_attn_mask = ops.concat([x_attn_mask, y_attn_mask], axis=0)
        bsz, src_len = x.shape[0], x_len + y_len
        _xy_padding_mask = (
            ar_xy_padding_mask.view(bsz, 1, 1, src_len)
            .broadcast_to((-1, self.num_head, -1, -1))
            .reshape(bsz * self.num_head, 1, src_len)
        )
        xy_attn_mask = xy_attn_mask.logical_or(_xy_padding_mask)
        new_attn_mask = ops.zeros_like(xy_attn_mask, dtype=x.dtype)
        new_attn_mask.masked_fill_(xy_attn_mask, float("-inf"))
        xy_attn_mask = new_attn_mask
        # x 和完整的 y 一次性输入模型
        xy_pos = ops.concat([x, y_pos], axis=1)
        xy_dec, _ = self.h(
            (xy_pos, None),
            mask=xy_attn_mask,
        )
        logits = self.ar_predict_layer(xy_dec[:, x_len:]).permute(0, 2, 1)
        # loss
        # from feiteng: 每次 duration 越多, 梯度更新也应该更多, 所以用 sum
        loss = ops.cross_entropy(logits, targets, reduction="sum")
        #acc = self.ar_accuracy_metric(logits.detach(), targets).item()
        acc=1
        return loss, acc

    # 需要看下这个函数和 forward 的区别以及没有 semantic 的时候 prompts 输入什么
    def infer(
        self,
        x,
        x_lens,
        prompts,
        bert_feature,
        top_p=1.0,
        top_k: int = -100,
        early_stop_num: int = -1,
        temperature: float = 1.0,
    ):
        x = self.ar_text_embedding(x)
        x = x + self.bert_proj(bert_feature.swapaxes(1, 2))
        x = self.ar_text_position(x)

        # AR Decoder
        y = prompts
        prefix_len = y.shape[1]
        x_len = x.shape[1]

        # 注意力 mask：float 构造 + pad，再转回 bool
        x_attn_mask_f32 = ops.zeros((x_len, x_len), dtype=ms.float32)
        stop = False
        for idx in tqdm(range(1500)):
            y_emb = self.ar_audio_embedding(y)
            y_pos = self.ar_audio_position(y_emb)

            # x 和逐渐增长的 y 一起输入给模型
            xy_pos = ops.concat([x, y_pos], axis=1)
            y_len = y.shape[1]

            x_attn_mask_pad = ops.pad(
                x_attn_mask_f32,
                (0, y_len),
                value=1.0,
            )
            y_attn_mask_f32 = ops.triu(
                ops.ones((y_len, y_len), dtype=ms.float32),
                diagonal=1,
            )
            y_attn_mask_pad = ops.pad(
                y_attn_mask_f32,
                (x_len, 0),
                value=0.0,
            )
            xy_attn_mask = ops.concat([x_attn_mask_pad, y_attn_mask_pad], axis=0)
            xy_attn_mask = xy_attn_mask > 0.5  # bool，供 src_mask 使用

            xy_dec = self.h(
                xy_pos,
                src_mask=xy_attn_mask,
            )
            logits = self.ar_predict_layer(xy_dec[:, -1])
            samples = topk_sampling(
                logits, top_k=top_k, top_p=top_p, temperature=temperature
            )

            if early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num:
                print("use early stop num:", early_stop_num)
                stop = True

            if ops.argmax(logits, dim=-1)[0] == self.EOS or samples[0, 0] == self.EOS:
                stop = True
            if stop:
                if prompts.shape[1] == y.shape[1]:
                    y = ops.concat([y, ops.zeros(samples.shape)], axis=1)
                    print("bad zero prediction")
                print(f"T2S Decoding EOS [{prefix_len} -> {y.shape[1]}]")
                break

            # 本次生成的 semantic_ids 和之前的 y 构成新的 y
            y = ops.concat([y, samples.to(ms.int32)], axis=1)

        return y, idx - 1


    def pad_y_eos(self, y, y_mask_int, eos_id):
        targets = ops.pad(y, (0, 1), value=0) + eos_id * ops.pad(
            y_mask_int, (0, 1), value=1
        )
        # 错位
        return targets[:, :-1], targets[:, 1:]

    def infer_panel(
        self,
        x,  #####全部文本token
        x_lens,
        prompts,  ####参考音频token
        bert_feature,
        top_k: int = -100,
        top_p: int = 100,
        early_stop_num: int = -1,
        temperature: float = 1.0,
    ):
        x = self.ar_text_embedding(x)
        x = x + self.bert_proj(bert_feature.swapaxes(1, 2))
        x = self.ar_text_position(x)

        # AR Decoder
        y = prompts.astype(ms.int32) if prompts is not None else ops.zeros((x.shape[0], 0), dtype=ms.int32)

        x_len = x.shape[1]
        # 注意：这里用 float32 构造 mask，pad 后再转 bool，避免 PadV3 类型错误
        x_attn_mask_f32 = ops.zeros((x_len, x_len), dtype=ms.float32)
        stop = False

        cache = {
            "all_stage": self.num_layers,
            "k": [None] * self.num_layers,
            "v": [None] * self.num_layers,
            "y_emb": None,
            "first_infer": 1,
            "stage": 0,
        }

        ###################  first step ##########################
        if y is not None and y.shape[1] > 0:
            y_emb = self.ar_audio_embedding(y)
            y_len = y_emb.shape[1]
            prefix_len = y.shape[1]
            y_pos = self.ar_audio_position(y_emb)
            xy_pos = ops.concat([x, y_pos], axis=1)
            cache["y_emb"] = y_emb
            ref_free = False
        else:
            y_emb = None
            y_len = 0
            prefix_len = 0
            y_pos = None
            xy_pos = x
            y = ops.zeros((x.shape[0], 0), dtype=ms.int32)
            ref_free = True

        # —— 注意力 mask：float 构造 + pad，再转回 bool ——
        x_attn_mask_pad = ops.pad(
            x_attn_mask_f32,
            (0, y_len),  # 右补 y_len 列为 1
            value=1.0,
        )
        y_attn_mask_f32 = ops.triu(
            ops.ones((y_len, y_len), dtype=ms.float32),
            diagonal=1,
        )
        y_attn_mask_pad = ops.pad(
            y_attn_mask_f32,
            (x_len, 0),  # 左补 x_len 列为 0
            value=0.0,
        )
        xy_attn_mask = ops.concat([x_attn_mask_pad, y_attn_mask_pad], axis=0)
        xy_attn_mask = xy_attn_mask > 0.5  # 转 bool

        for idx in tqdm(range(1500)):
            xy_dec, _ = self.h((xy_pos, None), mask=xy_attn_mask, cache=cache)
            logits = self.ar_predict_layer(xy_dec[:, -1])
            if idx == 0:  # 第一次跑不能 EOS，否则没有了
                logits = logits[:, :-1]  # 去掉 1024 终止符

            samples = sample(
                logits[0], y, top_k=top_k, top_p=top_p,
                repetition_penalty=1.35, temperature=temperature
            )[0].unsqueeze(0)

            # 累加 y（强制类型一致）
            y = ops.concat([y.astype(ms.int32), samples.squeeze(2).astype(ms.int32)], axis=1)

            # Early stop
            if early_stop_num != -1 and (y.shape[1] - prefix_len) > early_stop_num:
                print("use early stop num:", early_stop_num)
                stop = True

            if ops.argmax(logits, dim=-1)[0] == self.EOS or samples[0, 0] == self.EOS:
                stop = True
            if stop:
                if y.shape[1] == 0:
                    y = ops.concat([y.astype(ms.int32), ops.zeros(samples.shape, dtype=ms.int32)], axis=1)
                    print("bad zero prediction")
                print(f"T2S Decoding EOS [{prefix_len} -> {y.shape[1]}]")
                break

            # ############## update next step（缓存增量） ##############
            cache["first_infer"] = 0
            if cache["y_emb"] is not None:
                y_emb = ops.cat(
                    [cache["y_emb"], self.ar_audio_embedding(y[:, -1:])], axis=1
                )
                cache["y_emb"] = y_emb
                y_pos = self.ar_audio_position(y_emb)
                xy_pos = y_pos[:, -1:]
            else:
                y_emb = self.ar_audio_embedding(y[:, -1:])
                cache["y_emb"] = y_emb
                y_pos = self.ar_audio_position(y_emb)
                xy_pos = y_pos
            y_len = y_pos.shape[1]

            # 下一步的 mask（只需要一行，表示最后一个 token 的可见性）
            xy_attn_mask = ops.zeros((1, x_len + y_len), dtype=ms.bool_)

        if ref_free:
            return y[:, :-1], 0
        return y[:, :-1], idx - 1



