# -*- coding: utf-8 -*-
"""
GPT-SoVITS WebUI (MindSpore)
- 输出目录固定为项目同级 ./output/
- 点击合成后先显示“正在合成”，后台同步合成，落地后再返回 wav 路径给播放器
- 避免前端出现红色 Error，错误以状态栏文本给出
"""

import os, re, logging, json, time
from pathlib import Path
from datetime import datetime

import numpy as np
import librosa
import gradio as gr
import soundfile as sf

import mindspore as ms
from mindspore import ops
from mindspore.amp import auto_mixed_precision

# 日志降噪
logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)

# ==== 统一输出目录（项目根目录 ./output/） ====
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "output"
OUT_DIR.mkdir(exist_ok=True)

# ===== hotfix for mindnlp==0.3.0 + mindspore==2.3.1 =====
try:
    from mindnlp.injection import GroupNorm_hijack

    def _reduce_mean(self, x, axis):
        return ops.ReduceMean(keep_dims=True)(x, axis)

    def _reduce_sum(self, x, axis):
        return ops.ReduceSum(keep_dims=True)(x, axis)

    def _sqrt(self, x):
        return ops.Sqrt()(x)

    if not hasattr(GroupNorm_hijack, "reduce_mean"):
        GroupNorm_hijack.reduce_mean = _reduce_mean
    if not hasattr(GroupNorm_hijack, "reduce_sum"):
        GroupNorm_hijack.reduce_sum = _reduce_sum
    if not hasattr(GroupNorm_hijack, "sqrt"):
        GroupNorm_hijack.sqrt = _sqrt

    print("[Hotfix] Patched GroupNorm_hijack: reduce_mean/reduce_sum (keep_dims=True) and sqrt.")
except Exception as e:
    print("[Hotfix warning] Failed to patch GroupNorm_hijack:", e)
# =========================================================

from time import time as ttime
from mindnlp.transformers import AutoModelForMaskedLM, AutoTokenizer
from mindnlp.injection import set_global_fp16

from feature_extractor import cnhubert
from module.models import SynthesizerTrn
from AR.models.t2s_model import Text2SemanticDecoder
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from module.mel_processing import spectrogram_torch
from my_utils import load_audio
from tools.i18n.i18n import I18nAuto
import LangSegment

i18n = I18nAuto(language="zh_CN")

# ---------- 环境参数 ----------
if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]

is_half = eval(os.environ.get("is_half", "True"))
infer_ttswebui = int(os.environ.get("infer_ttswebui", 9872))
is_share = eval(os.environ.get("is_share", "False"))

# ---------- 权重路径读取 ----------
if os.path.exists("./gweight.txt"):
    with open("./gweight.txt", "r", encoding="utf-8") as f:
        gpt_path = os.environ.get("gpt_path", f.read())
else:
    gpt_path = os.environ.get(
        "gpt_path",
        "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232-ms.ckpt",
    )

if os.path.exists("./sweight.txt"):
    with open("./sweight.txt", "r", encoding="utf-8") as f:
        sovits_path = os.environ.get("sovits_path", f.read())
else:
    sovits_path = os.environ.get(
        "sovits_path", "GPT_SoVITS/pretrained_models/s2G488k-ms.ckpt"
    )

cnhubert_base_path = os.environ.get(
    "cnhubert_base_path", "GPT_SoVITS/pretrained_models/chinese-hubert-base"
)
bert_path = os.environ.get(
    "bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
)

# ---------- BERT ----------
tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path).to_float(ms.float16)
if is_half:
    bert_model = bert_model.half()
    set_global_fp16(True)

# ---------- HuBERT ----------
cnhubert.cnhubert_base_path = cnhubert_base_path
ssl_model = cnhubert.get_model().to_float(ms.float16)
ssl_model = ssl_model.half() if is_half else ssl_model

# ---------- 工具类 ----------
class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for k, v in input_dict.items():
            if isinstance(v, dict):
                v = DictToAttrRecursive(v)
            self[k] = v
            setattr(self, k, v)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

# ---------- SoVITS 加载 ----------
def change_sovits_weights(sovits_path_in):
    global vq_model, hps
    dict_s2 = ms.load_checkpoint(sovits_path_in)
    hps = json.loads(dict_s2["config"])
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).to_float(ms.float16)
    vq_model.quantizer.vq.layers[0]._codebook.inited = True
    vq_model.update_parameters_name()
    if "pretrained" not in sovits_path_in:
        # 兼容：有些 ckpt 不包含 enc_q
        try:
            del vq_model.enc_q
        except Exception:
            pass
    vq_model = vq_model.half() if is_half else vq_model
    vq_model.set_train(False)
    ms.load_param_into_net(vq_model, dict_s2)
    with open("./sweight.txt", "w", encoding="utf-8") as f:
        f.write(sovits_path_in)

change_sovits_weights(sovits_path)

# ---------- GPT 加载 ----------
def change_gpt_weights(gpt_path_in):
    global hz, max_sec, t2s_model, config
    hz = 50
    dict_s1 = ms.load_checkpoint(gpt_path_in)
    config = json.loads(dict_s1["config"])
    max_sec = config["data"]["max_sec"]
    t2s_model = Text2SemanticDecoder(config, top_k=3).to_float(ms.float16)
    ms.load_param_into_net(t2s_model, dict_s1)
    if is_half:
        t2s_model = t2s_model.half()
    t2s_model.set_train(False)
    total = sum([ops.size(p) for p in t2s_model.get_parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    with open("./gweight.txt", "w", encoding="utf-8") as f:
        f.write(gpt_path_in)

change_gpt_weights(gpt_path)

# ---------- 语音前处理 ----------
def get_spepc(hps_cfg, filename):
    audio = load_audio(filename, int(hps_cfg.data.sampling_rate))
    audio = ms.Tensor(audio).unsqueeze(0)
    spec = spectrogram_torch(
        audio,
        hps_cfg.data.filter_length,
        hps_cfg.data.sampling_rate,
        hps_cfg.data.hop_length,
        hps_cfg.data.win_length,
        center=False,
    )
    return spec

dict_language = {
    i18n("中文"): "all_zh",
    i18n("英文"): "en",
    i18n("日文"): "all_ja",
    i18n("中英混合"): "zh",
    i18n("日英混合"): "ja",
    i18n("多语种混合"): "auto",
}

def get_bert_feature(text, word2ph):
    inputs = ops.stop_gradient(tokenizer(text, return_tensors="ms"))
    res = bert_model(**inputs, output_hidden_states=True)
    res = ops.cat(res["hidden_states"][-3:-2], -1)[0][1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        phone_level_feature.append(res[i].repeat(word2ph[i], 1))
    phone_level_feature = ops.cat(phone_level_feature, axis=0)
    return phone_level_feature.T

def clean_text_inf(text, language):
    phones, word2ph, norm_text = clean_text(text, language)
    phones = cleaned_text_to_sequence(phones)
    return phones, word2ph, norm_text

dtype = ms.float16 if is_half else ms.float32

def get_bert_inf(phones, word2ph, norm_text, language):
    language = language.replace("all_", "")
    if language == "zh":
        return get_bert_feature(norm_text, word2ph)
    return ops.zeros((1024, len(phones)), dtype=dtype)

splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }

def get_first(text):
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    return re.split(pattern, text)[0].strip()

def get_phones_and_bert(text, language):
    if language in {"en", "all_zh", "all_ja"}:
        language = language.replace("all_", "")
        if language == "en":
            LangSegment.setfilters(["en"])
            formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
        else:
            formattext = text
        while "  " in formattext:
            formattext = formattext.replace("  ", " ")
        phones, word2ph, norm_text = clean_text_inf(formattext, language)
        if language == "zh":
            bert = get_bert_feature(norm_text, word2ph)
        else:
            bert = ops.zeros((1024, len(phones)), dtype=dtype)
    elif language in {"zh", "ja", "auto"}:
        textlist, langlist = [], []
        LangSegment.setfilters(["zh", "ja", "en", "ko"])
        if language == "auto":
            for tmp in LangSegment.getTexts(text):
                langlist.append("zh" if tmp["lang"] == "ko" else tmp["lang"])
                textlist.append(tmp["text"])
        else:
            for tmp in LangSegment.getTexts(text):
                langlist.append(tmp["lang"] if tmp["lang"] == "en" else language)
                textlist.append(tmp["text"])
        phones_list, bert_list, norm_text_list = [], [], []
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = clean_text_inf(textlist[i], lang)
            bert = get_bert_inf(phones, word2ph, norm_text, lang)
            phones_list.append(phones)
            norm_text_list.append(norm_text)
            bert_list.append(bert)
        bert = ops.cat(bert_list, axis=1)
        phones = sum(phones_list, [])
        norm_text = "".join(norm_text_list)
    return phones, bert.to(dtype), norm_text

def merge_short_text_in_array(texts, threshold):
    if len(texts) < 2:
        return texts
    result, buf = [], ""
    for s in texts:
        buf += s
        if len(buf) >= threshold:
            result.append(buf)
            buf = ""
    if len(buf) > 0:
        if len(result) == 0:
            result.append(buf)
        else:
            result[-1] += buf
    return result

def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i, j, n = 0, 0, len(todo_text)
    res = []
    while True:
        if i >= n:
            break
        if todo_text[i] in splits:
            i += 1
            res.append(todo_text[j:i])
            j = i
        else:
            i += 1
    return res

def cut1(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    idxs = list(range(0, len(inps), 4))
    idxs[-1] = None
    if len(idxs) > 1:
        out = []
        for k in range(len(idxs) - 1):
            out.append("".join(inps[idxs[k]:idxs[k+1]]))
        return "\n".join(out)
    return inp

def cut2(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    if len(inps) < 2:
        return inp
    out, s, buf = [], 0, ""
    for seg in inps:
        s += len(seg); buf += seg
        if s > 50:
            s = 0; out.append(buf); buf = ""
    if buf: out.append(buf)
    if len(out) > 1 and len(out[-1]) < 50:
        out[-2] = out[-2] + out[-1]; out = out[:-1]
    return "\n".join(out)

def cut3(inp): return "\n".join(["%s" % it for it in inp.strip("\n").strip("。").split("。")])
def cut4(inp): return "\n".join(["%s" % it for it in inp.strip("\n").strip(".").split(".")])
def cut5(inp):
    inp = inp.strip("\n")
    p = r'[,.;?!、，。？！;：…]'
    items = re.split(f'({p})', inp)
    merged = ["".join(g) for g in zip(items[::2], items[1::2])]
    if len(items) % 2 == 1:
        merged.append(items[-1])
    return "\n".join(merged)

# ---------- 权重文件列出 ----------
pretrained_sovits_name = "GPT_SoVITS/pretrained_models/s2G488k.pth"
pretrained_gpt_name = "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232-ms.ckpt"
SoVITS_weight_root = "SoVITS_weights"
GPT_weight_root = "GPT_weights"
os.makedirs(SoVITS_weight_root, exist_ok=True)
os.makedirs(GPT_weight_root, exist_ok=True)

def get_weights_names():
    s_names = [pretrained_sovits_name]
    for n in os.listdir(SoVITS_weight_root):
        if n.endswith(".ckpt"):
            s_names.append(f"{SoVITS_weight_root}/{n}")
    g_names = [pretrained_gpt_name]
    for n in os.listdir(GPT_weight_root):
        if n.endswith(".ckpt"):
            g_names.append(f"{GPT_weight_root}/{n}")
    return s_names, g_names

def custom_sort_key(s):
    parts = re.split(r"(\d+)", s)
    return [int(p) if p.isdigit() else p for p in parts]

def change_choices():
    s_names, g_names = get_weights_names()
    return (
        {"choices": sorted(s_names, key=custom_sort_key), "__type__": "update"},
        {"choices": sorted(g_names, key=custom_sort_key), "__type__": "update"},
    )

# ---------- 主合成：返回落地文件路径 ----------
def get_tts_wav(
    ref_wav_path, prompt_text, prompt_language, text, text_language,
    how_to_cut=i18n("不切"), top_k=20, top_p=0.6, temperature=0.6, ref_free=False
):
    if prompt_text is None or len(prompt_text) == 0:
        ref_free = True

    t0 = ttime()
    prompt_language = dict_language[prompt_language]
    text_language = dict_language[text_language]

    if not ref_free:
        prompt_text = prompt_text.strip("\n")
        if len(prompt_text) > 0 and (prompt_text[-1] not in splits):
            prompt_text += "。" if prompt_language != "en" else "."
        print(i18n("实际输入的参考文本:"), prompt_text)

    text = text.strip("\n")
    if len(text) > 0 and (text[0] not in splits and len(get_first(text)) < 4):
        text = "。" + text if text_language != "en" else "." + text
    print(i18n("实际输入的目标文本:"), text)

    wav16k, _ = librosa.load(ref_wav_path, sr=16000)
    if wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000:
        raise OSError(i18n("参考音频在3~10秒范围外，请更换！"))
    wav16k = ms.Tensor.from_numpy(wav16k)

    zero_wav = np.zeros(int(hps.data.sampling_rate * 0.3),
                        dtype=np.float16 if is_half else np.float32)
    zero_wav_ms = ms.Tensor.from_numpy(zero_wav)
    if is_half:
        wav16k = wav16k.half()
        zero_wav_ms = zero_wav_ms.half()
    wav16k = ops.cat([wav16k, zero_wav_ms])

    ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].swapaxes(1, 2)
    codes = vq_model.extract_latent(ssl_content)
    prompt_semantic = codes[0, 0]
    t1 = ttime()

    if how_to_cut == i18n("凑四句一切"):
        text = cut1(text)
    elif how_to_cut == i18n("凑50字一切"):
        text = cut2(text)
    elif how_to_cut == i18n("按中文句号。切"):
        text = cut3(text)
    elif how_to_cut == i18n("按英文句号.切"):
        text = cut4(text)
    elif how_to_cut == i18n("按标点符号切"):
        text = cut5(text)
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")

    print(i18n("实际输入的目标文本(切句后):"), text)
    texts = merge_short_text_in_array(text.split("\n"), 5)

    audio_opt = []
    if not ref_free:
        phones1, bert1, _ = get_phones_and_bert(prompt_text, prompt_language)

    for seg in texts:
        if len(seg.strip()) == 0:
            continue
        if seg[-1] not in splits:
            seg += "。" if text_language != "en" else "."
        print(i18n("实际输入的目标文本(每句):"), seg)
        phones2, bert2, norm_text2 = get_phones_and_bert(seg, text_language)
        print(i18n("前端处理后的文本(每句):"), norm_text2)

        if not ref_free:
            bert = ops.cat([bert1, bert2], 1)
            all_phoneme_ids = ms.Tensor(phones1 + phones2).unsqueeze(0)
        else:
            bert = bert2
            all_phoneme_ids = ms.Tensor(phones2).unsqueeze(0)

        bert = bert.unsqueeze(0)
        all_phoneme_len = ms.Tensor([all_phoneme_ids.shape[-1]])
        prompt = prompt_semantic.unsqueeze(0)
        t2 = ttime()

        pred_semantic, idx = t2s_model.infer_panel(
            all_phoneme_ids,
            all_phoneme_len,
            None if ref_free else prompt,
            bert,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            early_stop_num=hz * max_sec,
        )
        t3 = ttime()

        pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
        refer = get_spepc(hps, ref_wav_path)
        if is_half:
            refer = refer.half()

        audio = vq_model.decode(
            pred_semantic, ms.Tensor(phones2).unsqueeze(0), refer
        ).asnumpy()[0, 0]

        max_audio = np.abs(audio).max()
        if max_audio > 1:
            audio /= max_audio

        audio_opt.append(audio)
        audio_opt.append(zero_wav)
        t4 = ttime()
        print("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t3 - t2, t4 - t3))

    # 保存到 ./output/
    import soundfile as sf
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = OUT_DIR / f"output_{ts}.wav"
    final_audio = (np.concatenate(audio_opt, 0) * 32768).astype(np.int16)
    sf.write(str(out_path), final_audio, hps.data.sampling_rate)
    print(f"[Saved] 合成音频已保存到: {out_path}")

    return str(out_path)

# ---------- 包装生成器：先提示“正在合成”，完成后返回路径 ----------
def synthesize_with_waiting(ref_wav_path, prompt_text, prompt_language, text, text_language,
                            how_to_cut, top_k, top_p, temperature, ref_free):
    """
    用显式的 gr.update 来更新前端，兼容 gradio 3.38 的多输出生成器。
    第一次 yield 先把状态改为“正在合成…”，清空音频与下载；
    完成后第二次 yield 设置下载文件路径，同时更新状态（不自动播放）。
    """
    try:
        # 第一次：显示等待，清空播放器与下载
        yield gr.update(value=None), gr.update(value="⏳ 正在合成，请稍候..."), gr.update(value=None)

        # 启动后台线程执行合成，前端周期性轮询状态
        result = {"path": None, "error": None}

        def _worker():
            try:
                p = get_tts_wav(
                    ref_wav_path, prompt_text, prompt_language, text, text_language,
                    how_to_cut, top_k, top_p, temperature, ref_free
                )
                result["path"] = p
            except Exception as ex:
                result["error"] = str(ex)

        import threading
        th = threading.Thread(target=_worker, daemon=True)
        th.start()

        start_ts = time.time()
        # 轮询直到合成完成或报错，每0.5秒刷新一次状态
        while result["path"] is None and result["error"] is None:
            elapsed = time.time() - start_ts
            yield gr.update(value=None), gr.update(value=f"⏳ 正在合成（{elapsed:.1f}s）…"), gr.update(value=None)
            time.sleep(0.5)

        if result["error"] is not None:
            raise RuntimeError(result["error"])

        path = result["path"]
        # 合成完成：读取为numpy给播放器（不自动播放）+ 提供下载文件
        audio_np, sr = sf.read(path, dtype="float32", always_2d=False)
        if hasattr(audio_np, "ndim") and audio_np.ndim == 2 and audio_np.shape[1] == 1:
            audio_np = audio_np[:, 0]
        # 为避免 gradio 的 float32->int16 警告，这里转换为 int16
        if audio_np.dtype != np.int16:
            audio_np = np.clip(audio_np, -1.0, 1.0)
            audio_np = (audio_np * 32767.0).astype(np.int16)
        yield (sr, audio_np), gr.update(value=f"✅ 合成完成：{Path(path).name}"), gr.update(value=path)
    except Exception as e:
        # 失败：清空播放器与下载，显示错误
        yield gr.update(value=None), gr.update(value=f"❌ 合成失败：{e}"), gr.update(value=None)


# ---------- UI ----------
SoVITS_names, GPT_names = get_weights_names()

with gr.Blocks(title="GPT-SoVITS WebUI") as app:
    gr.Markdown(
        value=i18n("本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责. "
                   "如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录<b>LICENSE</b>.")
    )
    with gr.Group():
        gr.Markdown(value=i18n("模型切换"))
        with gr.Row():
            GPT_dropdown = gr.Dropdown(label=i18n("GPT模型列表"),
                                       choices=sorted(GPT_names, key=custom_sort_key),
                                       value=gpt_path, interactive=True)
            SoVITS_dropdown = gr.Dropdown(label=i18n("SoVITS模型列表"),
                                          choices=sorted(SoVITS_names, key=custom_sort_key),
                                          value=sovits_path, interactive=True)
            refresh_button = gr.Button(i18n("刷新模型路径"), variant="primary")
            refresh_button.click(fn=change_choices, inputs=[], outputs=[SoVITS_dropdown, GPT_dropdown])
            SoVITS_dropdown.change(change_sovits_weights, [SoVITS_dropdown], [])
            GPT_dropdown.change(change_gpt_weights, [GPT_dropdown], [])

        gr.Markdown(value=i18n("*请上传并填写参考信息"))
        with gr.Row():
            inp_ref = gr.Audio(label=i18n("请上传3~10秒内参考音频，超过会报错！"), type="filepath")
            with gr.Column():
                ref_text_free = gr.Checkbox(label=i18n("开启无参考文本模式。不填参考文本亦相当于开启。"),
                                            value=False, interactive=True, show_label=True)
                gr.Markdown(i18n("使用无参考文本模式时建议使用微调的GPT，听不清参考音频说的啥可以开，"
                                 "开启后无视填写的参考文本。"))
                prompt_text = gr.Textbox(label=i18n("参考音频的文本"), value="")
            prompt_language = gr.Dropdown(
                label=i18n("参考音频的语种"),
                choices=[i18n("中文"), i18n("英文"), i18n("日文"), i18n("中英混合"), i18n("日英混合"), i18n("多语种混合")],
                value=i18n("中文")
            )

        gr.Markdown(value=i18n("*请填写需要合成的目标文本和语种模式"))
        with gr.Row():
            text = gr.Textbox(label=i18n("需要合成的文本"), value="")
            text_language = gr.Dropdown(
                label=i18n("需要合成的语种"),
                choices=[i18n("中文"), i18n("英文"), i18n("日文"), i18n("中英混合"), i18n("日英混合"), i18n("多语种混合")],
                value=i18n("中文")
            )
            how_to_cut = gr.Radio(
                label=i18n("怎么切"),
                choices=[i18n("不切"), i18n("凑四句一切"), i18n("凑50字一切"),
                         i18n("按中文句号。切"), i18n("按英文句号.切"), i18n("按标点符号切")],
                value=i18n("凑四句一切"), interactive=True
            )
            with gr.Row():
                gr.Markdown(value=i18n("gpt采样参数(无参考文本时不要太低)："))
                top_k = gr.Slider(minimum=1, maximum=100, step=1, label=i18n("top_k"), value=5, interactive=True)
                top_p = gr.Slider(minimum=0, maximum=1, step=0.05, label=i18n("top_p"), value=1, interactive=True)
                temperature = gr.Slider(minimum=0, maximum=1, step=0.05,
                                        label=i18n("temperature"), value=1, interactive=True)

            inference_button = gr.Button(i18n("合成语音"), variant="primary")

            # 输出区域：状态在上、播放器在中、下载在下（纵向一列）
            with gr.Column():
                status_box = gr.Markdown(value="")
                output = gr.Audio(label=i18n("输出的语音"), type="numpy", autoplay=False)
                download_file = gr.File(label=i18n("下载合成音频"))

        inference_button.click(
            fn=synthesize_with_waiting,
            inputs=[inp_ref, prompt_text, prompt_language, text, text_language,
                    how_to_cut, top_k, top_p, temperature, ref_text_free],
            outputs=[output, status_box, download_file],
        )

        gr.Markdown(value=i18n("文本切分工具。太长的文本合成出来效果不一定好，所以太长建议先切。"
                               "合成会根据文本的换行分开合成再拼起来。"))
        with gr.Row():
            text_inp = gr.Textbox(label=i18n("需要合成的切分前文本"), value="")
            button1 = gr.Button(i18n("凑四句一切"), variant="primary")
            button2 = gr.Button(i18n("凑50字一切"), variant="primary")
            button3 = gr.Button(i18n("按中文句号。切"), variant="primary")
            button4 = gr.Button(i18n("按英文句号.切"), variant="primary")
            button5 = gr.Button(i18n("按标点符号切"), variant="primary")
            text_opt = gr.Textbox(label=i18n("切分后文本"), value="")
            button1.click(cut1, [text_inp], [text_opt])
            button2.click(cut2, [text_inp], [text_opt])
            button3.click(cut3, [text_inp], [text_opt])
            button4.click(cut4, [text_inp], [text_opt])
            button5.click(cut5, [text_inp], [text_opt])

    gr.Markdown(value=i18n("后续将支持转音素、手工修改音素、语音合成分步执行。"))

app.queue(concurrency_count=511, max_size=1022).launch(
    server_name="0.0.0.0",
    inbrowser=True,
    share=is_share,
    server_port=infer_ttswebui,
    quiet=True,
)
