# -*- coding: utf-8 -*-
"""
MindSpore 版 GPT-SoVITS 推理 API

- 依赖与实现参考 `GPT_SoVITS/inference_webui.py`，封装为 FastAPI。
- 主要端点：
  - POST /synthesize 进行语音合成并返回 WAV 音频流
  - POST /set_model 运行期切换 SoVITS/GPT 权重
  - GET  /healthz   健康检查

使用示例：
  uvicorn api_ms:app --host 0.0.0.0 --port 9881

请求示例（POST /synthesize）：
{
  "ref_wav_path": "./ref.wav",
  "prompt_text": "参考文本",
  "prompt_language": "中文",
  "text": "需要合成的文本。",
  "text_language": "中文",
  "how_to_cut": "凑四句一切",  # 可选：不切/凑四句一切/凑50字一切/按中文句号。切/按英文句号.切/按标点符号切
  "top_k": 5,
  "top_p": 1,
  "temperature": 1,
  "ref_free": false
}
"""

import os, re, json, logging, shutil, time, fcntl, sys, threading, configparser, datetime
from logging.handlers import TimedRotatingFileHandler
from io import BytesIO
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager
from urllib.parse import quote_plus, unquote_plus

import redis

# =========== 命令行参数预解析（需早于 MindSpore 与模型初始化） ===========
def _apply_cli_overrides():
    """
    目的：
    - 在模块导入阶段尽早读取命令行参数，设置显卡与端口/主机等环境变量，
      确保 MindSpore 与权重加载使用到正确的设备上下文。
    支持参数：
    - --device <idx>                   指定使用的 NPU(Ascend) 编号（等效于设置 infer_device_id）
    - --port <port>                    指定服务端口（等效设置 infer_api_port）
    - --host <host>                    指定服务绑定地址（设置 infer_api_host，默认 0.0.0.0）
    """
    try:
        argv = sys.argv[1:]
        def _get_flag_value(flags):
            for i, a in enumerate(argv):
                if a in flags and i + 1 < len(argv):
                    return argv[i + 1]
                # 支持 --flag=value 形式
                for f in flags:
                    if a.startswith(f + "="):
                        return a.split("=", 1)[1]
            return None
        dev = _get_flag_value(["--device"])
        if dev is not None and dev.strip() != "":
            os.environ["infer_device_id"] = dev.strip()
            # Ascend 常用环境变量，部分运行时会读取
            os.environ["ASCEND_DEVICE_ID"] = dev.strip()
            os.environ["DEVICE_ID"] = dev.strip()
        port = _get_flag_value(["--port"])
        if port is not None and port.strip() != "":
            os.environ["infer_api_port"] = port.strip()
        host = _get_flag_value(["--host"])
        if host is not None and host.strip() != "":
            os.environ["infer_api_host"] = host.strip()
    except Exception:
        # 任何解析异常均忽略，按默认值运行
        pass

_apply_cli_overrides()

def _ensure_nltk_data():
    """
    确认 root 根目录下存在 nltk_data 目录，避免运行时缺包。
    允许通过 NLTK_DATA/NLTK_DATA_HOME 环境变量自定义路径。
    """
    candidates: list[Path] = []
    for env_key in ("NLTK_DATA_HOME", "NLTK_DATA"):
        env_val = os.environ.get(env_key)
        if env_val:
            candidates.append(Path(env_val))
    try:
        candidates.append(Path.home() / "nltk_data")
    except Exception:
        pass
    candidates.append(Path("/root/nltk_data"))

    for path in candidates:
        if path and path.exists() and path.is_dir():
            os.environ["NLTK_DATA"] = str(path)
            print(f"[NLTK] nltk_data 目录已确认: {path}")
            return
    raise RuntimeError("未找到 nltk_data 目录，请在 root 根目录下创建（例如 /root/nltk_data）。")


_ensure_nltk_data()


PROJECT_ROOT = Path(__file__).resolve().parent
AUDIO_ROOT = PROJECT_ROOT / "audio"
AUDIO_ROOT.mkdir(exist_ok=True)
STORE_PATH = PROJECT_ROOT / "voice_models.json"
if not STORE_PATH.exists():
    STORE_PATH.write_text("[]", encoding="utf-8")

VOICE_CATEGORY_STORE = PROJECT_ROOT / "voice_categories.json"
if not VOICE_CATEGORY_STORE.exists():
    VOICE_CATEGORY_STORE.write_text("[]", encoding="utf-8")
OUT_DIR = PROJECT_ROOT / "output"
OUT_DIR.mkdir(exist_ok=True)
LOG_DIR = PROJECT_ROOT / "Logs"
LOG_DIR.mkdir(exist_ok=True)
VOICE_TASK_STORE = PROJECT_ROOT / "voice_task.json"
if not VOICE_TASK_STORE.exists():
    VOICE_TASK_STORE.write_text("[]", encoding="utf-8")

CONFIG_PATH = PROJECT_ROOT / "config.ini"
_config = configparser.ConfigParser()
if CONFIG_PATH.exists():
    try:
        _config.read(CONFIG_PATH, encoding="utf-8")
    except Exception:
        logging.getLogger("api_ms").warning("读取 config.ini 失败", exc_info=True)


def _cfg(section: str, option: str, fallback: str) -> str:
    try:
        if _config.has_option(section, option):
            return _config.get(section, option)
    except Exception:
        pass
    return fallback


def _parse_duration(raw: str, fallback: int) -> int:
    if not raw:
        return fallback
    raw = raw.strip().lower()
    try:
        return int(raw)
    except ValueError:
        units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
        num = raw[:-1]
        unit = raw[-1]
        if unit in units:
            try:
                return int(float(num) * units[unit])
            except ValueError:
                return fallback
    return fallback


REDIS_HOST = os.environ.get("REDIS_HOST", _cfg("redis", "host", "127.0.0.1"))
REDIS_PORT = int(os.environ.get("REDIS_PORT", _cfg("redis", "port", "6379")))
REDIS_DB = int(os.environ.get("REDIS_DB", _cfg("redis", "db", "0")))

default_retention = _cfg("cleanup", "retention", "604800")
TASK_RETENTION_SECONDS = _parse_duration(os.environ.get("TASK_RETENTION_SECONDS", default_retention), 604800)
TASK_CLEAN_INTERVAL_SECONDS = max(30, min(300, TASK_RETENTION_SECONDS // 10))

redis_client: Optional[redis.Redis] = None


import numpy as np
import librosa
import soundfile as sf

import mindspore as ms
from mindspore import ops
from time import time as ttime

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse, Response, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

# =========== 日志与依赖降噪 ==========
logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)
logger = logging.getLogger("uvicorn")

# =========== 统一响应结构 ===========
def _resp(code: int, message: str = "", data: dict | None = None):
    return JSONResponse({"code": code, "message": message, "data": (data if data is not None else {})}, status_code=200)

def _ok(data: dict | None = None, message: str = ""):
    return _resp(0, message, data)

def _err_failure(message: str):
    return _resp(1, message, {})

def _err_resource(message: str):
    return _resp(2, message, {})

def _err_param(message: str):
    return _resp(3, message, {})

def _page_ok(items: list, total: int, page: int, page_size: int, message: str = ""):
    return JSONResponse({
        "code": 0,
        "message": message,
        "data": {
            "total": total,
            "pageIndex": page,
            "pageSize": page_size,
            "data": items
        }
    }, status_code=200)

# =========== 依赖模块 ===========
from mindnlp.transformers import AutoModelForMaskedLM, AutoTokenizer
from mindnlp.injection import set_global_fp16

# 兼容在不同工作目录下运行，确保能找到 feature_extractor 等本地模块
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.append(_THIS_DIR)
_SUBDIR = os.path.join(_THIS_DIR, "GPT_SoVITS")
if _SUBDIR not in sys.path:
    sys.path.append(_SUBDIR)

from feature_extractor import cnhubert
from module.models import SynthesizerTrn
from AR.models.t2s_model import Text2SemanticDecoder
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from module.mel_processing import spectrogram_torch
from my_utils import load_audio
import LangSegment


# =========== 环境参数 ===========
if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]

is_half = eval(os.environ.get("is_half", "True"))
api_port = int(os.environ.get("infer_api_port", 9881))
api_host = os.environ.get("infer_api_host", "0.0.0.0")

# MindSpore 设备上下文（固定 Ascend，仅通过 --device 选择 NPU 编号）
device_id_env = os.environ.get("infer_device_id")

try:
    ms.set_context(device_target="Ascend", device_id=int(device_id_env or 0))
    try:
        _ctx_dev = ms.get_context("device_id")
        _ctx_tgt = ms.get_context("device_target")
        print(f"[Device] MindSpore context set: device_target={_ctx_tgt}, device_id={_ctx_dev}")
    except Exception:
        pass
except Exception:
    # 若未使用 GPU 或环境不支持，则保持 MindSpore 默认上下文
    pass

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
    sovits_path = os.environ.get("sovits_path", "GPT_SoVITS/pretrained_models/s2G488k-ms.ckpt")

cnhubert_base_path = os.environ.get("cnhubert_base_path", "GPT_SoVITS/pretrained_models/chinese-hubert-base")
bert_path = os.environ.get("bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large")


# =========== BERT 初始化 ===========
tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path).to_float(ms.float16)
if is_half:
    bert_model = bert_model.half()
    set_global_fp16(True)

# =========== HuBERT 初始化 ===========
cnhubert.cnhubert_base_path = cnhubert_base_path
ssl_model = cnhubert.get_model().to_float(ms.float16)
ssl_model = ssl_model.half() if is_half else ssl_model


# =========== 工具类 ===========
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


# =========== SoVITS / GPT 加载 ===========
def change_sovits_weights(sovits_path_in: str):
    global vq_model, hps
    dict_s2 = ms.load_checkpoint(sovits_path_in)
    hps = json.loads(dict_s2["config"])  # config 存成 json 字符串
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
    ).to_float(ms.float16)
    # 兼容某些 ckpt 无 enc_q
    try:
        vq_model.quantizer.vq.layers[0]._codebook.inited = True
        vq_model.update_parameters_name()
    except Exception:
        pass
    if "pretrained" not in sovits_path_in:
        try:
            del vq_model.enc_q
        except Exception:
            pass
    vq_model = vq_model.half() if is_half else vq_model
    vq_model.set_train(False)
    ms.load_param_into_net(vq_model, dict_s2)
    with open("./sweight.txt", "w", encoding="utf-8") as f:
        f.write(sovits_path_in)
    logger.info(f"[SoVITS] loaded: {sovits_path_in}")


def change_gpt_weights(gpt_path_in: str):
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
    with open("./gweight.txt", "w", encoding="utf-8") as f:
        f.write(gpt_path_in)
    logger.info(f"[GPT] loaded: {gpt_path_in}")


change_sovits_weights(sovits_path)
change_gpt_weights(gpt_path)


# =========== 语音前处理 ===========
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
    "中文": "all_zh",
    "英文": "en",
    "日文": "all_ja",
    "中英混合": "zh",
    "日英混合": "ja",
    "多语种混合": "auto",
    "all_zh": "all_zh",
    "en": "en",
    "all_ja": "all_ja",
    "zh": "zh",
    "ja": "ja",
    "auto": "auto",
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


splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…"}


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


def split_text_to_sentences(todo_text: str):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if len(todo_text) == 0:
        return []
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
    inps = split_text_to_sentences(inp)
    idxs = list(range(0, len(inps), 4))
    if not idxs:
        return inp
    idxs[-1] = None
    if len(idxs) > 1:
        out = []
        for k in range(len(idxs) - 1):
            out.append("".join(inps[idxs[k]:idxs[k + 1]]))
        return "\n".join(out)
    return inp


def cut2(inp):
    inp = inp.strip("\n")
    inps = split_text_to_sentences(inp)
    if len(inps) < 2:
        return inp
    out, s, buf = [], 0, ""
    for seg in inps:
        s += len(seg)
        buf += seg
        if s > 50:
            s = 0
            out.append(buf)
            buf = ""
    if buf:
        out.append(buf)
    if len(out) > 1 and len(out[-1]) < 50:
        out[-2] = out[-2] + out[-1]
        out = out[:-1]
    return "\n".join(out)


def cut3(inp):
    return "\n".join(["%s" % it for it in inp.strip("\n").strip("。").split("。")])


def cut4(inp):
    return "\n".join(["%s" % it for it in inp.strip("\n").strip(".").split(".")])


def cut5(inp):
    inp = inp.strip("\n")
    p = r'[,.;?!、，。？！;：…]'
    items = re.split(f'({p})', inp)
    merged = ["".join(g) for g in zip(items[::2], items[1::2])]
    if len(items) % 2 == 1:
        merged.append(items[-1])
    return "\n".join(merged)


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
    else:
        raise ValueError(f"Unsupported language: {language}")
    return phones, bert.to(dtype), norm_text


def get_first(text: str) -> str:
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    return re.split(pattern, text)[0].strip()


def build_wav_bytes(int16_audio: np.ndarray, sample_rate: int) -> bytes:
    wav_io = BytesIO()
    sf.write(wav_io, int16_audio, sample_rate, format='wav')
    return wav_io.getvalue()


def synthesize_once(
    ref_wav_path: str,
    prompt_text: Optional[str],
    prompt_language: str,
    text: str,
    text_language: str,
    how_to_cut: str = "凑四句一切",
    top_k: int = 5,
    top_p: float = 1.0,
    temperature: float = 1.0,
    ref_free: bool = False,
    save_user_id: Optional[str] = None,
    save_task_id: Optional[str] = None,
) -> bytes:
    if not os.path.exists(ref_wav_path):
        raise FileNotFoundError(f"ref_wav_path not found: {ref_wav_path}")

    prompt_language = dict_language[prompt_language]
    text_language = dict_language[text_language]

    if prompt_text is None or len(prompt_text) == 0:
        ref_free = True

    # 参考音频：检查 3~10 秒
    wav16k, _ = librosa.load(ref_wav_path, sr=16000)
    if wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000:
        raise OSError("参考音频需在 3~10 秒范围内")
    wav16k = ms.Tensor.from_numpy(wav16k)

    zero_wav = np.zeros(int(hps.data.sampling_rate * 0.3), dtype=np.float16 if is_half else np.float32)
    zero_wav_ms = ms.Tensor.from_numpy(zero_wav)
    if is_half:
        wav16k = wav16k.half()
        zero_wav_ms = zero_wav_ms.half()
    wav16k = ops.cat([wav16k, zero_wav_ms])

    # 打印与调整参考文本
    if not ref_free:
        prompt_text = prompt_text.strip("\n")
        if len(prompt_text) > 0 and (prompt_text[-1] not in splits):
            prompt_text += "。" if prompt_language != "en" else "."
        logger.info("实际输入的参考文本: %s", prompt_text)
        print("实际输入的参考文本:", prompt_text)

    # 打印与调整目标文本（前置符号优化）
    text = text.strip("\n")
    if len(text) > 0 and (text[0] not in splits and len(get_first(text)) < 4):
        text = ("。" if text_language != "en" else ".") + text
    logger.info("实际输入的目标文本: %s", text)
    print("实际输入的目标文本:", text)

    # 提取参考语义
    t0 = ttime()
    ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].swapaxes(1, 2)
    codes = vq_model.extract_latent(ssl_content)
    prompt_semantic = codes[0, 0]
    t1 = ttime()

    # 切句
    if how_to_cut == "凑四句一切":
        text = cut1(text)
    elif how_to_cut == "凑50字一切":
        text = cut2(text)
    elif how_to_cut == "按中文句号。切":
        text = cut3(text)
    elif how_to_cut == "按英文句号.切":
        text = cut4(text)
    elif how_to_cut == "按标点符号切":
        text = cut5(text)
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")
    logger.info("实际输入的目标文本(切句后): %s", text)
    print("实际输入的目标文本(切句后):", text)
    texts = merge_short_text_in_array(text.split("\n"), 5)
    logger.info("分段总数: %d", len(texts))
    print("分段总数:", len(texts))

    audio_opt = []
    if not ref_free:
        phones1, bert1, _ = get_phones_and_bert(prompt_text, prompt_language)

    seg_idx = 0
    total_segments = max(1, len(texts))
    for seg in texts:
        seg = seg.strip()
        if len(seg) == 0:
            continue
        if seg[-1] not in splits:
            seg += "。" if text_language != "en" else "."
        logger.info("实际输入的目标文本(每句): %s", seg)
        print("实际输入的目标文本(每句):", seg)
        phones2, bert2, norm_text2 = get_phones_and_bert(seg, text_language)
        logger.info("前端处理后的文本(每句): %s", norm_text2)
        print("前端处理后的文本(每句):", norm_text2)

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
        t4 = ttime()

        max_audio = np.abs(audio).max()
        if max_audio > 1:
            audio /= max_audio

        audio_opt.append(audio)
        audio_opt.append(zero_wav)
        logger.info("%.3f\t%.3f\t%.3f\t%.3f", t1 - t0, t2 - t1, t3 - t2, t4 - t3)
        print(f"{t1 - t0:.3f}\t{t2 - t1:.3f}\t{t3 - t2:.3f}\t{t4 - t3:.3f}")
        seg_idx += 1
        logger.info("进度: %d/%d (%.0f%%)", seg_idx, total_segments, seg_idx * 100.0 / total_segments)
        print("进度:", f"{seg_idx}/{total_segments}", f"({seg_idx * 100.0 / total_segments:.0f}%")

    final_audio = (np.concatenate(audio_opt, 0) * 32768).astype(np.int16)
    # 落地保存到 ./output/<user_id>/<task_id>/<task_id>.wav
    def _sanitize(s: Optional[str]) -> str:
        s = str(s or "").strip()
        return re.sub(r"[^a-zA-Z0-9_-]", "_", s) or "anonymous"
    uid = _sanitize(save_user_id) if save_user_id is not None else "anonymous"
    tid = _sanitize(save_task_id) if save_task_id is not None else datetime.now().strftime("T%Y%m%d%H%M%S")
    user_dir = OUT_DIR / uid / tid
    user_dir.mkdir(parents=True, exist_ok=True)
    out_path = user_dir / f"{tid}.wav"
    sf.write(str(out_path), final_audio, hps.data.sampling_rate)
    logger.info("[Saved] 合成音频已保存到: %s", out_path)
    print("[Saved] 合成音频已保存到:", str(out_path))
    return build_wav_bytes(final_audio, hps.data.sampling_rate)


# =========== 任务队列与进度 ==========
_task_queue_lock = threading.Lock()
_task_queue = []  # type: list[str]
_worker_started = False
_worker_start_lock = threading.Lock()  # 防止并发启动多个worker
_task_cancel_flags = {}  # task_id -> threading.Event
_single_worker_lock = threading.Lock()  # 兜底：即使误启多个线程，也保证同一时刻只跑一个任务


def _enqueue_task(task_id: str):
    with _task_queue_lock:
        _task_queue.append(task_id)


def _dequeue_task():
    with _task_queue_lock:
        if _task_queue:
            return _task_queue.pop(0)
    return None


def _task_worker_loop():
    while True:
        task_id = _dequeue_task()
        if not task_id:
            time.sleep(0.2)
            continue
        try:
            # 兜底串行锁，避免误启动多个线程时并发执行
            with _single_worker_lock:
                _run_task(task_id)
        except Exception:
            logger.exception("task failed: %s", task_id)
            # 任务日志记录异常
            try:
                _tl = _ensure_task_logger(task_id)
                _tl.exception("failed")
            except Exception:
                pass
            # 将任务状态标记为 error 并归档
            try:
                task = _get_task_from_index(task_id)
                if task is not None:
                    task["status"] = "error"
                    task["updated_at"] = int(time.time())
                    _update_task_index(task)
            except Exception:
                pass


def _ensure_worker():
    global _worker_started
    # 使用启动锁，避免高并发下重复启动
    with _worker_start_lock:
        if not _worker_started:
            th = threading.Thread(target=_task_worker_loop, daemon=True)
            th.start()
            _worker_started = True


def _run_task(task_id: str):
    # 读取任务（索引）
    task = _get_task_from_index(task_id)
    if task is None:
        return

    # 任务日志
    tlog = _ensure_task_logger(task_id)
    try:
        tlog.info("start user_id=%s model_id=%s", task.get("user_id"), task.get("model_id"))
    except Exception:
        pass

    # 状态改为运行中
    task["status"] = "run"
    task["updated_at"] = int(time.time())
    _update_task_index(task)

    # 参数
    model_id = task.get("model_id")
    text = task.get("text", "")
    text_language = task.get("text_language", "中文")
    how_to_cut = task.get("how_to_cut", "凑四句一切")
    prompt_text = task.get("prompt_text", "")
    prompt_language = task.get("prompt_language", "中文")
    ref_wav_path = task.get("ref_wav_path")

    # 若有model_id，补全参考信息
    if model_id and not ref_wav_path:
        model = _find_model(model_id)
        if model:
            ref_wav_path = model.get("refer_wav_path")
            if not prompt_text:
                prompt_text = model.get("prompt_text", "")
            if not task.get("prompt_language") and model.get("prompt_language"):
                prompt_language = model.get("prompt_language")

    # 参考文本是否为空
    ref_free = (prompt_text is None or len(prompt_text) == 0)
    # 运行前检查是否被取消
    if _is_cancelled(task_id):
        _mark_cancelled_and_cleanup(task_id, task)
        return
    # 预处理文本并打印
    if not ref_free:
        pt = (prompt_text or "").strip("\n")
        if len(pt) > 0 and (pt[-1] not in splits):
            pt += "。" if prompt_language != "en" else "."
        logger.info("实际输入的参考文本: %s", pt)
        print("实际输入的参考文本:", pt)
        prompt_text = pt

    text_adj = text.strip("\n")
    if len(text_adj) > 0 and (text_adj[0] not in splits and len(get_first(text_adj)) < 4):
        text_adj = ("。" if text_language != "en" else ".") + text_adj
    logger.info("实际输入的目标文本: %s", text_adj)
    print("实际输入的目标文本:", text_adj)

    # 切分并统计段数
    if how_to_cut == "凑四句一切":
        text_cut = cut1(text_adj)
    elif how_to_cut == "凑50字一切":
        text_cut = cut2(text_adj)
    elif how_to_cut == "按中文句号。切":
        text_cut = cut3(text_adj)
    elif how_to_cut == "按英文句号.切":
        text_cut = cut4(text_adj)
    elif how_to_cut == "按标点符号切":
        text_cut = cut5(text_adj)
    else:
        text_cut = text_adj
    while "\n\n" in text_cut:
        text_cut = text_cut.replace("\n\n", "\n")
    segs = [s.strip() for s in text_cut.split("\n") if len(s.strip()) > 0]
    logger.info("实际输入的目标文本(切句后): %s", text_cut)
    print("实际输入的目标文本(切句后):", text_cut)
    total_segments = len(segs)
    print("分段总数:", total_segments)
    if total_segments == 0:
        raise RuntimeError("空文本")

    # 参考语义
    if _is_cancelled(task_id):
        _mark_cancelled_and_cleanup(task_id, task)
        try: tlog.info("cancelled before feature extraction")
        except Exception: pass
        return
    wav16k, _ = librosa.load(ref_wav_path, sr=16000)
    if wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000:
        raise OSError("参考音频需在 3~10 秒范围内")
    wav16k = ms.Tensor.from_numpy(wav16k)
    zero_wav = np.zeros(int(hps.data.sampling_rate * 0.3), dtype=np.float16 if is_half else np.float32)
    zero_wav_ms = ms.Tensor.from_numpy(zero_wav)
    if is_half:
        wav16k = wav16k.half(); zero_wav_ms = zero_wav_ms.half()
    wav16k = ops.cat([wav16k, zero_wav_ms])
    if _is_cancelled(task_id):
        _mark_cancelled_and_cleanup(task_id, task)
        try: tlog.info("cancelled after feature stacking")
        except Exception: pass
        return
    ssl_content = ssl_model.model(wav16k.unsqueeze(0))["last_hidden_state"].swapaxes(1, 2)
    codes = vq_model.extract_latent(ssl_content)
    prompt_semantic = codes[0, 0]
    if not ref_free:
        phones1, bert1, _ = get_phones_and_bert(prompt_text, dict_language[prompt_language])

    audio_opt = []
    segments_done = 0
    for seg in segs:
        if _is_cancelled(task_id):
            _mark_cancelled_and_cleanup(task_id, task)
            try: tlog.info("cancelled during segments: %d/%d", segments_done, total_segments)
            except Exception: pass
            return
        if seg[-1] not in splits:
            seg += "。" if text_language != "en" else "."
        logger.info("实际输入的目标文本(每句): %s", seg)
        print("实际输入的目标文本(每句):", seg)
        phones2, bert2, _norm = get_phones_and_bert(seg, dict_language[text_language])
        logger.info("前端处理后的文本(每句): %s", _norm)
        print("前端处理后的文本(每句):", _norm)
        if not ref_free:
            bert = ops.cat([bert1, bert2], 1)
            all_phoneme_ids = ms.Tensor(phones1 + phones2).unsqueeze(0)
        else:
            bert = bert2
            all_phoneme_ids = ms.Tensor(phones2).unsqueeze(0)
        bert = bert.unsqueeze(0)
        all_phoneme_len = ms.Tensor([all_phoneme_ids.shape[-1]])
        prompt = prompt_semantic.unsqueeze(0)
        # 取消检查（推理前）
        if _is_cancelled(task_id):
            _mark_cancelled_and_cleanup(task_id, task)
            try: tlog.info("cancelled before infer_panel")
            except Exception: pass
            return
        pred_semantic, idx = t2s_model.infer_panel(
            all_phoneme_ids, all_phoneme_len, None if ref_free else prompt, bert,
            top_k=task.get("top_k", 5), top_p=task.get("top_p", 1.0), temperature=task.get("temperature", 1.0),
            early_stop_num=hz * max_sec,
        )
        pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)
        refer = get_spepc(hps, ref_wav_path)
        if is_half:
            refer = refer.half()
        # 取消检查（解码前）
        if _is_cancelled(task_id):
            _mark_cancelled_and_cleanup(task_id, task)
            try: tlog.info("cancelled before decode")
            except Exception: pass
            return
        audio = vq_model.decode(pred_semantic, ms.Tensor(phones2).unsqueeze(0), refer).asnumpy()[0, 0]
        max_audio = np.abs(audio).max()
        if max_audio > 1:
            audio /= max_audio
        audio_opt.append(audio); audio_opt.append(zero_wav)
        # 打印分段耗时（和 webui 类似四段）无法逐段严格复现 t0-t4，这里仅打印一次推理耗时
        # 若需要完整四段计时，可在上方添加时间戳类似 synthesize_once

        # 更新进度
        segments_done += 1
        progress = int(segments_done * 100 / total_segments)
        task["segments_done"] = segments_done
        task["total_segments"] = total_segments
        task["progress"] = progress
        task["updated_at"] = int(time.time())
        _update_task_index(task)
        # 控制台同步打印分段百分比
        logger.info("进度: %d/%d (%d%%)", segments_done, total_segments, progress)
        print("进度:", f"{segments_done}/{total_segments}", f"({progress}% )")
        try: tlog.info("progress %d/%d (%d%%)", segments_done, total_segments, progress)
        except Exception: pass

    # 直接写到 output/<user_id>/<task_id>/<task_id>.wav
    def _sanitize(s: Optional[str]) -> str:
        s = str(s or "").strip()
        return re.sub(r"[^a-zA-Z0-9_-]", "_", s) or "anonymous"
    uid = _sanitize(task.get("user_id"))
    tid = _sanitize(task_id)
    user_dir = OUT_DIR / uid / tid
    user_dir.mkdir(parents=True, exist_ok=True)
    out_path = user_dir / f"{tid}.wav"
    final_audio = (np.concatenate(audio_opt, 0) * 32768).astype(np.int16)
    sf.write(str(out_path), final_audio, hps.data.sampling_rate)

    # 完成（仅更新索引）
    task["status"] = "done"
    task["result_path"] = str(out_path)
    task["result_url"] = f"/output/{uid}/{tid}/{tid}.wav"
    task["updated_at"] = int(time.time())
    _update_task_index(task)
    print("[Saved] 合成音频已保存到:", str(out_path))
    try: tlog.info("done result=%s", str(out_path))
    except Exception: pass


def _get_cancel_event(task_id: str) -> threading.Event:
    ev = _task_cancel_flags.get(task_id)
    if ev is None:
        ev = threading.Event()
        _task_cancel_flags[task_id] = ev
    return ev


def _mark_cancel(task_id: str):
    _get_cancel_event(task_id).set()


def _is_cancelled(task_id: str) -> bool:
    ev = _task_cancel_flags.get(task_id)
    return ev.is_set() if ev else False


def _remove_from_index(task_id: str) -> Optional[dict]:
    items = _read_task_store()
    removed = None
    kept = []
    for it in items:
        if it.get("task_id") == task_id:
            removed = it
        else:
            kept.append(it)
    if removed is not None:
        _write_task_store(kept)
    return removed


def _mark_cancelled_and_cleanup(task_id: str, task: dict):
    task["status"] = "error"
    task["updated_at"] = int(time.time())
    _update_task_index(task)


@asynccontextmanager
async def _lifespan(app):
    redis_ready = _init_redis()
    if redis_ready:
        _redis_sync_local_tasks()
    _start_cleanup_thread()
    try:
        yield
    finally:
        _stop_cleanup_thread()


# =========== FastAPI ===========
# 文档分组与标题（用于 /docs 展示中文分组与名称）
_tags_metadata = [
    {"name": "音模管理", "description": "音模的新增、列表、删除等管理接口"},
    {"name": "任务队列", "description": "提交任务、查询任务、删除/取消任务等接口"},
    {"name": "页面", "description": "前端页面与重定向"},
    {"name": "其他", "description": "健康检查、内部使用接口等"},
]
app = FastAPI(
    title="GPT-SoVITS 推理服务",
    description="基于 MindSpore 的 GPT-SoVITS 文本转语音服务。",
    version="0.1.0",
    openapi_tags=_tags_metadata,
    lifespan=_lifespan,
)

app.mount("/audio", StaticFiles(directory=str(AUDIO_ROOT)), name="audio")
# 提供输出目录下载
app.mount("/output", StaticFiles(directory=str(OUT_DIR)), name="output")


# ---------- JSON 存储（进程内外锁） ----------

# 访问日志（每6小时切割）。/voice-tasks 持续请求量大，不记录
def _setup_access_logger():
    access_logger = logging.getLogger("api_ms_access")
    access_logger.setLevel(logging.INFO)
    handler = TimedRotatingFileHandler(str(LOG_DIR / "access.log"), when="h", interval=6, backupCount=28, encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s\t%(message)s")
    handler.setFormatter(fmt)
    access_logger.addHandler(handler)
    return access_logger

_access_logger = _setup_access_logger()

_task_loggers = {}  # task_id -> logger

def _ensure_task_logger(task_id: str) -> logging.Logger:
    """
    为指定任务创建独立日志文件：
    Logs/YYYYMMDD/HH/<task_id>.log
    记录任务的开始、进度、完成与异常。
    """
    lg = _task_loggers.get(task_id)
    if lg is not None:
        return lg
    now = datetime.now()
    date_dir = LOG_DIR / now.strftime("%Y%m%d") / now.strftime("%H")
    date_dir.mkdir(parents=True, exist_ok=True)
    log_path = date_dir / f"{task_id}.log"
    lg = logging.getLogger(f"task.{task_id}")
    lg.setLevel(logging.INFO)
    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s\t%(levelname)s\t%(message)s"))
    # 避免重复添加 handler
    if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == fh.baseFilename for h in lg.handlers):
        lg.addHandler(fh)
    lg.propagate = False
    _task_loggers[task_id] = lg
    return lg

@app.middleware("http")
async def access_log_middleware(request: Request, call_next):
    path = request.url.path
    # 跳过 /voice-tasks 列表轮询接口
    if path.startswith("/voice-tasks") and request.method == "GET":
        return await call_next(request)
    start = time.time()
    response = await call_next(request)
    duration_ms = int((time.time() - start) * 1000)
    try:
        client = request.client.host if request.client else "-"
        _access_logger.info(f"{client}\t{request.method} {path}\t{response.status_code}\t{duration_ms}ms")
    except Exception:
        pass
    return response
def _read_store() -> list:
    with open(STORE_PATH, "r", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
        try:
            data = json.load(f)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return data if isinstance(data, list) else []


def _write_store(items: list) -> None:
    tmp_path = STORE_PATH.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            json.dump(items, f, ensure_ascii=False, indent=2)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    os.replace(tmp_path, STORE_PATH)


def _read_categories() -> list:
    with open(VOICE_CATEGORY_STORE, "r", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
        try:
            data = json.load(f)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return data if isinstance(data, list) else []


def _write_categories(items: list) -> None:
    tmp_path = VOICE_CATEGORY_STORE.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            json.dump(items, f, ensure_ascii=False, indent=2)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    os.replace(tmp_path, VOICE_CATEGORY_STORE)


def _read_task_store() -> list:
    with open(VOICE_TASK_STORE, "r", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
        try:
            data = json.load(f)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return data if isinstance(data, list) else []


def _write_task_store(items: list) -> None:
    tmp_path = VOICE_TASK_STORE.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            json.dump(items, f, ensure_ascii=False, indent=2)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    os.replace(tmp_path, VOICE_TASK_STORE)


def _update_task_index(entry: dict):
    items = _read_task_store()
    # 去重后插入顶部
    items = [it for it in items if it.get("task_id") != entry.get("task_id")]
    items.insert(0, entry)
    # 按 created_at 逆序
    items = sorted(items, key=lambda x: x.get("created_at", 0), reverse=True)
    _write_task_store(items)
    _redis_upsert_task(entry)

def _get_task_from_index(task_id: str) -> Optional[dict]:
    items = _read_task_store()
    for it in items:
        if it.get("task_id") == task_id:
            return it
    return None


def _sanitize_model_id(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", model_id.strip())


def _normalize_category_id(cat_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", str(cat_id or "").strip())


def _find_model(model_id: str) -> Optional[dict]:
    items = _read_store()
    for it in items:
        if it.get("model_id") == model_id:
            return it
    return None


def _sanitize_fs_name(value: Optional[str]) -> str:
    s = str(value or "").strip()
    return re.sub(r"[^a-zA-Z0-9_-]", "_", s) or "anonymous"


def _clean_category_ids(cat_ids: list[str]) -> list[str]:
    cleaned = []
    for cid in cat_ids:
        norm = _normalize_category_id(cid)
        if norm and norm not in cleaned:
            cleaned.append(norm)
    return cleaned


def _ensure_categories_exist(cat_ids: list[str]) -> tuple[bool, list[str]]:
    if not cat_ids:
        return True, []
    existing = {c.get("category_id"): c for c in _read_categories()}
    missing = [cid for cid in cat_ids if cid not in existing]
    return len(missing) == 0, missing


# =========== Redis 与任务同步辅助 ===========
TASK_HASH_PREFIX = "tts:task"
TASK_INDEX_KEY = "tts:tasks:index"
USER_INDEX_PREFIX = "tts:user"
TASK_LOOKUP_PREFIX = "tts:task_lookup"


def _init_redis() -> bool:
    global redis_client
    if redis_client is not None:
        return True
    try:
        client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
        client.ping()
        redis_client = client
        logging.getLogger("api_ms").info("Redis 已连接：%s:%s/%s", REDIS_HOST, REDIS_PORT, REDIS_DB)
        return True
    except Exception as exc:
        logging.getLogger("api_ms").warning("Redis 不可用，继续使用本地存储：%s", exc)
        redis_client = None
        return False


def _get_redis_client() -> Optional[redis.Redis]:
    if redis_client is None:
        if not _init_redis():
            return None
    return redis_client


def _redis_quote(value: str) -> str:
    return quote_plus(str(value or ""))


def _redis_unquote(value: str) -> str:
    return unquote_plus(value or "")


def _redis_hash_key(user_id: str, task_id: str) -> str:
    return f"{TASK_HASH_PREFIX}:{_redis_quote(user_id)}:{_redis_quote(task_id)}"


def _redis_user_index_key(user_id: str) -> str:
    return f"{USER_INDEX_PREFIX}:{_redis_quote(user_id)}:index"


def _redis_lookup_key(task_id: str) -> str:
    return f"{TASK_LOOKUP_PREFIX}:{_redis_quote(task_id)}"


def _redis_member(user_id: str, task_id: str) -> str:
    return f"{_redis_quote(user_id)}:{_redis_quote(task_id)}"


def _redis_member_parse(member: str) -> tuple[str, str]:
    parts = member.split(":", 1)
    if len(parts) == 2:
        return _redis_unquote(parts[0]), _redis_unquote(parts[1])
    return _redis_unquote(member), ""


def _redis_upsert_task(entry: dict):
    client = _get_redis_client()
    if client is None:
        return
    try:
        user_id = str(entry.get("user_id", "")).strip()
        task_id = str(entry.get("task_id", "")).strip()
        if not user_id or not task_id:
            return
        updated_at = int(entry.get("updated_at", int(time.time())))
        payload = json.dumps(entry, ensure_ascii=False)
        mapping = {
            "task_id": task_id,
            "user_id": user_id,
            "status": entry.get("status", ""),
            "progress": str(entry.get("progress", 0)),
            "segments_done": str(entry.get("segments_done", 0)),
            "total_segments": str(entry.get("total_segments", 0)),
            "result_url": entry.get("result_url", "") or "",
            "error": entry.get("error", "") or "",
            "updated_at": str(updated_at),
            "created_at": str(entry.get("created_at", updated_at)),
            "payload": payload,
        }
        client.hset(_redis_hash_key(user_id, task_id), mapping=mapping)
        client.zadd(TASK_INDEX_KEY, {_redis_member(user_id, task_id): updated_at})
        client.zadd(_redis_user_index_key(user_id), {_redis_quote(task_id): updated_at})
        client.set(_redis_lookup_key(task_id), _redis_quote(user_id))
    except Exception:
        logging.getLogger("api_ms").warning("同步任务到 Redis 失败", exc_info=True)


def _redis_delete_task(user_id: Optional[str], task_id: str):
    client = _get_redis_client()
    if client is None:
        return
    try:
        if user_id:
            client.delete(_redis_hash_key(user_id, task_id))
            client.zrem(TASK_INDEX_KEY, _redis_member(user_id, task_id))
            client.zrem(_redis_user_index_key(user_id), _redis_quote(task_id))
        client.delete(_redis_lookup_key(task_id))
    except Exception:
        logging.getLogger("api_ms").warning("从 Redis 删除任务失败", exc_info=True)


def _redis_get_task(user_id: str, task_id: str) -> Optional[dict]:
    client = _get_redis_client()
    if client is None:
        return None
    try:
        data = client.hgetall(_redis_hash_key(user_id, task_id))
        if not data:
            return None
        payload = data.get("payload")
        if payload:
            try:
                return json.loads(payload)
            except Exception:
                pass
        result = {k: data.get(k) for k in data}
        result["task_id"] = result.get("task_id") or task_id
        result["user_id"] = result.get("user_id") or user_id
        for key in ("progress", "segments_done", "total_segments"):
            if key in result:
                try:
                    result[key] = int(float(result[key]))
                except Exception:
                    pass
        for key in ("updated_at", "created_at"):
            if key in result:
                try:
                    result[key] = int(float(result[key]))
                except Exception:
                    pass
        return result
    except Exception:
        logging.getLogger("api_ms").warning("从 Redis 获取任务失败", exc_info=True)
        return None


def _redis_get_task_by_task_id(task_id: str) -> Optional[dict]:
    client = _get_redis_client()
    if client is None:
        return None
    try:
        user_raw = client.get(_redis_lookup_key(task_id))
        if not user_raw:
            return None
        user_id = _redis_unquote(user_raw)
        return _redis_get_task(user_id, task_id)
    except Exception:
        logging.getLogger("api_ms").warning("通过 task_id 获取 Redis 任务失败", exc_info=True)
        return None


def _redis_list_user_tasks(user_id: str, page: int, page_size: int) -> tuple[Optional[list], int]:
    client = _get_redis_client()
    if client is None:
        return None, 0
    try:
        key = _redis_user_index_key(user_id)
        total = client.zcard(key)
        if total == 0:
            return [], 0
        start = (page - 1) * page_size
        end = start + page_size - 1
        members = client.zrevrange(key, start, end)
        items = []
        for member in members:
            tid = _redis_unquote(member)
            task = _redis_get_task(user_id, tid)
            if task:
                items.append(task)
        return items, total
    except Exception:
        logging.getLogger("api_ms").warning("从 Redis 分页读取任务失败", exc_info=True)
        return None, 0


def _redis_sync_local_tasks():
    client = _get_redis_client()
    if client is None:
        return
    for entry in _read_task_store():
        _redis_upsert_task(entry)


def _fetch_tasks_by_ids(user_id: str, ids: list[str]) -> list[dict]:
    ids = [str(i).strip() for i in ids if str(i).strip()]
    if not ids:
        return []
    results = []
    local_cache = None
    for task_id in ids:
        task = _redis_get_task(user_id, task_id)
        if task is None:
            if local_cache is None:
                local_cache = {
                    it.get("task_id"): it for it in _read_task_store()
                    if str(it.get("user_id", "")).strip() == str(user_id).strip()
                }
            task = local_cache.get(task_id)
        if task:
            results.append(task)
    return results


def _fetch_tasks_page(user_id: str, page: int, page_size: int) -> tuple[list[dict], int]:
    redis_items, redis_total = _redis_list_user_tasks(user_id, page, page_size)
    if redis_items is not None:
        return redis_items, redis_total
    items = [
        it for it in _read_task_store()
        if str(it.get("user_id", "")).strip() == str(user_id).strip()
    ]
    items = sorted(items, key=lambda x: x.get("created_at", 0), reverse=True)
    total = len(items)
    start = (page - 1) * page_size
    end = start + page_size
    return items[start:end], total


def _fetch_single_task(user_id: Optional[str], task_id: str) -> Optional[dict]:
    if user_id:
        task = _redis_get_task(user_id, task_id)
        if task:
            return task
    task = _redis_get_task_by_task_id(task_id)
    if task:
        return task
    local = _get_task_from_index(task_id)
    if local and (user_id is None or str(local.get("user_id", "")).strip() == str(user_id).strip()):
        return local
    return None


_cleanup_stop_event = threading.Event()
_cleanup_thread: Optional[threading.Thread] = None


def _start_cleanup_thread():
    global _cleanup_thread
    if TASK_RETENTION_SECONDS <= 0:
        return
    if _cleanup_thread and _cleanup_thread.is_alive():
        return
    _cleanup_stop_event.clear()
    _cleanup_thread = threading.Thread(target=_cleanup_loop, name="task-cleaner", daemon=True)
    _cleanup_thread.start()


def _stop_cleanup_thread():
    global _cleanup_thread
    if not _cleanup_thread:
        return
    _cleanup_stop_event.set()
    _cleanup_thread.join(timeout=5)
    _cleanup_thread = None


def _cleanup_loop():
    while not _cleanup_stop_event.is_set():
        try:
            _cleanup_expired_tasks_once()
        except Exception:
            logging.getLogger("api_ms").exception("任务自动清理失败")
        _cleanup_stop_event.wait(TASK_CLEAN_INTERVAL_SECONDS)


def _cleanup_expired_tasks_once():
    if TASK_RETENTION_SECONDS <= 0:
        return
    expire_before = int(time.time()) - TASK_RETENTION_SECONDS
    candidates: set[tuple[str, str]] = set()

    client = _get_redis_client()
    if client is not None:
        try:
            members = client.zrangebyscore(TASK_INDEX_KEY, 0, expire_before)
            for member in members:
                uid, tid = _redis_member_parse(member)
                if uid and tid:
                    candidates.add((uid, tid))
        except Exception:
            logging.getLogger("api_ms").warning("读取 Redis 过期任务失败", exc_info=True)

    for item in _read_task_store():
        updated_at = int(item.get("updated_at", 0))
        status = str(item.get("status", "")).lower()
        if updated_at and updated_at <= expire_before and status in {"done", "error", "succeeded", "failed"}:
            candidates.add((item.get("user_id"), item.get("task_id")))

    for uid, tid in candidates:
        entry = _fetch_single_task(uid, tid)
        status = str((entry or {}).get("status", "")).lower()
        if status and status not in {"done", "error", "succeeded", "failed"}:
            continue
        updated_at = int((entry or {}).get("updated_at", 0))
        if updated_at and updated_at > expire_before:
            continue
        _purge_task_record(uid, tid, reason="expired")


def _purge_task_record(user_id: Optional[str], task_id: str, reason: str):
    entry = _get_task_from_index(task_id)
    if entry is None:
        entry = _fetch_single_task(user_id, task_id)
    removed = _remove_from_index(task_id)
    _redis_delete_task(user_id, task_id)
    if entry is None:
        entry = {"user_id": user_id, "task_id": task_id}
    _cleanup_task_artifacts(entry)
    logging.getLogger("api_ms").info("[CLEANUP] user=%s task=%s reason=%s local=%s", user_id, task_id, reason, bool(removed))


def _cleanup_task_artifacts(task: dict):
    uid = _sanitize_fs_name(task.get("user_id"))
    tid = _sanitize_fs_name(task.get("task_id"))
    task_dir = OUT_DIR / uid / tid
    if task_dir.exists():
        shutil.rmtree(task_dir, ignore_errors=True)
    legacy_wav = OUT_DIR / uid / tid / f"{tid}.wav"
    if legacy_wav.exists():
        try:
            legacy_wav.unlink()
        except Exception:
            pass
    pattern = f"{tid}.log"
    for log_file in LOG_DIR.rglob(pattern):
        try:
            log_file.unlink()
        except Exception:
            pass


# ---------- Hotfix: GroupNorm_hijack 缺少 reduce_mean/reduce_sum/sqrt ----------
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


@app.get("/healthz", summary="健康检查", tags=["其他"])
async def healthz(request: Request):
    base = str(request.base_url).rstrip("/")
    return {"status": "ok", "index_url": f"{base}/index"}


@app.post("/set_model", summary="切换模型权重（内部）", tags=["其他"])
async def set_model(request: Request):
    req = await request.json()
    new_gpt = req.get("gpt_model_path")
    new_sovits = req.get("sovits_model_path")
    if not new_gpt and not new_sovits:
        return _err_param("缺少模型路径")
    try:
        if new_sovits:
            change_sovits_weights(new_sovits)
        if new_gpt:
            change_gpt_weights(new_gpt)
        return _ok({}, "Success")
    except Exception as e:
        logger.exception("set_model failed")
        return _err_failure(str(e))


@app.post("/synthesize", summary="直接合成（内部直连返回WAV）", tags=["其他"])
async def synthesize(request: Request):
    req = await request.json()

    ref_wav_path = req.get("ref_wav_path")
    prompt_text = req.get("prompt_text", "")
    prompt_language = req.get("prompt_language", "中文")
    text = req.get("text")
    text_language = req.get("text_language", "中文")
    how_to_cut = req.get("how_to_cut", "凑四句一切")
    top_k = int(req.get("top_k", 5))
    top_p = float(req.get("top_p", 1))
    temperature = float(req.get("temperature", 1))
    ref_free = bool(req.get("ref_free", False))
    model_id = req.get("model_id")
    save_user_id = req.get("user_id")  # 可选：用于保存输出目录
    save_task_id = req.get("task_id")  # 可选：用于保存输出目录

    # 若指定了音模ID，则作为默认值填充（显式传参优先生效）
    if model_id:
        model = _find_model(model_id)
        if model is None:
            return JSONResponse({"code": 404, "message": "音模不存在"}, status_code=404)
        if not ref_wav_path:
            ref_wav_path = model.get("refer_wav_path")
        if not prompt_text:
            prompt_text = model.get("prompt_text", "")
        if not req.get("prompt_language") and model.get("prompt_language"):
            prompt_language = model.get("prompt_language")

    if not ref_wav_path or not text:
        return _err_param("缺少参数: ref_wav_path 或 text")
    if prompt_language not in dict_language or text_language not in dict_language:
        return _err_param("语言不支持")

    try:
        wav_bytes = synthesize_once(
            ref_wav_path=ref_wav_path,
            prompt_text=prompt_text,
            prompt_language=prompt_language,
            text=text,
            text_language=text_language,
            how_to_cut=how_to_cut,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            ref_free=ref_free,
            save_user_id=save_user_id,
            save_task_id=save_task_id,
        )
        return Response(content=wav_bytes, media_type="audio/wav")
    except Exception as e:
        logger.exception("synthesize failed")
        return _err_failure(str(e))


# ---------- 音模管理接口 ----------
@app.post("/voice-models", summary="新增音模", tags=["音模管理"])
async def create_voice_model(
    request: Request,
    model_name: str = Form(...),
    model_id: str = Form(...),
    gender: int = Form(...),  # 1男 2女
    prompt_text: str = Form(""),
    prompt_language: str = Form("中文"),
    avatar: UploadFile = File(...),
    refer_wav: UploadFile = File(...),
    categories: List[str] = Form(None),
):
    model_id = _sanitize_model_id(model_id)
    if _find_model(model_id) is not None:
        return _err_resource("音模标识已存在")

    # 保存文件
    model_dir = AUDIO_ROOT / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    # 头像
    avatar_suffix = Path(avatar.filename).suffix or ".png"
    avatar_filename = f"avatar{avatar_suffix}"
    avatar_path = model_dir / avatar_filename
    with open(avatar_path, "wb") as f:
        f.write(await avatar.read())

    # 参考音频
    wav_suffix = Path(refer_wav.filename).suffix or ".wav"
    refer_filename = f"refer{wav_suffix}"
    refer_path = model_dir / refer_filename
    refer_bytes = await refer_wav.read()
    with open(refer_path, "wb") as f:
        f.write(refer_bytes)
    try:
        audio_buf = BytesIO(refer_bytes)
        wav16k, _ = librosa.load(audio_buf, sr=16000)
        if wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000:
            shutil.rmtree(model_dir, ignore_errors=True)
            return _err_param("参考音频需在 3~10 秒范围内")
    except Exception:
        shutil.rmtree(model_dir, ignore_errors=True)
        return _err_param("参考音频解析失败，请上传合法的音频文件（WAV/MP3/OGG 等）")

    base = str(request.base_url).rstrip('/')
    avatar_url = f"{base}/audio/{model_id}/{avatar_filename}"
    refer_wav_url = f"{base}/audio/{model_id}/{refer_filename}"

    cat_ids = _clean_category_ids(categories or [])
    ok, missing = _ensure_categories_exist(cat_ids)
    if not ok:
        return _err_param(f"分类不存在: {', '.join(missing)}")

    item = {
        "model_id": model_id,
        "model_name": model_name,
        "gender": int(gender),
        "prompt_text": prompt_text,
        "prompt_language": prompt_language,
        "avatar_path": str(avatar_path),
        "avatar_url": avatar_url,
        "refer_wav_path": str(refer_path),
        "refer_wav_url": refer_wav_url,
        "created_at": int(time.time()),
        "categories": cat_ids,
    }

    items = _read_store()
    items.append(item)
    # 按时间倒序
    items = sorted(items, key=lambda x: x.get("created_at", 0), reverse=True)
    _write_store(items)
    return _ok(item)


@app.get("/voice-models", summary="音模列表", tags=["音模管理"])
async def list_voice_models(request: Request, page: int = 1, page_size: int = 10, name: str = None, category: str = None):
    page = max(1, page)
    page_size = max(1, min(page_size, 100))
    items = _read_store()
    # 名称过滤（包含匹配，大小写不敏感）
    if name is not None and name.strip() != "":
        kw = name.strip().lower()
        def _match(it):
            n = (it.get("model_name") or "").lower()
            return kw in n
        items = [it for it in items if _match(it)]
    if category is not None and category.strip() != "":
        cat_norm = _normalize_category_id(category)
        items = [it for it in items if cat_norm in (it.get("categories") or [])]
    total = len(items)
    start = (page - 1) * page_size
    end = start + page_size
    base = str(request.base_url).rstrip('/')
    page_items = []
    for it in items[start:end]:
        model_id = it.get("model_id", "")
        # 尝试从已存字段推断头像文件名
        avatar_path = it.get("avatar_path", "")
        avatar_name = Path(avatar_path).name if avatar_path else "avatar.png"
        # 参考音频名
        wav_path = it.get("refer_wav_path", "")
        wav_name = Path(wav_path).name if wav_path else "refer.wav"
        it_with_url = {**it, "avatar_url": f"{base}/audio/{model_id}/{avatar_name}", "refer_wav_url": f"{base}/audio/{model_id}/{wav_name}"}
        it_with_url["categories"] = it_with_url.get("categories") or []
        page_items.append(it_with_url)
    return _page_ok(page_items, total, page, page_size)


@app.delete("/voice-models/{model_id}", summary="删除音模", tags=["音模管理"])
async def delete_voice_model(model_id: str):
    model_id = _sanitize_model_id(model_id)
    items = _read_store()
    kept = [it for it in items if it.get("model_id") != model_id]
    if len(kept) == len(items):
        return _err_resource("音模不存在")
    _write_store(kept)
    # 删除文件目录
    target_dir = AUDIO_ROOT / model_id
    if target_dir.exists():
        shutil.rmtree(target_dir, ignore_errors=True)
    return _ok({}, "deleted")


@app.post("/voice-models/{model_id}/categories", summary="设置音模分类", tags=["音模管理"])
async def update_voice_model_categories(model_id: str, request: Request):
    model_id = _sanitize_model_id(model_id)
    body = await request.json()
    cat_ids = _clean_category_ids(body.get("categories") or [])
    ok, missing = _ensure_categories_exist(cat_ids)
    if not ok:
        return _err_param(f"分类不存在: {', '.join(missing)}")
    items = _read_store()
    updated = None
    for it in items:
        if it.get("model_id") == model_id:
            it["categories"] = cat_ids
            updated = it
            break
    if updated is None:
        return _err_resource("音模不存在")
    _write_store(items)
    return _ok(updated)


@app.post("/voice-categories", summary="新增分类", tags=["音模管理"])
async def create_voice_category(request: Request):
    data = await request.json()
    category_id = _normalize_category_id(data.get("category_id", ""))
    category_name = str(data.get("category_name", "")).strip()
    description = str(data.get("description", "")).strip()
    if not category_id or not category_name:
        return _err_param("category_id 和 category_name 必填")
    items = _read_categories()
    if any(it.get("category_id") == category_id for it in items):
        return _err_resource("分类已存在")
    entry = {
        "category_id": category_id,
        "category_name": category_name,
        "description": description,
        "created_at": int(time.time()),
    }
    items.append(entry)
    items = sorted(items, key=lambda x: x.get("created_at", 0), reverse=True)
    _write_categories(items)
    return _ok(entry)


@app.get("/voice-categories", summary="分类列表", tags=["音模管理"])
async def list_voice_categories():
    return _ok(_read_categories())


@app.delete("/voice-categories/{category_id}", summary="删除分类", tags=["音模管理"])
async def delete_voice_category(category_id: str):
    category_id = _normalize_category_id(category_id)
    items = _read_categories()
    kept = [it for it in items if it.get("category_id") != category_id]
    if len(kept) == len(items):
        return _err_resource("分类不存在")
    _write_categories(kept)
    # 从音模中移除该分类
    models = _read_store()
    changed = False
    for it in models:
        cats = it.get("categories") or []
        if category_id in cats:
            it["categories"] = [c for c in cats if c != category_id]
            changed = True
    if changed:
        _write_store(models)
    return _ok({}, "deleted")


@app.get("/", summary="根路径重定向", tags=["页面"])
async def root_redirect():
    return RedirectResponse(url="/index")


@app.get("/index", summary="Index 页面", tags=["页面"])
async def index_page():
    html_path = PROJECT_ROOT / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>index.html not found</h1>")


@app.get("/manager", summary="管理页重定向", tags=["页面"])
async def manager_redirect():
    return RedirectResponse(url="/index")


# ---------- 任务接口 ----------
@app.post("/voice-tasks", summary="提交任务", tags=["任务队列"])
async def submit_task(request: Request):
    """
    前端提交任务：user_id(必填)、task_id(必填，前端生成)、model_id、text、text_language、how_to_cut、
    可选 prompt_text/prompt_language/ref_wav_path、top_k/top_p/temperature
    入队并返回当前任务状态。
    """
    body = await request.json()
    user_id = str(body.get("user_id", "")).strip()
    if not user_id:
        return _err_param("缺少 user_id")
    task_id = str(body.get("task_id", "")).strip()
    if not task_id:
        return _err_param("缺少 task_id")

    entry = {
        "task_id": task_id,
        "user_id": user_id,
        "status": "wait",
        "progress": 0,
        "segments_done": 0,
        "total_segments": 0,
        "created_at": int(time.time()),
        "updated_at": int(time.time()),
        # 透传参数
        "model_id": body.get("model_id"),
        "text": body.get("text", ""),
        "text_language": body.get("text_language", "中文"),
        "how_to_cut": body.get("how_to_cut", "凑四句一切"),
        "prompt_text": body.get("prompt_text", ""),
        "prompt_language": body.get("prompt_language", "中文"),
        "ref_wav_path": body.get("ref_wav_path"),
        "top_k": int(body.get("top_k", 5)),
        "top_p": float(body.get("top_p", 1.0)),
        "temperature": float(body.get("temperature", 1.0)),
    }
    _update_task_index(entry)
    _task_cancel_flags.pop(task_id, None)
    # 创建任务日志并记录入队
    tl = _ensure_task_logger(task_id)
    tl.info("wait user_id=%s model_id=%s text_len=%d", user_id, entry.get("model_id"), len(entry.get("text", "")))
    _enqueue_task(task_id)
    _ensure_worker()
    return _ok(entry)


@app.get("/voice-tasks", summary="查询任务（批量/分页）", tags=["任务队列"])
async def list_tasks(ids: str = None, task_ids: str = None, page: int = 1, page_size: int = 10, user_id: str = None, task_id: str = None):
    normalized_ids: list[str] = []
    for raw in (ids, task_ids):
        if raw:
            normalized_ids.extend([s.strip() for s in re.split(r"[\s,]+", raw) if s.strip()])
    if task_id:
        normalized_ids.append(str(task_id).strip())

    if normalized_ids:
        if user_id is None or str(user_id).strip() == "":
            return _err_param("缺少 user_id")
        results = _fetch_tasks_by_ids(str(user_id).strip(), normalized_ids)
        size = len(results)
        return _page_ok(results, size, 1, max(size, 1))

    if user_id is None or str(user_id).strip() == "":
        return _err_param("缺少 user_id")
    page = max(1, page)
    page_size = max(1, min(page_size, 100))
    items, total = _fetch_tasks_page(str(user_id).strip(), page, page_size)
    return _page_ok(items, total, page, page_size)


@app.post("/voice-tasks/query", summary="批量查询任务（JSON）", tags=["任务队列"])
async def list_tasks_query(payload: dict):
    page = int(payload.get("page", 1))
    page_size = int(payload.get("page_size", 10))
    user_id = str(payload.get("user_id", "")).strip()
    task_ids = payload.get("task_ids") or []
    if not user_id:
        return _err_param("缺少 user_id")
    if task_ids:
        ids = [str(t).strip() for t in task_ids if str(t).strip()]
        results = _fetch_tasks_by_ids(user_id, ids)
        size = len(results)
        return _page_ok(results, size, 1, max(size, 1))
    page = max(1, page)
    page_size = max(1, min(page_size, 100))
    items, total = _fetch_tasks_page(user_id, page, page_size)
    return _page_ok(items, total, page, page_size)


@app.get("/voice-tasks/{task_id}", summary="查询单个任务", tags=["任务队列"])
async def get_task(task_id: str):
    task = _redis_get_task_by_task_id(task_id)
    if task is None:
        task = _get_task_from_index(task_id)
    if task is not None:
        return _ok(task)
    return _err_resource("not found")


@app.delete("/voice-tasks/{task_id}", summary="删除/取消任务", tags=["任务队列"])
async def delete_task(task_id: str):
    # 允许删除运行中的任务：设置取消标记
    _mark_cancel(task_id)

    entry = _remove_from_index(task_id)
    if entry is None:
        entry = _redis_get_task_by_task_id(task_id)
    if entry is not None:
        _redis_delete_task(entry.get("user_id"), task_id)
        _cleanup_task_artifacts(entry)
        return _ok({"removed": True})
    _redis_delete_task(None, task_id)
    return _ok({"removed": False})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=api_host, port=api_port, workers=1)


