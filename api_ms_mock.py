# -*- coding: utf-8 -*-
"""
GPT-SoVITS MindSpore API 模拟服务（测试用）

与 api_ms.py 保持相同的 FastAPI 路由、请求/响应 JSON 结构、静态目录挂载方式，
但不加载 MindSpore / GPT-SoVITS 模型；任务队列在后台用线程模拟进度并生成占位 WAV。

启动示例：
  python api_ms_mock.py --host 0.0.0.0 --port 5400
  python api_ms_mock.py --port 5400 --root-path /ttsHttp --base-url https://36.136.54.165/media

URL 规则（每次响应时动态计算，绝不写入 JSON 固化）：
  - 同时配置 --base-url 与 --root-path：一律返回 base-url + root-path + /audio|/output|...
  - 未同时配置：一律返回 http://IP:端口 + /audio|/output|...

数据目录（默认 ./mock_runtime/，与正式服务隔离）：
  mock_runtime/audio、mock_runtime/output、voice_models.json 等

环境变量：
  MOCK_SEGMENT_DELAY_SEC  每段模拟耗时（秒，默认 0.3，未指定 time 时作兜底）
  MOCK_MAX_CONCURRENT     最大并发任务数（默认 4）
  MOCK_DEFAULT_SAMPLE_RATE  输出采样率（默认 32000）
  
  文档：测试模拟服务API.md
"""

from __future__ import annotations

import random
import argparse
import json
import logging
import math
import os
import re
import shutil
import struct
import sys
import threading
import time
import wave
from contextlib import asynccontextmanager
from datetime import datetime
from io import BytesIO
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import List, Optional
from urllib.parse import quote, quote_plus, urlparse

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# CLI / 环境
# ---------------------------------------------------------------------------

def _apply_cli_overrides():
    try:
        argv = sys.argv[1:]

        def _get_flag_value(flags):
            for i, a in enumerate(argv):
                if a in flags and i + 1 < len(argv):
                    return argv[i + 1]
                for f in flags:
                    if a.startswith(f + "="):
                        return a.split("=", 1)[1]
            return None

        for env_key, flags in (
            ("infer_api_port", ["--port"]),
            ("infer_api_host", ["--host"]),
            ("infer_api_root_path", ["--root-path", "--prefix"]),
            ("infer_api_base_url", ["--base-url", "--domain"]),
        ):
            val = _get_flag_value(flags)
            if val is not None and str(val).strip():
                os.environ[env_key] = str(val).strip()
    except Exception:
        pass


_apply_cli_overrides()

api_port = int(os.environ.get("infer_api_port", "9882"))
api_host = os.environ.get("infer_api_host", "0.0.0.0")
api_root_path = os.environ.get("infer_api_root_path", "")
api_base_url = os.environ.get("infer_api_base_url", "")
MOCK_SEGMENT_DELAY_SEC = float(os.environ.get("MOCK_SEGMENT_DELAY_SEC", "0.3"))
MOCK_MAX_CONCURRENT = max(1, int(os.environ.get("MOCK_MAX_CONCURRENT", "4")))
MOCK_DEFAULT_SAMPLE_RATE = int(os.environ.get("MOCK_DEFAULT_SAMPLE_RATE", "32000"))

PROJECT_ROOT = Path(__file__).resolve().parent
RUNTIME_ROOT = PROJECT_ROOT / "mock_runtime"
AUDIO_ROOT = RUNTIME_ROOT / "audio"
OUT_DIR = RUNTIME_ROOT / "output"
LOG_DIR = RUNTIME_ROOT / "Logs"
STORE_PATH = RUNTIME_ROOT / "voice_models.json"
VOICE_CATEGORY_STORE = RUNTIME_ROOT / "voice_categories.json"
VOICE_TASK_STORE = RUNTIME_ROOT / "voice_task.json"

for p in (RUNTIME_ROOT, AUDIO_ROOT, OUT_DIR, LOG_DIR):
    p.mkdir(parents=True, exist_ok=True)
for store, default in (
    (STORE_PATH, "[]"),
    (VOICE_CATEGORY_STORE, "[]"),
    (VOICE_TASK_STORE, "[]"),
):
    if not store.exists():
        store.write_text(default, encoding="utf-8")

logger = logging.getLogger("api_ms_mock")

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

dict_cut = {
    "不切": "cut0",
    "凑四句一切": "cut1",
    "凑50字一切": "cut2",
    "按中文句号。切": "cut3",
    "按英文句号.切": "cut4",
    "按标点符号切": "cut5",
    "cut0": "cut0",
    "cut1": "cut1",
    "cut2": "cut2",
    "cut3": "cut3",
    "cut4": "cut4",
    "cut5": "cut5",
}

_store_lock = threading.Lock()
_task_queue_lock = threading.Lock()
_task_queue: list[str] = []
_worker_started = False
_worker_start_lock = threading.Lock()
_task_cancel_flags: dict[str, threading.Event] = {}
_concurrency_sem = threading.Semaphore(MOCK_MAX_CONCURRENT)

# ---------------------------------------------------------------------------
# 响应封装（与 api_ms.py 一致）
# ---------------------------------------------------------------------------


def _resp(code: int, message: str = "", data: dict | None = None):
    return JSONResponse({"code": code, "message": message, "data": data if data is not None else {}}, status_code=200)


def _ok(data: dict | None = None, message: str = ""):
    return _resp(0, message, data)


def _err_failure(message: str):
    return _resp(1, message, {})


def _err_resource(message: str):
    return _resp(2, message, {})


def _err_param(message: str):
    return _resp(3, message, {})


def _page_ok(items: list, total: int, page: int, page_size: int, message: str = ""):
    return JSONResponse(
        {
            "code": 0,
            "message": message,
            "data": {"total": total, "pageIndex": page, "pageSize": page_size, "data": items},
        },
        status_code=200,
    )


# ---------------------------------------------------------------------------
# JSON 存储（跨平台锁，替代 fcntl）
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> list:
    with _store_lock:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    return data if isinstance(data, list) else []


def _write_json(path: Path, items: list) -> None:
    tmp_path = path.with_suffix(".tmp")
    with _store_lock:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)


def _read_store() -> list:
    items = _read_json(STORE_PATH)
    return [_model_for_store(it) if isinstance(it, dict) else it for it in items]


def _write_store(items: list) -> None:
    cleaned = [_model_for_store(it) if isinstance(it, dict) else it for it in items]
    _write_json(STORE_PATH, cleaned)


def _read_categories() -> list:
    return _read_json(VOICE_CATEGORY_STORE)


def _write_categories(items: list) -> None:
    _write_json(VOICE_CATEGORY_STORE, items)


def _read_task_store() -> list:
    return _read_json(VOICE_TASK_STORE)


def _write_task_store(items: list) -> None:
    _write_json(VOICE_TASK_STORE, items)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

# 0.0.0.0 / :: 仅用于服务 bind，不能写进返回给浏览器的资源 URL
_BIND_ONLY_HOSTS = frozenset({"0.0.0.0", "::", "[::]", "", "*"})


def _client_facing_host(host: str) -> str:
    h = (host or "").strip()
    if not h or h.lower() in _BIND_ONLY_HOSTS:
        return "127.0.0.1"
    return h


def _base_has_explicit_port(base: str) -> bool:
    if "://" not in base:
        return False
    netloc = base.split("://", 1)[1].split("/")[0]
    if netloc.startswith("["):
        return "]:" in netloc
    return ":" in netloc


def _authority_from_request(request: Request) -> Optional[str]:
    """从 nginx 转发的 Host / X-Forwarded-* 得到对外 origin（含 :8888 等端口）。"""
    forwarded_host = (request.headers.get("x-forwarded-host") or "").strip()
    host = forwarded_host.split(",")[0].strip() if forwarded_host else (request.headers.get("host") or "").strip()
    if not host:
        return None
    hostname = host.split(":")[0]
    if _client_facing_host(hostname) != hostname:
        return None
    proto = (request.headers.get("x-forwarded-proto") or request.url.scheme or "http").split(",")[0].strip()
    return f"{proto}://{host}"


def _external_url_configured() -> bool:
    """同时配置了 base-url 与 root-path 才走拼接 URL。"""
    return bool(api_base_url and api_root_path)


def _merge_port_into_base_url(base: str, request: Request) -> str:
    """
    仅 HTTP 非标准端口时补端口（如 :8888）。
    HTTPS 的 base-url 不再追加 nginx 内部端口（:8778/:8888），公网 URL 省略 443。
    """
    if _base_has_explicit_port(base):
        return base
    pu = urlparse(base if "://" in base else f"http://{base}")
    if pu.scheme == "https":
        return base
    auth = _authority_from_request(request)
    if not auth:
        return base
    pub_host = pu.hostname or ""
    req_pu = urlparse(auth)
    req_host = req_pu.hostname or ""
    if pub_host and req_host and pub_host != req_host:
        return base
    req_port = req_pu.port
    if req_port is None or req_port == 80 or req_port == api_port:
        return base
    host = pu.hostname or ""
    path = pu.path or ""
    return f"{pu.scheme}://{host}:{req_port}{path}"


def _internal_base_url(request: Optional[Request] = None) -> str:
    """内网直连：http://IP:端口（来自请求 Host，或启动参数）。"""
    if request is not None:
        try:
            host = (request.headers.get("host") or "").strip()
            if host:
                hostname = host.split(":")[0]
                if _client_facing_host(hostname) == hostname:
                    # 直连后端端口时固定 http，避免误用 x-forwarded-proto
                    scheme = "http" if f":{api_port}" in host else (request.url.scheme or "http")
                    return f"{scheme}://{host}".rstrip("/")
        except Exception:
            pass
    host = _client_facing_host(api_host)
    scheme = "https" if api_port == 443 else "http"
    if (scheme == "http" and api_port != 80) or (scheme == "https" and api_port != 443):
        host_with_port = f"{host}:{api_port}"
    else:
        host_with_port = host
    return f"{scheme}://{host_with_port}".rstrip("/")


def _public_base_url(request: Optional[Request] = None) -> str:
    base = f"{api_base_url.rstrip('/')}{api_root_path}".rstrip("/")
    if request is not None:
        merged = _merge_port_into_base_url(base, request).rstrip("/")
        # 经 HTTPS 代理时，若 base-url 为 https 则保持，不因后端 http 降级
        if request.headers.get("x-forwarded-proto", "").split(",")[0].strip().lower() == "https":
            if merged.startswith("http://") and base.startswith("https://"):
                merged = "https://" + merged[len("http://"):]
        return merged
    return base


def _page_root_url(request: Request) -> str:
    return f"{_get_base_url(request).rstrip('/')}/"


def _index_html_response() -> HTMLResponse:
    html_path = PROJECT_ROOT / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>index.html not found</h1><p>Mock API 运行中，可访问 /docs</p>")


def _get_base_url(request: Optional[Request] = None) -> str:
    """
    动态基础 URL（仅响应时拼接，不写入 JSON）：
    - 同时配置 --base-url + --root-path → 一律 base-url + root-path
    - 否则 → 一律 http://IP:端口
    """
    if _external_url_configured():
        return _public_base_url(request)
    return _internal_base_url(request)


def _external_path_prefixes() -> list[str]:
    """外网完整路径前缀（proxy_pass 不带 / 时，后端收到的路径前缀）。"""
    prefixes: list[str] = []
    if api_root_path:
        prefixes.append(api_root_path.rstrip("/"))
        if api_base_url:
            bu_path = (urlparse(api_base_url).path or "").rstrip("/")
            if bu_path:
                combined = f"{bu_path}{api_root_path}".rstrip("/")
                if combined not in prefixes:
                    prefixes.append(combined)
    return sorted(set(prefixes), key=len, reverse=True)


def _resolve_stripped_path(path: str, request: Request) -> str:
    """将 /media/ttsHttp/index 映射为 /index；已是 /index 则不变。"""
    prefixes = list(_external_path_prefixes())
    xf = (request.headers.get("x-forwarded-prefix") or "").strip().rstrip("/")
    if xf and xf not in prefixes:
        prefixes.append(xf)
    prefixes = sorted(set(prefixes), key=len, reverse=True)
    for prefix in prefixes:
        if not prefix:
            continue
        if path == prefix:
            return "/"
        if path.startswith(prefix + "/"):
            return path[len(prefix):] or "/"
    return path


def _build_resource_url(base_url: str, relative_path: str) -> str:
    rel = relative_path if relative_path.startswith("/") else f"/{relative_path}"
    parts = rel.split("/")
    encoded = "/".join(quote(p, safe="") if p else p for p in parts)
    return f"{base_url.rstrip('/')}{encoded}"


def _extract_relative_path(url: Optional[str]) -> str:
    """从相对路径或历史完整 URL 中提取 /audio/... 或 /output/...（去掉旧前缀）。"""
    if not url:
        return ""
    u = str(url).strip()
    if u.startswith(("http://", "https://")):
        u = urlparse(u).path or ""
    if not u.startswith("/"):
        u = f"/{u}"
    for marker in ("/audio/", "/output/"):
        idx = u.find(marker)
        if idx >= 0:
            return u[idx:]
    return u


def _model_relative_paths(item: dict) -> tuple[str, str]:
    model_id = item.get("model_id", "")
    avatar_name = Path(item.get("avatar_path", "")).name or "avatar.png"
    wav_name = Path(item.get("refer_wav_path", "")).name or "refer.wav"
    return f"/audio/{model_id}/{avatar_name}", f"/audio/{model_id}/{wav_name}"


def _model_for_store(item: dict) -> dict:
    """持久化只存路径，不存完整 URL（避免 base-url/root-path 变更后链接过期）。"""
    d = {k: v for k, v in item.items() if k not in ("avatar_url", "refer_wav_url")}
    return d


def _normalize_task_urls(task_data: dict, base_url: str) -> dict:
    if not task_data:
        return task_data
    result = task_data.copy()
    rel = _extract_relative_path(result.get("result_url"))
    if not rel and result.get("result_path"):
        uid = _sanitize_fs_name(result.get("user_id"))
        tid = _sanitize_fs_name(result.get("task_id"))
        rel = f"/output/{uid}/{tid}/{tid}.wav"
    if rel:
        result["result_url"] = _build_resource_url(base_url, rel)
    return result


def _normalize_model_urls(item: dict, base_url: str) -> dict:
    out = {**item}
    av_rel, wav_rel = _model_relative_paths(out)
    out["avatar_url"] = _build_resource_url(base_url, av_rel)
    out["refer_wav_url"] = _build_resource_url(base_url, wav_rel)
    return out


def _sanitize_model_id(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", model_id.strip())


def _normalize_category_id(cat_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", str(cat_id or "").strip())


def _sanitize_fs_name(value: Optional[str]) -> str:
    s = str(value or "").strip()
    if not s:
        return "anonymous"
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", s)
    if not sanitized or sanitized.replace("_", "").replace("-", "") == "":
        return quote_plus(s) or "anonymous"
    return sanitized


def _find_model(model_id: str) -> Optional[dict]:
    for it in _read_store():
        if it.get("model_id") == model_id:
            return it
    return None


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
    existing = {c.get("category_id") for c in _read_categories()}
    missing = [cid for cid in cat_ids if cid not in existing]
    return len(missing) == 0, missing


def _update_task_index(entry: dict):
    items = _read_task_store()
    items = [it for it in items if it.get("task_id") != entry.get("task_id")]
    items.insert(0, entry)
    items = sorted(items, key=lambda x: x.get("created_at", 0), reverse=True)
    _write_task_store(items)


def _get_task_from_index(task_id: str) -> Optional[dict]:
    for it in _read_task_store():
        if it.get("task_id") == task_id:
            return it
    return None


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


def _fetch_tasks_by_ids(user_id: str, ids: list[str]) -> list[dict]:
    ids = [str(i).strip() for i in ids if str(i).strip()]
    cache = {
        it.get("task_id"): it
        for it in _read_task_store()
        if str(it.get("user_id", "")).strip() == str(user_id).strip()
    }
    results = [cache[tid] for tid in ids if tid in cache]
    results.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    return results


def _fetch_tasks_page(user_id: str, page: int, page_size: int) -> tuple[list[dict], int]:
    items = [
        it
        for it in _read_task_store()
        if str(it.get("user_id", "")).strip() == str(user_id).strip()
    ]
    items = sorted(items, key=lambda x: x.get("created_at", 0), reverse=True)
    total = len(items)
    start = (page - 1) * page_size
    end = start + page_size
    return items[start:end], total


# ---------------------------------------------------------------------------
# 模拟音频
# ---------------------------------------------------------------------------


def _generate_mock_wav_bytes(duration_sec: float = 2.0, sample_rate: int = MOCK_DEFAULT_SAMPLE_RATE) -> bytes:
    """生成短促正弦波占位 WAV，不依赖 AI 模型。"""
    duration_sec = max(0.5, min(duration_sec, 30.0))
    n_samples = int(sample_rate * duration_sec)
    freq = 440.0
    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        for i in range(n_samples):
            val = int(32767 * 0.12 * math.sin(2 * math.pi * freq * i / sample_rate))
            frames.extend(struct.pack("<h", val))
        wf.writeframes(frames)
    return buf.getvalue()


def _write_mock_wav_file(path: Path, duration_sec: float = 2.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_generate_mock_wav_bytes(duration_sec))


def _mock_split_text(text: str, how_to_cut: str) -> list[str]:
    """简化切句，仅用于模拟分段进度。"""
    text = (text or "").strip()
    if not text:
        return []
    cut_key = dict_cut.get(how_to_cut, how_to_cut)
    if cut_key == "cut0":
        return [text]
    if cut_key == "cut3":
        parts = re.split(r"(?<=[。！？!?])", text)
    elif cut_key == "cut4":
        parts = re.split(r"(?<=[.!?])", text)
    elif cut_key == "cut5":
        parts = re.split(r"[\n,，;；、]+", text)
    elif cut_key == "cut2":
        parts = []
        chunk = 50
        for i in range(0, len(text), chunk):
            parts.append(text[i : i + chunk])
    else:
        # cut1 默认：按句号拆，每 4 句一组
        sentences = [s.strip() for s in re.split(r"(?<=[。！？!?])", text) if s.strip()]
        if not sentences:
            sentences = [text]
        parts = []
        for i in range(0, len(sentences), 4):
            parts.append("".join(sentences[i : i + 4]))
    segs = [p.strip() for p in parts if p and p.strip()]
    return segs or [text]


# ---------------------------------------------------------------------------
# 任务队列（模拟 worker）
# ---------------------------------------------------------------------------


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


def _mark_cancelled_and_cleanup(task_id: str, task: dict):
    task["status"] = "error"
    task["updated_at"] = int(time.time())
    _update_task_index(task)


def _cleanup_task_artifacts(task: dict):
    uid = _sanitize_fs_name(task.get("user_id"))
    tid = _sanitize_fs_name(task.get("task_id"))
    task_dir = OUT_DIR / uid / tid
    if task_dir.exists():
        shutil.rmtree(task_dir, ignore_errors=True)


def _resolve_mock_duration_sec(task: dict) -> float:
    """任务模拟总耗时（秒）：有 time 用指定值，否则随机。"""
    raw = task.get("time")
    if raw is not None and str(raw).strip() != "":
        try:
            return max(1.0, min(float(raw), 7200.0))
        except (TypeError, ValueError):
            pass
    return random.uniform(5.0, 45.0)


def _parse_submit_time(body: dict) -> tuple[Optional[float], Optional[str]]:
    """解析提交任务时的 time 参数，返回 (秒数, 错误信息)。"""
    if "time" not in body or body.get("time") is None or str(body.get("time")).strip() == "":
        return None, None
    try:
        sec = float(body.get("time"))
    except (TypeError, ValueError):
        return None, "time 必须为数字（秒）"
    if sec <= 0:
        return None, "time 必须为正数（秒）"
    return max(1.0, min(sec, 7200.0)), None


def _enqueue_task(task_id: str):
    with _task_queue_lock:
        _task_queue.append(task_id)


def _dequeue_task() -> Optional[str]:
    with _task_queue_lock:
        if _task_queue:
            return _task_queue.pop(0)
    return None


def _ensure_worker():
    global _worker_started
    with _worker_start_lock:
        if not _worker_started:
            threading.Thread(target=_task_worker_loop, daemon=True).start()
            _worker_started = True


def _run_task(task_id: str):
    task = _get_task_from_index(task_id)
    if task is None:
        return

    task["status"] = "run"
    task["updated_at"] = int(time.time())
    _update_task_index(task)

    text = task.get("text", "")
    how_to_cut = task.get("how_to_cut", "cut1")
    segs = _mock_split_text(text, how_to_cut)
    total_segments = max(len(segs), 1)
    task["total_segments"] = total_segments
    task["segments_done"] = 0
    task["progress"] = 0
    total_duration = _resolve_mock_duration_sec(task)
    task["mock_duration_sec"] = round(total_duration, 2)
    segment_sleep = total_duration / total_segments
    _update_task_index(task)

    for idx in range(total_segments):
        if _is_cancelled(task_id):
            _mark_cancelled_and_cleanup(task_id, task)
            return
        time.sleep(segment_sleep)
        task["segments_done"] = idx + 1
        task["progress"] = int((idx + 1) * 100 / total_segments)
        task["updated_at"] = int(time.time())
        _update_task_index(task)

    if _is_cancelled(task_id):
        _mark_cancelled_and_cleanup(task_id, task)
        return

    uid = _sanitize_fs_name(task.get("user_id"))
    tid = _sanitize_fs_name(task_id)
    user_dir = OUT_DIR / uid / tid
    out_path = user_dir / f"{tid}.wav"
    duration = max(1.0, len(text) * 0.05)
    _write_mock_wav_file(out_path, duration_sec=duration)

    task["status"] = "done"
    task["result_path"] = str(out_path)
    task["result_url"] = f"/output/{uid}/{tid}/{tid}.wav"
    task["progress"] = 100
    task["updated_at"] = int(time.time())
    _update_task_index(task)
    logger.info("[MOCK] task done: %s -> %s", task_id, out_path)


def _task_worker_loop():
    while True:
        task_id = _dequeue_task()
        if not task_id:
            time.sleep(0.2)
            continue
        _concurrency_sem.acquire()
        threading.Thread(target=_run_task_wrapper, args=(task_id,), daemon=True).start()


def _run_task_wrapper(task_id: str):
    try:
        _run_task(task_id)
    except Exception:
        logger.exception("mock task failed: %s", task_id)
        try:
            task = _get_task_from_index(task_id)
            if task is not None:
                task["status"] = "error"
                task["updated_at"] = int(time.time())
                _update_task_index(task)
        except Exception:
            pass
    finally:
        _concurrency_sem.release()


# ---------------------------------------------------------------------------
# 访问日志
# ---------------------------------------------------------------------------

_access_logger = logging.getLogger("api_ms_mock_access")


def _setup_access_logger():
    _access_logger.setLevel(logging.INFO)
    handler = TimedRotatingFileHandler(
        str(LOG_DIR / "access.log"), when="h", interval=6, backupCount=28, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s\t%(message)s"))
    _access_logger.addHandler(handler)


_setup_access_logger()


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    logging.basicConfig(level=logging.INFO)
    logger.info(
        "Mock API 已启动 | port=%s | runtime=%s | max_concurrent=%s | segment_delay=%.2fs",
        api_port,
        RUNTIME_ROOT,
        MOCK_MAX_CONCURRENT,
        MOCK_SEGMENT_DELAY_SEC,
    )
    yield


_tags_metadata = [
    {"name": "音模管理", "description": "音模的新增、列表、删除等管理接口（模拟）"},
    {"name": "任务队列", "description": "提交任务、查询任务、删除/取消任务等接口（模拟）"},
    {"name": "页面", "description": "前端页面与重定向"},
    {"name": "其他", "description": "健康检查、内部使用接口等（模拟）"},
]

# 注意：不要给 FastAPI 设置 root_path。
# nginx 两种写法均兼容（由 strip_proxy_prefix_middleware 处理）：
#   A) proxy_pass http://IP:5400/;     → 剥前缀，后端收到 /index
#   B) proxy_pass http://IP:5400;      → 不剥前缀，后端收到 /media/ttsHttp/index
# api_root_path / base-url 仅用于动态拼接返回 URL，不参与路由挂载。
app = FastAPI(
    title="GPT-SoVITS 推理服务（模拟）",
    description="测试用模拟 API，结构与 api_ms.py 一致，不调用 AI 模型。",
    version="mock-0.1.0",
    openapi_tags=_tags_metadata,
    lifespan=_lifespan,
)

app.mount("/audio", StaticFiles(directory=str(AUDIO_ROOT)), name="audio")
app.mount("/output", StaticFiles(directory=str(OUT_DIR)), name="output")


@app.middleware("http")
async def access_log_middleware(request: Request, call_next):
    path = request.url.path
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


@app.middleware("http")
async def strip_proxy_prefix_middleware(request: Request, call_next):
    """
    兼容 nginx proxy_pass 两种写法：
    - http://IP:5400/   （剥前缀）
    - http://IP:5400    （不剥前缀，整段 URI 转发）
    """
    path = request.scope.get("path", "") or ""
    new_path = _resolve_stripped_path(path, request)
    if new_path != path:
        request.scope["path"] = new_path
        request.scope["raw_path"] = new_path.encode("utf-8")
    return await call_next(request)


# ---------------------------------------------------------------------------
# 端点（与 api_ms.py 对齐）
# ---------------------------------------------------------------------------


@app.get("/healthz", summary="健康检查", tags=["其他"])
async def healthz(request: Request):
    base = _get_base_url(request)
    return {
        "status": "ok",
        "mock": True,
        "index_url": f"{base}/",
        "runtime_root": str(RUNTIME_ROOT),
    }


@app.post("/set_model", summary="切换模型权重（内部，模拟）", tags=["其他"])
async def set_model(request: Request):
    req = await request.json()
    if not req.get("gpt_model_path") and not req.get("sovits_model_path"):
        return _err_param("缺少模型路径")
    return _ok({}, "Success (mock: no model loaded)")


@app.post("/synthesize", summary="直接合成（模拟返回 WAV）", tags=["其他"])
async def synthesize(request: Request):
    req = await request.json()
    ref_wav_path = req.get("ref_wav_path")
    text = req.get("text")
    prompt_language = req.get("prompt_language", "all_zh")
    text_language = req.get("text_language", "all_zh")
    model_id = req.get("model_id")
    save_user_id = req.get("user_id")
    save_task_id = req.get("task_id")

    if model_id:
        model = _find_model(model_id)
        if model is None:
            return JSONResponse({"code": 404, "message": "音模不存在"}, status_code=404)
        if not ref_wav_path:
            ref_wav_path = model.get("refer_wav_path")

    if not ref_wav_path or not text:
        return _err_param("缺少参数: ref_wav_path 或 text")
    if prompt_language not in dict_language or text_language not in dict_language:
        return _err_param("语言不支持")
    if not os.path.exists(ref_wav_path):
        return _err_resource(f"参考音频不存在: {ref_wav_path}")

    duration = max(1.0, len(str(text)) * 0.05)
    wav_bytes = _generate_mock_wav_bytes(duration_sec=duration)
    if save_user_id and save_task_id:
        uid = _sanitize_fs_name(save_user_id)
        tid = _sanitize_fs_name(save_task_id)
        out_path = OUT_DIR / uid / tid / f"{tid}.wav"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(wav_bytes)
    return Response(content=wav_bytes, media_type="audio/wav")


@app.post("/voice-models", summary="新增音模", tags=["音模管理"])
async def create_voice_model(
    request: Request,
    model_name: str = Form(...),
    model_id: str = Form(...),
    gender: int = Form(...),
    prompt_text: str = Form(""),
    prompt_language: str = Form("all_zh"),
    avatar: UploadFile = File(...),
    refer_wav: UploadFile = File(...),
    categories: List[str] = Form(None),
):
    model_id = _sanitize_model_id(model_id)
    if _find_model(model_id) is not None:
        return _err_resource("音模标识已存在")

    model_dir = AUDIO_ROOT / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    avatar_suffix = Path(avatar.filename or "").suffix or ".png"
    avatar_filename = f"avatar{avatar_suffix}"
    avatar_path = model_dir / avatar_filename
    avatar_path.write_bytes(await avatar.read())

    wav_suffix = Path(refer_wav.filename or "").suffix or ".wav"
    refer_filename = f"refer{wav_suffix}"
    refer_path = model_dir / refer_filename
    refer_bytes = await refer_wav.read()
    if len(refer_bytes) < 100:
        shutil.rmtree(model_dir, ignore_errors=True)
        return _err_param("参考音频文件过小或无效")
    refer_path.write_bytes(refer_bytes)

    base = _get_base_url(request)
    cat_ids = _clean_category_ids(categories or [])
    ok, missing = _ensure_categories_exist(cat_ids)
    if not ok:
        shutil.rmtree(model_dir, ignore_errors=True)
        return _err_param(f"分类不存在: {', '.join(missing)}")

    item = _model_for_store({
        "model_id": model_id,
        "model_name": model_name,
        "gender": int(gender),
        "prompt_text": prompt_text,
        "prompt_language": prompt_language,
        "avatar_path": str(avatar_path),
        "refer_wav_path": str(refer_path),
        "created_at": int(time.time()),
        "categories": cat_ids,
    })
    items = _read_store()
    items.append(item)
    items = sorted(items, key=lambda x: x.get("created_at", 0), reverse=True)
    _write_store(items)
    return _ok(_normalize_model_urls(item, base))


@app.get("/voice-models", summary="音模列表", tags=["音模管理"])
async def list_voice_models(
    request: Request, page: int = 1, page_size: int = 10, name: str = None, category: str = None
):
    page = max(1, page)
    page_size = max(1, min(page_size, 100))
    items = _read_store()
    if name and name.strip():
        kw = name.strip().lower()
        items = [it for it in items if kw in (it.get("model_name") or "").lower()]
    if category and category.strip():
        cat_norm = _normalize_category_id(category)
        items = [it for it in items if cat_norm in (it.get("categories") or [])]
    total = len(items)
    start = (page - 1) * page_size
    end = start + page_size
    base = _get_base_url(request)
    page_items = []
    for it in items[start:end]:
        normalized = _normalize_model_urls(it, base)
        normalized["categories"] = normalized.get("categories") or []
        page_items.append(normalized)
    return _page_ok(page_items, total, page, page_size)


@app.delete("/voice-models/{model_id}", summary="删除音模", tags=["音模管理"])
async def delete_voice_model(model_id: str):
    model_id = _sanitize_model_id(model_id)
    items = _read_store()
    kept = [it for it in items if it.get("model_id") != model_id]
    if len(kept) == len(items):
        return _err_resource("音模不存在")
    _write_store(kept)
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
    base = _get_base_url(request)
    return _ok(_normalize_model_urls(updated, base))


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


@app.get("/", summary="管理页", tags=["页面"])
async def root_page():
    """直接返回页面，地址栏保持 / 不显示 index。"""
    return _index_html_response()


@app.get("/index", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
async def index_redirect_to_root(request: Request):
    return RedirectResponse(url=_page_root_url(request), status_code=301)


@app.get("/manager", summary="管理页重定向", tags=["页面"])
async def manager_redirect(request: Request):
    return RedirectResponse(url=_page_root_url(request), status_code=302)


@app.post("/voice-tasks", summary="提交任务", tags=["任务队列"])
async def submit_task(request: Request):
    body = await request.json()
    user_id = str(body.get("user_id", "")).strip()
    if not user_id:
        return _err_param("缺少 user_id")
    task_id = str(body.get("task_id", "")).strip()
    if not task_id:
        return _err_param("缺少 task_id")

    mock_time, time_err = _parse_submit_time(body)
    if time_err:
        return _err_param(time_err)

    entry = {
        "task_id": task_id,
        "user_id": user_id,
        "status": "wait",
        "progress": 0,
        "segments_done": 0,
        "total_segments": 0,
        "created_at": int(time.time()),
        "updated_at": int(time.time()),
        "model_id": body.get("model_id"),
        "text": body.get("text", ""),
        "text_language": body.get("text_language", "all_zh"),
        "how_to_cut": body.get("how_to_cut", "cut1"),
        "prompt_text": body.get("prompt_text", ""),
        "prompt_language": body.get("prompt_language", "all_zh"),
        "ref_wav_path": body.get("ref_wav_path"),
        "top_k": int(body.get("top_k", 5)),
        "top_p": float(body.get("top_p", 1.0)),
        "temperature": float(body.get("temperature", 1.0)),
    }
    if mock_time is not None:
        entry["time"] = mock_time
    _update_task_index(entry)
    _task_cancel_flags.pop(task_id, None)
    _get_cancel_event(task_id).clear()
    _enqueue_task(task_id)
    _ensure_worker()
    base = _get_base_url(request)
    return _ok(_normalize_task_urls(entry, base))


@app.get("/voice-tasks", summary="查询任务（批量/分页）", tags=["任务队列"])
async def list_tasks(
    request: Request,
    ids: str = None,
    task_ids: str = None,
    page: int = 1,
    page_size: int = 10,
    user_id: str = None,
    task_id: str = None,
):
    base = _get_base_url(request)
    normalized_ids: list[str] = []
    for raw in (ids, task_ids):
        if raw:
            normalized_ids.extend([s.strip() for s in re.split(r"[\s,]+", raw) if s.strip()])
    if task_id:
        normalized_ids.append(str(task_id).strip())

    if normalized_ids:
        if not user_id or not str(user_id).strip():
            return _err_param("缺少 user_id")
        results = _fetch_tasks_by_ids(str(user_id).strip(), normalized_ids)
        results = [_normalize_task_urls(t, base) for t in results]
        size = len(results)
        return _page_ok(results, size, 1, max(size, 1))

    if not user_id or not str(user_id).strip():
        return _err_param("缺少 user_id")
    page = max(1, page)
    page_size = max(1, min(page_size, 100))
    items, total = _fetch_tasks_page(str(user_id).strip(), page, page_size)
    items = [_normalize_task_urls(t, base) for t in items]
    return _page_ok(items, total, page, page_size)


@app.post("/voice-tasks/query", summary="批量查询任务（JSON）", tags=["任务队列"])
async def list_tasks_query(request: Request, payload: dict):
    base = _get_base_url(request)
    page = int(payload.get("page", 1))
    page_size = int(payload.get("page_size", 10))
    user_id = str(payload.get("user_id", "")).strip()
    task_ids = payload.get("task_ids") or []
    if not user_id:
        return _err_param("缺少 user_id")
    if task_ids:
        ids = [str(t).strip() for t in task_ids if str(t).strip()]
        results = _fetch_tasks_by_ids(user_id, ids)
        results = [_normalize_task_urls(t, base) for t in results]
        size = len(results)
        return _page_ok(results, size, 1, max(size, 1))
    page = max(1, page)
    page_size = max(1, min(page_size, 100))
    items, total = _fetch_tasks_page(user_id, page, page_size)
    items = [_normalize_task_urls(t, base) for t in items]
    return _page_ok(items, total, page, page_size)


@app.get("/voice-tasks/{task_id}", summary="查询单个任务", tags=["任务队列"])
async def get_task(request: Request, task_id: str):
    base = _get_base_url(request)
    task = _get_task_from_index(task_id)
    if task is not None:
        return _ok(_normalize_task_urls(task, base))
    return _err_resource("not found")


@app.delete("/voice-tasks/{task_id}", summary="删除/取消任务", tags=["任务队列"])
async def delete_task(task_id: str):
    _mark_cancel(task_id)
    entry = _remove_from_index(task_id)
    if entry is not None:
        _cleanup_task_artifacts(entry)
        return _ok({"removed": True})
    return _ok({"removed": False})


def _parse_args():
    parser = argparse.ArgumentParser(description="GPT-SoVITS Mock API（测试用）")
    parser.add_argument("--host", default=api_host)
    parser.add_argument("--port", type=int, default=api_port)
    parser.add_argument("--root-path", "--prefix", dest="root_path", default=api_root_path)
    parser.add_argument("--base-url", "--domain", dest="base_url", default=api_base_url)
    parser.add_argument(
        "--segment-delay",
        type=float,
        default=MOCK_SEGMENT_DELAY_SEC,
        help="未指定 time 时的兜底每段耗时（秒）",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=MOCK_MAX_CONCURRENT,
        help="最大并发执行任务数",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    api_host = args.host
    api_port = args.port
    api_root_path = args.root_path or ""
    api_base_url = args.base_url or ""
    os.environ["MOCK_SEGMENT_DELAY_SEC"] = str(args.segment_delay)
    os.environ["MOCK_MAX_CONCURRENT"] = str(max(1, args.max_concurrent))
    MOCK_SEGMENT_DELAY_SEC = args.segment_delay
    MOCK_MAX_CONCURRENT = max(1, args.max_concurrent)
    _concurrency_sem = threading.Semaphore(MOCK_MAX_CONCURRENT)

    import uvicorn

    uvicorn.run(app, host=api_host, port=api_port, workers=1)
