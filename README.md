# GPT-SoVITS-MindSpore-WebUI

一个基于 MindSpore 的 GPT-SoVITS 文本转语音与语音克隆服务，提供 Web 管理页面与 REST API。

- 零样本文本转语音：提供 3–10 秒参考音频即可合成
- 少样本微调体验：少量数据可进一步提升音色相似度与自然度
- 多语种支持：中文/英文/日文/混合文本
- Web 管理页：上传音模（头像/参考音频/元数据）、在线发起合成、任务队列管理
- 分类管理：可创建分类、将音模分配到多个分类，并按分类筛选
- REST API：任务排队、进度查询、取消任务、静态产物下载


## 目录结构（关键文件）

- `api_ms.py`：FastAPI 服务（TTS、音模管理、任务队列、静态目录、访问日志）
- `index.html`：管理页前端（深色风格，路径 `/index`，根路径 `/` 自动重定向到此页）
- `API_MS.md`：推理 API 文档（端点说明、参数示例、状态码约定、目录规范）
- `GPT_SoVITS/`：模型实现与权重转换工具（从 PyTorch 转 MindSpore）
- `audio/`：音模文件目录（`/audio/<model_id>/avatar.*`、`/audio/<model_id>/refer.*`）
- `voice_categories.json`：音模分类配置（可通过 API 与前端管理）
- `task/`、`done/`、`error/`：任务中间态/完成/失败的 JSON/WAV 与索引
- `output/`：直连模式与任务模式的输出 WAV（便于统一下载归档）
- `requirements.txt`：依赖清单；`install.sh`：安装脚本（可选）


## 环境要求

- 操作系统：Linux（推荐）
- Python：推荐 3.9（本仓库早期验证环境），亦可参考在线安装章节使用 3.10
- MindSpore：2.2.3（READMEV1 测试）或 2.3.1（在线安装示例，aarch64 包）
- 必备工具：`ffmpeg`


## 安装

### 方式 A：常规安装（推荐）

```bash
conda create -n GPTSoVits python=3.9
conda activate GPTSoVits

# 安装依赖
pip install -r requirements.txt

# 安装 ffmpeg（任选其一）
conda install ffmpeg -y
# 或 Ubuntu/Debian：
# sudo apt install ffmpeg
# sudo apt install libsox-dev
# conda install -c conda-forge 'ffmpeg<7'
```

#### 关于自定义 Python 路径与 ffmpeg 找不到的问题

如果你不是通过 `conda activate GPTSoVits` 启动服务，而是使用**自定义 Python 路径**，例如：

```bash
/root/autodl-tmp/env/GPTSoVits/bin/python api_ms.py --device 0 --host 0.0.0.0 --port 5400
```

即使已经在该环境里执行过 `conda install ffmpeg -y`，仍有可能在合成任务时遇到：

> FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'

原因是：当前进程的 `PATH` 中没有包含你自定义环境的 `bin` 目录，`ffmpeg-python` 在调用 `subprocess.Popen(["ffmpeg", ...])` 时找不到 `ffmpeg`。

**解决方法：在启动命令前显式把环境的 bin 目录加入 PATH**，示例如下（按你的实际路径替换）：

```bash
cd /root/autodl-tmp/GPT-SoVITS/GPT-SoVITS-mindspore && \
PATH="/root/autodl-tmp/env/GPTSoVits/bin:$PATH" \
/root/autodl-tmp/env/GPTSoVits/bin/python api_ms.py --device 0 --host 0.0.0.0 --port 5400
```

要点：
- 使用双引号：`PATH="/your/env/bin:$PATH"`，不要再额外嵌套引号；
- 不需要额外设置 `FFMPEG_PATH`，当前代码只依赖 `PATH` 中能找到 `ffmpeg`。

如需下载 MindSpore 对应的 `whl` 包，请参考官方文档或使用下述“在线安装（国内镜像）”流程。


### 方式 B：在线安装（国内镜像）

以下内容来自 `在线安装文档.txt`（原样收录）：

```bash
conda create -n GPTSoVits python=3.10
conda activate GPTSoVits
====================================
手动下载mindaudio放到本地
git clone https://ghproxy.cn/https://github.com/mindspore-lab/mindaudio.git
cd mindaudio
pip install .
===========================
回到项目根目录继续安装其它依赖：
安装剩余依赖（不再重新克隆 mindaudio）：
cd ..
pip install -r requirements.txt --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple

安装mindspore-2.3.1-cp310-cp310-linux_aarch64.whl（用本地离线包安装）

pip install mindspore-2.3.1-cp310-cp310-linux_aarch64.whl

pip install decorator attrs psutil jinja2 absl-py cloudpickle tornado ml-dtypes -i https://pypi.tuna.tsinghua.edu.cn/simple

pip install py3langid -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install packaging safetensors protobuf==3.20.3 sympy
pip install numpy==1.24.4 asttokens==2.2.1 astunparse==1.6.3 okenizers==0.13.3 addict datasets evaluate pyctcdecode regex 
pip install aiofiles aiohttp altair fastapi ffmpy httpx huggingface-hub markdown-it-py matplotlib mdit-py-plugins orjson pandas pydantic pydub python-multipart semantic-version typing-extensions uvicorn websockets fsspec resampy scikit-learn soundfile joblib pooch audioread
pip install proces nltk rapidfuzz future
pip install \"tokenizers>=0.15,<0.19\"
pip install \"gradio==3.41.2\" \"markdown-it-py<3\" linkify-it-py mdit-py-plugins
pip install addict datasets
 

将LangSegment（这是2.0版本的）放到python环境中（/root/miniconda3/envs/GPTSoVits/lib/python3.10/site-packages）

注意：`nltk_data` 目录的优先级为 `NLTK_DATA_HOME` → `NLTK_DATA` → `~/nltk_data` → `/root/nltk_data`。可根据需要将资源放在任一位置，只要确保该目录存在即可被 `api_ms.py` 自动检测。

conda install ffmpeg -y

之后就安装完毕了
```

提示：上面清单中个别包名在不同环境下可能略有差异，请根据实际情况适配（如 `tokenizers` 版本受平台限制）。


## 预训练模型

- 可从 `ModelScope` 下载 MindSpore 版本权重，放置至 `GPT_SoVITS/pretrained_models/`
- 或使用 `GPT_SoVITS/convert.py` 将 PyTorch 权重转换为 MindSpore 权重：

```bash
cd GPT-SoVITS-mindspore
python GPT_SoVITS/convert.py --g_path <path_to_GPT_ckpt> --s_path <path_to_SoVITS_ckpt>
```

可通过环境变量或权重提示文件指定权重路径：

- `gweight.txt`：当前 GPT 权重路径
- `sweight.txt`：当前 SoVITS 权重路径
- 环境变量：`gpt_path`、`sovits_path`、`cnhubert_base_path`、`bert_path`


## 启动

### 启动服务（Ascend NPU）

```bash
cd GPT-SoVITS-mindspore
# 使用第 1 张 NPU（Ascend）
python api_ms.py --device 1 --host 0.0.0.0 --port 9881
```

浏览器访问：

- 根路径 `/` 会重定向到 `/index`，加载 `index.html` 管理页面
- 静态目录：`/audio` `/task` `/done` `/error`


### 启动推理 API

上面命令即为服务端口（默认 `9881`），也可通过环境变量 `infer_api_port` 指定。

健康检查：

```bash
curl http://<host>:9881/healthz
```

切换权重：

```bash
curl -X POST http://<host>:9881/set_model \
  -H 'Content-Type: application/json' \
  -d '{ "gpt_model_path":"GPT_SoVITS/pretrained_models/xxx.ckpt", "sovits_model_path":"GPT_SoVITS/pretrained_models/yyy.ckpt" }'
```

直连合成（返回 WAV 字节流）：

```bash
curl -X POST http://<host>:9881/synthesize \
  -H 'Content-Type: application/json' \
  -d '{ "model_id":"example", "text":"你好世界。", "text_language":"中文", "how_to_cut":"凑四句一切" }' \
  --output out.wav
```


## API 速览

详见 `API_MS.md`。常用端点：

- 健康检查：`GET /healthz`
- 模型切换：`POST /set_model`
- 文本转语音：`POST /synthesize`（返回 `audio/wav`）
- 音模管理：
  - 新增：`POST /voice-models`（multipart，上传头像与参考音频）
  - 列表：`GET /voice-models?page=&page_size=&name=`（按名称过滤）
  - 删除：`DELETE /voice-models/{model_id}`
- 任务队列：
  - 提交：`POST /voice-tasks`
  - 列表：`GET /voice-tasks?ids=...` 或 `?page=&page_size=`
  - 查询：`GET /voice-tasks/{task_id}`
  - 删除：`DELETE /voice-tasks/{task_id}`（运行中即取消）

静态目录：

- `/audio`：头像与参考音频
- `/task`：任务中间态 json/wav
- `/done`：已完成任务 json/wav（可下载）
- `/error`：失败/取消任务 json


说明与排查：
- 仅 Ascend 场景：使用 `--device <idx>` 指定 NPU 编号（默认 0）。也可用环境变量替代：
  - `infer_device_id=1 ASCEND_DEVICE_ID=1 DEVICE_ID=1 python api_ms.py`
- 启动时会打印实际上下文：`[Device] MindSpore context set: device_target=Ascend, device_id=<n>`，若未打印对应编号，请检查是否正确传入 `--device` 或环境变量。
- 若依然使用 0 号 NPU，通常是底层运行时未读取到设备变量。本项目在启动时同步设置了 `ASCEND_DEVICE_ID` 与 `DEVICE_ID`，一般可解决。

## 使用要点与限制

- 参考音频需 3–10 秒，否则返回错误
- 语种枚举：`中文`、`英文`、`日文`、`中英混合`、`日英混合`、`