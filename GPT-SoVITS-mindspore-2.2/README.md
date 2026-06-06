# GPT-SoVITS-MindSpore-WebUI

> **当前文档版本：V3**

一个基于 MindSpore 的 GPT-SoVITS 文本转语音与语音克隆服务，提供 Web 管理页面与 REST API。

- 零样本文本转语音：提供 3–10 秒参考音频即可合成
- 少样本微调体验：少量数据可进一步提升音色相似度与自然度
- 多语种支持：中文/英文/日文/混合文本
- Web 管理页：上传音模（头像/参考音频/元数据）、在线发起合成、任务队列管理
- 分类管理：可创建分类、将音模分配到多个分类，并按分类筛选
- REST API：任务排队、进度查询、取消任务、静态产物下载

---

## V3 版本新增说明

以下为 **V3** 相对旧版（V1/V2）的主要新增与改造内容。

### V3 · 正式推理服务（`api_ms.py`）

| 类别 | 说明 |
|---|---|
| **动态资源 URL** | 支持 `--root-path` + `--base-url` 成对配置；响应中的 `avatar_url` / `refer_wav_url` / `result_url` 每次请求动态拼接，**不写入 JSON 持久化** |
| **nginx 反向代理兼容** | 中间件自动剥离代理前缀；兼容 `proxy_pass` 带/不带末尾 `/` 两种写法 |
| **页面入口** | `GET /` 直接返回管理页；`GET /index`、`/index.html` 301 重定向到 `/` |
| **任务存储改造** | 任务状态统一记录在 `voice_task.json`；产物 WAV 保存在 `output/<user_id>/<task_id>/<task_id>.wav` |
| **Redis 任务索引** | 任务查询 Redis 优先、JSON 兜底；支持任务分页、批量 ID 查询 |
| **任务列表分页** | `GET /voice-tasks?user_id=&page=&page_size=` 与 `POST /voice-tasks/query` 均支持分页（`page_size` 最大 100） |
| **跨平台文件锁** | Linux 使用 `fcntl`；Windows 使用 `threading.Lock`，可在 Windows 上跑接口层调试 |
| **HTTPS URL 修复** | 经 HTTPS 代理时不再错误拼接 nginx 内部端口（如 `:8778`） |
| **访问日志** | `Logs/access.log` 每 6 小时切割；`GET /voice-tasks` 轮询不记日志，避免刷屏 |
| **任务日志** | `Logs/<YYYYMMDD>/<HH>/<task_id>.log` 记录单任务排队、进度、完成与异常 |
| **启动参数扩展** | 新增 `--root-path`、`--base-url`（或环境变量 `infer_api_root_path`、`infer_api_base_url`） |

### V3 · 测试模拟服务（`api_ms_mock.py`）【V3 新增】

用于**不加载 MindSpore / 模型**的接口联调，路由与 JSON 结构与正式服务对齐。

| 类别 | 说明 |
|---|---|
| **独立数据目录** | 默认 `mock_runtime/`（音模、任务、日志与正式环境隔离） |
| **模拟任务队列** | 后台线程模拟进度，生成占位 WAV；支持 `--max-concurrent` 配置并发模拟数 |
| **任务 `time` 参数** | `POST /voice-tasks` 可选字段 `time`（秒），控制模拟总耗时；未传则随机 5～45 秒。**仅测试服务，不同步到正式服务** |
| **`mock_duration_sec`** | 任务执行后返回实际采用的模拟时长（仅测试服务） |
| **推荐测试端口** | `--port 5400`（代码默认 `9882`） |

### V3 · API 文档【V3 新增】

| 文档 | 对应程序 | 说明 |
|---|---|---|
| [正式服务API.md](./正式服务API.md) | `api_ms.py` | 正式服务全量接口说明（19 个路由） |
| [测试模拟服务API.md](./测试模拟服务API.md) | `api_ms_mock.py` | 测试服务全量接口说明，差异项已标注 |

> 旧文档 `API_MS.md`、`API_MOCK.md` 仍保留，**以 V3 两份新文档为准**。

---

## 目录结构（关键文件）

| 路径 | 说明 | 版本 |
|---|---|---|
| `api_ms.py` | 正式 FastAPI 服务（TTS、音模、任务队列、静态目录） | V3 |
| `api_ms_mock.py` | 测试模拟 FastAPI 服务（不加载模型） | **V3 新增** |
| `index.html` | 管理页前端；入口为 `/` | V3 |
| `正式服务API.md` | 正式服务 API 文档 | **V3 新增** |
| `测试模拟服务API.md` | 测试模拟服务 API 文档 | **V3 新增** |
| `GPT_SoVITS/` | 模型实现与权重转换工具 | — |
| `audio/` | 正式服务音模目录 | V3 |
| `output/` | 正式服务任务产物 WAV | V3 |
| `voice_models.json` | 音模索引 | V3 |
| `voice_categories.json` | 分类索引 | V3 |
| `voice_task.json` | 任务状态索引 | **V3** |
| `mock_runtime/` | 测试模拟服务数据目录 | **V3 新增** |
| `Logs/` | 访问日志与任务日志 | V3 |
| `config.ini` | Redis、任务清理等配置 | V3 |

> **V3 已废弃**旧目录方案：`task/`、`done/`、`error/` 不再作为任务状态主存储（产物统一在 `output/`）。

---

## 环境要求

- 操作系统：Linux（正式推理推荐）；Windows 可跑测试模拟服务或接口调试
- Python：推荐 3.9（本仓库早期验证环境），亦可参考在线安装章节使用 3.10
- MindSpore：2.2.3（READMEV1 测试）或 2.3.1（在线安装示例，aarch64 包）
- 必备工具：`ffmpeg`
- Redis：正式服务任务索引推荐（可选，不可用则 JSON 兜底）

---

## 安装

### 方式 A：常规安装（推荐）

```bash
conda create -n GPTSoVits python=3.9
conda activate GPTSoVits

pip install -r requirements.txt

# 安装 ffmpeg（任选其一）
conda install ffmpeg -y
```

#### 关于自定义 Python 路径与 ffmpeg 找不到的问题

若使用自定义 Python 路径启动，需把环境 `bin` 加入 `PATH`：

```bash
PATH="/your/env/bin:$PATH" python api_ms.py --device 0 --host 0.0.0.0 --port 9881
```

### 方式 B：在线安装（国内镜像）

详见原「在线安装文档」流程；`nltk_data` 目录会被 `api_ms.py` 自动检测（`NLTK_DATA_HOME` → `NLTK_DATA` → `~/nltk_data` → `/root/nltk_data`）。

---

## 预训练模型

- 可从 ModelScope 下载 MindSpore 权重，放置至 `GPT_SoVITS/pretrained_models/`
- 或使用 `GPT_SoVITS/convert.py` 将 PyTorch 权重转换为 MindSpore 权重

```bash
python GPT_SoVITS/convert.py --g_path <GPT_ckpt> --s_path <SoVITS_ckpt>
```

权重路径：`gweight.txt`、`sweight.txt`，或环境变量 `gpt_path`、`sovits_path`、`cnhubert_base_path`、`bert_path`。

---

## 启动

### V3 · 正式服务（Ascend NPU）

```bash
python api_ms.py --device 0 --host 0.0.0.0 --port 9881 \
  --root-path /ttsHttp --base-url https://your-domain/media
```

| 参数 | 环境变量 | 说明 |
|---|---|---|
| `--device` | `infer_device_id` | NPU 编号，默认 `0` |
| `--host` | `infer_api_host` | 绑定地址，默认 `0.0.0.0` |
| `--port` | `infer_api_port` | 端口，默认 `9881` |
| `--root-path` | `infer_api_root_path` | 外网 URL 路径前缀（V3） |
| `--base-url` | `infer_api_base_url` | 外网域名前缀（V3） |

浏览器访问：

- 管理页：`http://<host>:9881/`（V3：根路径即页面，非 `/index`）
- 静态资源：`/audio`、`/output`

### V3 · 测试模拟服务【V3 新增】

```powershell
python api_ms_mock.py --host 0.0.0.0 --port 5400 \
  --root-path /ttsHttp --base-url https://your-domain/media \
  --max-concurrent 4
```

不加载模型，用于前端 / nginx / 任务队列联调。详见 [测试模拟服务API.md](./测试模拟服务API.md)。

### V3 · nginx 反向代理示例

```nginx
location /media/ttsHttp/ {
    proxy_pass http://192.168.10.135:9881;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header X-Forwarded-Host $host;
}
```

测试模拟服务将端口改为 `5400` 即可。

---

## API 速览（V3）

完整说明见 [正式服务API.md](./正式服务API.md) 与 [测试模拟服务API.md](./测试模拟服务API.md)。

### 通用响应

```json
{ "code": 0, "message": "", "data": {} }
```

| code | 含义 |
|---|---|
| 0 | 成功 |
| 1 | 失败 |
| 2 | 资源错误 |
| 3 | 参数错误 |

### 端点一览（正式 / 测试结构一致）

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/healthz` | 健康检查 |
| POST | `/set_model` | 切换模型权重 |
| POST | `/synthesize` | 直接合成，返回 WAV |
| POST | `/voice-models` | 新增音模 |
| GET | `/voice-models` | 音模列表（分页/名称/分类筛选） |
| DELETE | `/voice-models/{model_id}` | 删除音模 |
| POST | `/voice-models/{model_id}/categories` | 设置音模分类 |
| GET | `/voice-categories` | 分类列表 |
| POST | `/voice-categories` | 新增分类 |
| DELETE | `/voice-categories/{category_id}` | 删除分类 |
| GET | `/` | 管理页 |
| POST | `/voice-tasks` | 提交任务 |
| GET | `/voice-tasks` | 查询任务（**分页**或批量 ID） |
| POST | `/voice-tasks/query` | 查询任务（JSON Body，**分页**或批量 ID） |
| GET | `/voice-tasks/{task_id}` | 查询单个任务 |
| DELETE | `/voice-tasks/{task_id}` | 删除/取消任务 |

### V3 · 任务列表分页

```
GET /voice-tasks?user_id=U10001&page=1&page_size=10
```

或：

```json
POST /voice-tasks/query
{ "user_id": "U10001", "page": 1, "page_size": 10 }
```

- `user_id` 必填；`page_size` 范围 1～100
- 传 `task_ids` 时走批量查询，不走分页

### V3 · 测试专用：`time` 参数（仅 `api_ms_mock.py`）

```json
POST /voice-tasks
{ "user_id": "U1", "task_id": "T1", "text": "测试", "time": 60 }
```

正式服务**不支持** `time`，且**不会**同步该参数。

### 静态资源（V3）

| 路径 | 说明 |
|---|---|
| `/audio/<model_id>/avatar.*` | 音模头像 |
| `/audio/<model_id>/refer.*` | 参考音频 |
| `/output/<user_id>/<task_id>/<task_id>.wav` | 任务合成产物 |

---

## 常用 curl 示例（V3）

```bash
# 健康检查
curl http://127.0.0.1:9881/healthz

# 提交任务
curl -X POST http://127.0.0.1:9881/voice-tasks \
  -H "Content-Type: application/json" \
  -d '{"user_id":"U1","task_id":"T1","model_id":"demo","text":"你好","text_language":"all_zh"}'

# 任务分页列表
curl "http://127.0.0.1:9881/voice-tasks?user_id=U1&page=1&page_size=10"

# 直连合成
curl -X POST http://127.0.0.1:9881/synthesize \
  -H "Content-Type: application/json" \
  -d '{"model_id":"demo","text":"你好世界","text_language":"all_zh"}' \
  --output out.wav
```

---

## 使用要点与限制

- 参考音频需 **3～10 秒**（正式服务校验；测试模拟服务校验较宽松）
- 正式服务任务**串行执行**（同一时刻最多 1 个推理任务）
- 语种标识符：`all_zh`、`en`、`zh`、`all_ja`、`ja`、`auto`（亦支持中文别名如「中文」「英文」）
- 切分策略：`cut0`～`cut5`（默认 `cut1` 凑四句一切）
- V3 动态 URL：须**同时**配置 `--root-path` 与 `--base-url` 才返回公网完整 URL；否则返回 `http(s)://IP:端口/...`

---

## 版本对照

| 项目 | V1/V2（旧） | V3（当前） |
|---|---|---|
| 任务目录 | `task/`、`done/`、`error/` | `voice_task.json` + `output/` |
| 页面入口 | `/index` | `/`（`/index` 重定向） |
| 公网 URL | 固定 host:port | `--root-path` + `--base-url` 动态拼接 |
| 测试模拟服务 | 无 | `api_ms_mock.py` + `mock_runtime/` |
| API 文档 | `API_MS.md` | `正式服务API.md`、`测试模拟服务API.md` |
| 任务分页 | 不完整/旧说明 | GET/POST 双入口，正式与测试均可用 |
| Redis | 无/部分 | 正式服务 Redis 优先 + JSON 兜底 |

---

## 说明与排查

- 启动正式服务时会打印：`[Device] MindSpore context set: device_target=Ascend, device_id=<n>`
- 若合成报 `ffmpeg` 找不到，检查 `PATH` 是否包含 conda 环境 `bin`
- 资源 URL 404：检查 nginx 配置与 `--root-path` / `--base-url` 是否成对、路径是否一致
- 联调建议：先在测试模拟服务验证接口与 nginx，再切换正式服务
