# 测试模拟服务 API 文档

> **对应程序**：`api_ms_mock.py`  
> **服务类型**：接口结构与正式服务一致，**不加载 AI 模型**，任务为线程模拟 + 占位 WAV  
> **用途**：先完成联调测试，通过后再将选定能力同步至 `api_ms.py`  
> **OpenAPI 文档**：启动后访问 `http://<host>:<port>/docs`

---

## ⚠️ 与正式服务差异速查（必读）

| 项目 | 测试模拟服务 | 正式服务 | 测试通过后是否同步 |
|---|---|---|---|
| 程序 | `api_ms_mock.py` | `api_ms.py` | — |
| 推理 | sleep + 占位 WAV | MindSpore 真实推理 | **否** |
| 数据目录 | `./mock_runtime/` | 项目根目录 | **否** |
| Redis | **无** | 有 | **否** |
| `--device` | **无** | 有 | **否** |
| 默认端口 | `9882`（建议 `--port 5400`） | `9881` | **否** |
| 任务并发 | 默认最多 **4** 个同时执行 | 串行，最多 **1** 个 | **否** |
| **POST `/voice-tasks` 的 `time`** | **有** | **无** | **绝不同步** |
| **`mock_duration_sec` 字段** | **有** | **无** | **否** |
| `/healthz` 的 `mock` 字段 | **有** | **无** | **否** |

---

## 目录

1. [服务说明](#1-服务说明)
2. [通用约定](#2-通用约定)
3. [动态 URL 规则](#3-动态-url-规则)
4. [静态资源](#4-静态资源)
5. [接口列表（全量）](#5-接口列表全量)
6. [接口详细说明](#6-接口详细说明)
   - 6.1 [GET /healthz](#61-get-healthz)
   - 6.2 [POST /set_model](#62-post-set_model)
   - 6.3 [POST /synthesize](#63-post-synthesize)
   - 6.4 [POST /voice-models](#64-post-voice-models)
   - 6.5 [GET /voice-models](#65-get-voice-models)
   - 6.6 [DELETE /voice-models/{model_id}](#66-delete-voice-modelsmodel_id)
   - 6.7 [POST /voice-models/{model_id}/categories](#67-post-voice-modelsmodel_idcategories)
   - 6.8 [POST /voice-categories](#68-post-voice-categories)
   - 6.9 [GET /voice-categories](#69-get-voice-categories)
   - 6.10 [DELETE /voice-categories/{category_id}](#610-delete-voice-categoriescategory_id)
   - 6.11 [GET /](#611-get-)
   - 6.12 [GET /index、GET /index.html](#612-get-indexget-indexhtml)
   - 6.13 [GET /manager](#613-get-manager)
   - 6.14 [POST /voice-tasks](#614-post-voice-tasks)
   - 6.15 [GET /voice-tasks](#615-get-voice-tasks)
   - 6.16 [POST /voice-tasks/query](#616-post-voice-tasksquery)
   - 6.17 [GET /voice-tasks/{task_id}](#617-get-voice-taskstask_id)
   - 6.18 [DELETE /voice-tasks/{task_id}](#618-delete-voice-taskstask_id)
7. [参数标识符附录](#7-参数标识符附录)
8. [数据存储与日志](#8-数据存储与日志)
9. [启动与环境变量](#9-启动与环境变量)
10. [测试检查清单](#10-测试检查清单)

---

## 1. 服务说明

| 项目 | 内容 |
|---|---|
| 代码默认端口 | `9882` |
| **推荐测试端口** | `5400`（启动时 `--port 5400`） |
| 默认绑定 | `0.0.0.0` |
| 最大并发任务 | 默认 `4`（`--max-concurrent` 可调） |
| 数据根目录 | `./mock_runtime/`（与正式环境隔离） |

**数据目录**

| 路径 | 用途 |
|---|---|
| `mock_runtime/audio/` | 音模文件 |
| `mock_runtime/output/` | 任务占位 WAV |
| `mock_runtime/voice_models.json` | 音模索引 |
| `mock_runtime/voice_categories.json` | 分类索引 |
| `mock_runtime/voice_task.json` | 任务索引 |
| `mock_runtime/Logs/` | 访问日志 |

---

## 2. 通用约定

与正式服务相同（见 [正式服务API.md](./正式服务API.md) §2）：

- 普通响应：`{ "code", "message", "data" }`
- 分页响应：`data.total` / `data.pageIndex` / `data.pageSize` / `data.data`
- code：`0` 成功、`1` 失败、`2` 资源错误、`3` 参数错误
- 任务状态：`wait` | `run` | `done` | `error`

---

## 3. 动态 URL 规则

与正式服务**一致**：

| 启动配置 | 资源 URL 形式 |
|---|---|
| **同时**配置 `--base-url` + `--root-path` | `{base-url}{root-path}/audio|output/...` |
| 未同时配置 | `http(s)://<IP>:<端口>/audio|output/...` |

JSON 索引**不持久化**完整 URL。

---

## 4. 静态资源

| 方法 | 路径模式 | 磁盘路径 |
|---|---|---|
| GET | `/audio/<model_id>/<文件名>` | `mock_runtime/audio/<model_id>/` |
| GET | `/output/<user_id>/<task_id>/<task_id>.wav` | `mock_runtime/output/...` |

---

## 5. 接口列表（全量）

共 **19** 个业务/页面路由（与正式服务数量一致，路由结构对齐）。

| 序号 | 方法 | 路径 | 分组 | 说明 | 备注 |
|---|---|---|---|---|---|
| 1 | GET | `/healthz` | 其他 | 健康检查 | 【与正式不同】含 `mock` |
| 2 | POST | `/set_model` | 其他 | 切换权重（模拟） | 【与正式不同】不加载模型 |
| 3 | POST | `/synthesize` | 其他 | 直接合成（模拟 WAV） | 【与正式不同】 |
| 4 | POST | `/voice-models` | 音模管理 | 新增音模 | 【与正式不同】音频时长校验宽松 |
| 5 | GET | `/voice-models` | 音模管理 | 音模列表 | 同正式 |
| 6 | DELETE | `/voice-models/{model_id}` | 音模管理 | 删除音模 | 同正式 |
| 7 | POST | `/voice-models/{model_id}/categories` | 音模管理 | 设置分类 | 同正式 |
| 8 | POST | `/voice-categories` | 音模管理 | 新增分类 | 同正式 |
| 9 | GET | `/voice-categories` | 音模管理 | 分类列表 | 同正式 |
| 10 | DELETE | `/voice-categories/{category_id}` | 音模管理 | 删除分类 | 同正式 |
| 11 | GET | `/` | 页面 | 管理页 | 同正式 |
| 12 | GET | `/index` | 页面 | 301 → `/` | 同正式 |
| 13 | GET | `/index.html` | 页面 | 301 → `/` | 同正式 |
| 14 | GET | `/manager` | 页面 | 302 → `/` | 同正式 |
| 15 | POST | `/voice-tasks` | 任务队列 | 提交任务 | 【测试专用】支持 `time` |
| 16 | GET | `/voice-tasks` | 任务队列 | 查询任务 | 【与正式不同】无 Redis |
| 17 | POST | `/voice-tasks/query` | 任务队列 | JSON 查询 | 【与正式不同】无 Redis |
| 18 | GET | `/voice-tasks/{task_id}` | 任务队列 | 单个任务 | 【与正式不同】仅 JSON |
| 19 | DELETE | `/voice-tasks/{task_id}` | 任务队列 | 删除/取消 | 同正式逻辑 |

---

## 6. 接口详细说明

---

### 6.1 GET /healthz

**说明**：服务存活探测。

| 项目 | 内容 |
|---|---|
| 方法 | GET |
| 路径 | `/healthz` |

**成功响应**（HTTP 200）：

```json
{
  "status": "ok",
  "mock": true,
  "index_url": "http://127.0.0.1:5400/",
  "runtime_root": "D:/.../mock_runtime"
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `status` | string | `"ok"` |
| **`mock`** | **bool** | **【测试专用】** 固定 `true`，表示模拟服务 |
| `index_url` | string | 管理页 URL |
| **`runtime_root`** | **string** | **【测试专用】** 运行时数据目录绝对路径 |

**【与正式不同】** 正式服务无 `mock`、`runtime_root` 字段。

---

### 6.2 POST /set_model

**说明**：模拟切换权重，**不加载任何模型文件**。

| 项目 | 内容 |
|---|---|
| 方法 | POST |
| 路径 | `/set_model` |
| Content-Type | `application/json` |

**请求 Body**

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `gpt_model_path` | string | 二选一 | 任意字符串即可 |
| `sovits_model_path` | string | 二选一 | 任意字符串即可 |

**成功响应**：

```json
{
  "code": 0,
  "message": "Success (mock: no model loaded)",
  "data": {}
}
```

**【与正式不同】** 正式服务会真实调用 `change_gpt_weights` / `change_sovits_weights`。

---

### 6.3 POST /synthesize

**说明**：同步返回**占位 WAV**，不调用 GPT/SoVITS。

| 项目 | 内容 |
|---|---|
| 方法 | POST |
| 路径 | `/synthesize` |
| Content-Type | `application/json` |

**请求 Body 字段**（与正式 §6.3 字段名相同）

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `ref_wav_path` | string | 是* | — | 参考音频路径，**须存在** |
| `text` | string | 是 | — | 待合成文本 |
| `text_language` | string | 否 | `all_zh` | 须在支持列表内 |
| `prompt_text` | string | 否 | `""` | 参考文本 |
| `prompt_language` | string | 否 | `all_zh` | 参考语种 |
| `how_to_cut` | string | 否 | `cut1` | 切分策略（模拟中不实际切句推理） |
| `top_k` | int | 否 | `5` | 模拟中忽略 |
| `top_p` | float | 否 | `1` | 模拟中忽略 |
| `temperature` | float | 否 | `1` | 模拟中忽略 |
| `ref_free` | bool | 否 | `false` | 模拟中忽略 |
| `model_id` | string | 否 | — | 可补全 `ref_wav_path` |
| `user_id` | string | 否 | — | 与 `task_id` 同传则保存 WAV |
| `task_id` | string | 否 | — | 保存路径用 |

**成功响应**：HTTP 200，`Content-Type: audio/wav`，Body 为占位 WAV。

- 时长约：`max(1.0, len(text) * 0.05)` 秒

**失败响应**：

| 情况 | code / HTTP | message |
|---|---|---|
| 音模不存在 | HTTP 404 | `音模不存在` |
| 缺少参数 | 3 | `缺少参数: ref_wav_path 或 text` |
| 语种不支持 | 3 | `语言不支持` |
| 参考文件不存在 | 2 | `参考音频不存在: <path>` |

**【与正式不同】** 正式服务返回真实合成语音；且用 librosa 校验参考音频 3～10 秒。

---

### 6.4 POST /voice-models

**说明**：创建音模（上传头像、参考音频）。

| 项目 | 内容 |
|---|---|
| 方法 | POST |
| 路径 | `/voice-models` |
| Content-Type | `multipart/form-data` |

**表单字段**（与正式 §6.4 相同）

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `model_name` | string | 是 | 音模名称 |
| `model_id` | string | 是 | 唯一标识，清洗为 `[a-zA-Z0-9_-]` |
| `gender` | int | 是 | `1` 男 / `2` 女 |
| `prompt_text` | string | 否 | 参考文本 |
| `prompt_language` | string | 否 | 默认 `all_zh` |
| `categories` | string[] | 否 | 分类 ID，须已存在 |
| `avatar` | file | 是 | 头像 |
| `refer_wav` | file | 是 | 参考音频 |

**【与正式不同】**

- 存储在 `mock_runtime/audio/<model_id>/`
- 参考音频仅校验文件大小 ≥ 100 字节，**不**校验 3～10 秒时长

**成功 / 失败响应**：结构与正式 §6.4 相同。

**磁盘路径**：`mock_runtime/audio/<model_id>/avatar.*`、`refer.*`

---

### 6.5 GET /voice-models

与正式服务 **§6.5** 完全相同（Query、`page`/`page_size`/`name`/`category`、分页响应结构）。

**【与正式不同】** 读取 `mock_runtime/voice_models.json`。

---

### 6.6 DELETE /voice-models/{model_id}

与正式服务 **§6.6** 完全相同。

删除目录：`mock_runtime/audio/<model_id>/`

---

### 6.7 POST /voice-models/{model_id}/categories

与正式服务 **§6.7** 完全相同。

---

### 6.8 POST /voice-categories

与正式服务 **§6.8** 完全相同。

存储：`mock_runtime/voice_categories.json`

---

### 6.9 GET /voice-categories

与正式服务 **§6.9** 完全相同。

---

### 6.10 DELETE /voice-categories/{category_id}

与正式服务 **§6.10** 完全相同。

---

### 6.11 GET /

与正式服务 **§6.11** 相同：返回 HTML 管理页。

---

### 6.12 GET /index、GET /index.html

与正式服务 **§6.12** 相同：HTTP 301 → `/`。

---

### 6.13 GET /manager

与正式服务 **§6.13** 相同：HTTP 302 → `/`。

---

### 6.14 POST /voice-tasks

**说明**：提交模拟合成任务。

| 项目 | 内容 |
|---|---|
| 方法 | POST |
| 路径 | `/voice-tasks` |
| Content-Type | `application/json` |

#### 与正式服务相同的字段

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `user_id` | string | **是** | — | 用户 ID |
| `task_id` | string | **是** | — | 任务 ID |
| `text` | string | 建议 | `""` | 待合成文本 |
| `text_language` | string | 否 | `all_zh` | 目标语种 |
| `model_id` | string | 建议 | — | 音模 ID |
| `how_to_cut` | string | 否 | `cut1` | 切分策略（影响模拟分段数） |
| `prompt_text` | string | 否 | `""` | 参考文本 |
| `prompt_language` | string | 否 | `all_zh` | 参考语种 |
| `ref_wav_path` | string | 否 | — | 参考音频路径 |
| `top_k` | int | 否 | `5` | 模拟中不参与推理 |
| `top_p` | float | 否 | `1.0` | 模拟中不参与推理 |
| `temperature` | float | 否 | `1.0` | 模拟中不参与推理 |

#### 【测试专用】正式服务没有的字段

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| **`time`** | **number** | **否** | **模拟任务总耗时（秒）**，如 `10`、`60`、`120`；范围 **1～7200**。**绝不同步到正式服务。** |

**`time` 行为**

| 情况 | 模拟总耗时 |
|---|---|
| 传入合法 `time` | 等于指定秒数 |
| 未传 / 空 | **随机 5～45 秒** |
| 非法（非数字、≤0） | 返回 code=3 |

**请求示例（含测试专用参数）**：

```json
{
  "user_id": "U10001",
  "task_id": "T20250101ABC123",
  "model_id": "demo",
  "text": "测试模拟合成的文本。",
  "text_language": "all_zh",
  "how_to_cut": "cut1",
  "time": 60
}
```

**成功响应（入队）**：

```json
{
  "code": 0,
  "message": "",
  "data": {
    "user_id": "U10001",
    "task_id": "T20250101ABC123",
    "status": "wait",
    "progress": 0,
    "segments_done": 0,
    "total_segments": 0,
    "time": 60,
    "model_id": "demo",
    "text": "测试模拟合成的文本。",
    "text_language": "all_zh",
    "how_to_cut": "cut1",
    "created_at": 1730000000,
    "updated_at": 1730000000,
    "result_url": null
  }
}
```

**执行中额外字段【测试专用】**：

| 字段 | 说明 |
|---|---|
| **`mock_duration_sec`** | 实际采用的模拟总秒数（float） |

**完成后**：

| 字段 | 说明 |
|---|---|
| `status` | `done` |
| `progress` | `100` |
| `result_url` | 占位 WAV 的 HTTP URL |
| `result_path` | `mock_runtime/output/<user_id>/<task_id>/<task_id>.wav` |

**【与正式不同】** 正式服务无 `time`、`mock_duration_sec`；产物为真实合成语音。

**失败示例**：

```json
{ "code": 3, "message": "缺少 user_id", "data": {} }
```

```json
{ "code": 3, "message": "time 必须为正数（秒）", "data": {} }
```

---

### 6.15 GET /voice-tasks

**说明**：查询任务（分页或多 ID）。

与正式服务 **§6.15** 请求参数、响应结构**相同**：

| 参数 | 说明 |
|---|---|
| `user_id` | 必填 |
| `page` / `page_size` | 分页 |
| `task_ids` / `ids` / `task_id` | 多 ID 模式 |

**【与正式不同】**

- 仅查询 `mock_runtime/voice_task.json`
- **不使用 Redis**

---

### 6.16 POST /voice-tasks/query

与正式服务 **§6.16** 请求 Body、响应结构**相同**。

**【与正式不同】** 无 Redis，仅 JSON 索引。

---

### 6.17 GET /voice-tasks/{task_id}

**说明**：按 task_id 查询单条任务。

| 项目 | 内容 |
|---|---|
| 方法 | GET |
| 路径 | `/voice-tasks/{task_id}` |

**成功**：`code=0`，`data` 为任务对象（含 `mock_duration_sec`、`time` 等测试字段，若存在）。

**失败**：

```json
{ "code": 2, "message": "not found", "data": {} }
```

**【与正式不同】** 正式服务先查 Redis 再查 JSON；模拟仅查 JSON。

---

### 6.18 DELETE /voice-tasks/{task_id}

与正式服务 **§6.18** 行为一致：

| 状态 | 行为 |
|---|---|
| 运行中 | 取消 → `error` → 清理 |
| 其他 | 删索引与 `mock_runtime/output/...` |

**成功响应**：

```json
{ "code": 0, "message": "", "data": { "removed": true } }
```

或 `removed: false`（任务不存在）。

**【与正式不同】** 不操作 Redis。

---

## 7. 参数标识符附录

与 [正式服务API.md](./正式服务API.md) §7 **完全相同**（语种、切分策略标识符及中文别名）。

---

## 8. 数据存储与日志

| 项目 | 测试模拟服务 |
|---|---|
| Redis | **不使用** |
| 音模索引 | `mock_runtime/voice_models.json` |
| 分类索引 | `mock_runtime/voice_categories.json` |
| 任务索引 | `mock_runtime/voice_task.json` |
| 访问日志 | `mock_runtime/Logs/access.log`（GET `/voice-tasks` 不记录） |
| 任务日志 | **无** 按任务独立文件（正式服务有） |

---

## 9. 启动与环境变量

### 9.1 推荐启动命令

```powershell
python api_ms_mock.py --host 0.0.0.0 --port 5400 \
  --root-path /ttsHttp --base-url https://36.136.54.165/media \
  --max-concurrent 4
```

### 9.2 命令行参数（全量）

| 参数 | 环境变量 | 说明 | 默认 |
|---|---|---|---|
| `--host` | `infer_api_host` | 绑定地址 | `0.0.0.0` |
| `--port` | `infer_api_port` | 端口 | `9882` |
| `--root-path` / `--prefix` | `infer_api_root_path` | URL 路径前缀 | 空 |
| `--base-url` / `--domain` | `infer_api_base_url` | URL 域名前缀 | 空 |
| **`--max-concurrent`** | **`MOCK_MAX_CONCURRENT`** | **【测试专用】** 最大并发任务数 | **`4`** |
| **`--segment-delay`** | **`MOCK_SEGMENT_DELAY_SEC`** | **【测试专用】** 兜底参数 | **`0.3`** |

### 9.3 【测试专用】环境变量

| 变量 | 说明 | 默认 |
|---|---|---|
| `MOCK_MAX_CONCURRENT` | 最大并发 | `4` |
| `MOCK_SEGMENT_DELAY_SEC` | 段延迟兜底 | `0.3` |
| `MOCK_DEFAULT_SAMPLE_RATE` | 占位 WAV 采样率 | `32000` |

### 9.4 nginx 示例

```nginx
location /media/ttsHttp/ {
    proxy_pass http://192.168.10.135:5400;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header X-Forwarded-Host $host;
}
```

---

## 10. 测试检查清单

联调时建议逐项验证：

- [ ] `GET /healthz` 返回 `"mock": true`
- [ ] `GET /` 管理页可打开；`/index` 301 到 `/`
- [ ] 配置 `--root-path` + `--base-url` 后，`avatar_url` / `result_url` 为公网完整路径
- [ ] `POST /voice-models` 上传成功
- [ ] `POST /voice-tasks` **不传** `time`：随机 5～45 秒内完成
- [ ] `POST /voice-tasks` **传** `"time": 10`：约 10 秒完成，有 `mock_duration_sec`
- [ ] `DELETE /voice-tasks/{id}` 可取消运行中任务
- [ ] `GET /output/.../*.wav` 可下载占位文件

**说明**：`time` 参数、`mock_duration_sec` 等仅用于测试模拟，**不同步**到正式服务。

---

**相关文档**：[正式服务API.md](./正式服务API.md)
