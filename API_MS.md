# GPT-SoVITS MindSpore 推理 API 文档（api_ms.py）

## 概览

- 基于 FastAPI 的文本转语音（TTS）服务，支持：
  - 参考音频 + 参考文本的语音克隆
  - 多语种文本处理与分句
  - 音模（参考音频/头像/元数据）管理
  - 任务队列、进度追踪、取消任务
- 静态资源挂载：头像/参考音频与任务产物可直接通过 HTTP 访问。

默认端口：`9881`。Base URL（示例）：`http://<host>:9881`

### 响应结构规范

- 单一返回：
```json
{ "code": 0, "message": "", "data": {} }
```
- 分页返回：
```json
{
  "code": 0,
  "message": "",
  "data": { "total": 9, "pageIndex": 1, "pageSize": 10, "data": [] }
}
```
- code 约定：0 成功；1 失败；2 资源错误；3 参数错误

### 端点速览（仅列出前端已调用的接口）

- 音模新增：POST `/voice-models`
- 音模列表：GET `/voice-models`（支持按名称查询 `?name=关键词`）
- 音模删除：DELETE `/voice-models/{model_id}`
- 提交任务：POST `/voice-tasks`
- 批量/分页任务：GET `/voice-tasks`（必须提供 `user_id`；可选 `task_id` 精准检索）
- 删除/取消任务：DELETE `/voice-tasks/{task_id}`
- 静态资源：GET `/output/<user_id>/<task_id>/<task_id>.wav`（任务产物下载）

---

## 健康检查

GET `/healthz`

响应：
```json
{ "status": "ok", "index_url": "http://<host>:9881/index" }
```

---

## 文本转语音（直连）

POST `/synthesize`

请求（application/json）：
```json
{
  "ref_wav_path": "./audio/<model_id>/refer.wav",     // 可选：与 model_id 二选一
  "prompt_text": "这是参考文本",                       // 可选
  "prompt_language": "中文",                          // 可选：中文/英文/日文/中英混合/日英混合/多语种混合
  "text": "需要合成的目标文本",                       // 必填
  "text_language": "中文",                            // 必填（同上枚举）
  "how_to_cut": "凑四句一切",                         // 可选：不切/凑四句一切/凑50字一切/按中文句号。切/按英文句号.切/按标点符号切
  "top_k": 5,                                         // 可选
  "top_p": 1.0,                                       // 可选
  "temperature": 1.0,                                 // 可选
  "ref_free": false,                                  // 可选：无参考文本模式
  "model_id": "your_model_id",                        // 可选：提供则自动填充 ref_wav_path/prompt_text/prompt_language
  "user_id": "U10001",                                // 可选：用于保存路径 output/<user_id>/<task_id>/<task_id>.wav
  "task_id": "T20250101ABC123"                        // 可选：用于保存路径；不传则自动生成
}
```

响应：音频流 `audio/wav`（返回整段 WAV 字节流）。

示例（保存为 out.wav）：
```bash
curl -X POST "http://<host>:9881/synthesize" \
  -H "Content-Type: application/json" \
  --data '{
    "model_id": "example",
    "text": "你好世界。",
    "text_language": "中文",
    "how_to_cut": "凑四句一切"
  }' --output out.wav
```

注意：
- `ref_wav_path` 需 3~10 秒长度；否则返回 500 错误。
- `model_id` 存在时会作为默认来源；显式传参优先生效。

---

## 音模管理（头像/参考音频/元数据）

### 新增音模

POST `/voice-models`（multipart/form-data）

表单字段：
- `model_name`（string）音模名（必填）
- `model_id`（string）音模标识（必填，仅字母数字`-_`）
- `gender`（int）1=男，2=女（必填）
- `prompt_text`（string）参考文本（可选）
- `prompt_language`（string）参考文本语种（默认：中文）
- `categories`（array[string]）分类 ID，多选；可留空
- `avatar`（file）头像（必填）
- `refer_wav`（file）参考音频（必填，3~10秒）

响应：
```json
{
  "code": 0,
  "data": {
    "model_id": "...",
    "model_name": "...",
    "gender": 1,
    "prompt_text": "...",
    "prompt_language": "中文",
    "avatar_url": "http://<host>:9881/audio/<model_id>/avatar.png",
    "refer_wav_url": "http://<host>:9881/audio/<model_id>/refer.wav",
    "created_at": 1730000000
  }
}
```

文件存储位置：`./audio/<model_id>/avatar.*` 与 `./audio/<model_id>/refer.*`

示例（curl 上传）：
```bash
curl -X POST "http://<host>:9881/voice-models" \
  -F model_name="示例音模" \
  -F model_id="example" \
  -F gender=1 \
  -F prompt_text="这是参考文本" \
  -F prompt_language="中文" \
  -F avatar=@./avatar.png \
  -F refer_wav=@./refer.wav
```

### 列表音模（分页/分类）

GET `/voice-models?page=1&page_size=10&name=关键词&category=<cat_id>`

响应：
```json
{
  "code": 0,
  "message": "",
  "data": {
    "total": 12,
    "pageIndex": 1,
    "pageSize": 10,
    "data": [
      {
        "model_id": "...",
        "model_name": "...",
        "gender": 1,
        "prompt_text": "...",
        "prompt_language": "中文",
        "avatar_url": "...",
        "refer_wav_url": "...",
        "created_at": 1730000000
      }
    ]
  }
}
```

说明：
- `name` 为可选的名称关键字（不区分大小写、包含匹配），与分页参数可同时使用。

### 删除音模 / 设置分类

- DELETE `/voice-models/{model_id}`

响应：
```json
{ "code": 0, "message": "deleted" }
```

说明：删除后对应 `./audio/<model_id>` 目录被移除。

- POST `/voice-models/{model_id}/categories`

请求：
```json
{ "categories": ["cat_a","cat_b"] }
```

说明：分类必须已存在；传空数组则清空分类。

### 分类管理

- GET `/voice-categories`：返回所有分类列表
- POST `/voice-categories`

```json
{ "category_id": "cn", "category_name": "中文主播", "description": "示例" }
```

- DELETE `/voice-categories/{category_id}`：删除分类并自动从音模中移除该分类

---

## 任务队列（排队合成/进度/取消）

### 提交任务

POST `/voice-tasks`

请求（application/json）：
```json
{
  "user_id": "U10001",                     // 必填：用户ID
  "task_id": "T20250101ABC123",            // 必填：前端生成（时间戳+随机）
  "model_id": "your_model_id",             // 建议提供
  "text": "需要合成的文本",                  // 必填
  "text_language": "中文",                   // 必填
  "how_to_cut": "凑四句一切",                // 可选（同直连）
  "prompt_text": "参考文本",                 // 可选
  "prompt_language": "中文",                 // 可选
  "ref_wav_path": "./audio/<model_id>/refer.wav", // 可选
  "top_k": 5,
  "top_p": 1.0,
  "temperature": 1.0
}
```

响应（初始入队条目）：
```json
{
  "code": 0,
  "data": {
    "user_id": "U10001",
    "task_id": "...",
    "status": "wait",             // wait|run|done|error
    "progress": 0,                 // 百分比
    "segments_done": 0,            // 已完成段数
    "total_segments": 0,           // 总段数（切句后）
    "created_at": 1730000000,
    "updated_at": 1730000000,
    "result_url": null             // 完成后提供 /output/<user_id>/<task_id>/<task_id>.wav
  }
}
```

说明：
- 任务仅在索引文件中记录状态 `./voice_task.json`；不再使用 `task/done/error` 目录。
- 产物 WAV 直接保存在 `./output/<user_id>/<task_id>/<task_id>.wav`（前端以 `result_url` 下载/预览）。
- 每个任务的日志：`./Logs/<YYYYMMDD>/<HH>/<task_id>.log`（记录排队、开始、进度、完成与异常，按小时分目录）

### 批量 / 分页查询任务

#### GET `/voice-tasks`
- **分页模式**：必须带 `user_id`、`page`、`page_size`。示例：`/voice-tasks?user_id=U10001&page=1&page_size=10`
- **多 ID 模式**：带 `user_id` 与 `task_ids`（逗号/空格分隔）。示例：`/voice-tasks?user_id=U10001&task_ids=T1,T2`
  - 返回的 `pageIndex` 固定为 1，`pageSize` 为匹配数量。

#### POST `/voice-tasks/query`
Body：
```json
{
  "user_id": "U10001",
  "task_ids": ["T1","T2"],   // 可选；为空则走分页
  "page": 1,
  "page_size": 20
}
```

说明：
- 两种模式都共用 Redis 优先、JSON 兜底的查询逻辑。
- `task_ids` 必须搭配 `user_id`，支持海量 ID 查询，避免 URL 过长。

### 查询 / 删除单个任务

- GET `/voice-tasks/{task_id}`：返回单条任务详情（优先 Redis，回退本地 JSON）。
- DELETE `/voice-tasks/{task_id}`：取消/删除；若正在运行会标记取消，完成后清理输出与日志。

### 删除任务（包含取消运行中任务）

DELETE `/voice-tasks/{task_id}`

行为：
- 运行中：设置取消标记，任务在最近的检查点立即中断，状态写为 `error`；随后移除索引与临时文件。
- 已完成/排队：移除索引与对应目录下的 json/wav。

响应：
```json
{ "code": 0, "removed": true }
```

---

## 前端管理页面

GET `/manager`

- 提供简易管理 UI：上传音模、音模列表（含参考音频试听/详情/删除）、选择音模发起合成或提交任务、任务列表（进度/下载/删除）。
- 任务列表每 2 秒自动刷新。

---

## 静态目录

- `/audio`：头像与参考音频（示例：`/audio/<model_id>/avatar.png`、`/audio/<model_id>/refer.wav`）
- `/output`：最终产物（示例：`/output/<user_id>/<task_id>/<task_id>.wav`）

---

## 其他说明

- 语种枚举（`prompt_language`/`text_language`）：`中文`、`英文`、`日文`、`中英混合`、`日英混合`、`多语种混合`
- 参考音频时长限制：3~10 秒
- 输出目录：`./output/`（`synthesize` 直连模式与任务模式皆会保存）
- 环境变量：
  - `is_half`（默认 True）半精模式
  - `infer_api_port`（默认 9881）端口
  - `infer_api_host`（默认 0.0.0.0）绑定地址
  - `infer_device_id`（默认 0）Ascend 设备编号
  - `ASCEND_DEVICE_ID` / `DEVICE_ID`（与 `infer_device_id` 同步，供底层运行时读取）
  - `gpt_path` / `sovits_path` / `cnhubert_base_path` / `bert_path`

### 启动参数

```bash
# 仅 Ascend 场景：指定第 1 张 NPU 并启动服务
python api_ms.py --device 1 --host 0.0.0.0 --port 9881
```

说明：
- 若不传 `--device`，默认使用 0 号 NPU
- 启动时会打印当前上下文：`[Device] MindSpore context set: device_target=Ascend, device_id=<n>`

### 访问日志

- 路径：`./Logs/access.log`
- 切割：每 6 小时滚动（一天 4 份），保留约 7 天（28 份）
- 格式：`时间戳\t客户端IP\t方法 路径\t状态码\t耗时ms`
- 忽略：GET `/voice-tasks`（轮询接口）不记录，避免日志过量

---

## 常见错误

- 400 缺少参数：检查 `text`、`text_language` 与参考来源（`model_id` 或 `ref_wav_path`）
- 404 音模不存在：检查 `model_id`
- 500 模型/依赖错误：确认权重路径、`ffmpeg`/CUDA/MindSpore 环境

---

## 状态码与返回类型（约定）

- 除 `/synthesize` 返回音频流外，其余接口统一以 JSON 返回业务码：
  - code=0 成功；code=1 失败；code=2 资源错误；code=3 参数错误
  - HTTP 状态通常为 200（业务码区分结果），少量兜底错误可能返回 4xx/5xx

返回类型：
- JSON：除 `/synthesize` 外的所有接口
- audio/wav：`/synthesize`（整段 WAV 字节流）


