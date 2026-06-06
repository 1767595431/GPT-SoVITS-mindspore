# 正式服务 API 文档

> **对应程序**：`api_ms.py`  
> **服务类型**：MindSpore + GPT-SoVITS 真实语音合成（Ascend NPU）  
> **OpenAPI 文档**：启动后访问 `http://<host>:<port>/docs`

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
10. [常见错误](#10-常见错误)

---

## 1. 服务说明

| 项目 | 内容 |
|---|---|
| 默认端口 | `9881` |
| 默认绑定 | `0.0.0.0` |
| 并发执行 | **串行**：同一时刻最多 **1** 个任务在跑推理（`_single_worker_lock`） |
| 任务 `time` 参数 | **不支持** |

**数据目录（项目根目录）**

| 路径 | 用途 |
|---|---|
| `./audio/` | 音模头像、参考音频 |
| `./output/` | 任务合成产物 WAV |
| `./voice_models.json` | 音模索引 |
| `./voice_categories.json` | 分类索引 |
| `./voice_task.json` | 任务索引（Redis 兜底同步） |
| `./Logs/` | 访问日志、任务日志 |

---

## 2. 通用约定

### 2.1 HTTP 与 Content-Type

- 业务接口（除 `/synthesize` 成功时返回 WAV）均返回 **JSON**，HTTP 状态码通常为 **200**。
- 以响应体中的 `code` 区分业务成功/失败。

### 2.2 普通响应结构

```json
{
  "code": 0,
  "message": "",
  "data": {}
}
```

### 2.3 分页响应结构

```json
{
  "code": 0,
  "message": "",
  "data": {
    "total": 100,
    "pageIndex": 1,
    "pageSize": 10,
    "data": []
  }
}
```

### 2.4 业务码 code

| code | 含义 | 典型 message |
|---|---|---|
| 0 | 成功 | `""` 或 `"deleted"` / `"updated"` 等 |
| 1 | 失败 | 推理异常、内部错误等 |
| 2 | 资源错误 | 音模不存在、任务 not found 等 |
| 3 | 参数错误 | 缺少必填项、参考音频时长不符等 |

### 2.5 任务状态 status

| 值 | 说明 |
|---|---|
| `wait` | 已入队，等待执行 |
| `run` | 正在推理合成 |
| `done` | 完成，可下载 `result_url` |
| `error` | 失败或已取消 |

---

## 3. 动态 URL 规则

响应中的 `avatar_url`、`refer_wav_url`、`result_url` **每次请求动态拼接**，**不写入** JSON 持久化。

| 启动配置 | 资源 URL 形式 |
|---|---|
| **同时**配置 `--base-url` 与 `--root-path` | `{base-url}{root-path}/audio/...` 或 `.../output/...` |
| 未同时配置 | `http(s)://<客户端可见主机>:<端口>/audio/...` 或 `/output/...` |

示例：

- 内网直连：`http://192.168.1.10:9881/output/U1/T1/T1.wav`
- 公网代理：`https://36.136.54.165/media/ttsHttp/output/U1/T1/T1.wav`

---

## 4. 静态资源

由 FastAPI `StaticFiles` 挂载，**无需**经过业务 JSON 接口。

| 方法 | 路径模式 | 说明 |
|---|---|---|
| GET | `/audio/<model_id>/<文件名>` | 音模头像、参考音频，如 `avatar.png`、`refer.wav` |
| GET | `/output/<user_id>/<task_id>/<task_id>.wav` | 任务合成产物 |

磁盘路径：

- 音模：`./audio/<model_id>/`
- 产物：`./output/<user_id>/<task_id>/<task_id>.wav`

---

## 5. 接口列表（全量）

共 **19** 个业务/页面路由（不含静态文件目录列举）。

| 序号 | 方法 | 路径 | 分组 | 说明 |
|---|---|---|---|---|
| 1 | GET | `/healthz` | 其他 | 健康检查 |
| 2 | POST | `/set_model` | 其他 | 切换 GPT/SoVITS 权重（内部） |
| 3 | POST | `/synthesize` | 其他 | 直接合成，返回 WAV 流（内部） |
| 4 | POST | `/voice-models` | 音模管理 | 新增音模 |
| 5 | GET | `/voice-models` | 音模管理 | 音模列表（分页/筛选） |
| 6 | DELETE | `/voice-models/{model_id}` | 音模管理 | 删除音模 |
| 7 | POST | `/voice-models/{model_id}/categories` | 音模管理 | 设置音模分类 |
| 8 | POST | `/voice-categories` | 音模管理 | 新增分类 |
| 9 | GET | `/voice-categories` | 音模管理 | 分类列表 |
| 10 | DELETE | `/voice-categories/{category_id}` | 音模管理 | 删除分类 |
| 11 | GET | `/` | 页面 | 管理页 HTML |
| 12 | GET | `/index` | 页面 | 301 重定向到 `/` |
| 13 | GET | `/index.html` | 页面 | 301 重定向到 `/` |
| 14 | GET | `/manager` | 页面 | 302 重定向到 `/` |
| 15 | POST | `/voice-tasks` | 任务队列 | 提交合成任务 |
| 16 | GET | `/voice-tasks` | 任务队列 | 查询任务（分页或多 ID） |
| 17 | POST | `/voice-tasks/query` | 任务队列 | 查询任务（JSON Body） |
| 18 | GET | `/voice-tasks/{task_id}` | 任务队列 | 查询单个任务 |
| 19 | DELETE | `/voice-tasks/{task_id}` | 任务队列 | 删除/取消任务 |

---

## 6. 接口详细说明

---

### 6.1 GET /healthz

**说明**：服务存活探测。

| 项目 | 内容 |
|---|---|
| 方法 | GET |
| 路径 | `/healthz` |
| 请求体 | 无 |
| Query | 无 |

**成功响应**（HTTP 200，**非**统一 `{code,message,data}` 格式）：

```json
{
  "status": "ok",
  "index_url": "http://127.0.0.1:9881/"
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `status` | string | 固定 `"ok"` |
| `index_url` | string | 管理页完整 URL（按动态 URL 规则计算） |

**curl 示例**：

```bash
curl "http://127.0.0.1:9881/healthz"
```

---

### 6.2 POST /set_model

**说明**：运行期切换 GPT 或 SoVITS 模型权重文件（内部调试用）。

| 项目 | 内容 |
|---|---|
| 方法 | POST |
| 路径 | `/set_model` |
| Content-Type | `application/json` |

**请求 Body 字段**

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `gpt_model_path` | string | 二选一 | GPT 权重路径 |
| `sovits_model_path` | string | 二选一 | SoVITS 权重路径 |

至少提供其中一个路径。

**请求示例**：

```json
{
  "gpt_model_path": "/path/to/s1bert25hz-...ckpt",
  "sovits_model_path": "/path/to/s2G488k.pth"
}
```

**成功响应**：

```json
{
  "code": 0,
  "message": "Success",
  "data": {}
}
```

**失败响应示例**：

```json
{ "code": 3, "message": "缺少模型路径", "data": {} }
```

```json
{ "code": 1, "message": "具体异常信息", "data": {} }
```

---

### 6.3 POST /synthesize

**说明**：不走任务队列，**同步**调用模型合成，直接返回 WAV 二进制流。

| 项目 | 内容 |
|---|---|
| 方法 | POST |
| 路径 | `/synthesize` |
| Content-Type | `application/json` |

**请求 Body 字段**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `ref_wav_path` | string | 是* | — | 参考音频路径；若提供 `model_id` 且音模存在，可从音模补全 |
| `text` | string | 是 | — | 待合成文本 |
| `text_language` | string | 是 | `all_zh` | 目标文本语种，须在支持列表内（见 §7） |
| `prompt_text` | string | 否 | `""` | 参考文本；空则 `ref_free` 效果 |
| `prompt_language` | string | 否 | `all_zh` | 参考文本语种 |
| `how_to_cut` | string | 否 | `cut1` | 切分策略（见 §7） |
| `top_k` | int | 否 | `5` | 采样参数 |
| `top_p` | float | 否 | `1` | 采样参数 |
| `temperature` | float | 否 | `1` | 采样参数 |
| `ref_free` | bool | 否 | `false` | 是否无参考文本模式 |
| `model_id` | string | 否 | — | 音模 ID；用于补全 `ref_wav_path`、`prompt_text`、`prompt_language` |
| `user_id` | string | 否 | — | 若与 `task_id` 同时提供，额外保存 WAV 到 output 目录 |
| `task_id` | string | 否 | — | 保存目录名，见上 |

\* 当 `model_id` 有效且音模含参考音频时，`ref_wav_path` 可被补全。

**请求示例**：

```json
{
  "ref_wav_path": "./audio/demo/refer.wav",
  "prompt_text": "这是参考文本。",
  "prompt_language": "all_zh",
  "text": "需要合成的目标文本。",
  "text_language": "all_zh",
  "how_to_cut": "cut1",
  "top_k": 5,
  "top_p": 1.0,
  "temperature": 1.0,
  "ref_free": false
}
```

**成功响应**：

- HTTP 200
- `Content-Type: audio/wav`
- Body：WAV 二进制数据（非 JSON）

**失败响应**（JSON）：

| 情况 | code | 示例 message |
|---|---|---|
| 音模不存在 | 404（HTTP） | `{"code":404,"message":"音模不存在"}` |
| 缺少参数 | 3 | `缺少参数: ref_wav_path 或 text` |
| 语种不支持 | 3 | `语言不支持` |
| 推理失败 | 1 | 异常信息字符串 |

---

### 6.4 POST /voice-models

**说明**：上传头像与参考音频，创建音模记录。

| 项目 | 内容 |
|---|---|
| 方法 | POST |
| 路径 | `/voice-models` |
| Content-Type | `multipart/form-data` |

**表单字段**

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `model_name` | string | 是 | 音模显示名称 |
| `model_id` | string | 是 | 唯一标识；入库前会清洗为仅 `[a-zA-Z0-9_-]` |
| `gender` | int | 是 | `1`=男，`2`=女 |
| `prompt_text` | string | 否 | 参考文本，默认空 |
| `prompt_language` | string | 否 | 默认 `all_zh` |
| `categories` | string[] | 否 | 分类 ID 列表；**须已存在**，可多次传同名 form 字段 |
| `avatar` | file | 是 | 头像图片 |
| `refer_wav` | file | 是 | 参考音频；**时长须在 3～10 秒**（16kHz 采样点数 48000～160000） |

**成功响应 data 字段**

| 字段 | 类型 | 说明 |
|---|---|---|
| `model_id` | string | 音模 ID |
| `model_name` | string | 名称 |
| `gender` | int | 1 或 2 |
| `prompt_text` | string | 参考文本 |
| `prompt_language` | string | 存储的语种标识符 |
| `avatar_url` | string | 动态拼接的头像 URL |
| `refer_wav_url` | string | 动态拼接的参考音频 URL |
| `created_at` | int | Unix 时间戳（秒） |
| `categories` | string[] | 分类 ID 列表 |
| `avatar_path` | string | 磁盘路径（内部字段，可能出现） |
| `refer_wav_path` | string | 磁盘路径（内部字段，可能出现） |

**成功示例**：

```json
{
  "code": 0,
  "message": "",
  "data": {
    "model_id": "demo",
    "model_name": "演示音模",
    "gender": 1,
    "prompt_text": "你好",
    "prompt_language": "all_zh",
    "avatar_url": "http://127.0.0.1:9881/audio/demo/avatar.png",
    "refer_wav_url": "http://127.0.0.1:9881/audio/demo/refer.wav",
    "created_at": 1730000000,
    "categories": ["cn"]
  }
}
```

**失败响应**

| 情况 | code | message 示例 |
|---|---|---|
| 音模 ID 已存在 | 2 | `音模标识已存在` |
| 参考音频时长不符 | 3 | `参考音频需在 3~10 秒范围内` |
| 音频解析失败 | 3 | `参考音频解析失败，请上传合法的音频文件（WAV/MP3/OGG 等）` |
| 分类不存在 | 3 | `分类不存在: xxx` |

**curl 示例**：

```bash
curl -X POST "http://127.0.0.1:9881/voice-models" \
  -F "model_name=演示音模" \
  -F "model_id=demo" \
  -F "gender=1" \
  -F "prompt_text=你好" \
  -F "prompt_language=all_zh" \
  -F "avatar=@./avatar.png" \
  -F "refer_wav=@./refer.wav"
```

**磁盘存储**：`./audio/<model_id>/avatar.<ext>`、`./audio/<model_id>/refer.<ext>`

---

### 6.5 GET /voice-models

**说明**：分页查询音模，支持名称关键字与分类筛选。

| 项目 | 内容 |
|---|---|
| 方法 | GET |
| 路径 | `/voice-models` |

**Query 参数**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `page` | int | 否 | `1` | 页码，最小 1 |
| `page_size` | int | 否 | `10` | 每页条数，范围 1～100 |
| `name` | string | 否 | — | 名称关键字，**不区分大小写、包含匹配** |
| `category` | string | 否 | — | 分类 ID，仅返回包含该分类的音模 |

**成功响应**：分页结构，`data.data` 为音模数组，每项字段同 §6.4，且含 `categories`。

**请求示例**：

```
GET /voice-models?page=1&page_size=10&name=主播&category=cn
```

**成功示例**：

```json
{
  "code": 0,
  "message": "",
  "data": {
    "total": 2,
    "pageIndex": 1,
    "pageSize": 10,
    "data": [
      {
        "model_id": "demo",
        "model_name": "演示音模",
        "gender": 1,
        "prompt_text": "你好",
        "prompt_language": "all_zh",
        "avatar_url": "http://127.0.0.1:9881/audio/demo/avatar.png",
        "refer_wav_url": "http://127.0.0.1:9881/audio/demo/refer.wav",
        "created_at": 1730000000,
        "categories": ["cn"]
      }
    ]
  }
}
```

---

### 6.6 DELETE /voice-models/{model_id}

**说明**：删除音模记录及 `./audio/<model_id>/` 目录。

| 项目 | 内容 |
|---|---|
| 方法 | DELETE |
| 路径 | `/voice-models/{model_id}` |

**路径参数**

| 参数 | 类型 | 说明 |
|---|---|---|
| `model_id` | string | 音模 ID（会做安全字符清洗） |

**成功响应**：

```json
{
  "code": 0,
  "message": "deleted",
  "data": {}
}
```

**失败响应**：

```json
{ "code": 2, "message": "音模不存在", "data": {} }
```

---

### 6.7 POST /voice-models/{model_id}/categories

**说明**：为指定音模设置分类列表（覆盖式更新）。

| 项目 | 内容 |
|---|---|
| 方法 | POST |
| 路径 | `/voice-models/{model_id}/categories` |
| Content-Type | `application/json` |

**路径参数**：`model_id`（string）

**请求 Body**

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `categories` | string[] | 是 | 分类 ID 数组；传 `[]` 表示清空；**所有 ID 须已存在** |

**请求示例**：

```json
{
  "categories": ["cn", "hot"]
}
```

**成功响应**：`data` 为更新后的音模对象（含动态 URL）。

```json
{
  "code": 0,
  "message": "",
  "data": {
    "model_id": "demo",
    "model_name": "演示音模",
    "categories": ["cn", "hot"],
    "avatar_url": "...",
    "refer_wav_url": "..."
  }
}
```

**失败响应**

| 情况 | code | message |
|---|---|---|
| 音模不存在 | 2 | `音模不存在` |
| 分类不存在 | 3 | `分类不存在: xxx` |

---

### 6.8 POST /voice-categories

**说明**：新增一条音模分类。

| 项目 | 内容 |
|---|---|
| 方法 | POST |
| 路径 | `/voice-categories` |
| Content-Type | `application/json` |

**请求 Body**

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `category_id` | string | 是 | 分类 ID；清洗为 `[a-zA-Z0-9_-]` |
| `category_name` | string | 是 | 分类显示名 |
| `description` | string | 否 | 描述，默认空 |

**请求示例**：

```json
{
  "category_id": "cn",
  "category_name": "中文主播",
  "description": "示例分类"
}
```

**成功响应**：

```json
{
  "code": 0,
  "message": "",
  "data": {
    "category_id": "cn",
    "category_name": "中文主播",
    "description": "示例分类",
    "created_at": 1730000000
  }
}
```

**失败响应**

| 情况 | code | message |
|---|---|---|
| 缺少必填 | 3 | `category_id 和 category_name 必填` |
| 已存在 | 2 | `分类已存在` |

---

### 6.9 GET /voice-categories

**说明**：返回全部分类列表（不分页）。

| 项目 | 内容 |
|---|---|
| 方法 | GET |
| 路径 | `/voice-categories` |
| 参数 | 无 |

**成功响应**：

```json
{
  "code": 0,
  "message": "",
  "data": [
    {
      "category_id": "cn",
      "category_name": "中文主播",
      "description": "示例",
      "created_at": 1730000000
    }
  ]
}
```

---

### 6.10 DELETE /voice-categories/{category_id}

**说明**：删除分类，并自动从所有音模的 `categories` 中移除该 ID。

| 项目 | 内容 |
|---|---|
| 方法 | DELETE |
| 路径 | `/voice-categories/{category_id}` |

**路径参数**：`category_id`（string）

**成功响应**：

```json
{
  "code": 0,
  "message": "deleted",
  "data": {}
}
```

**失败响应**：

```json
{ "code": 2, "message": "分类不存在", "data": {} }
```

---

### 6.11 GET /

**说明**：返回内置管理页面 HTML。

| 项目 | 内容 |
|---|---|
| 方法 | GET |
| 路径 | `/` |
| 响应类型 | `text/html` |

**成功响应**：HTML 文档体（`index.html` 内容）。

---

### 6.12 GET /index、GET /index.html

**说明**：重定向到管理页根路径 `/`（地址栏不保留 index）。

| 项目 | 内容 |
|---|---|
| 方法 | GET |
| 路径 | `/index` 或 `/index.html` |
| 响应 | HTTP **301**，`Location` 为动态计算的站点根 URL + `/` |

---

### 6.13 GET /manager

**说明**：管理页入口别名，重定向到 `/`。

| 项目 | 内容 |
|---|---|
| 方法 | GET |
| 路径 | `/manager` |
| 响应 | HTTP **302**，`Location` 为站点根 URL + `/` |

---

### 6.14 POST /voice-tasks

**说明**：提交语音合成任务，写入队列，后台串行执行推理。

| 项目 | 内容 |
|---|---|
| 方法 | POST |
| 路径 | `/voice-tasks` |
| Content-Type | `application/json` |

**请求 Body 字段**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `user_id` | string | **是** | — | 用户 ID |
| `task_id` | string | **是** | — | 任务 ID，建议前端生成（时间戳+随机） |
| `text` | string | 建议填 | `""` | 待合成文本；空文本执行时会报错 |
| `text_language` | string | 否 | `all_zh` | 目标语种（§7） |
| `model_id` | string | 建议 | — | 音模 ID；可自动补全参考音频与参考文本 |
| `how_to_cut` | string | 否 | `cut1` | 切分策略（§7） |
| `prompt_text` | string | 否 | `""` | 参考文本 |
| `prompt_language` | string | 否 | `all_zh` | 参考文本语种 |
| `ref_wav_path` | string | 否 | — | 参考音频路径；有 `model_id` 时可从音模补全 |
| `top_k` | int | 否 | `5` | GPT 采样 |
| `top_p` | float | 否 | `1.0` | GPT 采样 |
| `temperature` | float | 否 | `1.0` | GPT 采样 |

**注意**：**不支持** `time` 字段；传入会被忽略，不参与任何逻辑。

**请求示例**：

```json
{
  "user_id": "U10001",
  "task_id": "T20250101ABC123",
  "model_id": "demo",
  "text": "需要合成的长文本内容。",
  "text_language": "all_zh",
  "how_to_cut": "cut1",
  "prompt_text": "参考文本",
  "prompt_language": "all_zh",
  "top_k": 5,
  "top_p": 1.0,
  "temperature": 1.0
}
```

**成功响应 data（入队瞬间）**：

| 字段 | 类型 | 说明 |
|---|---|---|
| `user_id` | string | 用户 ID |
| `task_id` | string | 任务 ID |
| `status` | string | 固定 `wait` |
| `progress` | int | `0` |
| `segments_done` | int | `0` |
| `total_segments` | int | `0`（执行后更新为切分段数） |
| `model_id` | string | 音模 ID |
| `text` | string | 文本 |
| `text_language` | string | 语种 |
| `how_to_cut` | string | 切分策略 |
| `prompt_text` | string | 参考文本 |
| `prompt_language` | string | 参考语种 |
| `ref_wav_path` | string | 参考音频路径 |
| `top_k` / `top_p` / `temperature` | number | 采样参数 |
| `created_at` | int | 创建时间戳 |
| `updated_at` | int | 更新时间戳 |
| `result_url` | string/null | 入队时为 `null`；完成后为完整 URL |

**失败响应**：

```json
{ "code": 3, "message": "缺少 user_id", "data": {} }
```

```json
{ "code": 3, "message": "缺少 task_id", "data": {} }
```

**任务完成后 data 额外字段**：

| 字段 | 说明 |
|---|---|
| `status` | `done` |
| `progress` | `100` |
| `result_path` | 磁盘路径，如 `./output/U10001/Txxx/Txxx.wav` |
| `result_url` | 可下载的完整 HTTP URL |

---

### 6.15 GET /voice-tasks

**说明**：按用户查询任务，支持**分页**或**多 task_id** 两种模式。

| 项目 | 内容 |
|---|---|
| 方法 | GET |
| 路径 | `/voice-tasks` |

**Query 参数**

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `user_id` | string | **是** | 用户 ID |
| `page` | int | 分页模式 | 页码，默认 `1` |
| `page_size` | int | 分页模式 | 每页条数，默认 `10`，最大 `100` |
| `task_ids` | string | 多 ID 模式 | 逗号或空格分隔的任务 ID，如 `T1,T2 T3` |
| `ids` | string | 多 ID 模式 | 与 `task_ids` 等价，可并用 |
| `task_id` | string | 多 ID 模式 | 单个任务 ID，与上面参数可叠加 |

**模式一：分页**（未传任何 task id 列表时）

```
GET /voice-tasks?user_id=U10001&page=1&page_size=10
```

- 返回该用户任务按 `updated_at` 倒序分页。
- 查询顺序：**Redis 优先**，JSON `voice_task.json` 兜底。

**模式二：多 ID**

```
GET /voice-tasks?user_id=U10001&task_ids=T1,T2
```

- `pageIndex` 固定为 `1`
- `pageSize` 为命中条数
- `total` 为命中条数

**缺少 user_id**：

```json
{ "code": 3, "message": "缺少 user_id", "data": {} }
```

**成功响应（分页）**：分页结构，`data.data` 中每项为任务对象（含动态 `result_url`）。

**任务对象常见字段**：同 §6.14，另含执行中的 `segments_done`、`total_segments`、`progress`。

---

### 6.16 POST /voice-tasks/query

**说明**：与 GET `/voice-tasks` 功能相同，参数放在 JSON Body，适合 task_id 列表很长的情况。

| 项目 | 内容 |
|---|---|
| 方法 | POST |
| 路径 | `/voice-tasks/query` |
| Content-Type | `application/json` |

**请求 Body**

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|---|---|---|---|---|
| `user_id` | string | **是** | — | 用户 ID |
| `task_ids` | string[] | 否 | `[]` | 有值时按 ID 批量查询 |
| `page` | int | 否 | `1` | 无 `task_ids` 时分页页码 |
| `page_size` | int | 否 | `10` | 无 `task_ids` 时每页条数 |

**请求示例（批量 ID）**：

```json
{
  "user_id": "U10001",
  "task_ids": ["T1", "T2", "T3"]
}
```

**请求示例（分页）**：

```json
{
  "user_id": "U10001",
  "page": 1,
  "page_size": 20
}
```

**响应**：与 §6.15 相同（分页结构）。

---

### 6.17 GET /voice-tasks/{task_id}

**说明**：按 `task_id` 查询单条任务（**不区分 user_id**）。

| 项目 | 内容 |
|---|---|
| 方法 | GET |
| 路径 | `/voice-tasks/{task_id}` |

**路径参数**：`task_id`（string）

**查询顺序**：Redis → JSON 索引。

**成功响应**：

```json
{
  "code": 0,
  "message": "",
  "data": {
    "task_id": "T20250101ABC123",
    "user_id": "U10001",
    "status": "run",
    "progress": 50,
    "segments_done": 2,
    "total_segments": 4,
    "result_url": null,
    "created_at": 1730000000,
    "updated_at": 1730000100
  }
}
```

**失败响应**：

```json
{ "code": 2, "message": "not found", "data": {} }
```

---

### 6.18 DELETE /voice-tasks/{task_id}

**说明**：删除或取消任务。

| 项目 | 内容 |
|---|---|
| 方法 | DELETE |
| 路径 | `/voice-tasks/{task_id}` |

**路径参数**：`task_id`（string）

**行为说明**

| 任务状态 | 行为 |
|---|---|
| `run`（运行中） | 设置取消标记；推理在检查点中断；状态记为 `error`；清理产物与 Redis |
| `wait` / `done` / `error` | 从索引移除；删除 `./output/<user_id>/<task_id>/` 下文件；清理 Redis |

**成功响应**：

```json
{
  "code": 0,
  "message": "",
  "data": {
    "removed": true
  }
}
```

任务不存在时：

```json
{
  "code": 0,
  "message": "",
  "data": {
    "removed": false
  }
}
```

---

## 7. 参数标识符附录

### 7.1 语种（`prompt_language` / `text_language`）

以下标识符均可传入（中文名为兼容别名，内部会映射）：

| 标识符 | 中文别名 | 说明 |
|---|---|---|
| `all_zh` | 中文 | 中文 |
| `en` | 英文 | 英文 |
| `zh` | 中英混合 | 中英混合 |
| `all_ja` | 日文 | 日文 |
| `ja` | 日英混合 | 日英混合 |
| `auto` | 多语种混合 | 多语种混合 |

### 7.2 切分策略（`how_to_cut`）

| 标识符 | 中文别名 | 说明 |
|---|---|---|
| `cut0` | 不切 | 不切分 |
| `cut1` | 凑四句一切 | 默认 |
| `cut2` | 凑50字一切 | 按字数 |
| `cut3` | 按中文句号。切 | 中文句号 |
| `cut4` | 按英文句号.切 | 英文句号 |
| `cut5` | 按标点符号切 | 标点切分 |

---

## 8. 数据存储与日志

### 8.1 Redis

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `REDIS_HOST` | `127.0.0.1` | 主机 |
| `REDIS_PORT` | `6379` | 端口 |
| `REDIS_DB` | `0` | 库编号 |

任务查询 **Redis 优先**，连接失败时使用本地 JSON。

### 8.2 任务保留清理

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `TASK_RETENTION_SECONDS` | `604800`（7天） | 过期任务自动清理周期，可在 `config.ini` 配置 |

### 8.3 日志

| 类型 | 路径 | 说明 |
|---|---|---|
| 访问日志 | `./Logs/access.log` | 每 6 小时切割；**GET `/voice-tasks` 不记录**（避免轮询刷屏） |
| 任务日志 | `./Logs/<YYYYMMDD>/<HH>/<task_id>.log` | 排队、进度、完成、异常 |

---

## 9. 启动与环境变量

### 9.1 启动命令

```bash
python api_ms.py --device 0 --host 0.0.0.0 --port 9881 \
  --root-path /ttsHttp --base-url https://36.136.54.165/media
```

### 9.2 命令行参数

| 参数 | 环境变量 | 说明 |
|---|---|---|
| `--device` | `infer_device_id` | Ascend NPU 编号 |
| `--host` | `infer_api_host` | 绑定地址 |
| `--port` | `infer_api_port` | 端口 |
| `--root-path` / `--prefix` | `infer_api_root_path` | URL 路径前缀 |
| `--base-url` / `--domain` | `infer_api_base_url` | URL 域名前缀 |

### 9.3 其他环境变量

| 变量 | 说明 |
|---|---|
| `is_half` | 半精度，默认 `True` |
| `gpt_path` / `sovits_path` | 模型权重路径 |
| `cnhubert_base_path` / `bert_path` | 依赖模型路径 |
| `ASCEND_DEVICE_ID` / `DEVICE_ID` | 与 `--device` 同步 |

### 9.4 nginx 反向代理示例

```nginx
location /media/ttsHttp/ {
    proxy_pass http://192.168.10.135:9881;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header X-Forwarded-Host $host;
}
```

---

## 10. 常见错误

| 现象 | code | 处理建议 |
|---|---|---|
| 缺少 `user_id` / `task_id` | 3 | 检查请求 JSON |
| 音模/分类/任务不存在 | 2 | 确认 ID |
| 参考音频 3～10 秒 | 3 | 重新录制参考音频 |
| 推理 OOM / MindSpore 报错 | 1 | 检查 NPU、权重、并发（勿多任务并行推理） |
| 资源 URL 404 | — | 检查 nginx 与 `--root-path`/`--base-url` 是否成对配置 |

---

**相关文档**：[测试模拟服务API.md](./测试模拟服务API.md)（联调测试用）
