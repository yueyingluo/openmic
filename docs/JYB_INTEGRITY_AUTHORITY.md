# 可移植审核接口

本版移除了历史实验的真实 WAV 哈希、指定句子和用户回复常量。未提供外部审核凭据时，逐句机器内容检查、端点检查均必须通过。原 `gate` 和 `audio` 两个核心文件没有改动。

主要接口：

```python
load_authority(path, expected_sha256, plan, assets_root)
human_accepted(entry, *, authority=None)
content_accepted(entry, *, authority=None)
ensure_units(plan, backend, out, assets_root, ..., authority=None, verbatim_rescue=False)
assemble(plan, accepted, assets_root, out, *, authority=None)
render_reviewed(plan, audio_by_id, profile="safe_course", human_reviews=None, *, authority=None)
```

`load_authority` 返回已核验的 `ReviewAuthority`，不接受 entry 自带的 `passed=True` 作为外部授权。CLI 对应 `--authority PATH --authority-sha256 HASH`，两个参数必须同时给出。加载后修改凭据、候选记录、WAV 或独立 ASR 文件，会拒绝继续使用。无人工审核时，`render_reviewed` 直接调用原 `render_integrity`。

## 凭据格式

顶层字段如下。`canonical_sha` 使用排序键、紧凑分隔符、UTF-8、禁止 NaN 的 JSON。

```json
{
  "schema": "openmic.integrity-authority/1",
  "plan_sha256": "canonical_sha(plan)",
  "source_script_sha256": "plan 中的原稿哈希",
  "reviews": [],
  "trials": [],
  "maximum_new_tts_calls": 0
}
```

每条 `reviews` 必须包含：`id`、`kind`、`unit_id`、`text`、`audio_sha256`、`decision="accept"`、`reviewer`、`review_text`、`candidate_record`、`candidate_record_sha256`、`machine_edge_sha256`、`machine_content_sha256`。所有路径相对 `assets_root`，不得越界。候选记录自身也必须绑定当前 plan/source 和原句。

- `human_listening`：再提供 `head_only_exemption=true`。只允许原报告中单独的首部风险，不能豁免尾部、格式或静音错误。审核人应实际核听同一 WAV 的完整原句。原机器 `False` 或未提供的内容报告继续保留，软件不会声称审核已由它独立证实。
- `orthography`：再提供 `secondary_asr_record`、`secondary_record_sha256`、`substitution={reference_index, reference, hypothesis}`。两个不同识别器必须绑定同一 WAV/原句并给出相同结果；只能是审核人明确指定的一个替换、零删除和零插入。不能全局忽略同音字。原严格内容失败和 CER 不改写。

先运行正常流程留下候选，核听或审查后创建独立凭据；不要修改原候选记录来制造通过。CLI 在 `accepted.json` 中附加审核 proof，原候选文件保持原样。外部凭据被原样保存在输出的 `authorities/<SHA>.json`，便于之后核对。人工渲染的 adapter 标为 `openmic.integrity-listening/2`，与历史实验区分。

## 可选采样设置

默认每句最多两次原控制和两次 plain 1.0 请求；复用缓存为 attempt 0，不产生新 TTS 请求。`--verbatim-rescue` 才允许额外一次 attempt 5 的逐字朗读请求。未启用时，不会自动执行此候选。

attempt 6/7 只来自凭据的 `trials`，每项必须绑定 `unit_id`、完整 `text`、`attempt`、`reviewer`、`review_text`、`voice`、`instruction`、`speed`、`maximum_new_tts_calls=1` 和 `transform`。顶层预算必须等于条目数。支持两种变换：

- `{"kind":"terminal_pause","suffix":"……"}`：明确记录末尾标点改变。
- `{"kind":"focus_spans","spans":[[start,end]]}`：按原文半开区间添加 `<strong>`，不改字词或原标点，区间不能重叠。

新候选仍须通过原机器检查；试验请求授权不代表音频通过。`terminal_control_approved=True` 布尔参数不再授权任何试验，需改用明确的凭据。中断、失败候选会消耗其持久化 attempt，恢复不会重置预算。请求预留/中断未知状态不应作为冷启动性能数据。

## 验证边界

可移植测试只使用生成的 PCM、临时 JSON 和 Mock ASR，不需要历史真实 WAV、模型权重或 API。它们验证来源绑定、拒绝错误放行、采样预算、原始 PCM 和停顿规则；不代表历史 123/124 数据已用本版代码重新运行，也不代表听感改善已获证明。历史 `test_stage_content_gate.py` 依赖实验目录的 `tools/asr_check.py`，未迁入；对应的内容拒绝、缓存恢复机制由可移植 integrity 测试覆盖。
