# AutoGen 0.2 真实模型 Smoke Test

- 日期：2026-09-25
- AutoGen：`autogen-agentchat==0.2.40`
- 模型：读取本地 `.env` 的 `LLM_MODEL`
- 输入主题：`我的网购经历`
- 风格：`roast`
- 时长：3 分钟
- 受众：大学生

## 结果

真实 API 已依次完成以下流程：

```text
UserRequest
  → ComedyDirector
  → AudienceAnalyzer
  → JokeWriter
  → PerformanceCoach
  → QualityController
  → APPROVED
```

- 五个核心 Agent 均成功响应；
- `GroupChatManager` 按自定义路由正确切换发言者；
- JokeWriter 生成了带 `[SETUP]`、`[PUNCHLINE]`、`[CALLBACK]` 的完整稿件；
- PerformanceCoach 添加了停顿、重音和情绪标记；
- QualityController 给出的三项评分为 8.5、9.0、8.5，报告均值为 8.67，并输出 `DECISION: APPROVED`；
- 未触发返工分支，因此 `revision_count=0`。

## 本轮发现的问题

1. **评分解析格式差异**：模型把评分放进 Markdown 表格，初版只识别 `SCORE: 8.5`，导致 CLI 误显示 `0.0`。现已兼容冒号格式和 Markdown 表格，并增加回归测试。
2. **响应时间超标**：从首个 Agent 开始生成到质检结束约 120 秒，未达到课程目标 `<30 秒`。当前先保证链路正确，后续需测试短提示词、更短输出、模型选择、缓存或并行化。
3. **AutoGen 成本告警**：当前模型名不在 AutoGen 0.2 内置价格表中，因此框架显示成本为 0；这不代表 API 免费。后续应自行记录 token 和实际供应商费用。

## 当前完成边界

这次验证只证明了“主题输入 → 五 Agent 文本协作 → 表演标记 → 质量判定”链路。以下部分仍未实现，不能称为完整课程交付：

- CFunSet 数据读取、检索或训练；
- ChatTTS/VALL-E X 等真实语音合成；
- 16 kHz、实时率与 MOS 评测；
- Streamlit/FastAPI 界面；
- 五个主题样例和完整实验报告。

## 2026-09-25 TTS 与页面增量验证

- 使用同一个 SiliconFlow endpoint 和 API Key 调用 `FunAudioLLM/CosyVoice2-0.5B`；
- 系统音色：`alex`；
- 输出：WAV、16-bit、单声道、16 kHz；
- 短句样例实际音频时长：8.2 秒；
- 已修正 SiliconFlow 流式 WAV 中的 RIFF/data 长度占位值；
- Streamlit 本地健康检查返回 `ok`；
- Streamlit AppTest 已验证 Mock 生成、三个结果指标、TTS 文本框和“生成 16 kHz WAV”按钮。

本轮验证了 TTS 服务连通性和页面骨架；尚未对完整 3–5 分钟稿件进行语音质量、实时率或 MOS 评测。
