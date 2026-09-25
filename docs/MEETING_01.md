# 第一次项目会议议程（建议 60 分钟）

## 会前准备

每个人会前完成三件事：

1. 阅读课程项目页和本 PRD；
2. 在本机跑通 Mock 流程和测试；
3. 对自己最想负责和最不想负责的模块各写一句理由。

## 会议目标

会议结束时必须留下明确结论，而不是只“交流了一下”：

- 向全员同步 AutoGen 0.2 与 Python 版本；
- 确认五个人的主责与 Reviewer；
- 确认 M1 的接口、截止时间和 Demo 标准；
- 确认模型 API、GPU 和费用边界；
- 把未决问题指定给具体负责人和答复日期。

## 议程

| 时间 | 内容 | 产出 |
|---:|---|---|
| 0–5 分钟 | 组长说明目标与评分结构 | 共同目标 |
| 5–15 分钟 | 现场跑 M0 Mock Demo | 所有人理解数据流 |
| 15–25 分钟 | 同步 AutoGen 0.2、决定模型和基本接口 | 技术决策记录 |
| 25–40 分钟 | 按兴趣和能力确定主责/Reviewer | RACI 初稿 |
| 40–50 分钟 | 拆 M1 Issues 和验收标准 | 每人一个可合并任务 |
| 50–60 分钟 | 风险、资源和下次会议 | Owner + Deadline |

## 建议现场决策表

| 决策 | 选项 | 结论 | Owner |
|---|---|---|---|
| AutoGen API | 0.2 / 0.4 | **已定 0.2** | 组长 |
| 模型 | DeepSeek API / 其他 | 待定 | 待定 |
| TTS baseline | ChatTTS / 其他 | 待定 | 待定 |
| UI | 仅 Streamlit / Streamlit + FastAPI | 待定 | 待定 |
| 仓库可见性 | private / public | 待定 | 组长 |
| 周会时间 | 固定时段 | 待定 | 全员 |

## M1 可直接创建的 GitHub Issues

1. `feat/autogen-baseline`：锁定版本并跑通五 Agent 一轮对话；
2. `feat/cfunset-loader`：完成合法数据获取、样本检查和最小检索接口；
3. `feat/tts-smoke-test`：一句带停顿标记的中文文本生成 16 kHz WAV；
4. `feat/quality-schema`：定义结构化评分结果和最小人工评测表；
5. `feat/streamlit-shell`：输入表单、Agent 轨迹占位、脚本和音频区域。

## 会后 10 分钟内

- 在 PRD 分工表填姓名；
- 为每个 Issue 填 Owner、Reviewer、验收条件和截止时间；
- 把会议决定提交到 `docs/`，不要只留在聊天记录里。
