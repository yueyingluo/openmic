# OpenMic

OpenMic 是认知推理课程项目的协作仓库。目标是用 5 个核心智能体，把一个生活主题转换为中文脱口秀脚本和具有表演控制的语音。

课程要求：[OpenMic 项目页](https://yzhu.io/courses/core/projects/openmic/)

## 当前状态

这是 **M0 最小骨架**：

- 已定义主题、风格、时长、受众等统一输入；
- 已定义 5 个 Agent 的职责和固定协作顺序；
- 默认使用 Mock 引擎，无 API Key 也能跑通；
- 已提供基础测试、PR 模板、立项 PRD 和首次会议议程；
- 已接入 AutoGen 0.2 的真实 GroupChat；尚未接入 CFunSet、TTS、Web UI。

## 流程

```text
ComedyDirector
      ↓
AudienceAnalyzer
      ↓
JokeWriter
      ↓
PerformanceCoach
      ↓
QualityController
      ├── APPROVED → 输出脚本
      └── REVISION_REQUIRED → JokeWriter（最多重写 2 次）
```

## 安装

要求 Python 3.9+。

```bash
cd openmic
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## 30 秒跑起 Mock

Mock 不调用外部模型，适合全员验证环境和接口：

```bash
PYTHONPATH=src python -m openmic \
  --topic "我的网购经历" \
  --style roast \
  --duration 3 \
  --audience "大学生"
```

## 运行 AutoGen 0.2

配置 OpenAI-compatible 模型端点；以下以 DeepSeek 为例：

```bash
export OPENMIC_API_KEY="你的 Key"
export OPENMIC_BASE_URL="https://api.deepseek.com"
export OPENMIC_MODEL="deepseek-chat"

openmic \
  --backend autogen \
  --topic "我的网购经历" \
  --style roast \
  --duration 3 \
  --audience "大学生"
```

程序会自动读取仓库根目录的 `.env`，并兼容 `LLM_API_KEY`、`LLM_BASE_URL`、`LLM_MODEL` 这组三个变量名。

AutoGen 流程实际创建 5 个 `ConversableAgent`，再由 `GroupChatManager` 按固定路由组织发言。`UserRequest` 是不调用 LLM 的输入代理，不算第六个智能角色。

运行测试：

```bash
cd openmic
PYTHONPATH=src python -m unittest discover -s tests -v
```

## 文档

- [产品需求文档](docs/PRD.md)
- [第一次项目会议](docs/MEETING_01.md)
- [协作规范](CONTRIBUTING.md)

## AutoGen 版本说明

本仓库已锁定 `autogen-agentchat~=0.2`，对应课程页点名的 `ConversableAgent`、`GroupChat` 和 `GroupChatManager`。全组不要混装 AutoGen 0.4 示例，也不要按照旧博客直接安装 `pyautogen`。
