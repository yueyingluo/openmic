from openmic.agents import AgentSpec
from openmic.engines.base import AgentEngine
from openmic.models import AgentMessage, QualityDecision, WorkflowState


class MockAgentEngine(AgentEngine):
    """Deterministic engine for local development and CI; it never calls an LLM."""

    def generate(self, spec: AgentSpec, state: WorkflowState) -> AgentMessage:
        req = state.request
        if spec.name == "ComedyDirector":
            content = (
                f"围绕“{req.topic}”建立一条逐步升级的叙事线；风格={req.style}，"
                f"目标时长={req.duration_minutes}分钟。"
            )
        elif spec.name == "AudienceAnalyzer":
            content = f"目标受众为{req.audience}；优先使用共同生活经验，避免身份攻击。"
        elif spec.name == "JokeWriter":
            content = (
                f"[开场] 大家好，今天聊聊{req.topic}。\n"
                "[setup] 我原以为事情会很简单。\n"
                "[punchline] 后来发现，简单的是我的想象。\n"
                "[callback] 所以真正需要退款的，其实是我的想象力。"
            )
        elif spec.name == "PerformanceCoach":
            content = "表演标记：开场中速；包袱前停顿0.8秒；重读“我的想象”；笑点后停顿2秒。"
        else:
            raise ValueError(f"Mock generate 不支持角色：{spec.name}")
        return AgentMessage(agent=spec.name, content=content)

    def evaluate(self, spec: AgentSpec, state: WorkflowState) -> QualityDecision:
        if spec.name != "QualityController":
            raise ValueError("evaluate 只能由 QualityController 调用")
        return QualityDecision(
            approved=True,
            score=7.5,
            feedback="Mock 验收通过；接入真实模型后替换为多维评分。",
        )

