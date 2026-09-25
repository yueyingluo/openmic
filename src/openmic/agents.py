from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class AgentSpec:
    name: str
    responsibility: str
    system_prompt: str


AGENT_SPECS: Dict[str, AgentSpec] = {
    "ComedyDirector": AgentSpec(
        name="ComedyDirector",
        responsibility="制定主题策略、叙事主线和风格边界",
        system_prompt="你是中文脱口秀总导演。输出清晰的创作策略，不直接写完整稿件。",
    ),
    "AudienceAnalyzer": AgentSpec(
        name="AudienceAnalyzer",
        responsibility="分析目标受众、文化语境和内容风险",
        system_prompt="你是受众分析师。识别共鸣点、知识门槛和冒犯风险。",
    ),
    "JokeWriter": AgentSpec(
        name="JokeWriter",
        responsibility="创作 setup-punchline、回调和口语化脚本",
        system_prompt="你是中文脱口秀编剧。围绕策略写出自然的铺垫、包袱与回调。",
    ),
    "PerformanceCoach": AgentSpec(
        name="PerformanceCoach",
        responsibility="添加语速、停顿、重音和情绪表演标记",
        system_prompt="你是表演指导。只添加可被 TTS 执行的表演标记并说明节奏。",
    ),
    "QualityController": AgentSpec(
        name="QualityController",
        responsibility="评估幽默度、文化适配和结构完整性",
        system_prompt="你是质检员。给出量化评分、是否通过及可执行的修改意见。",
    ),
}

