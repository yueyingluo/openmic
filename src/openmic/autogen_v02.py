"""AutoGen 0.2 implementation of the five-agent OpenMic group chat."""

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from openmic.agents import AGENT_SPECS
from openmic.models import AgentMessage, ProjectRequest, WorkflowResult


@dataclass(frozen=True)
class AutoGenConfig:
    api_key: str
    base_url: Optional[str]
    model: str
    temperature: float = 0.7
    timeout_seconds: int = 120

    @classmethod
    def from_env(cls) -> "AutoGenConfig":
        load_dotenv()
        api_key = (
            os.getenv("OPENMIC_API_KEY")
            or os.getenv("LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if not api_key:
            raise ValueError(
                "缺少 OPENMIC_API_KEY。请参考 .env.example 导出环境变量；"
                "不要把真实 Key 提交到 Git。"
            )
        return cls(
            api_key=api_key,
            base_url=(
                os.getenv("OPENMIC_BASE_URL")
                or os.getenv("LLM_BASE_URL")
                or os.getenv("OPENAI_BASE_URL")
            ),
            model=(
                os.getenv("OPENMIC_MODEL")
                or os.getenv("LLM_MODEL")
                or "deepseek-chat"
            ),
        )

    def llm_config(self) -> Dict[str, Any]:
        model_config: Dict[str, Any] = {
            "model": self.model,
            "api_key": self.api_key,
        }
        if self.base_url:
            model_config["base_url"] = self.base_url
        return {
            "config_list": [model_config],
            "temperature": self.temperature,
            "timeout": self.timeout_seconds,
            "cache_seed": None,
        }


class SpeakerRouter:
    """Deterministic speaker transitions with a bounded revision loop."""

    def __init__(self, agents_by_name: Dict[str, Any], max_revisions: int = 2):
        self.agents_by_name = agents_by_name
        self.max_revisions = max_revisions
        self.revision_count = 0

    def __call__(self, last_speaker: Any, groupchat: Any) -> Optional[Any]:
        transitions = {
            "UserRequest": "ComedyDirector",
            "ComedyDirector": "AudienceAnalyzer",
            "AudienceAnalyzer": "JokeWriter",
            "JokeWriter": "PerformanceCoach",
            "PerformanceCoach": "QualityController",
        }
        next_name = transitions.get(last_speaker.name)
        if next_name:
            return self.agents_by_name[next_name]

        if last_speaker.name == "QualityController":
            last_content = str(groupchat.messages[-1].get("content", ""))
            needs_revision = "DECISION: REVISION_REQUIRED" in last_content
            if needs_revision and self.revision_count < self.max_revisions:
                self.revision_count += 1
                return self.agents_by_name["JokeWriter"]
            return None

        raise ValueError(f"未知的发言者：{last_speaker.name}")


def _shared_context(request: ProjectRequest) -> str:
    return (
        "\n\n当前任务约束："
        f"主题={request.topic}；风格={request.style}；"
        f"目标时长={request.duration_minutes}分钟；受众={request.audience}。"
        "只处理自己职责范围内的工作，并显式引用必要的上游结论。"
    )


def _system_messages(request: ProjectRequest) -> Dict[str, str]:
    context = _shared_context(request)
    return {
        "ComedyDirector": AGENT_SPECS["ComedyDirector"].system_prompt
        + context
        + "\n输出：主题切入点、叙事主线、三段式结构、风格边界。控制在300字内。",
        "AudienceAnalyzer": AGENT_SPECS["AudienceAnalyzer"].system_prompt
        + context
        + "\n输出：共鸣点、知识门槛、措辞偏好、冒犯风险和修改建议。控制在300字内。",
        "JokeWriter": AGENT_SPECS["JokeWriter"].system_prompt
        + context
        + (
            "\n根据导演与受众分析写完整脚本，显式使用 [SETUP]、[PUNCHLINE]、"
            "[CALLBACK] 标签。目标约800-1000个汉字。收到质检修改意见时必须重写，而不是解释。"
        ),
        "PerformanceCoach": AGENT_SPECS["PerformanceCoach"].system_prompt
        + context
        + (
            "\n保留脚本文字并插入 [PAUSE=0.8]、[PAUSE=2.0]、[EMPHASIS]、"
            "[EMOTION=...] 等可解析标记。不要另写分析、表格或重复说明。"
        ),
        "QualityController": AGENT_SPECS["QualityController"].system_prompt
        + context
        + (
            "\n分别给出 HUMOR_SCORE、CULTURE_SCORE、STRUCTURE_SCORE（0-10）。"
            "总分达到 7.0 且无明显安全问题则通过。最后一行必须且只能是 "
            "`DECISION: APPROVED` 或 `DECISION: REVISION_REQUIRED`。控制在350字内。"
        ),
    }


class AutoGenV02Workflow:
    """Build and run the course-required AutoGen 0.2 GroupChat."""

    def __init__(self, config: AutoGenConfig, max_revisions: int = 2):
        self.config = config
        self.max_revisions = max_revisions

    def run(self, request: ProjectRequest) -> WorkflowResult:
        try:
            from autogen import ConversableAgent, GroupChat, GroupChatManager, UserProxyAgent
        except ImportError as exc:
            raise RuntimeError(
                "未安装 AutoGen 0.2。请运行 `python -m pip install -e .`。"
            ) from exc

        llm_config = self.config.llm_config()
        prompts = _system_messages(request)
        agents_by_name: Dict[str, Any] = {}
        for name in AGENT_SPECS:
            agents_by_name[name] = ConversableAgent(
                name=name,
                system_message=prompts[name],
                llm_config=llm_config,
                human_input_mode="NEVER",
                code_execution_config=False,
            )

        # Input transport only; this is not a sixth intelligent project role.
        user_proxy = UserProxyAgent(
            name="UserRequest",
            human_input_mode="NEVER",
            llm_config=False,
            code_execution_config=False,
            default_auto_reply="",
        )
        agents_by_name["UserRequest"] = user_proxy

        router = SpeakerRouter(agents_by_name, max_revisions=self.max_revisions)
        groupchat = GroupChat(
            agents=[user_proxy] + [agents_by_name[name] for name in AGENT_SPECS],
            messages=[],
            max_round=6 + 3 * self.max_revisions,
            speaker_selection_method=router,
            allow_repeat_speaker=False,
            send_introductions=False,
        )
        manager = GroupChatManager(
            groupchat=groupchat,
            llm_config=llm_config,
            human_input_mode="NEVER",
        )
        request_message = json.dumps(
            {
                "topic": request.topic,
                "style": request.style,
                "duration_minutes": request.duration_minutes,
                "audience": request.audience,
            },
            ensure_ascii=False,
        )
        user_proxy.initiate_chat(
            manager,
            message="请启动 OpenMic 创作流程。用户需求：" + request_message,
            clear_history=True,
        )

        transcript = [
            AgentMessage(
                agent=str(message.get("name") or message.get("role") or "unknown"),
                content=str(message.get("content", "")),
            )
            for message in groupchat.messages
            if (message.get("name") or "") != "UserRequest"
        ]
        qc_messages: List[str] = [
            message.content
            for message in transcript
            if message.agent == "QualityController"
        ]
        final_qc = qc_messages[-1] if qc_messages else ""
        return WorkflowResult(
            approved="DECISION: APPROVED" in final_qc,
            score=_extract_score(final_qc),
            messages=transcript,
            revision_count=router.revision_count,
        )


def _extract_score(content: str) -> float:
    labels = ("HUMOR", "CULTURE", "STRUCTURE")
    scores: List[float] = []
    for label in labels:
        patterns = (
            rf"{label}_SCORE\s*[:：]\s*\**\s*(\d+(?:\.\d+)?)",
            rf"{label}_SCORE[^|\n]*\|\s*\**\s*(\d+(?:\.\d+)?)",
        )
        for pattern in patterns:
            match = re.search(pattern, content, flags=re.IGNORECASE)
            if match:
                scores.append(float(match.group(1)))
                break
    return sum(scores) / len(scores) if scores else 0.0
