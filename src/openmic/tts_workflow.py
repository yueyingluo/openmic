"""Additive AutoGen 0.2 workflow with an executable coach protocol.

The five roles and router are preserved. This entry point constrains the
writer/coach contracts and tolerates Markdown QC scores.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from openmic.agents import AGENT_SPECS
from openmic.autogen_v02 import (
    AutoGenConfig, AutoGenV02Workflow, SpeakerRouter, _extract_score,
    _qc_has_actionable_issues, _qc_quality_passes, _system_messages,
)
from openmic.humor_adaptation import choose_source_mode, source_similarity
from openmic.humor_craft import analyze_craft_plan, extract_craft_plan
from openmic.humor_data import CFunRetriever, HumorExample, curate_stage_reference_bundle
from openmic.humor_evaluation import script_contract_issues
from openmic.humor_generation import clean_writer_script
from openmic.models import AgentMessage, ProjectRequest, WorkflowResult
from openmic.tts_plan import EMOTIONS


def protocol_system_messages(
    request: ProjectRequest, *, content_mode: str = "baseline",
    examples: Optional[List[HumorExample]] = None,
    cfunset_mechanisms: Optional[List[dict]] = None,
    source_mode: str = "original",
) -> Dict[str, str]:
    """Return fresh prompts; never mutate or monkeypatch the original module."""
    if content_mode not in {"baseline", "cfunset"}:
        raise ValueError("content_mode 必须为 baseline 或 cfunset")
    prompts = _system_messages(
        request, examples=examples, cfunset_mechanisms=cfunset_mechanisms,
        source_mode=source_mode,
    )
    if content_mode == "cfunset":
        # Preserve the shared writer's references, source policy and creative
        # constraints. TTS adds an output contract, not a new story quota.
        prompts["JokeWriter"] += (
            "\n下游TTS交接格式：整个回复只输出一个 <SCRIPT> 到 </SCRIPT> 之间的完整稿件，"
            "标签外不写标题或解释。保留上述创作路线、参考与时长上限；不为TTS补足字数、"
            "包袱或回调数量。结构标签使用大写 [SETUP]、[PUNCHLINE]，有自然回调才用"
            " [CALLBACK]，可选 [INTRO]；标记之外的正文仍由本轮创作策略决定。"
            "本角色不写情绪、语速、停顿或重音标记，这些交给PerformanceCoach。"
        )
        prompts["PerformanceCoach"] += (
            "\n下游可执行TTS协议：保留最近一版 JokeWriter 的全部正文字词及先后顺序，"
            "不删字、不增字、不改写、不插入语气词。保留原有结构标签，不新增包袱或回调，"
            "不因字数或时长扩写正文。整个回复只输出一个 <SCRIPT> 到 </SCRIPT> 之间的完整稿件，"
            "边界外不写解释，禁止Markdown代码围栏。每个原有结构段独立一行，"
            "结构标签保持大写；包袱和原有回调前单独一行 [PAUSE=0.8]，结束后单独一行"
            " [PAUSE=2.0]，不用省略号替代。情绪标记放在结构标签之后、正文之前，"
            "只能写 [EMOTION=枚举值]，枚举值只允许："
            + ", ".join(EMOTIONS)
            + "。默认neutral，优先克制的calm或self_deprecating，不创造其他情绪名。"
            "重音只能写成对闭合的 [EMPHASIS]目标词[/EMPHASIS]，保留目标词在原句中的位置，"
            "不因重音另拆语音行。可选 [SPEED=1.05] 或 [SPEED=0.95]，置于该行正文之前。"
            "禁止HTML标签和其他自定义控制标签。只添加上述表演标记，不改变创作内容。"
        )
        return prompts
    prompts["JokeWriter"] = (
        AGENT_SPECS["JokeWriter"].system_prompt
        + f"\n主题={request.topic}；风格={request.style}；目标时长={request.duration_minutes}分钟；"
          f"受众={request.audience}。"
        + "\n围绕导演与受众结论创作原创中文脱口秀，正文目标900–1050个汉字，不含标记和标点。"
          "整个回复仅输出一个<SCRIPT>到</SCRIPT>包围的完整稿件，标签外不写解释或标题。"
          "使用大写[INTRO]（可选）、[SETUP]、[PUNCHLINE]、[CALLBACK]。"
          "完整稿件约12–18个语音段，每段独立一行；包含4–6个清楚的包袱及至少1个回调。"
          "每段要服务于连贯主题，铺垫和反转分开；反转关键词应尽量落在包袱段后部。"
          "保持口语化、克制自嘲，笑点指向自己的行为或具体情境。"
          "不以职业、身份、疾病或弱势处境作为贬低对象，不使用自伤、轻生类比制造笑点。"
          "数字尽量用中文书写，避免难读外语缩写。"
          "本角色不添加情绪/停顿/重音标记，这些交给PerformanceCoach。"
          "若收到质量修改意见，重写完整稿件并保留有效结构，不输出解释。"
    )
    prompts["PerformanceCoach"] = (
        AGENT_SPECS["PerformanceCoach"].system_prompt
        + f"\n主题={request.topic}；风格={request.style}；目标时长={request.duration_minutes}分钟；"
          f"受众={request.audience}。"
        + "\n保留最近一版 JokeWriter 的全部正文字词及先后顺序，不删字、不增字、不改写、不插入语气词。"
          "必须保留其结构标签和完整段落，不能换成表演说明。"
          "只添加能够被下游解析器执行的标记。下面是必须遵守的输出协议："
          "\n1. 整个回复只能是一个 <SCRIPT> 到 </SCRIPT> 之间的完整脚本。"
          "边界外禁止任何确认、解释、建议、标题和总结；禁止 Markdown 代码围栏。"
          "\n2. 必须保留并使用大写 [SETUP]、[PUNCHLINE]、[CALLBACK] 结构标记。"
          "开场可用 [INTRO]。若上游遗漏结构标记，按原文语义补齐，但不可丢失正文。"
          "每个完整 setup、punchline、callback 各占一行，每个语音行以其结构标记开头。"
          "至少有四个 [PUNCHLINE] 和一个 [CALLBACK]；不要把多处笑点合并成一个包袱行。"
          "\n3. 核心包袱和回调前单独一行 [PAUSE=0.8]，结束后单独一行 [PAUSE=2.0]。"
          "不要用省略号代替停顿标记。"
          "\n4. 情绪标记只能写 [EMOTION=枚举值]，枚举值只允许："
        + ", ".join(EMOTIONS)
        + "。默认 neutral；优先克制的 calm 或 self_deprecating。"
          "情绪标记放在该行结构标记之后、正文之前。不得创造其他英文或中文情绪词。"
          "\n5. 重音只能写 [EMPHASIS]目标词[/EMPHASIS]，必须成对闭合。"
          "重音词仍放在原句内，不因重音而拆成多个语音行；禁止输出 HTML 标签。"
          "\n6. 可选语速标记 [SPEED=1.05] 或 [SPEED=0.95]，放在该行正文之前。"
          "不得使用其他自定义标签，不得输出解析器不认识的控制指令。"
          "\n输出形态示例（实际必须保留整份稿件，不要只输出示例）："
          "\n<SCRIPT>\n[SETUP]原文的完整铺垫。\n[PAUSE=0.8]"
          "\n[PUNCHLINE][EMOTION=self_deprecating]原文的[EMPHASIS]反转词[/EMPHASIS]。"
          "\n[PAUSE=2.0]\n</SCRIPT>"
    )
    return prompts


def extract_quality_score(content: str) -> float:
    """Keep score-label underscores while removing Markdown wrappers."""
    return _extract_score(content.replace("**", "").replace("__", "").replace("`", ""))


class JybWorkflow:
    def __init__(
        self, config: AutoGenConfig, max_revisions: int = 2, generation_feedback: str = "",
        *, content_mode: str = "baseline", cfunset_index: Optional[Path] = None,
        adaptation_probability: Optional[float] = None,
        adaptation_seed: Optional[int] = None, source_mode: Optional[str] = None,
    ):
        if not isinstance(max_revisions, int) or isinstance(max_revisions, bool) or not 0 <= max_revisions <= 2:
            raise ValueError("max_revisions 必须是 0、1 或 2，避免无上限返工")
        self.config, self.max_revisions = config, max_revisions
        self.generation_feedback = generation_feedback
        # Reuse the shared constructor's validation and environment handling.
        # Constructing this object does not run a workflow or make API calls.
        settings = AutoGenV02Workflow(
            config, max_revisions=max_revisions, content_mode=content_mode,
            cfunset_index=cfunset_index, adaptation_probability=adaptation_probability,
            adaptation_seed=adaptation_seed, source_mode=source_mode,
        )
        self.content_mode = settings.content_mode
        self.cfunset_index = settings.cfunset_index
        self.adaptation_probability = settings.adaptation_probability
        self.adaptation_seed = settings.adaptation_seed
        self.source_mode = settings.source_mode

    def run(self, request: ProjectRequest,
            on_message: Optional[Callable[[AgentMessage], None]] = None) -> WorkflowResult:
        try:
            from autogen import ConversableAgent, GroupChat, GroupChatManager, UserProxyAgent
        except ImportError as exc:
            raise RuntimeError("未安装 AutoGen 0.2，请在项目环境执行 python -m pip install -e .") from exc
        llm_config = self.config.llm_config()
        examples: List[HumorExample] = []
        mechanisms: List[dict] = []
        reference_selection: Dict[str, Any] = {}
        policy = choose_source_mode(self.content_mode, self.adaptation_probability, self.adaptation_seed)
        source_mode = self.source_mode or policy["requested"]
        if self.source_mode:
            policy.update(requested=self.source_mode, draw=None, reason="explicit_or_checkpoint")
        fallback_reason = ""
        if self.content_mode == "cfunset":
            if self.cfunset_index is None:
                raise ValueError("cfunset 模式需要 cfunset_index 或 OPENMIC_CFUNSET_INDEX")
            candidates = CFunRetriever(self.cfunset_index).retrieve(
                request.topic, context=request.context, k=30,
            )
            examples, mechanisms = curate_stage_reference_bundle(
                request.topic, candidates, self.config, context=request.context,
                k=1 if source_mode == "adaptation" else 2,
                for_adaptation=source_mode == "adaptation",
                selection_trace=reference_selection,
            )
            if source_mode == "adaptation" and not examples:
                source_mode = "original"
                fallback_reason = "没有能保留原梗且符合用户事实的参考，回退自主创作"
        prompts = protocol_system_messages(
            request, content_mode=self.content_mode, examples=examples,
            cfunset_mechanisms=mechanisms, source_mode=source_mode,
        )
        if self.generation_feedback:
            feedback = "\n上一轮本地校验失败原因：" + self.generation_feedback + "。请修复并仅输出完整稿件，不输出修复说明。"
            prompts["JokeWriter"] += feedback
            prompts["PerformanceCoach"] += (
                "\n本轮Writer已经收到修订要求。你只给最近一版Writer全文加表演标记。"
                "不要再次根据字数、时长、内容建议增删正文；不自行补写结尾或回调句。"
            )
        agents: Dict[str, Any] = {
            name: ConversableAgent(name=name, system_message=prompts[name], llm_config=llm_config,
                                   human_input_mode="NEVER", code_execution_config=False)
            for name in AGENT_SPECS
        }
        user = UserProxyAgent(name="UserRequest", human_input_mode="NEVER", llm_config=False,
                              code_execution_config=False, default_auto_reply="")
        agents["UserRequest"] = user
        router = SpeakerRouter(agents, max_revisions=self.max_revisions, on_message=on_message)
        chat = GroupChat(agents=[user] + [agents[name] for name in AGENT_SPECS], messages=[],
                         max_round=6 + 3 * self.max_revisions, speaker_selection_method=router,
                         allow_repeat_speaker=False, send_introductions=False)
        manager = GroupChatManager(groupchat=chat, llm_config=llm_config, human_input_mode="NEVER")
        payload = {"topic": request.topic, "style": request.style,
                   "duration_minutes": request.duration_minutes, "audience": request.audience,
                   "context": request.context}
        user.initiate_chat(manager, message="请启动 OpenMic 创作流程。用户需求：" +
                           json.dumps(payload, ensure_ascii=False), clear_history=True)
        transcript = [AgentMessage(agent=str(m.get("name") or m.get("role") or "unknown"),
                                   content=str(m.get("content", "")))
                      for m in chat.messages if m.get("name") != "UserRequest"]
        final_qc = next((m.content for m in reversed(transcript) if m.agent == "QualityController"), "")
        metadata: Dict[str, Any] = {}
        validation_issues = ()
        if self.content_mode == "cfunset":
            writer_script = next((m.content for m in reversed(transcript) if m.agent == "JokeWriter"), "")
            writer_script = clean_writer_script(writer_script)
            validation_issues = tuple(script_contract_issues(
                writer_script, request.duration_minutes, request.audience,
            ))
            director = next((m.content for m in transcript if m.agent == "ComedyDirector"), "")
            craft_plan = extract_craft_plan(director)
            metadata = {
                "validation_issues": validation_issues,
                "cfunset_example_ids": tuple(item.id for item in examples),
                "cfunset_mechanisms": tuple(mechanisms),
                "craft_plan": craft_plan,
                "craft_diagnostics": analyze_craft_plan(craft_plan, writer_script),
                "source_policy": {**policy, "effective": source_mode, "fallback_reason": fallback_reason},
                "adaptation_source_id": examples[0].id if source_mode == "adaptation" and examples else None,
                "adaptation_similarity": (
                    source_similarity(writer_script, examples[0].setup, examples[0].punchline)
                    if source_mode == "adaptation" and examples else None
                ),
                "adaptation_reference": asdict(examples[0]) if source_mode == "adaptation" and examples else None,
                "reference_selection": reference_selection,
            }
        return WorkflowResult(approved=("DECISION: APPROVED" in final_qc
                                        and _qc_quality_passes(final_qc)
                                        and not _qc_has_actionable_issues(final_qc)
                                        and not validation_issues),
                              score=extract_quality_score(final_qc), messages=transcript,
                              revision_count=router.revision_count, **metadata)
