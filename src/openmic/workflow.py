from typing import Callable, Optional

from openmic.agents import AGENT_SPECS
from openmic.engines.base import AgentEngine
from openmic.engines.mock import MockAgentEngine
from openmic.models import AgentMessage, ProjectRequest, WorkflowResult, WorkflowState


class OpenMicWorkflow:
    """A bounded workflow that prevents uncontrolled agent conversations."""

    def __init__(self, engine: Optional[AgentEngine] = None, max_revisions: int = 2):
        self.engine = engine or MockAgentEngine()
        self.max_revisions = max_revisions

    def run(
        self,
        request: ProjectRequest,
        on_message: Optional[Callable[[AgentMessage], None]] = None,
    ) -> WorkflowResult:
        state = WorkflowState(request=request)

        for name in ("ComedyDirector", "AudienceAnalyzer", "JokeWriter", "PerformanceCoach"):
            message = self.engine.generate(AGENT_SPECS[name], state)
            state.messages.append(message)
            if on_message:
                on_message(message)

        while True:
            decision = self.engine.evaluate(AGENT_SPECS["QualityController"], state)
            quality_message = AgentMessage(
                agent="QualityController",
                content=(
                    f"status={'APPROVED' if decision.approved else 'REVISION_REQUIRED'}; "
                    f"score={decision.score:.1f}; feedback={decision.feedback}"
                ),
            )
            state.messages.append(quality_message)
            if on_message:
                on_message(quality_message)
            if decision.approved or state.revision_count >= self.max_revisions:
                return WorkflowResult(
                    approved=decision.approved,
                    score=decision.score,
                    messages=list(state.messages),
                    revision_count=state.revision_count,
                )

            state.revision_count += 1
            for name in ("JokeWriter", "PerformanceCoach"):
                message = self.engine.generate(AGENT_SPECS[name], state)
                state.messages.append(message)
                if on_message:
                    on_message(message)
