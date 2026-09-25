from typing import Optional

from openmic.agents import AGENT_SPECS
from openmic.engines.base import AgentEngine
from openmic.engines.mock import MockAgentEngine
from openmic.models import AgentMessage, ProjectRequest, WorkflowResult, WorkflowState


class OpenMicWorkflow:
    """A bounded workflow that prevents uncontrolled agent conversations."""

    def __init__(self, engine: Optional[AgentEngine] = None, max_revisions: int = 2):
        self.engine = engine or MockAgentEngine()
        self.max_revisions = max_revisions

    def run(self, request: ProjectRequest) -> WorkflowResult:
        state = WorkflowState(request=request)

        for name in ("ComedyDirector", "AudienceAnalyzer", "JokeWriter", "PerformanceCoach"):
            state.messages.append(self.engine.generate(AGENT_SPECS[name], state))

        while True:
            decision = self.engine.evaluate(AGENT_SPECS["QualityController"], state)
            state.messages.append(
                AgentMessage(
                    agent="QualityController",
                    content=(
                        f"status={'APPROVED' if decision.approved else 'REVISION_REQUIRED'}; "
                        f"score={decision.score:.1f}; feedback={decision.feedback}"
                    ),
                )
            )
            if decision.approved or state.revision_count >= self.max_revisions:
                return WorkflowResult(
                    approved=decision.approved,
                    score=decision.score,
                    messages=list(state.messages),
                    revision_count=state.revision_count,
                )

            state.revision_count += 1
            state.messages.append(self.engine.generate(AGENT_SPECS["JokeWriter"], state))
            state.messages.append(self.engine.generate(AGENT_SPECS["PerformanceCoach"], state))

