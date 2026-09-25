from abc import ABC, abstractmethod

from openmic.agents import AgentSpec
from openmic.models import AgentMessage, QualityDecision, WorkflowState


class AgentEngine(ABC):
    """Framework-neutral boundary. The AutoGen adapter will implement this API."""

    @abstractmethod
    def generate(self, spec: AgentSpec, state: WorkflowState) -> AgentMessage:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, spec: AgentSpec, state: WorkflowState) -> QualityDecision:
        raise NotImplementedError

