"""No-network checks for the additive five-agent coach protocol."""

import sys
import types
import unittest
from unittest.mock import patch

from openmic.agents import AGENT_SPECS
from openmic.autogen_v02 import AutoGenConfig, _system_messages
from openmic.models import ProjectRequest
from openmic.tts_plan import EMOTIONS, compile_script
from openmic.tts_workflow import JybWorkflow, extract_quality_score, protocol_system_messages


class JybWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.request = ProjectRequest("我的网购经历", style="roast")
        self.config = AutoGenConfig("unit-test-no-network", "https://example.invalid/v1", "fake-model")

    def test_coach_contract_is_explicit_and_other_prompts_are_unchanged(self):
        original = _system_messages(self.request)
        prompts = protocol_system_messages(self.request)
        coach = prompts["PerformanceCoach"]
        for token in ("<SCRIPT>", "</SCRIPT>", "[SETUP]", "[PUNCHLINE]", "[CALLBACK]",
                      "[PAUSE=0.8]", "[PAUSE=2.0]", "[EMPHASIS]", "[/EMPHASIS]",
                      "每个语音行", "枚举值只允许", "禁止 Markdown"):
            with self.subTest(token=token):
                self.assertIn(token, coach)
        for emotion in EMOTIONS:
            self.assertIn(emotion, coach)
        self.assertNotIn("[EMOTION=...]", coach)
        writer = prompts["JokeWriter"]
        for token in ("900–1050", "12–18", "4–6", "职业", "自伤", "<SCRIPT>", "</SCRIPT>"):
            self.assertIn(token, writer)
        for name in AGENT_SPECS:
            if name not in {"PerformanceCoach", "JokeWriter"}:
                self.assertEqual(prompts[name], original[name])
        self.assertEqual(_system_messages(self.request), original)

    def test_qc_bold_labels_numbers_and_inline_code_parse(self):
        content = "**HUMOR_SCORE**: **8.5**\n**CULTURE_SCORE**：9.0\n`STRUCTURE_SCORE`: 8.0"
        self.assertEqual(extract_quality_score(content), 8.5)
        self.assertEqual(extract_quality_score("没有给出评分"), 0)
        table = "| **HUMOR_SCORE** | **8** |\n| CULTURE_SCORE | 7 |\n| STRUCTURE_SCORE | 9 |"
        self.assertEqual(extract_quality_score(table), 8)

    def test_revision_bound_is_finite(self):
        for value in (-1, 3, 100, True, 1.5):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    JybWorkflow(self.config, max_revisions=value)

    def test_content_revision_feedback_is_only_sent_to_writer(self):
        module = self._fake_module(approve=True)
        feedback = "必须删去冗余正文并把稿件缩至九百字"
        with patch.dict(sys.modules, {"autogen": module}):
            JybWorkflow(self.config, generation_feedback=feedback).run(self.request)
        writer = next(agent for agent in module.created if agent.name == "JokeWriter")
        coach = next(agent for agent in module.created if agent.name == "PerformanceCoach")
        self.assertIn(feedback, writer.options["system_message"])
        self.assertNotIn(feedback, coach.options["system_message"])
        self.assertIn("不要再次根据字数", coach.options["system_message"])

    def _fake_module(self, *, approve):
        module = types.ModuleType("autogen")
        script = "<SCRIPT>\n[SETUP]铺垫。\n[PAUSE=0.8]\n[PUNCHLINE][EMOTION=calm]反转。\n[PAUSE=2.0]\n</SCRIPT>"
        qc = ("**HUMOR_SCORE**: 8.5\n**CULTURE_SCORE**: 9\n**STRUCTURE_SCORE**: 8\n"
              + ("DECISION: APPROVED" if approve else "DECISION: REVISION_REQUIRED"))
        module.created = []

        class Agent:
            def __init__(self, **kwargs):
                self.name, self.options = kwargs["name"], kwargs
                module.created.append(self)

        class User(Agent):
            def initiate_chat(self, manager, *, message, clear_history):
                chat = manager.chat
                module.initial_message = message
                module.clear_history = clear_history
                chat.messages.append({"name": self.name, "content": message})
                previous = self
                # The actual bounded router drives this mock exactly as in production.
                for _ in range(chat.options["max_round"] + 1):
                    speaker = chat.router(previous, chat)
                    if speaker is None:
                        return
                    chat.messages.append({"name": speaker.name,
                                          "content": qc if speaker.name == "QualityController" else
                                                     script if speaker.name == "PerformanceCoach" else "上游结果"})
                    previous = speaker
                raise AssertionError("router failed to terminate")

        class Chat:
            def __init__(self, **kwargs):
                self.messages = kwargs["messages"]
                self.router, self.options = kwargs["speaker_selection_method"], kwargs
                module.chat = self

        class Manager:
            def __init__(self, **kwargs):
                self.chat = kwargs["groupchat"]

        module.ConversableAgent, module.UserProxyAgent = Agent, User
        module.GroupChat, module.GroupChatManager = Chat, Manager
        return module

    def test_constructed_workflow_uses_five_roles_and_streams_messages(self):
        module, observed = self._fake_module(approve=True), []
        with patch.dict(sys.modules, {"autogen": module}):
            result = JybWorkflow(self.config).run(self.request, on_message=observed.append)
        expected = list(AGENT_SPECS)
        self.assertEqual([message.agent for message in result.messages], expected)
        self.assertEqual([message.agent for message in observed], expected)
        self.assertTrue(result.approved)
        self.assertEqual(result.score, 8.5)
        self.assertEqual(result.revision_count, 0)
        self.assertEqual(module.chat.options["max_round"], 12)
        self.assertFalse(module.chat.options["allow_repeat_speaker"])
        self.assertTrue(module.clear_history)
        for agent in module.created:
            self.assertFalse(agent.options["code_execution_config"])
            self.assertEqual(agent.options["human_input_mode"], "NEVER")
        coach = next(agent for agent in module.created if agent.name == "PerformanceCoach")
        self.assertIn("枚举值只允许", coach.options["system_message"])
        self.assertIn("我的网购经历", module.initial_message)
        compile_script(next(message.content for message in result.messages if message.agent == "PerformanceCoach"))

    def test_rejected_workflow_stops_after_two_revisions(self):
        module = self._fake_module(approve=False)
        with patch.dict(sys.modules, {"autogen": module}):
            result = JybWorkflow(self.config).run(self.request)
        self.assertFalse(result.approved)
        self.assertEqual(result.revision_count, 2)
        self.assertEqual(sum(message.agent == "JokeWriter" for message in result.messages), 3)
        self.assertEqual(sum(message.agent == "QualityController" for message in result.messages), 3)
        self.assertEqual(len(result.messages), 11)


if __name__ == "__main__":
    unittest.main()
