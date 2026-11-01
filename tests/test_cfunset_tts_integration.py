"""No-network regression tests for the current CFunSet-to-TTS handoff."""

import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openmic.agents import AGENT_SPECS
from openmic.autogen_v02 import AutoGenConfig, _system_messages
from openmic.humor_data import HumorExample
from openmic.models import ProjectRequest
from openmic.tts_plan import Speech, compile_script
from openmic.tts_review import normalized_spoken_text
from openmic.tts_workflow import JybWorkflow, protocol_system_messages


WRITER = "<SCRIPT>\n[SETUP]我网购买了一个杯子，觉得这次终于能少花钱了。\n[PUNCHLINE]结果为了包邮，我又买了两个杯子。\n</SCRIPT>"
COACH = (
    "<SCRIPT>\n[SETUP][SPEED=1.05]我网购买了一个杯子，觉得这次终于能少花钱了。"
    "\n[PAUSE=0.8]\n[PUNCHLINE][EMOTION=calm]结果为了包邮，我又买了"
    "[EMPHASIS]两个杯子[/EMPHASIS]。\n[PAUSE=2.0]\n</SCRIPT>"
)
APPROVED = "HUMOR_SCORE: 8\nCULTURE_SCORE: 8\nSTRUCTURE_SCORE: 8\nDECISION: APPROVED"


def fake_autogen(writer=WRITER, coach=COACH, qc=APPROVED):
    module = types.ModuleType("autogen")
    module.created = []

    class Agent:
        def __init__(self, **options):
            self.name = options["name"]
            self.options = options
            module.created.append(self)

    class User(Agent):
        def initiate_chat(self, manager, *, message, clear_history):
            module.request_message = message
            chat = manager.chat
            chat.messages.append({"name": self.name, "content": message})
            last = self
            for _ in range(chat.options["max_round"] + 1):
                speaker = chat.router(last, chat)
                if speaker is None:
                    return
                content = {
                    "ComedyDirector": '{"events": [], "story": "购物经历"}',
                    "AudienceAnalyzer": "保留日常购物逻辑。",
                    "JokeWriter": writer,
                    "PerformanceCoach": coach,
                    "QualityController": qc,
                }[speaker.name]
                chat.messages.append({"name": speaker.name, "content": content})
                last = speaker
            raise AssertionError("The bounded router did not terminate")

    class Chat:
        def __init__(self, **options):
            self.options = options
            self.messages = options["messages"]
            self.router = options["speaker_selection_method"]
            module.chat = self

    class Manager:
        def __init__(self, **options):
            self.chat = options["groupchat"]

    module.ConversableAgent = Agent
    module.UserProxyAgent = User
    module.GroupChat = Chat
    module.GroupChatManager = Manager
    return module


class CFunSetTTSIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.config = AutoGenConfig("no-network-test", "https://example.invalid/v1", "fake-model")
        self.request = ProjectRequest("我的网购经历", "roast", 1, "大学生", "没有额外收费。")
        self.example = HumorExample(
            "selected-train-reference", "keyword_generation",
            "主题：购物。我原以为买一个就可以了。", "最后带回了三个。", "train",
        )
        self.cards = [{
            "id": self.example.id,
            "expected": "只需要一个",
            "turning_quote": "最后带回了三个",
            "changed": "为了节省反而买多",
            "information_flow": [{"holder": "听众", "before": "只买一个", "after": "买了三个",
                                  "evidence_quote": "最后带回了三个"}],
        }]

    def run_cfunset(self, *, module=None, examples=None, probability=1, source_mode=None,
                    max_revisions=2, feedback=""):
        module = module or fake_autogen()
        selected = [self.example] if examples is None else examples
        observed = []

        def select(topic, candidates, config, **options):
            options["selection_trace"].update(
                candidate_ids=[self.example.id], selected_ids=[item.id for item in selected],
                reason="fixture selection", attempts=[{"response": "fixture"}],
            )
            return selected, self.cards if selected else []

        with patch.dict(sys.modules, {"autogen": module}), \
             patch("openmic.tts_workflow.CFunRetriever") as retriever, \
             patch("openmic.tts_workflow.curate_stage_reference_bundle", side_effect=select) as curator:
            retriever.return_value.retrieve.return_value = [self.example]
            result = JybWorkflow(
                self.config, max_revisions=max_revisions, generation_feedback=feedback,
                content_mode="cfunset", cfunset_index=Path("fixture-index.jsonl"),
                adaptation_probability=probability, adaptation_seed=7, source_mode=source_mode,
            ).run(self.request, on_message=observed.append)
            retrieval_call = retriever.return_value.retrieve.call_args
            curation_options = curator.call_args.kwargs
        return result, module, observed, retrieval_call, curation_options

    def test_cfunset_prompts_append_protocol_without_replacing_reference_or_source_policy(self):
        for mode in ("original", "adaptation"):
            with self.subTest(mode=mode):
                shared = _system_messages(self.request, examples=[self.example],
                                          cfunset_mechanisms=self.cards, source_mode=mode)
                prompts = protocol_system_messages(
                    self.request, content_mode="cfunset", examples=[self.example],
                    cfunset_mechanisms=self.cards, source_mode=mode,
                )
                self.assertTrue(prompts["JokeWriter"].startswith(shared["JokeWriter"]))
                self.assertTrue(prompts["PerformanceCoach"].startswith(shared["PerformanceCoach"]))
                for role in ("ComedyDirector", "AudienceAnalyzer", "QualityController"):
                    self.assertEqual(prompts[role], shared[role])
                for role in ("ComedyDirector", "JokeWriter"):
                    self.assertIn(self.example.id, prompts[role])
                    self.assertIn(self.cards[0]["turning_quote"], prompts[role])
                self.assertNotIn("900–1050", prompts["JokeWriter"])
                self.assertNotIn("至少有四个", prompts["PerformanceCoach"])
                self.assertIn("不新增包袱或回调", prompts["PerformanceCoach"])
                self.assertIn("不删字、不增字、不改写", prompts["PerformanceCoach"])

    def test_default_baseline_keeps_historical_writer_and_coach_protocol(self):
        prompts = protocol_system_messages(self.request)
        self.assertIn("900–1050", prompts["JokeWriter"])
        self.assertIn("至少有四个", prompts["PerformanceCoach"])
        shared = _system_messages(self.request)
        for role in ("ComedyDirector", "AudienceAnalyzer", "QualityController"):
            self.assertEqual(prompts[role], shared[role])

    def test_selected_reference_reaches_actual_writer_and_full_metadata_is_retained(self):
        result, module, observed, retrieval, curation = self.run_cfunset()
        self.assertTrue(result.approved)
        self.assertEqual(result.validation_issues, ())
        self.assertEqual(result.cfunset_example_ids, (self.example.id,))
        self.assertEqual(result.cfunset_mechanisms, tuple(self.cards))
        self.assertEqual(result.reference_selection["selected_ids"], [self.example.id])
        self.assertEqual(result.adaptation_source_id, self.example.id)
        self.assertEqual(result.adaptation_reference["setup"], self.example.setup)
        self.assertIsNotNone(result.adaptation_similarity)
        self.assertIsNotNone(result.craft_diagnostics)
        self.assertEqual(result.source_policy["effective"], "adaptation")
        self.assertEqual(retrieval.kwargs, {"context": self.request.context, "k": 30})
        self.assertEqual(curation["k"], 1)
        self.assertTrue(curation["for_adaptation"])
        writer = next(agent for agent in module.created if agent.name == "JokeWriter")
        self.assertIn(self.example.id, writer.options["system_message"])
        self.assertIn(self.cards[0]["turning_quote"], writer.options["system_message"])
        self.assertEqual([item.agent for item in observed], list(AGENT_SPECS))
        self.assertEqual(json.loads(module.request_message.split("用户需求：", 1)[1])["context"], self.request.context)
        self.assertEqual(normalized_spoken_text(WRITER), normalized_spoken_text(COACH))
        speech = [event for event in compile_script(COACH) if isinstance(event, Speech)]
        self.assertEqual(len({event.beat for event in speech if event.role == "punchline"}), 1)
        self.assertFalse(any(event.role == "callback" for event in speech))

    def test_explicit_original_policy_preserves_cfunset_references_without_false_adaptation(self):
        result, module, _, _, curation = self.run_cfunset(source_mode="original")
        self.assertTrue(result.approved)
        self.assertEqual(result.cfunset_example_ids, (self.example.id,))
        self.assertEqual(result.source_policy["requested"], "original")
        self.assertEqual(result.source_policy["effective"], "original")
        self.assertEqual(result.source_policy["reason"], "explicit_or_checkpoint")
        self.assertIsNone(result.adaptation_source_id)
        self.assertIsNone(result.adaptation_reference)
        self.assertEqual(curation["k"], 2)
        self.assertFalse(curation["for_adaptation"])
        writer = next(agent for agent in module.created if agent.name == "JokeWriter")
        self.assertIn(self.example.id, writer.options["system_message"])

    def test_no_compatible_adaptation_records_empty_selection_and_original_fallback(self):
        result, _, _, _, _ = self.run_cfunset(examples=[])
        self.assertEqual(result.cfunset_example_ids, ())
        self.assertEqual(result.reference_selection["selected_ids"], [])
        self.assertEqual(result.source_policy["requested"], "adaptation")
        self.assertEqual(result.source_policy["effective"], "original")
        self.assertTrue(result.source_policy["fallback_reason"])
        self.assertIsNone(result.adaptation_source_id)

    def test_high_average_cannot_hide_low_humor_and_router_stops_after_two_revisions(self):
        low_humor = "HUMOR_SCORE: 6\nCULTURE_SCORE: 10\nSTRUCTURE_SCORE: 10\nDECISION: APPROVED"
        result, _, observed, _, _ = self.run_cfunset(module=fake_autogen(qc=low_humor))
        self.assertFalse(result.approved)
        self.assertGreater(result.score, 7)
        self.assertEqual(result.revision_count, 2)
        self.assertEqual(sum(message.agent == "JokeWriter" for message in observed), 3)
        self.assertEqual(sum(message.agent == "QualityController" for message in observed), 3)
        self.assertEqual(len(result.messages), 11)

    def test_approval_with_concrete_repairs_is_not_reported_as_approved_at_limit(self):
        qc = APPROVED.replace("DECISION:", "修改意见：结尾仍需修改。\nDECISION:")
        result, _, _, _, _ = self.run_cfunset(module=fake_autogen(qc=qc), max_revisions=0)
        self.assertFalse(result.approved)
        self.assertEqual(result.revision_count, 0)

    def test_shared_empty_callback_contract_failure_is_retained_despite_model_approval(self):
        writer = WRITER.replace("</SCRIPT>", "[CALLBACK]\n</SCRIPT>")
        coach = COACH.replace("</SCRIPT>", "[CALLBACK]\n</SCRIPT>")
        result, _, _, _, _ = self.run_cfunset(module=fake_autogen(writer=writer, coach=coach))
        self.assertFalse(result.approved)
        self.assertTrue(any("空回调" in issue for issue in result.validation_issues))

    def test_revision_feedback_reaches_writer_without_overwriting_sources_or_coach_text(self):
        feedback = "fixture-local-feedback"
        _, module, _, _, _ = self.run_cfunset(feedback=feedback)
        prompts = {agent.name: agent.options.get("system_message", "") for agent in module.created}
        self.assertIn(feedback, prompts["JokeWriter"])
        self.assertIn(self.example.id, prompts["JokeWriter"])
        for role in ("ComedyDirector", "AudienceAnalyzer", "PerformanceCoach", "QualityController"):
            self.assertNotIn(feedback, prompts[role])

    def test_missing_index_fails_before_reference_selection_and_agent_construction(self):
        module = fake_autogen()
        with patch.dict(os.environ, {"OPENMIC_CFUNSET_INDEX": ""}), \
             patch.dict(sys.modules, {"autogen": module}), \
             patch("openmic.tts_workflow.curate_stage_reference_bundle") as curator:
            with self.assertRaisesRegex(ValueError, "cfunset_index"):
                JybWorkflow(self.config, content_mode="cfunset").run(self.request)
            curator.assert_not_called()
        self.assertEqual(module.created, [])


if __name__ == "__main__":
    unittest.main()
