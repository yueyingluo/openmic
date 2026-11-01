"""Exercise the adapted CLI, batch, review and UI without external requests.

These tests live outside source_snapshot so the original project tests remain
byte-identical. Reference metadata is a local fixture, never a claimed API run.
"""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from openmic.autogen_v02 import AutoGenConfig
from openmic.models import AgentMessage, ProjectRequest, WorkflowResult
from openmic.tts_cli import main as cli_main
from openmic.tts_review import review_script


RUN = Path(__file__).resolve().parents[1]
APP = RUN / "src/openmic/tts_app.py"
CONFIG = AutoGenConfig("test-only-not-a-real-key", "https://example.invalid/v1", "fixture-model")
WRITER = "<SCRIPT>\n[SETUP]我准备给室友发进度，先把文件名改成最终版。\n[PUNCHLINE]改完以后，我终于有了一份名字已经完成的作业。\n</SCRIPT>"
COACH = WRITER.replace("[PUNCHLINE]", "[PAUSE=0.8]\n[PUNCHLINE][EMOTION=calm]")
COACH = COACH.replace("作业。", "[EMPHASIS]作业[/EMPHASIS]。\n[PAUSE=2.0]")
QC = "HUMOR_SCORE: 8\nCULTURE_SCORE: 9\nSTRUCTURE_SCORE: 8\nDECISION: APPROVED"


def fixture_result():
    return WorkflowResult(
        approved=True, score=25 / 3, revision_count=0,
        messages=[AgentMessage("ComedyDirector", "这是本地测试策略。"),
                  AgentMessage("AudienceAnalyzer", "这是本地受众测试。"),
                  AgentMessage("JokeWriter", WRITER),
                  AgentMessage("PerformanceCoach", COACH),
                  AgentMessage("QualityController", QC)],
        cfunset_example_ids=("fixture-reference-17",),
        cfunset_mechanisms=({"example_id": "fixture-reference-17", "mechanism": "预期变化"},),
        craft_plan={"goal": "本地测试"}, craft_diagnostics={"fixture": True},
        source_policy={"requested": "adaptation", "effective": "adaptation", "draw": None},
        adaptation_source_id="fixture-reference-17",
        adaptation_similarity={"ratio": 0.2},
        adaptation_reference={"id": "fixture-reference-17", "setup": "测试铺垫", "punchline": "测试揭底"},
        reference_selection={"candidate_ids": ["fixture-reference-17"],
                             "selected_ids": ["fixture-reference-17"], "fixture": True},
    )


def trace_fixture():
    return asdict(fixture_result())


class NoNetworkCase(unittest.TestCase):
    def setUp(self):
        for target in ("requests.sessions.Session.request", "httpx.Client.send", "httpx.AsyncClient.send"):
            mocker = patch(target, side_effect=AssertionError("No external API calls allowed in this test"))
            mocker.start()
            self.addCleanup(mocker.stop)


class CFunSetCLIEntryTests(NoNetworkCase):
    def generate(self, path, *, result=None, additional=()):
        with patch("openmic.autogen_v02.AutoGenConfig.from_env", return_value=CONFIG), \
                patch("openmic.tts_workflow.JybWorkflow") as workflow, \
                contextlib.redirect_stdout(io.StringIO()):
            workflow.return_value.run.return_value = result or fixture_result()
            cli_main(["generate", "--backend", "autogen", "--topic", "小组合作",
                      "--content-mode", "cfunset", "--cfunset-index", "local-reference-index.jsonl",
                      "--context", "室友只是忘记发送文件，不是没有完成。", "--source-mode", "adaptation",
                      "--adaptation-seed", "23", "--out", str(path), *additional])
            return workflow

    def test_cli_forwards_reference_policy_and_user_facts(self):
        with tempfile.TemporaryDirectory() as directory:
            workflow = self.generate(Path(directory) / "draft.txt")
        workflow.assert_called_once_with(
            CONFIG, content_mode="cfunset", cfunset_index=Path("local-reference-index.jsonl"),
            source_mode="adaptation", adaptation_seed=23,
        )
        request = workflow.return_value.run.call_args.args[0]
        self.assertIsInstance(request, ProjectRequest)
        self.assertEqual(request.context, "室友只是忘记发送文件，不是没有完成。")
        self.assertEqual(request.topic, "小组合作")

    def test_cli_saves_complete_reference_lineage_and_independent_review(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "draft.txt"
            expected = fixture_result()
            self.generate(path, result=expected)
            trace = json.loads(path.with_suffix(".trace.json").read_text(encoding="utf-8"))
            for key in ("cfunset_example_ids", "cfunset_mechanisms", "craft_plan", "craft_diagnostics",
                        "source_policy", "adaptation_source_id", "adaptation_similarity",
                        "adaptation_reference", "reference_selection", "validation_issues"):
                self.assertEqual(trace[key], json.loads(json.dumps(asdict(expected)[key], ensure_ascii=False)), key)
            self.assertEqual(trace["model"], CONFIG.model)
            self.assertEqual(trace["content_mode"], "cfunset")
            self.assertEqual(trace["request"]["context"], "室友只是忘记发送文件，不是没有完成。")
            self.assertTrue(path.read_text(encoding="utf-8").startswith("<SCRIPT>\n"))
            review = json.loads(path.with_suffix(".review.json").read_text(encoding="utf-8"))
            self.assertTrue(review["passed"], review["errors"])
            self.assertTrue(review["writer_coach_text_match"])
            self.assertEqual(review["callbacks"], 0)
            self.assertEqual(review["content_mode"], "cfunset")

    def test_cli_keeps_failed_fidelity_review_visible(self):
        result = fixture_result()
        result.messages[3] = AgentMessage("PerformanceCoach", COACH.replace("室友", "老师"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "draft.txt"
            self.generate(path, result=result)
            review = json.loads(path.with_suffix(".review.json").read_text(encoding="utf-8"))
            self.assertFalse(review["passed"])
            self.assertFalse(review["writer_coach_text_match"])
            self.assertTrue(path.with_suffix(".trace.json").is_file())

    def test_mock_cfunset_is_rejected_before_config_or_workflow(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "draft.txt"
            with patch("openmic.autogen_v02.AutoGenConfig.from_env") as config, \
                    patch("openmic.tts_workflow.JybWorkflow") as real, \
                    patch("openmic.workflow.OpenMicWorkflow") as mock, \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                cli_main(["generate", "--backend", "mock", "--content-mode", "cfunset",
                          "--topic", "网购", "--out", str(path)])
            self.assertEqual(raised.exception.code, 1)
            config.assert_not_called()
            real.assert_not_called()
            mock.assert_not_called()
            self.assertFalse(path.exists())

    def test_legacy_generate_keeps_baseline_default_and_strict_length_review(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "draft.txt"
            with patch("openmic.autogen_v02.AutoGenConfig.from_env", return_value=CONFIG), \
                    patch("openmic.tts_workflow.JybWorkflow") as workflow, \
                    contextlib.redirect_stdout(io.StringIO()):
                workflow.return_value.run.return_value = fixture_result()
                cli_main(["generate", "--backend", "autogen", "--topic", "网购", "--out", str(path)])
            self.assertEqual(workflow.call_args.kwargs["content_mode"], "baseline")
            review = json.loads(path.with_suffix(".review.json").read_text(encoding="utf-8"))
            self.assertFalse(review["passed"])
            self.assertTrue(any("850" in error for error in review["errors"]))
            self.assertTrue(any("CALLBACK" in error for error in review["errors"]))


class CFunSetBatchEntryTests(NoNetworkCase):
    def test_batch_passes_each_topics_context_and_shared_reference_options(self):
        from openmic.tts_batch import main as batch_main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            topics = root / "topics.json"
            topics.write_text(json.dumps([
                {"id": "first", "topic": "网购", "style": "roast", "duration_minutes": 3,
                 "audience": "大学生", "context": "只是看错价格。"},
                {"id": "second", "topic": "食堂", "style": "observational", "duration_minutes": 4,
                 "audience": "同学"},
            ], ensure_ascii=False), encoding="utf-8")
            argv = ["tts_batch", "generate", "--topics", str(topics), "--out", str(root / "drafts"),
                    "--backend", "autogen", "--content-mode", "cfunset", "--cfunset-index", str(root / "index.jsonl"),
                    "--source-mode", "original", "--adaptation-seed", "31"]
            with patch.object(sys, "argv", argv), patch("openmic.tts_batch._generate") as generate:
                batch_main()
            self.assertEqual(generate.call_count, 2)
            first, second = [call.args[0] for call in generate.call_args_list]
            self.assertEqual(first.context, "只是看错价格。")
            self.assertEqual(second.context, "")
            self.assertEqual([first.out.name, second.out.name], ["first.txt", "second.txt"])
            for args in (first, second):
                self.assertEqual(args.content_mode, "cfunset")
                self.assertEqual(args.cfunset_index, root / "index.jsonl")
                self.assertEqual(args.source_mode, "original")
                self.assertEqual(args.adaptation_seed, 31)

    def test_batch_mock_cfunset_fails_without_creating_a_fake_draft(self):
        from openmic.tts_batch import main as batch_main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            topics = root / "topics.json"
            topics.write_text(json.dumps([{"id": "one", "topic": "网购", "style": "roast",
                                           "duration_minutes": 3, "audience": "同学"}], ensure_ascii=False), encoding="utf-8")
            with patch.object(sys, "argv", ["tts_batch", "generate", "--backend", "mock", "--content-mode", "cfunset",
                                             "--topics", str(topics), "--out", str(root / "drafts")]), \
                    patch("openmic.autogen_v02.AutoGenConfig.from_env") as config, \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                batch_main()
            self.assertEqual(raised.exception.code, 2)
            config.assert_not_called()
            self.assertFalse((root / "drafts").exists())


class CFunSetReviewBoundaryTests(NoNetworkCase):
    def test_cfunset_short_story_can_omit_callback_but_original_defaults_stay_strict(self):
        trace = trace_fixture()
        relaxed = review_script(COACH, trace, min_characters=1, min_punchlines=0, min_callbacks=0)
        self.assertTrue(relaxed["passed"], relaxed["errors"])
        self.assertEqual(relaxed["callbacks"], 0)
        original = review_script(COACH, trace)
        self.assertFalse(original["passed"])
        self.assertTrue(any("850" in error for error in original["errors"]))
        self.assertTrue(any("CALLBACK" in error for error in original["errors"]))

    def test_relaxing_story_quota_does_not_relax_word_fidelity_or_qc(self):
        cases = []
        missing_word = trace_fixture()
        missing_word["messages"][2]["content"] = WRITER.replace("室友", "朋友")
        cases.append((COACH, missing_word))
        rejected = trace_fixture()
        rejected["approved"] = False
        cases.append((COACH, rejected))
        weak_qc = trace_fixture()
        weak_qc["messages"][-1]["content"] = QC.replace("HUMOR_SCORE: 8", "HUMOR_SCORE: 1")
        cases.append((COACH, weak_qc))
        cases.append((COACH.replace("[EMOTION=calm]", "[EMOTION=invented]"), trace_fixture()))
        for script, trace in cases:
            with self.subTest(script=script[:20], approved=trace["approved"]):
                result = review_script(script, trace, min_characters=1, min_punchlines=0, min_callbacks=0)
                self.assertFalse(result["passed"])
                self.assertTrue(result["errors"])


class CFunSetUIEntryTests(NoNetworkCase):
    @staticmethod
    def field(app, group, label):
        return next(widget for widget in getattr(app, group) if widget.label == label)

    def test_ui_exposes_content_mode_reference_index_and_user_facts(self):
        with patch.dict("os.environ", {"OPENMIC_CFUNSET_INDEX": "isolated-run/cfunset-index.jsonl"}):
            app = AppTest.from_file(str(APP)).run(timeout=20)
        self.assertEqual(len(app.exception), 0)
        mode = self.field(app, "selectbox", "文本生成方式")
        self.assertEqual(mode.value, "原 TTS 长稿协议")
        self.assertIn("团队 CFunSet 检索增强", mode.options)
        self.assertEqual(self.field(app, "text_input", "CFunSet 索引路径").value,
                         "isolated-run/cfunset-index.jsonl")
        self.assertEqual(self.field(app, "text_input", "已知事实（可留空）").value, "")

    def test_ui_rejects_mock_reference_generation_without_api_calls(self):
        app = AppTest.from_file(str(APP)).run(timeout=20)
        self.field(app, "selectbox", "文本生成方式").set_value("团队 CFunSet 检索增强")
        with patch("openmic.autogen_v02.AutoGenConfig.from_env") as config, \
                patch("openmic.tts_workflow.JybWorkflow") as workflow:
            self.field(app, "button", "生成并载入脚本").click().run(timeout=20)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("CFunSet 模式请选真实 AutoGen" in message.value for message in app.error))
        config.assert_not_called()
        workflow.assert_not_called()

    def test_ui_passes_reference_index_and_facts_and_displays_metadata(self):
        app = AppTest.from_file(str(APP)).run(timeout=20)
        self.field(app, "radio", "脚本后端").set_value("真实 AutoGen")
        self.field(app, "selectbox", "文本生成方式").set_value("团队 CFunSet 检索增强")
        self.field(app, "text_input", "CFunSet 索引路径").set_value("local-ui-index.jsonl")
        self.field(app, "text_input", "已知事实（可留空）").set_value("同学没有偷拿东西。")
        with patch("openmic.autogen_v02.AutoGenConfig.from_env", return_value=CONFIG), \
                patch("openmic.tts_workflow.JybWorkflow") as workflow:
            workflow.return_value.run.return_value = fixture_result()
            self.field(app, "button", "生成并载入脚本").click().run(timeout=20)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0, [message.value for message in app.error])
        workflow.assert_called_once_with(CONFIG, content_mode="cfunset", cfunset_index=Path("local-ui-index.jsonl"))
        request = workflow.return_value.run.call_args.args[0]
        self.assertEqual(request.context, "同学没有偷拿东西。")
        self.assertIn("名字已经完成", app.text_area[0].value)
        displayed_json = "\n".join(str(item.value) for item in app.json)
        self.assertIn("reference_selection", displayed_json)
        self.assertIn("fixture-reference-17", displayed_json)


if __name__ == "__main__":
    unittest.main()
