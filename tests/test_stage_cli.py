import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openmic.tts_cli import main as shared_tts
from openmic.tts_stage_cli import main
from openmic.tts_stage_plan import build_plan
from openmic.autogen_v02 import AutoGenConfig

SCRIPT = "<SCRIPT>\n[SETUP]我先把作业文件名改成最终版本。\n[PUNCHLINE]改完以后，只有名字已经完成。\n</SCRIPT>"
QC = "HUMOR_SCORE: 8\nCULTURE_SCORE: 8\nSTRUCTURE_SCORE: 8\nDECISION: APPROVED"


class StageCLITests(unittest.TestCase):
    def files(self, folder):
        script = folder / "script.txt"
        trace = folder / "trace.json"
        script.write_text(SCRIPT, encoding="utf-8")
        trace.write_text(json.dumps({"approved": True, "messages": [
            {"agent": "JokeWriter", "content": SCRIPT}, {"agent": "PerformanceCoach", "content": SCRIPT},
            {"agent": "QualityController", "content": QC}]}), encoding="utf-8")
        return script, trace

    def test_public_dispatch_enters_stage_without_legacy_argument_parsing(self):
        with patch("openmic.tts_stage_cli.main") as stage:
            shared_tts(["stage", "plan", "--script", "x.txt"])
        stage.assert_called_once_with(["plan", "--script", "x.txt"])

    def test_checked_rules_plan_and_all_tone_profiles_resume_without_rewriting(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            folder = Path(temporary)
            script, trace = self.files(folder)
            plan = folder / "plan.json"
            main(["plan", "--script", str(script), "--trace", str(trace), "--out", str(plan)])
            self.assertIn("units", json.loads(plan.read_text(encoding="utf-8")))
            args = ["synth", "--script", str(script), "--trace", str(trace), "--plan", str(plan),
                    "--backend", "tone", "--cache", str(folder / "cache"), "--out", str(folder / "out")]
            main(args)
            audio = folder / "out/stage_club/audio.wav"
            before = audio.read_bytes()
            with patch("openmic.tts_stage_cli.render_stage", side_effect=AssertionError("Already completed")):
                main(args)
            self.assertEqual(audio.read_bytes(), before)
            self.assertFalse((folder / "out/.stage-synthesis.lock").exists())

    def test_invalid_plan_is_rejected_before_paid_backend(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            script, trace = self.files(folder)
            plan = build_plan(SCRIPT)
            plan["units"][0]["text"] += "各位朋友"
            plan_path = folder / "plan.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            with patch("openmic.tts_stage_cli.SiliconFlowBackend") as backend, self.assertRaises(ValueError):
                main(["synth", "--script", str(script), "--trace", str(trace), "--plan", str(plan_path),
                      "--backend", "siliconflow", "--out", str(folder / "out")])
            backend.assert_not_called()

    def test_failed_word_fidelity_cannot_be_replaced_by_an_approved_boolean(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            script, trace = self.files(folder)
            data = json.loads(trace.read_text(encoding="utf-8"))
            data["messages"][0]["content"] = SCRIPT.replace("文件名", "文件")
            trace.write_text(json.dumps(data), encoding="utf-8")
            with patch("openmic.tts_stage_cli.request_plan") as planner, self.assertRaises(ValueError):
                main(["plan", "--script", str(script), "--trace", str(trace), "--planner", "llm", "--out", str(folder / "plan.json")])
            planner.assert_not_called()

    def test_slow_planner_uses_own_timeout_without_changing_shared_config(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            folder = Path(temporary)
            script, trace = self.files(folder)
            config = AutoGenConfig(api_key="LOCAL_TEST_ONLY", base_url="https://example.invalid/v1", model="fixture")
            with patch("openmic.autogen_v02.AutoGenConfig.from_env", return_value=config), \
                 patch("openmic.tts_stage_cli.request_plan", return_value=build_plan(SCRIPT)) as planner:
                main(["plan", "--script", str(script), "--trace", str(trace), "--planner", "llm",
                      "--planner-timeout", "300", "--out", str(folder / "plan.json")])
            self.assertEqual(config.timeout_seconds, 120)
            self.assertEqual(planner.call_args.args[1].timeout_seconds, 300)
            self.assertEqual(planner.call_args.args[0], SCRIPT)

    def test_invalid_planner_timeout_stops_before_network(self):
        with patch("openmic.tts_stage_cli.request_plan") as planner, contextlib.redirect_stderr(io.StringIO()):
            for value in ("0", "601", "nan"):
                with self.subTest(value=value), self.assertRaises(SystemExit):
                    main(["plan", "--script", "unused.txt", "--out", "unused.json", "--planner-timeout", value])
        planner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
