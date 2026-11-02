"""Public command routing and strict defaults without keys or model requests."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from openmic.tts_cli import main as public_main
from openmic.tts_integrity_cli import main, parameters, DEFAULT_VERBATIM_INSTRUCTION
from openmic.tts_stage_plan import build_plan
from tests.integrity_fixtures import SCRIPT, FixtureBackend, build_fixture_plan, write_json


class IntegrityCLITests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.script, self.stage = self.root / "script.txt", self.root / "stage.json"
        self.script.write_text(SCRIPT, encoding="utf-8"); write_json(self.stage, build_plan(SCRIPT))

    def args(self, command, out):
        return [command, "--script", str(self.script), "--stage-plan", str(self.stage),
                "--out", str(out), "--manual-script"]

    def test_public_integrity_plan_command_is_offline(self):
        target = self.root / "plan.json"
        with patch("openmic.tts_integrity_cli.SiliconFlowBackend") as backend, patch("requests.post", side_effect=AssertionError("No network")):
            public_main(["integrity", *self.args("plan", target)])
        backend.assert_not_called()
        self.assertEqual(json.loads(target.read_text(encoding="utf-8")), build_fixture_plan())

    def test_missing_authority_digest_stops_before_backend_or_file_reads(self):
        with contextlib.redirect_stderr(io.StringIO()), patch("openmic.tts_integrity_cli.SiliconFlowBackend") as backend:
            with self.assertRaises(SystemExit) as error:
                main([*self.args("synth", self.root / "run"), "--authority", "not-opened.json"])
        self.assertEqual(error.exception.code, 2); backend.assert_not_called()

    def test_synthetic_public_synth_records_synthetic_not_real_evidence(self):
        backend = FixtureBackend()
        text = build_fixture_plan()["units"][0]["text"]
        with patch("openmic.tts_integrity_cli.SiliconFlowBackend", return_value=backend), \
             patch("openmic.tts_integrity_cli.TTSConfig.from_env", return_value=object()), \
             patch("openmic.tts_integrity_cli.transcribe", return_value={"transcript": text, "model": "fixture-asr"}), \
             patch("requests.post", side_effect=AssertionError("No network")):
            main(self.args("synth", self.root / "run"))
        self.assertEqual(backend.calls, 1)
        metrics = json.loads((self.root / "run/safe_club/metrics.json").read_text(encoding="utf-8"))
        self.assertTrue(metrics["synthetic"]); self.assertIsNone(metrics["authority_sha256"])
        accepted = json.loads((self.root / "run/accepted.json").read_text(encoding="utf-8"))
        self.assertNotIn("human_review", accepted[0]); self.assertNotIn("orthography_review", accepted[0])

    def test_verbatim_text_is_explicit_and_unicode_intact(self):
        plan = build_fixture_plan(); backend = FixtureBackend()
        self.assertEqual(DEFAULT_VERBATIM_INSTRUCTION, "请逐字朗读。")
        self.assertEqual(parameters(plan["units"][0], backend, 5)["instruction"], DEFAULT_VERBATIM_INSTRUCTION)
        for attempt in (6, 7):
            with self.assertRaises(ValueError):
                parameters(plan["units"][0], backend, attempt)


if __name__ == "__main__": unittest.main()
