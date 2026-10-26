"""Exercise the integrated public CLI and the unchanged legacy mock route.

Subprocesses use this checkout's source and fail immediately on network access
or any attempt to import the former standalone package.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openmic.cli import main


ROOT = Path(__file__).resolve().parents[1]


class IntegratedTTSCLITests(unittest.TestCase):
    def run_public_cli(self, arguments, *, block_numpy=False):
        with tempfile.TemporaryDirectory() as temporary:
            guard = Path(temporary)
            (guard / "sitecustomize.py").write_text(
                "import builtins, sys\n"
                "original_import = builtins.__import__\n"
                "def checked_import(name, *args, **kwargs):\n"
                "    if name == 'openmic_jyb' or name.startswith('openmic_jyb.'):\n"
                "        raise AssertionError('Former standalone package must not be imported')\n"
                + ("    if name == 'numpy' or name.startswith('numpy.'):\n"
                   "        raise AssertionError('Legacy CLI must not import performance dependencies')\n"
                   if block_numpy else "")
                + "    return original_import(name, *args, **kwargs)\n"
                "builtins.__import__ = checked_import\n"
                "def audit(event, args):\n"
                "    if event in {'socket.connect', 'socket.getaddrinfo'}:\n"
                "        raise AssertionError('No network allowed in CLI integration tests')\n"
                "sys.addaudithook(audit)\n",
                encoding="utf-8",
            )
            environment = dict(os.environ)
            environment.update(
                PYTHONPATH=os.pathsep.join([str(guard), str(ROOT / "src")]),
                PYTHONUTF8="1",
                PYTHONIOENCODING="utf-8",
                PYTHONDONTWRITEBYTECODE="1",
            )
            return subprocess.run(
                [sys.executable, "-m", "openmic", *arguments], cwd=str(ROOT),
                env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", timeout=45,
            )

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + "\n" + result.stderr)

    def test_shared_entry_dispatches_tts_arguments_without_legacy_workflow(self):
        with patch("openmic.tts_cli.main") as performance, patch("openmic.cli.OpenMicWorkflow") as legacy:
            main(["tts", "plan", "--script", "example.txt", "--out", "plan.json"])
        performance.assert_called_once_with(["plan", "--script", "example.txt", "--out", "plan.json"])
        legacy.assert_not_called()

    def test_legacy_help_has_tts_entry_without_importing_numpy(self):
        result = self.run_public_cli(["--help"], block_numpy=True)
        self.assert_success(result)
        self.assertIn("tts", result.stdout)

    def test_original_mock_command_remains_usable_without_numpy(self):
        result = self.run_public_cli(
            ["--backend", "mock", "--topic", "我的网购经历", "--style", "roast"], block_numpy=True)
        self.assert_success(result)
        for role in ("ComedyDirector", "AudienceAnalyzer", "JokeWriter", "PerformanceCoach", "QualityController"):
            self.assertIn("[" + role + "]", result.stdout)
        self.assertIn("Result:", result.stdout)

    def test_module_entry_writes_plan_from_current_checkout(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "plan.json"
            result = self.run_public_cli(
                ["tts", "plan", "--script", "examples/jyb_demo.txt", "--out", str(out)])
            self.assert_success(result)
            events = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(any(isinstance(event.get("text"), str) for event in events))
            self.assertTrue(any(isinstance(event.get("seconds"), (int, float)) for event in events))

    def test_module_entry_renders_all_four_tone_conditions(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "conditions"
            result = self.run_public_cli(
                ["tts", "synth", "--script", "examples/jyb_demo.txt", "--backend", "tone",
                 "--conditions", "baseline", "timing", "emotion", "performance",
                 "--cache", str(Path(folder) / "cache"), "--out", str(out)])
            self.assert_success(result)
            hashes = set()
            for condition in ("baseline", "timing", "emotion", "performance"):
                metadata = json.loads((out / condition / "metrics.json").read_text(encoding="utf-8"))
                self.assertIs(metadata["synthetic"], True)
                self.assertEqual(metadata["condition"], condition)
                self.assertTrue((out / condition / "audio.wav").is_file())
                hashes.add(metadata["spoken_text_sha256"])
            self.assertEqual(len(hashes), 1)


if __name__ == "__main__":
    unittest.main()
