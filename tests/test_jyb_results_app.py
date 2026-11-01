"""Local fixtures only; these tests never read credentials or call a model API."""

import json
import os
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "src" / "openmic" / "tts_results_app.py"


class ViewerTests(unittest.TestCase):
    def test_missing_outputs_is_friendly(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"JYB_RUN_ROOT": folder}):
            app = AppTest.from_file(str(APP)).run(timeout=15)
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any("尚未创建" in item.value for item in app.warning))

    def test_empty_topic_shows_pending(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"JYB_RUN_ROOT": folder}):
            (Path(folder) / "outputs").mkdir()
            app = AppTest.from_file(str(APP)).run(timeout=15)
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.selectbox[0].options), 6)
            self.assertEqual(len(app.tabs), 4)
            self.assertEqual(len(app.metric), 0)
            self.assertTrue(any("尚未完成" in item.value for item in app.info))

    def test_saved_audio_and_trace_need_no_api(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"JYB_RUN_ROOT": folder}):
            root = Path(folder)
            topic = root / "outputs" / "real" / "shopping"
            topic.mkdir(parents=True)
            (topic / "script.txt").write_text("[SETUP]测试正文。", encoding="utf-8")
            for condition in ("baseline", "timing", "emotion", "performance"):
                target = topic / condition
                target.mkdir()
                with wave.open(str(target / "audio.wav"), "wb") as audio:
                    audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                    audio.writeframes(b"\x01\x00" * 1600)
                (target / "metrics.json").write_text(json.dumps({
                    "synthetic": False, "duration_seconds": 0.1,
                    "inserted_silence_seconds": 0, "wall_seconds": 0.02,
                    "first_speech_chunk_seconds": 0.01, "cache_hits": 1,
                    "valid_cold_tts_benchmark": False, "in_target_duration": False,
                }), encoding="utf-8")
            trace = root / "outputs" / "scripts" / "shopping.trace.json"
            trace.parent.mkdir()
            trace.write_text(json.dumps({"approved": True, "score": 8.2,
                                        "generation_seconds": 60,
                                        "messages": [{"agent": "PerformanceCoach", "content": "演示轨迹"}]}), encoding="utf-8")
            with patch("requests.sessions.Session.request", side_effect=AssertionError("No API calls allowed")):
                app = AppTest.from_file(str(APP)).run(timeout=15)
                app.selectbox[0].set_value("五 Agent 稿｜我的网购经历").run(timeout=15)
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.tabs), 4)
            self.assertEqual(len(app.metric), 15)
            self.assertTrue(any("模型评分是脚本质检" in item.value for item in app.caption))
            self.assertTrue(any("测试正文" in item.value for item in app.code))


if __name__ == "__main__":
    unittest.main()
