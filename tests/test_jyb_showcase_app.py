import io
import json
import os
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "src" / "openmic" / "tts_showcase_app.py"


class ShowcaseTests(unittest.TestCase):
    def test_completed_audio_loads_without_key_or_api_calls(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            topic = root / "real_demo_20261026" / "shopping"
            topic.mkdir(parents=True)
            (topic / "script.txt").write_text("[SETUP]固定测试稿。", encoding="utf-8")
            for condition in ("baseline", "timing", "emotion", "performance"):
                target = topic / condition
                target.mkdir()
                buffer = io.BytesIO()
                with wave.open(buffer, "wb") as audio:
                    audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                    audio.writeframes(b"\x01\x00" * 1600)
                (target / "audio.wav").write_bytes(buffer.getvalue())
                (target / "metrics.json").write_text(json.dumps({
                    "synthetic": False, "condition": condition, "duration_seconds": 0.1,
                    "inserted_silence_seconds": 0, "wall_seconds": 0.02,
                    "cache_hits": 1, "valid_cold_tts_benchmark": False}), encoding="utf-8")
                (target / "timeline.json").write_text("[]", encoding="utf-8")
            with patch.dict(os.environ, {"OPENMIC_SHOWCASE_ROOT": folder, "TTS_API_KEY": "", "OPENMIC_API_KEY": ""}), \
                    patch("requests.sessions.Session.request", side_effect=AssertionError("viewer must not call APIs")):
                app = AppTest.from_file(str(APP)).run(timeout=10)
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.tabs), 4)
            self.assertEqual(len(app.metric), 12)
            self.assertEqual(len(app.get("audio")), 4)
            self.assertTrue(any("真实中文语音" in item.value for item in app.success))
            self.assertTrue(any("不能作为冷启动" in item.value for item in app.caption))
            self.assertEqual(app.code[0].value, "[SETUP]固定测试稿。")

    def test_missing_directory_is_friendly(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"OPENMIC_SHOWCASE_ROOT": folder}):
            app = AppTest.from_file(str(APP)).run(timeout=10)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("尚未找到" in item.value for item in app.warning))

    def test_partially_generated_topic_does_not_crash(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "real_showcase_20261026" / "shopping").mkdir(parents=True)
            with patch.dict(os.environ, {"OPENMIC_SHOWCASE_ROOT": folder}):
                app = AppTest.from_file(str(APP)).run(timeout=10)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.tabs), 4)
        self.assertEqual(sum("生成中" in item.value for item in app.info), 4)
        self.assertTrue(app.selectbox[0].value.startswith("固定稿样例｜"))
        self.assertTrue(any("预置原创稿，不是五 Agent" in item.value for item in app.caption))

    def test_autogen_selection_shows_saved_trace_as_model_assessment(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "real_autogen_20261026" / "shopping").mkdir(parents=True)
            trace_dir = root / "execution" / "autogen_protocol"
            trace_dir.mkdir(parents=True)
            roles = ["ComedyDirector", "AudienceAnalyzer", "JokeWriter", "PerformanceCoach", "QualityController"]
            (trace_dir / "shopping.trace.json").write_text(json.dumps({
                "approved": True, "score": 8.1667, "generation_seconds": 66.16,
                "messages": [{"agent": role, "content": role + "测试轨迹"} for role in roles],
            }, ensure_ascii=False), encoding="utf-8")
            with patch.dict(os.environ, {"OPENMIC_SHOWCASE_ROOT": folder}):
                app = AppTest.from_file(str(APP)).run(timeout=10)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.selectbox[0].value, "真实五Agent稿｜我的网购经历")
        self.assertEqual([(item.label, item.value) for item in app.metric],
                         [("模型质检", "通过"), ("模型评分", "8.17"), ("稿件生成耗时", "66.16 秒")])
        self.assertTrue(any("不代表真人听评" in item.value for item in app.caption))
        self.assertTrue(any("仍待组内内容审核" in item.value for item in app.caption))
        for role in roles:
            self.assertTrue(any(role + "测试轨迹" in item.value for item in app.markdown))


if __name__ == "__main__":
    unittest.main()
