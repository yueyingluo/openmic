import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from openmic.models import AgentMessage, WorkflowResult
from openmic.tts_app import _workflow_script


APP_PATH = Path(__file__).resolve().parents[1] / "src" / "openmic" / "tts_app.py"


class JybAppTests(unittest.TestCase):
    def test_preview_and_tone_render_need_no_key(self):
        app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
        app.text_area[0].set_value("[SETUP]我以为能省钱。\n[PUNCHLINE]最后省下的是力气。")
        next(button for button in app.button if button.label == "预览表演计划").click().run(timeout=10)
        self.assertEqual(len(app.exception), 0)
        self.assertGreater(len(app.json), 0)
        next(button for button in app.button if button.label == "生成对照音频").click().run(timeout=10)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.tabs), 3)
        self.assertEqual(len(app.metric), 9)
        app.text_area[0].set_value("[SETUP]修改过的新稿件。").run(timeout=10)
        self.assertEqual(len(app.metric), 0)
        self.assertTrue(any("此前的结果" in item.value for item in app.warning))

    def test_unknown_marker_shows_parse_error(self):
        app = AppTest.from_file(str(APP_PATH)).run(timeout=10)
        app.text_area[0].set_value("[UNKNOWN]测试")
        next(button for button in app.button if button.label == "预览表演计划").click().run(timeout=10)
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("未知标记" in item.value for item in app.error))

    def test_mock_uses_writer_not_coach_explanation(self):
        result = WorkflowResult(True, 7.5, [AgentMessage("JokeWriter", "[SETUP]正文"),
                                          AgentMessage("PerformanceCoach", "表演标记：开场中速")], 0)
        self.assertEqual(_workflow_script(result, "mock"), "[SETUP]正文")


if __name__ == "__main__":
    unittest.main()
