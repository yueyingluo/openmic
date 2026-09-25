import unittest

from streamlit.testing.v1 import AppTest


class StreamlitAppTests(unittest.TestCase):
    def test_mock_generation_renders_tts_controls(self) -> None:
        app = AppTest.from_file("src/openmic/streamlit_app.py").run(timeout=10)
        app.radio[0].set_value("mock")
        app.button[0].click().run(timeout=10)

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.title[0].value, "🎙️ OpenMic")
        self.assertEqual(
            [(metric.label, metric.value) for metric in app.metric],
            [("质检结果", "通过"), ("平均分", "7.50"), ("返工次数", "0")],
        )
        self.assertIn("生成 16 kHz WAV", [button.label for button in app.button])
        self.assertEqual(len(app.text_area), 1)
        self.assertIn("实时协作过程", [item.value for item in app.subheader])
        self.assertEqual(len(app.chat_message), 5)


if __name__ == "__main__":
    unittest.main()
