import io
import struct
import unittest
import wave

from openmic.tts import (
    SiliconFlowTTS,
    TTSConfig,
    TTSRequestError,
    extract_spoken_script,
    normalize_wav_header,
    prepare_tts_text,
)


class FakeResponse:
    status_code = 200
    with io.BytesIO() as _buffer:
        with wave.open(_buffer, "wb") as _wav_file:
            _wav_file.setnchannels(1)
            _wav_file.setsampwidth(2)
            _wav_file.setframerate(16000)
            _wav_file.writeframes(b"\x00\x00" * 1600)
        content = _buffer.getvalue()
    text = ""


class FakeSession:
    def __init__(self):
        self.call = None

    def post(self, url, **kwargs):
        self.call = (url, kwargs)
        return FakeResponse()


class TTSTests(unittest.TestCase):
    def test_extracts_explicit_script_boundary(self) -> None:
        text = "好的，已完成。\n<SCRIPT>\n[PAUSE=0.8]大家好。\n</SCRIPT>\n如需调整请告诉我。"
        self.assertEqual(extract_spoken_script(text), "[PAUSE=0.8]大家好。")

    def test_removes_performance_coach_boilerplate(self) -> None:
        text = """好的，表演指导已到位。以下是经过表演标记编排的最终脚本，可直接用于 TTS 音频生成。所有标记均已内嵌，仅保留脚本文字与表演指令。

[PAUSE=0.8]大家好，今天聊聊我的网购经历。

[PUNCHLINE]省了二十块，却多花了二百八，我管这叫理财。

如需调整某一段的语气、语速，或增删某个标记，请直接告诉我，我来微调。"""
        output = extract_spoken_script(text)
        self.assertTrue(output.startswith("[PAUSE=0.8]大家好"))
        self.assertTrue(output.endswith("我管这叫理财。"))
        self.assertNotIn("表演指导已到位", output)
        self.assertNotIn("如需调整", output)

    def test_prepare_text_removes_internal_markers(self) -> None:
        output = prepare_tts_text(
            "[SETUP]你好[PAUSE=0.8][EMOTION=happy]世界[EMPHASIS]"
        )
        self.assertIn("<|endofprompt|>", output)
        self.assertNotIn("[SETUP]", output)
        self.assertNotIn("[EMOTION", output)
        self.assertIn("……", output)

    def test_siliconflow_payload_uses_16khz_wav(self) -> None:
        session = FakeSession()
        config = TTSConfig(
            api_key="secret",
            base_url="https://api.siliconflow.cn/v1",
        )
        result = SiliconFlowTTS(config, session=session).synthesize("测试语音")

        url, kwargs = session.call
        self.assertEqual(url, "https://api.siliconflow.cn/v1/audio/speech")
        self.assertEqual(kwargs["json"]["sample_rate"], 16000)
        self.assertEqual(kwargs["json"]["response_format"], "wav")
        self.assertEqual(result.media_type, "audio/wav")
        self.assertEqual(result.channels, 1)
        self.assertAlmostEqual(result.duration_seconds, 0.1, places=3)
        self.assertIsNotNone(result.generation_seconds)

    def test_rejects_empty_text_after_script_cleaning(self) -> None:
        session = FakeSession()
        config = TTSConfig(
            api_key="secret",
            base_url="https://api.siliconflow.cn/v1",
        )
        with self.assertRaises(ValueError):
            SiliconFlowTTS(config, session=session).synthesize(
                "<SCRIPT>\n</SCRIPT>"
            )

    def test_inspect_wav_rejects_wrong_sample_rate(self) -> None:
        config = TTSConfig(
            api_key="secret",
            base_url="https://api.siliconflow.cn/v1",
        )

        class WrongRateResponse(FakeResponse):
            with io.BytesIO() as _buffer:
                with wave.open(_buffer, "wb") as _wav_file:
                    _wav_file.setnchannels(1)
                    _wav_file.setsampwidth(2)
                    _wav_file.setframerate(24000)
                    _wav_file.writeframes(b"\x00\x00" * 2400)
                content = _buffer.getvalue()

        class WrongRateSession(FakeSession):
            def post(self, url, **kwargs):
                self.call = (url, kwargs)
                return WrongRateResponse()

        with self.assertRaises(TTSRequestError):
            SiliconFlowTTS(config, session=WrongRateSession()).synthesize("测试语音")

    def test_normalizes_streaming_wav_sizes(self) -> None:
        audio = (
            b"RIFF"
            + struct.pack("<I", 0xFFFFFFA6)
            + b"WAVE"
            + b"fmt "
            + struct.pack("<I", 4)
            + b"test"
            + b"data"
            + struct.pack("<I", 0xFFFFFF00)
            + b"123456"
        )
        normalized = normalize_wav_header(audio)
        self.assertEqual(struct.unpack_from("<I", normalized, 4)[0], len(audio) - 8)
        self.assertEqual(struct.unpack_from("<I", normalized, 28)[0], 6)


if __name__ == "__main__":
    unittest.main()
