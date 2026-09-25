import unittest
import struct

from openmic.tts import (
    SiliconFlowTTS,
    TTSConfig,
    normalize_wav_header,
    prepare_tts_text,
)


class FakeResponse:
    status_code = 200
    content = b"RIFFfake-wave"
    text = ""


class FakeSession:
    def __init__(self):
        self.call = None

    def post(self, url, **kwargs):
        self.call = (url, kwargs)
        return FakeResponse()


class TTSTests(unittest.TestCase):
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
