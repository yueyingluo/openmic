import unittest

from openmic.tts_asr_check import find_cached_transcription


class ASRPortabilityTests(unittest.TestCase):
    def setUp(self):
        self.row = {"audio": r"C:\old\topic\performance\audio.wav", "audio_sha256": "audio",
                    "reference_sha256": "words", "model": "asr", "condition": "performance"}

    def test_content_cache_does_not_require_windows_path_to_exist(self):
        self.assertIs(find_cached_transcription([self.row], "audio", "words", "asr", "performance"), self.row)

    def test_changed_content_or_failed_response_is_not_reused(self):
        for audio, text, model, condition in (("changed", "words", "asr", "performance"),
                                              ("audio", "changed", "asr", "performance"),
                                              ("audio", "words", "changed", "performance"),
                                              ("audio", "words", "asr", "baseline")):
            self.assertIsNone(find_cached_transcription([self.row], audio, text, model, condition))
        self.assertIsNone(find_cached_transcription([{**self.row, "error": "timeout"}],
                                                   "audio", "words", "asr", "performance"))


if __name__ == "__main__":
    unittest.main()
