"""Behaviour checks for jyb's strict plan, PCM assembly and clean baseline.

The fake backend produces a waveform, never speech; these tests do not establish
real API availability, voice quality, MOS, or inference speed.
"""

import io
import json
import struct
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np
import requests

from openmic.tts import TTSConfig, TTSRequestError
from openmic.tts_backends import (
    SAMPLE_RATE,
    SiliconFlowBackend,
    decode_wav,
    encode_wav,
)
from openmic.tts_plan import Pause, Speech, compile_script, plain_text
from openmic.tts_render import render


def audible_wav():
    # Leading/trailing silence exercises shared trimming; bipolar speech-shaped
    # samples keep a nonzero signal without tying the tests to a TTS provider.
    signal = np.tile(np.array([1800, -1800], dtype=np.int16), 800)
    return encode_wav(np.pad(signal, (80, 80)))


class CountingBackend:
    synthetic = False
    identity = {"backend": "fake-for-unit-tests", "version": 1}

    def __init__(self):
        self.calls = []

    def synthesize(self, text, *, instruction, speed):
        self.calls.append({"text": text, "instruction": instruction, "speed": speed})
        return audible_wav()


class FakeResponse:
    def __init__(self, status=200, content=None, text=""):
        self.status_code = status
        self.content = audible_wav() if content is None else content
        self.text = text


class RecordingSession:
    def __init__(self, response=None, error=None):
        self.response = response or FakeResponse()
        self.error = error
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.error is not None:
            raise self.error
        return self.response


class PlanTests(unittest.TestCase):
    def test_boundaries_strip_commentary_and_automatic_pauses(self):
        plan = compile_script("说明\n<SCRIPT>[SETUP]铺垫。[PUNCHLINE]包袱。</SCRIPT>\n总结")
        self.assertEqual([plain_text(e.text) for e in plan if isinstance(e, Speech)],
                         ["铺垫。", "包袱。"])
        self.assertEqual([e.seconds for e in plan if isinstance(e, Pause)], [0.8, 2.0])

    def test_explicit_zero_suppresses_both_automatic_boundaries(self):
        plan = compile_script("[SETUP]铺垫。[PAUSE=0][PUNCHLINE]包袱。[PAUSE=0]")
        pauses = [e for e in plan if isinstance(e, Pause)]
        self.assertEqual([e.seconds for e in pauses], [0, 0])
        self.assertTrue(all(e.reason == "explicit" for e in pauses))

    def test_explicit_duration_overrides_instead_of_adding(self):
        plan = compile_script("[SETUP]铺垫。[PAUSE=1.3][PUNCHLINE]包袱。[PAUSE=2.4]")
        self.assertEqual([e.seconds for e in plan if isinstance(e, Pause)], [1.3, 2.4])

    def test_each_punchline_and_callback_gets_one_pause_pair(self):
        plan = compile_script("[SETUP]甲。[PUNCHLINE]乙。[SETUP]丙。[CALLBACK]丁。")
        self.assertEqual([e.seconds for e in plan if isinstance(e, Pause)],
                         [0.8, 2.0, 0.8, 2.0])

    def test_shared_punchline_boundary_uses_max_not_sum(self):
        plan = compile_script("[SETUP]甲。[PUNCHLINE]乙。[CALLBACK]丙。")
        self.assertEqual([e.seconds for e in plan if isinstance(e, Pause)], [0.8, 2.0, 2.0])

    def test_split_within_same_beat_does_not_invent_extra_punchline_pauses(self):
        plan = compile_script("[SETUP]甲。[PUNCHLINE]乙，[EMOTION=happy]丙。\n丁。")
        self.assertEqual([e.seconds for e in plan if isinstance(e, Pause)], [0.8, 2.0])

    def test_unknown_malformed_or_unsafe_controls_fail(self):
        for source in ("正文[LOUD]", "正文[PAUSE=x]", "正文[PAUSE=-1]",
                       "正文[PAUSE=nan]", "正文[SPEED=2]", "正文[EMOTION=unknown]",
                       "正文[PAUSE=1", "正文]", "<b>正文</b>",
                       "正文<|endofprompt|>", "正文[/EMPHASIS]", "[PAUSE=2]"):
            with self.subTest(source=source):
                with self.assertRaises(ValueError):
                    compile_script(source)

    def test_closed_emphasis_scopes_only_the_marked_word(self):
        plan = compile_script("[SETUP]只重读[EMPHASIS]这里[/EMPHASIS]，后面正常。")
        speech = [e for e in plan if isinstance(e, Speech)][0]
        self.assertEqual(speech.text, "只重读<strong>这里</strong>，后面正常。")
        self.assertEqual(plain_text(speech.text), "只重读这里，后面正常。")

    def test_legacy_emphasis_ends_at_newline_or_control(self):
        plan = compile_script("[SETUP][EMPHASIS]第一行\n第二行[EMPHASIS]重点[EMOTION=happy]平常。")
        speech = [e for e in plan if isinstance(e, Speech)]
        self.assertEqual(speech[0].text, "<strong>第一行</strong>")
        self.assertEqual(speech[1].text, "第二行<strong>重点</strong>")
        self.assertEqual(speech[2].text, "平常。")
        self.assertFalse(speech[2].emphasis)

    def test_chinese_emotion_and_speed_persist_until_next_beat(self):
        plan = compile_script("[SETUP][EMOTION=开心][SPEED=1.2]甲。\n乙。[PUNCHLINE]丙。")
        speech = [e for e in plan if isinstance(e, Speech)]
        self.assertEqual([(e.emotion, e.speed) for e in speech],
                         [("happy", 1.2), ("happy", 1.2), ("neutral", 0.95)])

    def test_filler_insertion_is_optional_and_bounded(self):
        source = "".join("[SETUP]" + "这是比较长的铺垫内容，" * 4 for _ in range(3))
        normal = [e for e in compile_script(source) if isinstance(e, Speech)]
        filled = [e for e in compile_script(source, add_fillers=True) if isinstance(e, Speech)]
        self.assertEqual([e.text for e in normal[:2]], [e.text for e in filled[:2]])
        self.assertEqual(filled[2].text, "那个，" + normal[2].text)


class RenderTests(unittest.TestCase):
    def test_08_and_20_pauses_are_exact_zero_samples(self):
        plan = compile_script("[SETUP]铺垫。[PUNCHLINE]包袱。")
        result = render(plan, CountingBackend())
        signal = decode_wav(result.audio)
        pauses = [r for r in result.timeline if r["kind"] == "pause"]
        self.assertEqual([r["end_frame"] - r["start_frame"] for r in pauses], [12800, 32000])
        for row in pauses:
            self.assertTrue(np.all(signal[row["start_frame"]:row["end_frame"]] == 0))
        self.assertAlmostEqual(result.metrics["inserted_silence_seconds"], 2.8)
        self.assertEqual(result.timeline[-1]["end_frame"], len(signal))
        for left, right in zip(result.timeline, result.timeline[1:]):
            self.assertEqual(left["end_frame"], right["start_frame"])

    def test_zero_pause_does_not_create_frames(self):
        result = render(compile_script("[SETUP]甲。[PAUSE=0][PUNCHLINE]乙。[PAUSE=0]"),
                        CountingBackend())
        self.assertEqual(result.metrics["inserted_silence_seconds"], 0)
        self.assertTrue(all(r["start_frame"] == r["end_frame"]
                            for r in result.timeline if r["kind"] == "pause"))

    def test_baseline_is_neutral_and_drops_emphasis_speed_and_silence(self):
        backend = CountingBackend()
        plan = compile_script("[SETUP][EMOTION=angry][SPEED=1.2]我很[EMPHASIS]冷静[/EMPHASIS]。")
        result = render(plan, backend, condition="baseline")
        self.assertEqual(backend.calls, [{"text": "我很冷静。",
                                         "instruction": "请用自然平静的语气说。", "speed": 1.0}])
        self.assertEqual(result.metrics["inserted_silence_seconds"], 0)

    def test_baseline_and_emotion_use_ordinary_boundaries_without_edge_padding(self):
        # Ordinary short boundaries keep the segmented baseline from sounding
        # unnaturally joined after all exterior provider silence is trimmed.
        plan = compile_script("[SETUP]甲。\n乙。\n丙。")
        for condition in ("baseline", "emotion"):
            with self.subTest(condition=condition):
                result = render(plan, CountingBackend(), condition=condition)
                self.assertEqual(result.timeline[0]["kind"], "speech")
                self.assertEqual(result.timeline[-1]["kind"], "speech")
                pauses = [row for row in result.timeline if row["kind"] == "pause"]
                self.assertEqual([row["end_frame"] - row["start_frame"] for row in pauses],
                                 [1920, 1920])
                signal = decode_wav(result.audio)
                for row in pauses:
                    self.assertTrue(np.all(signal[row["start_frame"]:row["end_frame"]] == 0))

    def test_baseline_replaces_performance_pauses_with_ordinary_boundaries(self):
        plan = compile_script("[SETUP]甲。[PAUSE=0][PUNCHLINE]乙。[PAUSE=2]")
        for condition in ("baseline", "emotion"):
            with self.subTest(condition=condition):
                result = render(plan, CountingBackend(), condition=condition)
                pauses = [row for row in result.timeline if row["kind"] == "pause"]
                self.assertEqual([row["end_frame"] - row["start_frame"] for row in pauses], [1920])
                self.assertEqual(result.timeline[-1]["kind"], "speech")

    def test_timing_and_performance_share_ordinary_boundaries_but_respect_explicit_pause(self):
        plan = compile_script("[SETUP]甲。\n乙。[PAUSE=0]丙。[PAUSE=0.6]丁。")
        for condition in ("timing", "performance"):
            with self.subTest(condition=condition):
                result = render(plan, CountingBackend(), condition=condition)
                pauses = [row for row in result.timeline if row["kind"] == "pause"]
                self.assertEqual([row["end_frame"] - row["start_frame"] for row in pauses],
                                 [1920, 0, 9600])
                self.assertEqual(result.timeline[0]["kind"], "speech")
                self.assertEqual(result.timeline[-1]["kind"], "speech")

    def test_emotion_condition_applies_only_emotion(self):
        backend = CountingBackend()
        result = render(compile_script("[PUNCHLINE][EMOTION=开心][SPEED=1.2]真[EMPHASIS]开心[/EMPHASIS]。"),
                        backend, condition="emotion")
        self.assertEqual(backend.calls[0], {"text": "真开心。", "instruction": "请用高兴的语气说。", "speed": 1.0})
        self.assertEqual(result.metrics["inserted_silence_seconds"], 0)

    def test_timing_condition_keeps_pauses_with_neutral_speech(self):
        backend = CountingBackend()
        result = render(compile_script("[PUNCHLINE][EMOTION=angry][SPEED=1.2]甲。"),
                        backend, condition="timing")
        self.assertEqual(backend.calls[0]["instruction"], "请用自然平静的语气说。")
        self.assertEqual(backend.calls[0]["speed"], 1.0)
        self.assertAlmostEqual(result.metrics["inserted_silence_seconds"], 2.8)

    def test_performance_preserves_selected_controls(self):
        backend = CountingBackend()
        render(compile_script("[SETUP][EMOTION=开心][SPEED=1.2]真[EMPHASIS]开心[/EMPHASIS]。"), backend)
        self.assertEqual(backend.calls[0], {"text": "真<strong>开心</strong>。",
                                          "instruction": "请用高兴的语气说。", "speed": 1.2})

    def test_cold_warm_cache_and_corrupt_cache_refetch(self):
        plan = compile_script("[SETUP]唯一一句测试文本。")
        backend = CountingBackend()
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            cold = render(plan, backend, cache_dir=cache)
            warm = render(plan, backend, cache_dir=cache)
            self.assertEqual(len(backend.calls), 1)
            self.assertEqual(cold.audio, warm.audio)
            self.assertEqual((cold.metrics["api_calls"], cold.metrics["cache_hits"]), (1, 0))
            self.assertEqual((warm.metrics["api_calls"], warm.metrics["cache_hits"]), (0, 1))
            self.assertTrue(cold.metrics["valid_cold_tts_benchmark"])
            self.assertFalse(warm.metrics["valid_cold_tts_benchmark"])
            next(cache.glob("*.wav")).write_bytes(b"bad cached payload")
            recovered = render(plan, backend, cache_dir=cache)
            self.assertEqual(len(backend.calls), 2)
            self.assertEqual(recovered.audio, cold.audio)
            self.assertEqual(recovered.metrics["cache_hits"], 0)

    def test_different_actual_controls_do_not_share_cache(self):
        backend = CountingBackend()
        plan = compile_script("[SETUP][EMOTION=angry][SPEED=1.2]同样正文。")
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            render(plan, backend, condition="baseline", cache_dir=cache)
            render(plan, backend, condition="performance", cache_dir=cache)
        self.assertEqual(len(backend.calls), 2)

    def test_valid_format_but_silent_cache_is_refetched(self):
        backend = CountingBackend()
        plan = compile_script("[SETUP]正文。")
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            render(plan, backend, cache_dir=cache)
            next(cache.glob("*.wav")).write_bytes(encode_wav(np.zeros(1600, dtype=np.int16)))
            result = render(plan, backend, cache_dir=cache)
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(result.metrics["cache_hits"], 0)

    def test_segment_callbacks_have_playable_wav_and_matching_timeline(self):
        observed = []
        result = render(compile_script("[SETUP]甲。[PUNCHLINE]乙。"), CountingBackend(),
                        on_chunk=lambda audio, row: observed.append((audio, row)))
        self.assertEqual([row for _, row in observed], result.timeline)
        for audio, row in observed:
            self.assertEqual(len(decode_wav(audio)), row["end_frame"] - row["start_frame"])


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.config = TTSConfig(api_key="unit-test-secret-key", base_url="https://example.invalid/v1")

    def test_fake_api_payload_neutral_baseline_bypasses_old_prompt(self):
        session = RecordingSession()
        backend = SiliconFlowBackend(self.config, session=session)
        render(compile_script("[SETUP][EMOTION=angry]正文。"), backend, condition="baseline")
        url, args = session.calls[0]
        self.assertEqual(url, "https://example.invalid/v1/audio/speech")
        self.assertEqual(args["json"]["input"], "请用自然平静的语气说。<|endofprompt|>正文。")
        self.assertNotIn("幽默、自嘲", args["json"]["input"])
        self.assertEqual(args["json"]["sample_rate"], 16000)
        self.assertEqual(args["json"]["response_format"], "wav")
        self.assertEqual(args["headers"]["Authorization"], "Bearer unit-test-secret-key")
        self.assertNotIn("unit-test-secret-key", json.dumps(backend.identity))

    def test_provider_network_error_does_not_expose_secret(self):
        session = RecordingSession(error=requests.RequestException("Bearer unit-test-secret-key"))
        backend = SiliconFlowBackend(self.config, session=session)
        with self.assertRaises(TTSRequestError) as raised:
            backend.synthesize("正文", instruction="自然", speed=1)
        self.assertNotIn("unit-test-secret-key", str(raised.exception))

    def test_provider_http_error_does_not_expose_response_body(self):
        session = RecordingSession(response=FakeResponse(status=401, text="unit-test-secret-key"))
        backend = SiliconFlowBackend(self.config, session=session)
        with self.assertRaises(TTSRequestError) as raised:
            backend.synthesize("正文", instruction="自然", speed=1)
        self.assertNotIn("unit-test-secret-key", str(raised.exception))
        self.assertIn("401", str(raised.exception))

    def test_invalid_audio_and_wrong_format_are_rejected(self):
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(2)
            handle.setsampwidth(2)
            handle.setframerate(24000)
            handle.writeframes(b"\x00\x00" * 100)
        for data in (b"unit-test-secret-key is not WAV", buffer.getvalue(), encode_wav(np.array([], dtype=np.int16))):
            with self.subTest(data_length=len(data)):
                with self.assertRaises(ValueError):
                    decode_wav(data)

    def test_declared_pcm_truncation_is_rejected_before_header_normalization(self):
        audio = audible_wav()
        with self.assertRaises(ValueError):
            decode_wav(audio[:-40])

    def test_partial_pcm_frame_is_rejected(self):
        audio = bytearray(audible_wav()[:-1])
        struct.pack_into("<I", audio, 4, len(audio) - 8)
        struct.pack_into("<I", audio, 40, len(audio) - 44)
        with self.assertRaises(ValueError):
            decode_wav(bytes(audio))

    def test_streaming_placeholder_lengths_are_repaired(self):
        audio = bytearray(audible_wav())
        original = decode_wav(bytes(audio))
        struct.pack_into("<I", audio, 4, 0xFFFFFFFF)
        struct.pack_into("<I", audio, 40, 0xFFFFFFFF)
        self.assertTrue(np.array_equal(decode_wav(bytes(audio)), original))

    def test_trailing_list_chunk_does_not_become_speech(self):
        original = audible_wav()
        audio = bytearray(original + b"LIST" + struct.pack("<I", 4) + b"INFO")
        struct.pack_into("<I", audio, 4, len(audio) - 8)
        self.assertTrue(np.array_equal(decode_wav(bytes(audio)), decode_wav(original)))


if __name__ == "__main__":
    unittest.main()
