"""Local signal fixtures verify stage control integrity, never listening quality."""

import copy
import hashlib
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np

from openmic.tts_backends import ToneBackend, decode_wav, encode_wav
from openmic.tts_plan import Speech, compile_script, plain_text
from openmic.tts_render import render
from openmic.tts_stage_plan import build_plan
from openmic.tts_stage_render import (
    HEADROOM_PEAK, INTENT_GAINS_DB, INTENT_SPEEDS, PAUSE_SECONDS, PROFILES,
    STAGE_INSTRUCTION, render_stage,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = """<SCRIPT>
[SETUP][EMOTION=calm][SPEED=1.03]大家好，我最近认真研究怎样省钱。
[SETUP]今天打开购物车，我先看了一眼余额。
[PAUSE=0.8]
[PUNCHLINE][EMOTION=surprised][SPEED=0.95]后来发现，省下来的只有[EMPHASIS]出门的力气[/EMPHASIS]。
[PAUSE=2.0]
[SETUP]于是我决定先收起手机。
[CALLBACK][EMOTION=self_deprecating]购物车替我宣布，这就算努力过了。
[PAUSE=2.0]
</SCRIPT>"""


class SignalBackend:
    synthetic = True
    identity = {"backend": "unit-test-waveform", "voice": "test-only:alex", "sample_rate": 16000}

    def __init__(self, peak=2600):
        self.calls = []
        self.peak = peak

    def synthesize(self, text, *, instruction, speed):
        self.calls.append({"text": text, "instruction": instruction,
                           "speed": speed, "voice": self.identity["voice"]})
        signal = np.tile(np.array([self.peak, -self.peak], dtype="<i2"), 800)
        return encode_wav(np.pad(signal, (100, 100)))


class StageRenderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="stage_render_test_")
        self.addCleanup(temporary.cleanup)
        self.temporary = Path(temporary.name)
        self.plan = build_plan(SCRIPT)

    def controlled_plan(self):
        plan = copy.deepcopy(self.plan)
        plan["units"][1].update(intent="contrast", focus="余额", boundary_before="breath")
        plan["units"][2].update(intent="reveal", focus="力气", boundary_before="turn")
        plan["units"][3].update(intent="aside", focus=None, boundary_before="flow")
        plan["units"][4].update(intent="button", focus="努力", boundary_before="turn")
        return plan

    def test_fixed_persona_and_voice_for_all_three_profiles(self):
        for profile in PROFILES:
            with self.subTest(profile=profile):
                backend = SignalBackend()
                result = render_stage(self.plan, backend, profile)
                self.assertEqual({call["instruction"] for call in backend.calls}, {STAGE_INSTRUCTION})
                self.assertEqual({call["voice"] for call in backend.calls}, {"test-only:alex"})
                self.assertEqual(result.metrics["actual_instruction"], STAGE_INSTRUCTION)
                speech = [row for row in result.timeline if row["kind"] == "speech"]
                self.assertTrue(all(row["actual_instruction"] == STAGE_INSTRUCTION for row in speech))
                self.assertTrue(any(row["original_emotion"] == "surprised" for row in speech))
                self.assertTrue(all("emotion" not in row for row in speech))
                self.assertTrue(result.metrics["synthetic"])
                self.assertFalse(result.metrics["valid_cold_tts_benchmark"])

    def test_prompt_only_changes_instruction_not_text_speed_gain_or_wave_assembly(self):
        old_backend, new_backend = SignalBackend(), SignalBackend()
        old = render(compile_script(SCRIPT), old_backend, condition="performance")
        new = render_stage(self.plan, new_backend, "stage_prompt")
        self.assertEqual(new.audio, old.audio)
        self.assertEqual([(call["text"], call["speed"]) for call in old_backend.calls],
                         [(call["text"], call["speed"]) for call in new_backend.calls])
        self.assertTrue(any("<strong>" in call["text"] for call in new_backend.calls))
        self.assertEqual(new.metrics["global_headroom_gain"], 1.0)
        self.assertTrue(all(row["actual_gain_db"] == 0 for row in new.timeline if row["kind"] == "speech"))
        self.assertNotEqual([call["instruction"] for call in old_backend.calls],
                            [call["instruction"] for call in new_backend.calls])
        old_pauses = [(row["start_frame"], row["end_frame"]) for row in old.timeline if row["kind"] == "pause"]
        new_pauses = [(row["start_frame"], row["end_frame"]) for row in new.timeline if row["kind"] == "pause"]
        self.assertEqual(new_pauses, old_pauses)

    def test_prompt_only_keeps_explicit_zero_and_original_ordinary_boundaries(self):
        script = "<SCRIPT>[SETUP]甲。\n乙。[PAUSE=0][PUNCHLINE]丙。[PAUSE=0]</SCRIPT>"
        plan = build_plan(script)
        expected = render(compile_script(script), SignalBackend(), condition="performance")
        actual = render_stage(plan, SignalBackend(), "stage_prompt")
        self.assertEqual(actual.audio, expected.audio)
        self.assertEqual([row["requested_seconds"] for row in actual.timeline if row["kind"] == "pause"],
                         [0.12, 0.0, 0.0])

    def test_four_conditions_preserve_source_body_hash_and_exact_source_slices(self):
        source_texts = [plain_text(event.text) for event in compile_script(SCRIPT) if isinstance(event, Speech)]
        body_hash = hashlib.sha256("".join(source_texts).encode("utf-8")).hexdigest()
        original = render(compile_script(SCRIPT), SignalBackend(), condition="performance")
        results = [render_stage(self.controlled_plan(), SignalBackend(), profile) for profile in PROFILES]
        self.assertTrue(all(result.metrics["spoken_text_sha256"] == body_hash for result in results))
        self.assertTrue(all(result.metrics["legacy_spoken_text_sha256"] == original.metrics["spoken_text_sha256"]
                            for result in results))
        for result in results:
            self.assertEqual("".join(row["text"] for row in result.timeline if row["kind"] == "speech"),
                             "".join(source_texts))
            for row in result.timeline:
                if row["kind"] == "speech":
                    self.assertEqual(row["text"], source_texts[row["source_index"]][row["source_start"]:row["source_end"]])
                    self.assertEqual(plain_text(row["actual_input"]), row["text"])

    def test_full_profiles_use_intent_speed_one_native_focus_and_fixed_linear_gain(self):
        plan, backend = self.controlled_plan(), SignalBackend()
        result = render_stage(plan, backend, "stage_course")
        speech = [row for row in result.timeline if row["kind"] == "speech"]
        for row, unit, call in zip(speech, plan["units"], backend.calls):
            self.assertEqual(call["speed"], INTENT_SPEEDS[unit["intent"]])
            self.assertEqual(row["actual_gain_db"], INTENT_GAINS_DB[unit["intent"]])
            self.assertEqual(call["text"].count("<strong>"), int(bool(unit["focus"])))
            self.assertEqual(call["text"].count("</strong>"), int(bool(unit["focus"])))
            if unit["focus"]:
                self.assertIn("<strong>" + unit["focus"] + "</strong>", call["text"])
            self.assertIn("not independently verified", row["focus_execution"])

    def test_course_and_club_share_final_speech_pcm_and_all_requests_from_cache(self):
        plan, backend = self.controlled_plan(), SignalBackend(peak=32760)
        cache = self.temporary / "cache"
        course = render_stage(plan, backend, "stage_course", cache_dir=cache)
        count = len(backend.calls)
        club = render_stage(plan, backend, "stage_club", cache_dir=cache)
        self.assertEqual(len(backend.calls), count)
        self.assertEqual(club.metrics["api_calls"], 0)
        self.assertEqual(club.metrics["cache_hits"], len(plan["units"]))
        self.assertEqual(course.metrics["speech_pcm_sha256"], club.metrics["speech_pcm_sha256"])
        self.assertEqual(course.metrics["global_headroom_gain"], club.metrics["global_headroom_gain"])
        self.assertLess(course.metrics["global_headroom_gain"], 1)
        self.assertNotEqual(course.audio, club.audio)
        self.assertGreater(course.metrics["inserted_silence_seconds"], club.metrics["inserted_silence_seconds"])
        self.assertEqual(course.metrics["speech_seconds"], club.metrics["speech_seconds"])

    def test_pauses_are_exact_zeros_and_reaction_uses_max_not_sum(self):
        plan = self.controlled_plan()
        for profile in ("stage_course", "stage_club"):
            result = render_stage(plan, SignalBackend(), profile)
            signal = decode_wav(result.audio)
            self.assertEqual(result.timeline[0]["kind"], "speech")
            self.assertEqual(result.timeline[-1]["kind"], "pause")
            pauses = [row for row in result.timeline if row["kind"] == "pause"]
            self.assertEqual([row["pause_type"] for row in pauses], ["breath", "turn", "reaction", "turn", "reaction"])
            for row in pauses:
                self.assertEqual(row["requested_seconds"], PAUSE_SECONDS[profile][row["pause_type"]])
                self.assertEqual(row["end_frame"] - row["start_frame"], round(row["requested_seconds"] * 16000))
                self.assertTrue(np.all(signal[row["start_frame"]:row["end_frame"]] == 0))
            self.assertEqual(result.timeline[-1]["end_frame"], len(signal))
            self.assertTrue(all(left["end_frame"] == right["start_frame"]
                                for left, right in zip(result.timeline, result.timeline[1:])))

    def test_non_reveal_ending_has_no_tail_pause_and_first_boundary_is_ignored(self):
        plan = self.controlled_plan()
        plan["units"][0]["boundary_before"] = "reaction"
        plan["units"][-1]["intent"] = "setup"
        for profile in ("stage_course", "stage_club"):
            result = render_stage(plan, SignalBackend(), profile)
            self.assertEqual(result.timeline[0]["kind"], "speech")
            self.assertEqual(result.timeline[-1]["kind"], "speech")

    def test_headroom_preserves_relative_unit_gain_without_clipping_or_leveling(self):
        backend = SignalBackend(peak=32760)
        result = render_stage(self.controlled_plan(), backend, "stage_club")
        signal = decode_wav(result.audio).astype(np.int32)
        self.assertLessEqual(int(np.max(np.abs(signal))), HEADROOM_PEAK)
        speech = [row for row in result.timeline if row["kind"] == "speech"]
        peaks = [int(np.max(np.abs(signal[row["start_frame"]:row["end_frame"]]))) for row in speech]
        self.assertGreater(peaks[4], peaks[3])  # button +1.2 dB vs aside -0.8 dB
        self.assertAlmostEqual(peaks[4] / peaks[3], 10 ** (2 / 20), delta=0.0002)
        self.assertTrue(all(row["global_headroom_gain"] == result.metrics["global_headroom_gain"] for row in speech))

    def test_final_pcm_callbacks_match_saved_audio_and_wait_for_global_headroom(self):
        backend, received = SignalBackend(peak=32760), []
        result = render_stage(self.controlled_plan(), backend, "stage_course",
                              on_chunk=lambda audio, row: received.append((audio, row, len(backend.calls))))
        final = decode_wav(result.audio)
        self.assertEqual(len(received), len(result.timeline))
        self.assertEqual(received[0][2], len(self.plan["units"]))
        for audio, row, _ in received:
            self.assertTrue(np.array_equal(decode_wav(audio), final[row["start_frame"]:row["end_frame"]]))
        self.assertGreaterEqual(result.metrics["first_speech_chunk_seconds"], result.metrics["first_raw_speech_ready_seconds"])

    def test_validated_raw_wav_cache_refetches_corrupt_and_silent_entries(self):
        backend, cache = SignalBackend(), self.temporary / "cache"
        first = render_stage(self.plan, backend, "stage_course", cache_dir=cache)
        key = next(row["cache_key"] for row in first.timeline if row["kind"] == "speech")
        for invalid in (b"broken WAV", encode_wav(np.zeros(400, dtype="<i2"))):
            with self.subTest(kind=len(invalid)):
                (cache / (key + ".wav")).write_bytes(invalid)
                before = len(backend.calls)
                result = render_stage(self.plan, backend, "stage_course", cache_dir=cache)
                self.assertEqual(len(backend.calls), before + 1)
                self.assertEqual(result.metrics["api_calls"], 1)
                self.assertEqual(result.metrics["cache_hits"], len(self.plan["units"]) - 1)
                self.assertEqual(result.audio, first.audio)

    def test_concurrent_cache_writes_use_unique_atomic_temporary_files(self):
        class DifferentResponses(SignalBackend):
            def __init__(self):
                super().__init__()
                self.lock = threading.Lock()

            def synthesize(self, text, *, instruction, speed):
                with self.lock:
                    peak = 2000 + len(self.calls) * 100
                    self.calls.append({"text": text, "instruction": instruction, "speed": speed, "peak": peak})
                signal = np.tile(np.array([peak, -peak], dtype="<i2"), 800)
                return encode_wav(np.pad(signal, (100, 100)))

        cache, backend = self.temporary / "shared-cache", DifferentResponses()
        barrier = threading.Barrier(2)
        original_write = Path.write_bytes

        def simultaneous_write(path, data):
            count = original_write(path, data)
            if path.suffix == ".tmp":
                barrier.wait(timeout=10)
            return count

        with patch.object(Path, "write_bytes", simultaneous_write), ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(render_stage, self.plan, backend, "stage_course", cache) for _ in range(2)]
            results = [future.result(timeout=20) for future in futures]
        self.assertEqual(results[0].metrics["speech_pcm_sha256"], results[1].metrics["speech_pcm_sha256"])
        self.assertEqual(sum(result.metrics["api_calls"] for result in results), len(backend.calls))
        self.assertEqual(len(backend.calls), 2 * len(self.plan["units"]))
        self.assertEqual(sum(result.metrics["cache_winner_reuses"] for result in results), len(self.plan["units"]))
        self.assertTrue(all(result.metrics["cache_hits"] == 0 for result in results))
        self.assertEqual(len(list(cache.glob("*.wav"))), len(self.plan["units"]))
        self.assertFalse(list(cache.glob("*.tmp")))
        self.assertFalse(list(cache.glob("*.lock")))
        for path in cache.glob("*.wav"):
            self.assertGreater(len(decode_wav(path.read_bytes())), 0)

    def test_invalid_plan_and_parameters_fail_before_any_backend_request(self):
        cases = []
        wrong_words = copy.deepcopy(self.plan)
        wrong_words["units"][0]["text"] += "新增的词"
        cases.append((wrong_words, "stage_course", None))
        cases.append((self.plan, "unknown", None))
        cases.append((self.plan, "stage_course", "not callable"))
        for plan, profile, callback in cases:
            backend = SignalBackend()
            with self.subTest(profile=profile, callback=callback):
                with self.assertRaises(ValueError):
                    render_stage(plan, backend, profile, on_chunk=callback)
                self.assertFalse(backend.calls)

    def test_tone_backend_remains_explicitly_synthetic(self):
        result = render_stage(self.plan, ToneBackend(), "stage_club")
        self.assertTrue(result.metrics["synthetic"])
        self.assertFalse(result.metrics["valid_cold_tts_benchmark"])
        self.assertIsNotNone(result.metrics["first_model_segment_seconds"])
        self.assertEqual(result.metrics["sample_rate"], 16000)
        self.assertEqual(result.metrics["channels"], 1)
        self.assertEqual(result.metrics["sample_width"], 2)


if __name__ == "__main__":
    unittest.main()
