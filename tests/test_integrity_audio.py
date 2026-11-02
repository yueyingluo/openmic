"""Signal fixtures test preservation and timing, without requests or speech claims."""

import copy
import hashlib
import io
import json
import struct
import unittest
import wave

import numpy as np

from openmic.tts_backends import decode_wav, encode_wav
from openmic.tts_integrity_audio import (
    HEAD_RMS_LIMIT_DBFS, TAIL_RMS_LIMIT_DBFS, inspect_edges, render_integrity,
)
from openmic.tts_integrity_plan import build_integrity_plan
from openmic.tts_stage_plan import build_plan


SCRIPT = """<SCRIPT>
[SETUP]我先把第一句话完整说出来。接下来我再把第二句话讲清楚。
[PAUSE=0.5]
[PUNCHLINE]原来最困难的事情只是开始。可是直到现在我还没有开始。
[PAUSE=0.0]
[SETUP]最后我还是回到普通的这一句。
</SCRIPT>"""


def fixture():
    # Nonzero quiet edges demonstrate that no low-energy samples get removed
    # or faded. The body deliberately contains sharp peaks and exact zeros.
    head = np.tile(np.array([40, -40], dtype="<i2"), 160)
    body = np.tile(np.array([18000, -32768, 0, 32767, -900, 1700], dtype="<i2"), 137)
    tail = np.tile(np.array([20, -20], dtype="<i2"), 240)
    return np.concatenate((head, body, tail))


class IntegrityAudioTests(unittest.TestCase):
    def setUp(self):
        base = build_plan(SCRIPT)
        base["units"][1]["boundary_before"] = "turn"
        self.plan = build_integrity_plan(SCRIPT, base)
        self.raw = fixture()
        self.audio = {unit["id"]: encode_wav(self.raw + (unit["id"] % 2)) for unit in self.plan["units"]}

    def test_edge_report_has_exact_pcm_identity_and_finite_json(self):
        report = inspect_edges(encode_wav(self.raw))
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["raw_pcm_sha256"], hashlib.sha256(self.raw.tobytes()).hexdigest())
        self.assertEqual(report["nframes"], len(self.raw))
        self.assertEqual(report["head_rms_limit_dbfs"], HEAD_RMS_LIMIT_DBFS)
        self.assertEqual(report["tail_rms_limit_dbfs"], TAIL_RMS_LIMIT_DBFS)
        json.dumps(report, allow_nan=False)

    def test_digital_silence_edges_are_null_dbfs_but_not_failed(self):
        raw = self.raw.copy()
        raw[:160] = raw[-160:] = 0
        report = inspect_edges(encode_wav(raw))
        self.assertTrue(report["passed"])
        self.assertIsNone(report["head_rms_dbfs"])
        self.assertIsNone(report["tail_rms_dbfs"])

    def test_hot_head_rejected_without_repair(self):
        raw = self.raw.copy()
        raw[:160] = 1000
        report = inspect_edges(encode_wav(raw))
        self.assertFalse(report["passed"])
        self.assertGreater(report["head_rms_dbfs"], -35)
        self.assertTrue(any("onset" in error for error in report["errors"]))

    def test_hot_tail_rejected_without_padding_or_fade(self):
        raw = self.raw.copy()
        raw[-160:] = 500
        report = inspect_edges(encode_wav(raw))
        self.assertFalse(report["passed"])
        self.assertGreater(report["tail_rms_dbfs"], -40)
        self.assertTrue(any("ending" in error for error in report["errors"]))
        bad = dict(self.audio)
        bad[0] = encode_wav(raw)
        with self.assertRaisesRegex(ValueError, "Rejected raw audio for unit 0"):
            render_integrity(self.plan, bad)

    def test_entirely_silent_clip_is_rejected(self):
        report = inspect_edges(encode_wav(np.zeros(1000, dtype="<i2")))
        self.assertFalse(report["passed"])
        self.assertTrue(any("entirely" in error for error in report["errors"]))

    def test_format_and_truncation_are_rejected(self):
        for rate, channels, width in ((24000, 1, 2), (16000, 2, 2), (16000, 1, 1)):
            with self.subTest(rate=rate, channels=channels, width=width):
                stream = io.BytesIO()
                with wave.open(stream, "wb") as handle:
                    handle.setframerate(rate)
                    handle.setnchannels(channels)
                    handle.setsampwidth(width)
                    handle.writeframes(bytes(3200))
                self.assertFalse(inspect_edges(stream.getvalue())["passed"])
        good = encode_wav(self.raw)
        for bad in (b"", b"not wave", good[:-2], good + b"lost trailing PCM"):
            with self.subTest(length=len(bad)):
                self.assertFalse(inspect_edges(bad)["passed"])

    def test_placeholder_header_is_not_silently_repaired(self):
        audio = bytearray(encode_wav(self.raw))
        struct.pack_into("<I", audio, 4, 0x7FFFFFFF)
        self.assertFalse(inspect_edges(bytes(audio))["passed"])

    def test_second_data_chunk_is_not_silently_dropped(self):
        original = encode_wav(self.raw)
        extra = b"data" + struct.pack("<I", 2) + b"\x01\x00"
        audio = bytearray(original + extra)
        struct.pack_into("<I", audio, 4, len(audio) - 8)
        report = inspect_edges(bytes(audio))
        self.assertFalse(report["passed"])
        self.assertTrue(any("multiple" in error for error in report["errors"]))

    def test_every_speech_sample_is_original_including_weak_edges_and_peaks(self):
        result = render_integrity(self.plan, self.audio)
        combined = decode_wav(result.audio)
        rows = [row for row in result.timeline if row["kind"] == "speech"]
        self.assertEqual([row["unit_id"] for row in rows], list(range(len(self.audio))))
        expected_speech = []
        for row in rows:
            original = decode_wav(self.audio[row["unit_id"]])
            actual = combined[row["start_frame"]:row["end_frame"]]
            self.assertTrue(np.array_equal(actual, original))
            self.assertEqual(row["raw_pcm_sha256"], hashlib.sha256(original.tobytes()).hexdigest())
            self.assertEqual((row["trimmed_frames"], row["fade_frames"], row["local_gain_db"]), (0, 0, 0))
            expected_speech.append(original.tobytes())
        self.assertEqual(result.metrics["speech_pcm_sha256"], hashlib.sha256(b"".join(expected_speech)).hexdigest())
        self.assertFalse(result.metrics["original_stage_intent_gain_executed"])
        self.assertIsNone(result.metrics["synthetic"])

    def test_profiles_share_identical_speech_pcm_and_all_pause_samples_are_zero(self):
        course = render_integrity(self.plan, self.audio, "safe_course")
        club = render_integrity(self.plan, self.audio, "safe_club")
        self.assertEqual(course.metrics["speech_pcm_sha256"], club.metrics["speech_pcm_sha256"])
        self.assertGreater(course.metrics["duration_seconds"], club.metrics["duration_seconds"])
        for result in (course, club):
            samples = decode_wav(result.audio)
            cursor = 0
            for row in result.timeline:
                self.assertEqual(row["start_frame"], cursor)
                cursor = row["end_frame"]
                if row["kind"] == "pause":
                    self.assertEqual(row["nframes"], round(row["requested_seconds"] * 16000))
                    self.assertFalse(np.any(samples[row["start_frame"]:row["end_frame"]]))
            self.assertEqual(cursor, len(samples))

    def test_internal_reveal_sentences_do_not_repeat_reaction_gap(self):
        result = render_integrity(self.plan, self.audio)
        internal = [row for row in result.timeline if row["kind"] == "pause" and row.get("base_unit_changed") is False]
        self.assertGreaterEqual(len(internal), 2)
        self.assertTrue(all(row["requested_seconds"] == .36 for row in internal))
        reactions = [row for row in result.timeline if row["kind"] == "pause" and row.get("reaction_seconds") == 2.0]
        self.assertEqual(len(reactions), 1)
        self.assertEqual(reactions[0]["source_pause_before_seconds"], 0)

    def test_source_and_planned_gaps_take_maximum_instead_of_sum(self):
        course = render_integrity(self.plan, self.audio)
        club = render_integrity(self.plan, self.audio, "safe_club")
        for result, expected in ((course, .8), (club, .5)):
            rows = [row for row in result.timeline if row["kind"] == "pause" and row.get("source_pause_before_seconds") == .5]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["requested_seconds"], expected)

    def test_start_end_guards_and_final_base_reaction_use_maximum(self):
        base = build_plan("<SCRIPT>[PAUSE=0.6][SETUP]现在把这一句话完整说出来。[PAUSE=0.7]</SCRIPT>")
        script = "<SCRIPT>[PAUSE=0.6][SETUP]现在把这一句话完整说出来。[PAUSE=0.7]</SCRIPT>"
        plan = build_integrity_plan(script, base)
        audio = {unit["id"]: encode_wav(self.raw) for unit in plan["units"]}
        result = render_integrity(plan, audio)
        self.assertEqual(result.timeline[0]["requested_seconds"], .6)
        self.assertEqual(result.timeline[-1]["requested_seconds"], .7)
        script = "<SCRIPT>[PUNCHLINE]原来最困难的事情只是开始。</SCRIPT>"
        plan = build_integrity_plan(script, build_plan(script))
        audio = {unit["id"]: encode_wav(self.raw) for unit in plan["units"]}
        for profile in ("safe_course", "safe_club"):
            # The original compiler's automatic 2-second tail remains a floor.
            result = render_integrity(plan, audio, profile)
            self.assertEqual(result.timeline[-1]["requested_seconds"], 2.0)

    def test_missing_extra_bool_or_nonbyte_audio_is_rejected(self):
        bool_key = {False: self.audio[0], **{key: value for key, value in self.audio.items() if key != 0}}
        for bad in ({}, {**self.audio, 999: encode_wav(self.raw)}, bool_key, {**self.audio, 0: bytearray(self.audio[0])}):
            with self.subTest(keys=list(bad)):
                with self.assertRaises(ValueError):
                    render_integrity(self.plan, bad)

    def test_audio_mapping_insertion_order_does_not_reorder_speech(self):
        reversed_mapping = dict(reversed(list(self.audio.items())))
        result = render_integrity(self.plan, reversed_mapping)
        self.assertEqual([row["unit_id"] for row in result.timeline if row["kind"] == "speech"], list(self.audio))

    def test_modified_plan_or_unsupported_profile_is_rejected(self):
        plan = copy.deepcopy(self.plan)
        plan["units"][0]["text"] += "篡改"
        with self.assertRaises(ValueError):
            render_integrity(plan, self.audio)
        with self.assertRaisesRegex(ValueError, "profile"):
            render_integrity(self.plan, self.audio, "stage_club")

    def test_render_does_not_mutate_plan_or_input_mapping(self):
        before_plan = copy.deepcopy(self.plan)
        before_audio = dict(self.audio)
        result = render_integrity(self.plan, self.audio)
        self.assertEqual(self.plan, before_plan)
        self.assertEqual(self.audio, before_audio)
        json.dumps(result.metrics, allow_nan=False)
        json.dumps(result.timeline, allow_nan=False)
        self.assertFalse(result.metrics["native_phoneme_completeness_verified"])


if __name__ == "__main__":
    unittest.main()
