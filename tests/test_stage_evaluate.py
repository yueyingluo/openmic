"""Synthetic PCM validates engineering and blank forms, never speech quality."""

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import wave

import numpy as np

from openmic.tts_stage_evaluate import CONDITIONS, DIMENSIONS, analyze_file, build_blind, match_topic


def write_wave(path, samples, rate=16000, channels=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as writer:
        writer.setparams((channels, 2, rate, 0, "NONE", "not compressed"))
        writer.writeframes(np.asarray(samples, dtype="<i2").tobytes())


def fixture_topic(root, name="topic"):
    directory = root / name
    time = np.arange(32000) / 16000
    # Vary level and crest factor; silence belongs to the unchanged test waveform.
    for i, condition in enumerate(CONDITIONS):
        signal = np.sin(2 * np.pi * (180 + i * 7) * time) * (1600 + i * 2200)
        signal[8000:12000] = 0
        target = directory / condition
        write_wave(target / "audio.wav", signal)
        (target / "metrics.json").write_text(json.dumps({"condition": condition}), encoding="utf-8")
        (target / "timeline.json").write_text("[]", encoding="utf-8")
    (directory / "legacy/reference_provenance.json").write_text(
        json.dumps({"copied_reference": True, "new_api_request": False}), encoding="utf-8")
    return directory


class StageEvaluationTests(unittest.TestCase):
    def test_strict_wave_and_no_perceptual_claims(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "silence.wav"
            write_wave(path, np.zeros(16000))
            measured = analyze_file(path)
            self.assertIsNone(measured["integrated_lufs"])
            self.assertEqual(measured["loudness_status"], "unmeasurable_silence")
            self.assertEqual(measured["exact_zero_fraction"], 1)
            self.assertFalse(measured["perceptual_quality_estimated"])
            self.assertFalse(measured["f0_measured"])
            self.assertIsNone(measured["rms_frames"]["active_p90_minus_p10_db"])
            write_wave(root / "wrong.wav", np.ones(16000), rate=24000)
            with self.assertRaisesRegex(ValueError, "16000"):
                analyze_file(root / "wrong.wav")
            path.write_bytes(path.read_bytes()[:-2])
            with self.assertRaisesRegex(ValueError, "RIFF"):
                analyze_file(path)

    def test_global_match_preserves_original_and_pcm_length(self):
        with tempfile.TemporaryDirectory() as temporary:
            topic = fixture_topic(Path(temporary))
            originals = {condition: (topic / condition / "audio.wav").read_bytes() for condition in CONDITIONS}
            report = match_topic(topic)
            self.assertEqual(report["status"], "passed")
            self.assertFalse(report["legacy_new_tts_request_this_run"])
            for condition in CONDITIONS:
                row = report["conditions"][condition]
                self.assertEqual((topic / condition / "audio.wav").read_bytes(), originals[condition])
                self.assertLessEqual(row["lufs_error_db"], 0.12)
                self.assertLessEqual(row["matched"]["approx_true_peak_dbtp"], -1)
                self.assertTrue(row["samples_unchanged_in_count"])
                self.assertTrue((topic / condition / "levels.json").is_file())
                self.assertAlmostEqual(row["gain_db"], report["target_lufs"] - row["original"]["integrated_lufs"])
                with wave.open(str(topic / condition / "audio.wav"), "rb") as reader:
                    source_pcm = np.frombuffer(reader.readframes(reader.getnframes()), dtype="<i2")
                with wave.open(str(topic / condition / "matched.wav"), "rb") as reader:
                    matched_pcm = np.frombuffer(reader.readframes(reader.getnframes()), dtype="<i2")
                np.testing.assert_array_equal(
                    matched_pcm, np.rint(source_pcm.astype(np.float64) * row["linear_gain"]).astype("<i2"))
            matched_loudness = [row["matched"]["integrated_lufs"] for row in report["conditions"].values()]
            self.assertLess(max(matched_loudness) - min(matched_loudness), 0.12)

    def test_high_crest_factor_reduces_common_target_instead_of_limiting(self):
        with tempfile.TemporaryDirectory() as temporary:
            topic = fixture_topic(Path(temporary))
            signal = np.sin(2 * np.pi * 210 * np.arange(32000) / 16000) * 80
            signal[15000] = 31000
            write_wave(topic / "legacy/audio.wav", signal)
            report = match_topic(topic)
            expected = min(-20.0, *(row["original"]["integrated_lufs"] - 1.1 -
                                     row["original"]["approx_true_peak_dbtp"]
                                     for row in report["conditions"].values()))
            self.assertAlmostEqual(report["target_lufs"], expected)
            self.assertLess(report["target_lufs"], -20)
            self.assertTrue(all(row["matched"]["approx_true_peak_dbtp"] <= -1
                                for row in report["conditions"].values()))

    def test_missing_condition_is_error_without_matched_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            topic = fixture_topic(Path(temporary))
            (topic / "stage_club/audio.wav").unlink()
            with self.assertRaises(FileNotFoundError):
                match_topic(topic)
            self.assertEqual(list(topic.rglob("matched.wav")), [])

    def test_blank_blind_forms_balanced_and_no_condition_leak(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = root / "runs"
            for name in ("first", "second"):
                match_topic(fixture_topic(runs, name))
            out = root / "blind"
            summary = build_blind(runs, out, listeners=8, seed=91)
            self.assertEqual(summary["human_ratings_collected"], 0)
            self.assertEqual(summary["pairs_per_listener"], 2)
            key = json.loads((out / "investigator/answer_key.json").read_text(encoding="utf-8"))
            for topic in ("first", "second"):
                rows = [row for row in key["rows"] if row["topic"] == topic]
                self.assertEqual(sum(row["A"] == "legacy" for row in rows), 4)
            csv_files = list((out / "public/listeners").glob("*/ratings.csv"))
            self.assertEqual(len(csv_files), 8)
            for path in csv_files:
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                for row in rows:
                    self.assertTrue(all(row[name] == "" for name in (*DIMENSIONS, "comment")))
                    self.assertTrue((path.parent / row["audio_a"]).is_file())
                    self.assertTrue((path.parent / row["audio_b"]).is_file())
            for path in (out / "public").rglob("*"):
                if path.is_file() and path.suffix != ".wav":
                    text = path.read_text(encoding="utf-8-sig")
                    self.assertNotIn("stage_club", text)
                    self.assertNotIn("legacy", text)
            with self.assertRaises(FileExistsError):
                build_blind(runs, out)

    def test_blind_rejects_audio_changed_after_matching(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = root / "runs"
            topic = fixture_topic(runs)
            match_topic(topic)
            path = topic / "stage_club/matched.wav"
            original = path.read_bytes()
            changed = bytearray(original)
            changed[-3] ^= 1
            path.write_bytes(changed)
            self.assertNotEqual(hashlib.sha256(original).hexdigest(), hashlib.sha256(changed).hexdigest())
            with self.assertRaisesRegex(ValueError, "changed"):
                build_blind(runs, root / "blind")
            self.assertFalse((root / "blind").exists())


if __name__ == "__main__":
    unittest.main()
