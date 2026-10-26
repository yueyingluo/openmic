"""Small fixtures exercise anonymization, missing data and listener-level pairs."""

import csv
import hashlib
import json
import shutil
import tempfile
import unittest
import wave
from pathlib import Path

from openmic.tts_evaluate import DIMENSIONS, RATING_COLUMNS, pack, summarize


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.runs = self.root / "runs"
        for topic in ("shopping", "campus"):
            for condition in ("baseline", "performance"):
                self.make_run(topic, condition)

    def make_run(self, topic, condition, synthetic=False):
        directory = self.runs / topic / condition
        directory.mkdir(parents=True, exist_ok=True)
        with wave.open(str(directory / "audio.wav"), "wb") as handle:
            handle.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            handle.writeframes(b"\x01\x00" * 160)
        (directory / "metrics.json").write_text(json.dumps({
            "condition": condition, "backend": {"backend": "fixture"},
            "synthetic": synthetic, "in_target_duration": False}), encoding="utf-8")

    def write_ratings(self, path, rows):
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RATING_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    def update_metadata(self, topic, condition, **changes):
        path = self.runs / topic / condition / "metrics.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata.update(changes)
        path.write_text(json.dumps(metadata), encoding="utf-8")

    def package(self):
        target = self.root / "blind"
        manifest = pack(self.runs, target, listeners=2)
        with (target / "investigator/key.csv").open(encoding="utf-8-sig", newline="") as handle:
            key = list(csv.DictReader(handle))
        return target, key, manifest

    def test_package_anonymizes_and_separates_answer_key(self):
        target, key, manifest = self.package()
        public = target / "participant_package"
        self.assertEqual(manifest["samples"], 4)
        self.assertEqual(manifest["not_in_target_duration_count"], 4)
        self.assertEqual(manifest["missing_text_hash_count"], 4)
        self.assertFalse(manifest["topic_match_validation"]["shopping"]["same_text_verified"])
        self.assertFalse((public / "key.csv").exists())
        self.assertEqual(len(list((public / "audio").glob("sample*.wav"))), 4)
        public_names = " ".join(str(path.relative_to(public)) for path in public.rglob("*"))
        self.assertNotIn("baseline", public_names)
        self.assertNotIn("performance", public_names)
        orders = []
        for participant in ("p001", "p002"):
            with (public / (participant + "_playlist.csv")).open(encoding="utf-8-sig", newline="") as handle:
                orders.append([row["sample_id"] for row in csv.DictReader(handle)])
        self.assertEqual(orders[0], list(reversed(orders[1])))
        self.assertEqual({row["topic"] for row in key}, {"shopping", "campus"})

    def test_rejects_synthetic_unless_explicit_practice(self):
        self.make_run("shopping", "baseline", synthetic=True)
        target = self.root / "practice"
        with self.assertRaisesRegex(ValueError, "synthetic"):
            pack(self.runs, target)
        self.assertFalse(target.exists())
        result = pack(self.runs, target, allow_synthetic=True)
        self.assertTrue(result["synthetic_practice_only"])
        self.assertTrue(json.loads((target / "participant_package/manifest.json").read_text(
            encoding="utf-8"))["synthetic_practice_only"])

    def test_summary_uses_available_dimensions_and_listener_level_pairs(self):
        target, key, _ = self.package()
        rows = []
        for participant in ("p001", "p002"):
            for sample in key:
                score = 2 if sample["condition"] == "baseline" else (4 if participant == "p001" else 3)
                rows.append({"participant_id": participant, "sample_id": sample["sample_id"],
                             "topic": "untrusted topic is ignored", "humor": score,
                             "naturalness": score if participant == "p001" else ""})
        self.write_ratings(target / "participant_package/filled.csv", rows)
        result = summarize(target / "participant_package", target / "investigator/key.csv",
                           self.root / "summary")
        self.assertEqual(result["ratings_count"], 8)
        naturalness = result["per_condition"]["performance"]["dimensions"]["naturalness"]
        self.assertEqual(naturalness["n"], 2)
        self.assertEqual(naturalness["listeners"], 1)
        paired = result["paired_performance_minus_baseline"]["humor"]
        self.assertEqual(paired["n_listeners"], 2)
        self.assertEqual(paired["n_listener_topic_pairs"], 4)
        self.assertEqual(paired["mean"], 1.5)
        self.assertEqual(paired["sample_variance"], 0.5)
        self.assertIsNone(result["per_condition"]["baseline"]["dimensions"]["emotion"]["mean"])

    def test_rejects_bad_scores_unknown_samples_and_duplicate_rows(self):
        target, key, _ = self.package()
        path = self.root / "ratings.csv"
        row = {"participant_id": "p001", "sample_id": key[0]["sample_id"], "humor": 4}
        cases = [([{**row, "humor": "4.5"}], "整数"),
                 ([{**row, "sample_id": "missing"}], "未知"),
                 ([{**row, "participant_id": "张三"}], "匿名"),
                 ([row, row], "重复")]
        for rows, message in cases:
            with self.subTest(message=message):
                self.write_ratings(path, rows)
                with self.assertRaisesRegex(ValueError, message):
                    summarize(path, target / "investigator/key.csv", self.root / "invalid")

    def test_blank_templates_are_not_invented_results(self):
        target, _, _ = self.package()
        with self.assertRaisesRegex(ValueError, "没有已填写"):
            summarize(target / "participant_package", target / "investigator/key.csv",
                      self.root / "empty")

    def test_matching_text_hashes_are_verified_per_topic(self):
        for topic in ("shopping", "campus"):
            text_hash = hashlib.sha256(topic.encode("utf-8")).hexdigest()
            for condition in ("baseline", "performance"):
                self.update_metadata(topic, condition, spoken_text_sha256=text_hash,
                                     plan_sha256="fixture-plan")
        _, key, manifest = self.package()
        self.assertEqual(manifest["missing_text_hash_count"], 0)
        self.assertTrue(all(row["same_text_verified"] for row in
                            manifest["topic_match_validation"].values()))
        self.assertTrue(all(row["spoken_text_sha256"] for row in key))

    def test_mismatched_text_or_backend_rejected_before_writing(self):
        self.update_metadata("shopping", "baseline", spoken_text_sha256="a" * 64)
        self.update_metadata("shopping", "performance", spoken_text_sha256="b" * 64)
        target = self.root / "mismatched"
        with self.assertRaisesRegex(ValueError, "同稿对照"):
            pack(self.runs, target)
        self.assertFalse(target.exists())
        self.update_metadata("shopping", "performance", spoken_text_sha256="a" * 64,
                             backend={"backend": "fixture", "voice": "different-voice"})
        with self.assertRaisesRegex(ValueError, "backend identity"):
            pack(self.runs, target)
        self.assertFalse(target.exists())

    def test_duplicate_topic_condition_rejected(self):
        repeated = self.runs / "shopping" / "repeat" / "performance"
        shutil.copytree(self.runs / "shopping" / "performance", repeated)
        target = self.root / "duplicates"
        with self.assertRaisesRegex(ValueError, "重复 topic/condition"):
            pack(self.runs, target)
        self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
