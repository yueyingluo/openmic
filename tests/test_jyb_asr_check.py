import hashlib
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from openmic.tts_asr_check import edit_distance, load_reference, normalize_text
from openmic.tts_plan import Speech, compile_script, plain_text


class ContentCheckTests(unittest.TestCase):
    def test_punctuation_and_spacing_do_not_create_errors(self):
        self.assertEqual(normalize_text("你好，TTS！ 16k"), normalize_text("你好tts16k"))

    def test_insertions_deletions_and_substitutions_are_counted(self):
        self.assertEqual(edit_distance("甲乙丙", "甲丁丙戊"), 2)
        self.assertEqual(edit_distance("甲乙丙", "甲丙"), 1)
        self.assertEqual(edit_distance("", "甲乙"), 2)
        self.assertEqual(edit_distance("甲乙", ""), 2)

    def fixture(self, directory, *, fillers=False):
        source = "".join("[SETUP]" + "这是比较长的铺垫内容，" * 4 for _ in range(3))
        plan = [asdict(event) for event in compile_script(source, add_fillers=fillers)]
        topic = Path(directory) / "topic"
        condition = topic / "performance"
        condition.mkdir(parents=True)
        (topic / "script.txt").write_text(source, encoding="utf-8")
        (topic / "plan.json").write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
        text_hash = hashlib.sha256("\n".join(plain_text(event["text"]) for event in plan if "text" in event).encode("utf-8")).hexdigest()
        plan_hash = hashlib.sha256(json.dumps(plan, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        return condition / "metrics.json", {"spoken_text_sha256": text_hash, "plan_sha256": plan_hash}, plan

    def test_reference_uses_rendered_filler_plan_instead_of_raw_script(self):
        with tempfile.TemporaryDirectory() as directory:
            metric, metadata, plan = self.fixture(directory, fillers=True)
            result = load_reference(metric, metadata)
            self.assertIn("那个，", result["expected_text"])
            self.assertTrue(result["reference_verified"])
            self.assertTrue(result["reference_source"].endswith("plan.json"))
            self.assertEqual(result["reference_sha256"], metadata["spoken_text_sha256"])

    def test_mismatched_plan_or_text_is_rejected_before_paid_asr(self):
        with tempfile.TemporaryDirectory() as directory:
            metric, metadata, plan = self.fixture(directory)
            for name in ("spoken_text_sha256", "plan_sha256"):
                with self.subTest(name=name):
                    broken = {**metadata, name: "0" * 64}
                    with self.assertRaises(ValueError):
                        load_reference(metric, broken)

    def test_legacy_timeline_reference_can_still_be_hash_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            metric, metadata, plan = self.fixture(directory)
            metric.parent.parent.joinpath("plan.json").unlink()
            rows = [{**event, "kind": "speech" if "text" in event else "pause"} for event in plan]
            metric.parent.joinpath("timeline.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            result = load_reference(metric, metadata)
            self.assertTrue(result["reference_verified"])
            self.assertTrue(result["reference_source"].endswith("timeline.json"))


if __name__ == "__main__":
    unittest.main()
