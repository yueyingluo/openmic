"""Acceptance checks use temporary fixtures, never a service or credentials."""

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from openmic.tts_acceptance import _design_status, inspect_runs, main
from openmic.tts_backends import ToneBackend, decode_wav, encode_wav
from openmic.tts_evaluate import pack
from openmic.tts_plan import compile_script
from openmic.tts_render import render, save_result


class FixtureBackend(ToneBackend):
    # This false flag exercises real-result validation, not a claim that tests
    # called a real provider. The acceptance report explicitly states that limit.
    synthetic = False
    identity = {"backend": "unit-test-fixture", "voice": "test-only"}


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.runs = self.root / "runs"
        self.plan = compile_script("[SETUP]这是铺垫。[PUNCHLINE][EMOTION=calm]这是[EMPHASIS]包袱[/EMPHASIS]。")
        topic = self.runs / "shopping"
        topic.mkdir(parents=True)
        (topic / "plan.json").write_text(json.dumps([asdict(event) for event in self.plan], ensure_ascii=False), encoding="utf-8")
        for condition in ("baseline", "timing", "emotion", "performance"):
            result = render(self.plan, FixtureBackend(), condition=condition, cache_dir=self.root / "cache")
            save_result(result, topic / condition)

    def modify_json(self, path, change):
        data = json.loads(path.read_text(encoding="utf-8"))
        change(data)
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def test_short_demo_is_valid_without_required_duration(self):
        report = inspect_runs(self.runs)
        self.assertTrue(report["hard_checks_passed"])
        self.assertEqual(report["sample_count"], 4)
        self.assertEqual(report["topic_count"], 1)
        self.assertFalse(report["five_topics_present"])
        self.assertFalse(report["performance_duration_target_passed"])
        self.assertTrue(report["complete_four_condition_design"])
        self.assertTrue(report["all_topics_have_performance"])
        self.assertFalse(report["five_topic_deliverable_passed"])
        self.assertEqual(report["matched_topics"][0]["baseline_timing_speech_pcm"], "identical")
        self.assertTrue(report["human_review_pending"])
        self.assertFalse(inspect_runs(self.runs, require_duration=True)["hard_checks_passed"])

    def test_missing_performance_is_reported_despite_existing_files_passing(self):
        (self.runs / "shopping/performance/metrics.json").unlink()
        report = inspect_runs(self.runs)
        self.assertTrue(report["hard_checks_passed"])
        self.assertEqual(report["sample_count"], 3)
        self.assertFalse(report["complete_four_condition_design"])
        self.assertEqual(report["missing_conditions_by_topic"], {"shopping": ["performance"]})
        self.assertFalse(report["all_topics_have_performance"])
        self.assertFalse(report["performance_duration_target_passed"])
        self.assertFalse(report["five_topic_deliverable_passed"])

    def test_nineteen_valid_files_do_not_make_a_complete_five_topic_delivery(self):
        # These are already-validated sample report fixtures. Duration is true
        # for every present sample, reproducing the partially finished real run.
        rows = [{"topic": topic, "condition": condition, "duration_in_target": True}
                for topic in ("shopping", "canteen", "deadline", "fitness", "groupwork")
                for condition in ("baseline", "timing", "emotion", "performance")]
        complete = _design_status(rows, hard_checks_passed=True)
        self.assertTrue(complete["five_topic_deliverable_passed"])
        self.assertTrue(complete["complete_four_condition_design"])
        rows = [row for row in rows if (row["topic"], row["condition"]) != ("shopping", "performance")]
        partial = _design_status(rows, hard_checks_passed=True)
        self.assertFalse(partial["performance_duration_target_passed"])
        self.assertFalse(partial["five_topic_deliverable_passed"])
        self.assertFalse(partial["complete_four_condition_design"])
        self.assertEqual(partial["missing_conditions_by_topic"], {"shopping": ["performance"]})
        # Emotion is an additional ablation: a 15-file mandatory design can be
        # complete for delivery while explicitly not being a four-condition set.
        mandatory = [row for row in rows if row["condition"] != "emotion"]
        mandatory.append({"topic": "shopping", "condition": "performance", "duration_in_target": True})
        status = _design_status(mandatory, hard_checks_passed=True)
        self.assertTrue(status["five_topic_deliverable_passed"])
        self.assertFalse(status["complete_four_condition_design"])
        self.assertFalse(_design_status(mandatory, hard_checks_passed=False)["five_topic_deliverable_passed"])

    def test_pause_nonzero_or_wrong_length_fails(self):
        path = self.runs / "shopping/timing"
        timeline = json.loads((path / "timeline.json").read_text(encoding="utf-8"))
        pause = next(row for row in timeline if row["kind"] == "pause")
        samples = decode_wav((path / "audio.wav").read_bytes())
        samples[pause["start_frame"] + 10] = 100
        (path / "audio.wav").write_bytes(encode_wav(samples))
        report = inspect_runs(self.runs)
        row = next(row for row in report["samples"] if row["condition"] == "timing")
        self.assertFalse(row["checks"]["exact_zero_sample_pauses"])
        self.assertFalse(report["hard_checks_passed"])

    def test_changed_pcm_breaks_timing_only_comparison(self):
        path = self.runs / "shopping/timing/audio.wav"
        samples = decode_wav(path.read_bytes())
        samples[100] += 1
        path.write_bytes(encode_wav(samples))
        report = inspect_runs(self.runs)
        self.assertEqual(report["matched_topics"][0]["baseline_timing_speech_pcm"], "different")
        self.assertFalse(report["hard_checks_passed"])

    def test_plan_hash_backend_and_metadata_conflicts_fail(self):
        path = self.runs / "shopping/performance/metrics.json"
        self.modify_json(path, lambda row: row.update(spoken_text_sha256="0" * 64, backend={"voice": "changed"}, duration_seconds=99))
        report = inspect_runs(self.runs)
        self.assertFalse(report["hard_checks_passed"])
        self.assertTrue(report["matched_topics"][0]["errors"])
        row = next(row for row in report["samples"] if row["condition"] == "performance")
        self.assertFalse(row["checks"]["saved_plan_text_hash"])
        self.assertFalse(row["checks"]["duration_metadata"])

    def test_truncated_wav_and_discontinuous_timeline_fail(self):
        baseline = self.runs / "shopping/baseline/audio.wav"
        baseline.write_bytes(baseline.read_bytes()[:-10])
        timeline = self.runs / "shopping/performance/timeline.json"
        self.modify_json(timeline, lambda rows: rows[1].update(start_frame=rows[1]["start_frame"] + 1))
        self.assertFalse(inspect_runs(self.runs)["hard_checks_passed"])

    def test_payload_and_synthetic_failures_are_visible(self):
        timeline = self.runs / "shopping/performance/timeline.json"
        self.modify_json(timeline, lambda rows: rows[0].update(actual_instruction="wrong"))
        metrics = self.runs / "shopping/emotion/metrics.json"
        self.modify_json(metrics, lambda row: row.update(synthetic=True))
        report = inspect_runs(self.runs)
        performance = next(row for row in report["samples"] if row["condition"] == "performance")
        emotion = next(row for row in report["samples"] if row["condition"] == "emotion")
        self.assertFalse(performance["checks"]["declared_control_payloads"])
        self.assertFalse(emotion["checks"]["declared_real_audio"])

    def test_cold_benchmark_and_blank_blind_forms_are_reported(self):
        benchmark = self.root / "benchmarks/shopping/performance"
        result = render(self.plan, FixtureBackend(), condition="performance")
        save_result(result, benchmark)
        blind = self.root / "blind"
        pack(self.runs, blind, listeners=2)
        report = inspect_runs(self.runs, benchmarks=self.root / "benchmarks", blind=blind)
        self.assertTrue(report["hard_checks_passed"])
        self.assertTrue(report["benchmarks"]["samples"][0]["valid_cold_benchmark"])
        self.assertTrue(report["blind_package"]["passed"])
        self.assertTrue(report["blind_package"]["rating_templates_blank"])
        self.assertEqual(report["blind_package"]["filled_rating_rows"], 0)
        self.assertTrue(report["human_review_pending"])
        self.modify_json(benchmark / "metrics.json", lambda row: row.update(cache_hits=1))
        report = inspect_runs(self.runs, benchmarks=self.root / "benchmarks")
        self.assertFalse(report["benchmarks"]["samples"][0]["valid_cold_benchmark"])

    def test_cli_writes_reports_and_empty_runs_fail(self):
        out = self.root / "reports"
        self.assertEqual(main(["--runs", str(self.runs), "--out", str(out)]), 0)
        for name in ("acceptance.json", "samples.csv", "README.md"):
            self.assertTrue((out / name).is_file())
        self.assertEqual(main(["--runs", str(self.root / "missing"), "--out", str(self.root / "empty.json")]), 1)
        self.assertTrue((self.root / "empty.json").is_file())


if __name__ == "__main__":
    unittest.main()
