"""Portable synthetic review-authority regressions; no actual user claims."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from openmic.tts_backends import decode_wav
from openmic.tts_integrity_audio import render_integrity
from openmic.tts_integrity_cli import IntegrityError, content_accepted, ensure_units, assemble
from openmic.tts_integrity_listening import human_accepted, load_authority, render_reviewed
from tests.integrity_fixtures import (FixtureBackend, approved_entry, audio_fixture, authority_fixture,
    build_fixture_plan, digest, record_fixture, write_json)

class IntegrityListeningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan = build_fixture_plan()
        self.path, self.record = record_fixture(self.root, self.plan)
        self.authority, self.document = authority_fixture(self.root, self.plan, self.path, self.record)
        self.entry = approved_entry(self.root, self.plan, self.path, self.record, self.authority)
        self.audio = {0: (self.path.parent / "audio.wav").read_bytes()}

    def test_entry_cannot_authorize_itself_without_external_authority(self):
        self.assertFalse(human_accepted(self.entry))
        self.assertFalse(content_accepted(self.entry))
        with self.assertRaises(ValueError):
            render_reviewed(self.plan, self.audio, human_reviews={0: self.entry})

    def test_explicit_review_retains_false_edge_and_absent_content(self):
        self.assertTrue(human_accepted(self.entry, authority=self.authority))
        self.assertFalse(self.entry["edge"]["passed"])
        self.assertIsNone(self.entry["content"])

    def test_manifest_digest_is_independently_required(self):
        with self.assertRaisesRegex(ValueError, "SHA256"):
            load_authority(self.root / "authority.json", "0" * 64, self.plan, self.root)

    def test_changed_manifest_after_approval_is_rejected(self):
        self.document["reviews"][0]["review_text"] += " Changed."
        write_json(self.root / "authority.json", self.document)
        with self.assertRaisesRegex(ValueError, "changed"):
            human_accepted(self.entry, authority=self.authority)

    def test_changed_sentence_id_wav_or_self_review_is_rejected(self):
        for key, value in (("unit_id", True), ("text", "Different words."), ("audio_sha256", "0" * 64)):
            entry = copy.deepcopy(self.entry); entry[key] = value
            self.assertFalse(human_accepted(entry, authority=self.authority))
        entry = copy.deepcopy(self.entry); entry["human_review"]["review_text"] += " Self-authorized."
        self.assertFalse(human_accepted(entry, authority=self.authority))

    def test_same_words_new_plan_cannot_reuse_review(self):
        plan = copy.deepcopy(self.plan); plan["source_tail_pause_seconds"] += .5
        with self.assertRaisesRegex(ValueError, "another plan"):
            render_reviewed(plan, self.audio, human_reviews={0: self.entry}, authority=self.authority)

    def test_changed_candidate_record_is_rejected(self):
        record = copy.deepcopy(self.record); record["status"] = "pretend-pass"
        write_json(self.path, record)
        with self.assertRaisesRegex(ValueError, "record changed"):
            human_accepted(self.entry, authority=self.authority)

    def test_changed_reviewed_wav_is_rejected(self):
        (self.path.parent / "audio.wav").write_bytes(audio_fixture())
        with self.assertRaisesRegex(ValueError, "WAV changed"):
            human_accepted(self.entry, authority=self.authority)

    def test_tail_risk_cannot_be_authorized(self):
        other = self.root / "other"
        path, record = record_fixture(other, self.plan, audio=audio_fixture(hot_head=True, hot_tail=True))
        with self.assertRaisesRegex(ValueError, "head-only"):
            authority_fixture(other, self.plan, path, record)

    def test_no_review_delegates_exactly_to_frozen_renderer(self):
        audio = {0: audio_fixture()}
        expected, actual = render_integrity(self.plan, audio, "safe_club"), render_reviewed(self.plan, audio, "safe_club", {})
        self.assertEqual((actual.audio, actual.metrics, actual.timeline), (expected.audio, expected.metrics, expected.timeline))

    def test_reviewed_render_preserves_every_raw_sample_and_machine_failure(self):
        result = render_reviewed(self.plan, self.audio, "safe_club", {0: self.entry}, authority=self.authority)
        signal = decode_wav(result.audio); row = next(item for item in result.timeline if item["kind"] == "speech")
        self.assertTrue(np.array_equal(signal[row["start_frame"]:row["end_frame"]], decode_wav(self.audio[0])))
        self.assertFalse(row["edge_check"]["passed"]); self.assertIsNone(row["machine_content_check"])
        self.assertTrue(row["human_review_passed"])
        self.assertEqual(result.metrics["user_reviewed_unit_ids"], [0]); self.assertFalse(result.metrics["all_machine_edges_passed"])

    def test_profiles_share_raw_speech_and_keep_original_reports(self):
        results = [render_reviewed(self.plan, self.audio, profile, {0: self.entry}, authority=self.authority) for profile in ("safe_course", "safe_club")]
        self.assertEqual(results[0].metrics["speech_pcm_sha256"], results[1].metrics["speech_pcm_sha256"])
        self.assertEqual(self.record, json.loads(self.path.read_text(encoding="utf-8")))

    def test_reviewed_resume_never_sends_new_tts_or_edits_candidate(self):
        backend = FixtureBackend()
        accepted = ensure_units(self.plan, backend, self.root / "run", self.root, authority=self.authority, asr_function=lambda _: self.fail("No new ASR"))
        self.assertEqual(backend.calls, 0); self.assertEqual(self.record, json.loads(self.path.read_text(encoding="utf-8")))
        assemble(self.plan, accepted, self.root, self.root / "run", authority=self.authority)
        with self.assertRaises(IntegrityError):
            assemble(self.plan, accepted, self.root, self.root / "run")

    def test_false_or_unknown_review_entries_cannot_count_as_authorization(self):
        for reviews in ({0: None}, {True: self.entry}, {1: self.entry}):
            with self.subTest(keys=list(reviews)):
                with self.assertRaises(ValueError):
                    render_reviewed(self.plan, self.audio, "safe_club", reviews, authority=self.authority)

    def test_explicit_new_authority_rebind_is_persisted_without_changing_pcm(self):
        out = self.root / "run"; backend = FixtureBackend()
        first = ensure_units(self.plan, backend, out, self.root, authority=self.authority)
        assemble(self.plan, first, self.root, out, authority=self.authority)
        original_pcm_file = (out / "safe_club/audio.wav").read_bytes()
        document = copy.deepcopy(self.document)
        document["reviews"][0]["review_text"] += " Explicit new approved review revision."
        authority_path = self.root / "authority.json"; write_json(authority_path, document)
        authority = load_authority(authority_path, digest(authority_path.read_bytes()), self.plan, self.root)
        rebound = ensure_units(self.plan, backend, out, self.root, authority=authority)
        saved = json.loads((out / "accepted.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, rebound)
        self.assertEqual(saved[0]["human_review"]["authority_sha256"], authority.sha256)
        self.assertNotEqual(authority.sha256, self.authority.sha256)
        assemble(self.plan, rebound, self.root, out, authority=authority)
        timeline = json.loads((out / "safe_club/timeline.json").read_text(encoding="utf-8"))
        reviewed = next(row for row in timeline if row["kind"] == "speech")
        self.assertEqual(reviewed["human_review"], saved[0]["human_review"])
        self.assertEqual((out / "safe_club/audio.wav").read_bytes(), original_pcm_file)
        self.assertEqual(self.record, json.loads(self.path.read_text(encoding="utf-8")))
        self.assertEqual(backend.calls, 0)

if __name__ == "__main__": unittest.main()
