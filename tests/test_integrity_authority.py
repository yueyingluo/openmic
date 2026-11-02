"""Exact external review/trial scope, with generated PCM and mocked ASR only."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from openmic.tts_integrity_cli import IntegrityError, assemble, content_accepted, ensure_units, orthography_review
from openmic.tts_integrity_listening import load_authority
from tests.integrity_fixtures import (FixtureBackend, approved_entry, audio_fixture, authority_fixture,
    build_fixture_plan, digest, record_fixture, trial_fixture, write_json)


class IntegrityAuthorityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.plan = build_fixture_plan("<SCRIPT>[SETUP]今天吃饭。</SCRIPT>")
        text = self.plan["units"][0]["text"]
        self.transcript = text.replace("今", "金", 1)
        self.path, self.record = record_fixture(self.root, self.plan, audio=audio_fixture(), transcript=self.transcript)
        self.authority, self.document = authority_fixture(self.root, self.plan, self.path, self.record, kind="orthography")
        self.entry = approved_entry(self.root, self.plan, self.path, self.record, self.authority, "orthography")

    def reload(self, document):
        path = self.root / "authority.json"
        write_json(path, document)
        return load_authority(path, digest(path.read_bytes()), self.plan, self.root)

    def test_exact_two_recognizer_review_keeps_machine_failure(self):
        self.assertFalse(self.entry["content"]["passed"])
        self.assertFalse(content_accepted(self.entry))
        self.assertTrue(content_accepted(self.entry, authority=self.authority))
        proof = orthography_review(self.entry["text"], self.transcript, self.transcript,
            self.entry["audio_sha256"], authority=self.authority, entry=self.entry)
        self.assertEqual(proof, self.entry["orthography_review"])

    def test_one_off_review_is_not_a_global_same_character_waiver(self):
        entry = copy.deepcopy(self.entry)
        entry["text"] = "今晚吃饭。"
        self.assertFalse(content_accepted(entry, authority=self.authority))
        entry = copy.deepcopy(self.entry); entry["audio_sha256"] = "0" * 64
        self.assertFalse(content_accepted(entry, authority=self.authority))

    def test_changed_independent_asr_file_is_rejected(self):
        path = self.root / "secondary.json"
        record = json.loads(path.read_text(encoding="utf-8")); record["transcript"] = "Other words."
        write_json(path, record)
        with self.assertRaisesRegex(ValueError, "ASR evidence changed"):
            content_accepted(self.entry, authority=self.authority)

    def test_same_recognizer_cannot_claim_independent_agreement(self):
        path = self.root / "secondary.json"
        record = json.loads(path.read_text(encoding="utf-8")); record["model"] = self.record["asr"]["model"]
        write_json(path, record)
        self.document["reviews"][0]["secondary_record_sha256"] = digest(path.read_bytes())
        with self.assertRaisesRegex(ValueError, "two distinct"):
            self.reload(self.document)

    def test_other_wav_asr_cannot_be_bound_by_updating_only_manifest_hash(self):
        path = self.root / "secondary.json"
        record = json.loads(path.read_text(encoding="utf-8")); record["audio_sha256"] = "0" * 64
        write_json(path, record)
        self.document["reviews"][0]["secondary_record_sha256"] = digest(path.read_bytes())
        with self.assertRaisesRegex(ValueError, "Both recognizers"):
            self.reload(self.document)

    def test_review_cannot_waive_deletion_or_unlisted_substitution(self):
        other = self.root / "deletion"
        path, record = record_fixture(other, self.plan, audio=audio_fixture(), transcript="天吃饭。")
        with self.assertRaises(ValueError):
            authority_fixture(other, self.plan, path, record, kind="orthography")
        document = copy.deepcopy(self.document)
        document["reviews"][0]["substitution"]["reference_index"] = 1
        with self.assertRaises(ValueError):
            self.reload(document)

    def test_source_plan_and_candidate_digest_all_bind_authority(self):
        for field, value in (("source_script_sha256", "0" * 64), ("plan_sha256", "0" * 64)):
            document = copy.deepcopy(self.document); document[field] = value
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "plan and source"):
                    self.reload(document)
        document = copy.deepcopy(self.document)
        document["reviews"][0]["candidate_record_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "record digest"):
            self.reload(document)

    def test_review_path_cannot_escape_assets_root(self):
        document = copy.deepcopy(self.document); document["reviews"][0]["candidate_record"] = "../outside.json"
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.reload(document)

    def test_duplicate_json_keys_are_not_silently_accepted(self):
        path = self.root / "duplicate.json"; path.write_text('{"schema":"x","schema":"y"}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_authority(path, digest(path.read_bytes()), self.plan, self.root)

    def test_explicit_trial_changes_only_declared_input_and_consumes_once(self):
        root = self.root / "trial"; root.mkdir()
        trial = trial_fixture(self.plan)
        authority, _ = authority_fixture(root, self.plan, trials=[trial])
        backend = FixtureBackend()
        for _ in range(2):
            with self.assertRaises(IntegrityError):
                ensure_units(self.plan, backend, root / "run", root, authority=authority,
                             asr_function=lambda _: {"transcript": "完全不同。"})
            self.assertEqual(backend.calls, 5)
        record = json.loads((root / "run/candidates/000/06/record.json").read_text(encoding="utf-8"))
        self.assertEqual(record["params"]["text"], self.plan["units"][0]["text"] + "……")
        self.assertEqual(record["text"], self.plan["units"][0]["text"])
        self.assertEqual(record["trial_authority_sha256"], authority.sha256)
        self.assertFalse(record["content"]["passed"])
        self.assertTrue((root / "run/authorities" / (authority.sha256 + ".json")).is_file())

    def test_focus_trial_uses_explicit_offsets_not_hardcoded_syllables(self):
        root = self.root / "focus"; root.mkdir()
        trial = trial_fixture(self.plan, attempt=7, transform={"kind": "focus_spans", "spans": [[0, 2]]})
        authority, _ = authority_fixture(root, self.plan, trials=[trial])
        backend = FixtureBackend()
        with self.assertRaises(IntegrityError):
            ensure_units(self.plan, backend, root / "run", root, authority=authority,
                         asr_function=lambda _: {"transcript": "完全不同。"})
        self.assertEqual(backend.calls, 5)
        self.assertEqual(backend.requests[-1]["text"], "<strong>今天</strong>吃饭。")

    def test_successful_trial_reuses_saved_authority_without_new_requests(self):
        root = self.root / "trial_pass"; root.mkdir()
        authority, _ = authority_fixture(root, self.plan, trials=[trial_fixture(self.plan)])
        backend = FixtureBackend(); answers = iter(["完全不同。"] * 4 + [self.plan["units"][0]["text"]])
        accepted = ensure_units(self.plan, backend, root / "run", root, authority=authority,
                                asr_function=lambda _: {"transcript": next(answers)})
        self.assertEqual(backend.calls, 5)
        assemble(self.plan, accepted, root, root / "run")
        resumed = ensure_units(self.plan, backend, root / "run", root,
                               asr_function=lambda _: self.fail("Accepted result must not call ASR"))
        self.assertEqual(backend.calls, 5)
        self.assertEqual(resumed, accepted)
        snapshot = root / "run/authorities" / (authority.sha256 + ".json")
        snapshot.write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            assemble(self.plan, accepted, root, root / "run")

    def test_trial_budget_bool_or_overlapping_spans_are_rejected(self):
        for transform, budget in (({"kind": "focus_spans", "spans": [[0, 2], [1, 3]]}, 1),
                                   ({"kind": "terminal_pause", "suffix": "other words"}, 1),
                                   ({"kind": "terminal_pause", "suffix": "……"}, True)):
            root = self.root / str(len(list(self.root.iterdir()))); root.mkdir()
            trial = trial_fixture(self.plan, transform=transform); trial["maximum_new_tts_calls"] = budget
            with self.subTest(transform=transform, budget=budget):
                with self.assertRaises(ValueError):
                    authority_fixture(root, self.plan, trials=[trial])


if __name__ == "__main__": unittest.main()
