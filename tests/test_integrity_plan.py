import copy
import hashlib
import json
import unittest

from openmic.tts_integrity_plan import (
    build_integrity_plan, sentence_slices, validate_execution_integrity_plan,
    validate_integrity_plan,
)
from openmic.tts_stage_plan import build_plan


SCRIPT = "<SCRIPT>\n[PAUSE=0.8]\n[SETUP]第一句话。好。第三句呢？\n[PAUSE=1.2]\n[PUNCHLINE]答案是[EMPHASIS]明天[/EMPHASIS]。你信吗？！\n[PAUSE=2.0]\n</SCRIPT>"


class IntegrityPlanTests(unittest.TestCase):
    def setUp(self):
        self.base = build_plan(SCRIPT)
        self.plan = build_integrity_plan(SCRIPT, self.base)

    def test_every_short_sentence_is_independent_and_body_is_exact(self):
        self.assertEqual([u["text"] for u in self.plan["units"]], ["第一句话。", "好。", "第三句呢？", "答案是明天。", "你信吗？！"])
        self.assertEqual("".join(u["text"] for u in self.plan["units"]), "".join(s["text"] for s in self.base["source_speeches"]))
        self.assertTrue(validate_integrity_plan(self.plan, SCRIPT, self.base)["original_script_verified"])

    def test_offsets_are_original_speech_offsets(self):
        for unit in self.plan["units"]:
            source = self.base["source_speeches"][unit["source_index"]]["text"]
            self.assertEqual(source[unit["start"]:unit["end"]], unit["text"])
        self.assertEqual([u["id"] for u in self.plan["units"]], list(range(5)))

    def test_paired_nested_quotations_and_book_titles_stay_whole(self):
        text = '他说：“真的？每天都去？”然后答：“当然！”。《谁？我！》也行。结束。'
        parts = [text[start:end] for start, end in sentence_slices(text)]
        self.assertEqual(parts, ['他说：“真的？每天都去？”', '然后答：“当然！”。', '《谁？我！》也行。', '结束。'])

    def test_ascii_questions_decimal_and_abbreviation(self):
        text = "价格3.14元。Mr. Wang问OK?!然后走了。"
        self.assertEqual([text[a:b] for a,b in sentence_slices(text)], ["价格3.14元。", "Mr. Wang问OK?!", "然后走了。"])

    def test_whitespace_is_preserved_and_attached_to_prior_sentence(self):
        text = "好。  再说？\n继续。"
        parts = [text[a:b] for a,b in sentence_slices(text)]
        self.assertEqual(parts, ["好。  ", "再说？\n", "继续。"])
        self.assertEqual("".join(parts), text)

    def test_focus_inherits_only_into_its_sentence(self):
        focused = [u for u in self.plan["units"] if u["focus"] is not None]
        self.assertEqual([(u["text"],u["focus"]) for u in focused], [("答案是明天。", "明天")])

    def test_original_pauses_apply_only_to_first_sentence_and_tail(self):
        self.assertEqual([u["source_pause_before_seconds"] for u in self.plan["units"]], [.8,0,0,1.2,0])
        self.assertEqual(self.plan["source_tail_pause_seconds"], 2)
        self.assertEqual([u["boundary_before"] for u in self.plan["units"]], ["flow","sentence","sentence","flow","sentence"])

    def test_continuous_source_pauses_use_max_not_sum(self):
        script = "[PAUSE=0.2][PAUSE=0.8][SETUP]一句。短句。[PAUSE=1.2][PAUSE=0.4]"
        plan = build_integrity_plan(script, build_plan(script))
        self.assertEqual([u["source_pause_before_seconds"] for u in plan["units"]], [.8,0])
        self.assertEqual(plan["source_tail_pause_seconds"], 1.2)

    def test_explicit_zero_stays_zero_in_plan(self):
        script = "[PAUSE=0][SETUP]一句。再说。[PAUSE=0]"
        plan = build_integrity_plan(script, build_plan(script))
        self.assertEqual(plan["units"][0]["source_pause_before_seconds"], 0.0)
        self.assertEqual(plan["source_tail_pause_seconds"], 0.0)

    def test_existing_two_base_units_keep_their_boundaries(self):
        first, second = "甲"*14+"。", "乙"*14+"。"
        script = "[PAUSE=0.8][SETUP]"+first+second
        base = build_plan(script, {"segments":[{"source_index":0,"parts":[
            {"text":first,"intent":"setup","focus":None,"boundary_before":"breath"},
            {"text":second,"intent":"contrast","focus":None,"boundary_before":"turn"}]}]})
        plan = build_integrity_plan(script, base)
        self.assertEqual([u["boundary_before"] for u in plan["units"]], ["breath","turn"])
        self.assertEqual([u["source_pause_before_seconds"] for u in plan["units"]], [.8,0])

    def test_unbalanced_quote_refuses_ambiguous_splitting(self):
        script = "[SETUP]他说：“还没结束。然后走了。"
        with self.assertRaisesRegex(ValueError, "quotation"):
            build_integrity_plan(script, build_plan(script))

    def test_execution_validator_does_not_claim_original_script_verified(self):
        report = validate_execution_integrity_plan(self.plan)
        self.assertTrue(report["passed"])
        self.assertFalse(report["original_script_verified"])

    def test_mutated_body_offsets_annotations_and_hash_are_rejected(self):
        for mutate in (
            lambda p:p["units"][0].update(text="别的正文。"),
            lambda p:p["units"][1].update(start=0),
            lambda p:p["units"][0].update(intent="reveal"),
            lambda p:p.update(base_stage_plan_sha256="0"*64),
            lambda p:p["units"][0].update(source_pause_before_seconds=.1),
        ):
            with self.subTest(mutate=mutate):
                changed = copy.deepcopy(self.plan);mutate(changed)
                self.assertFalse(validate_integrity_plan(changed, SCRIPT, self.base)["passed"])

    def test_bool_numbers_extra_fields_duplicate_missing_and_nonfinite_rejected(self):
        for mutate in (
            lambda p:p["units"][0].update(id=True),
            lambda p:p["units"][0].update(extra=1),
            lambda p:p["units"][1].update(id=0),
            lambda p:p["units"].pop(),
            lambda p:p.update(source_tail_pause_seconds=float("nan")),
            lambda p:p["units"][1].update(source_pause_before_seconds=.1),
        ):
            with self.subTest(mutate=mutate):
                changed = copy.deepcopy(self.plan);mutate(changed)
                self.assertFalse(validate_execution_integrity_plan(changed)["passed"])

    def test_base_hash_matches_documented_canonical_json(self):
        expected = hashlib.sha256(json.dumps(self.base,ensure_ascii=False,sort_keys=True,allow_nan=False).encode()).hexdigest()
        self.assertEqual(self.plan["base_stage_plan_sha256"], expected)


if __name__ == "__main__":
    unittest.main()
