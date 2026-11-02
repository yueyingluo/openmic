"""No-network checks of immutable prose and bounded stage annotations."""

import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openmic.tts_stage_plan import (
    StagePlanError, build_plan, request_plan, validate_execution_plan, validate_plan,
)


FIRST = "我原来每天都说自己很快就会开始认真写作业。"
SECOND = "后来我发现最先写好的其实是那张特别漂亮的计划表。"
ENDING = "最后完成的是这张表格而不是我的作业。"
SCRIPT = "<SCRIPT>\n[SETUP]" + FIRST + SECOND + "\n[PUNCHLINE]" + ENDING + "\n</SCRIPT>"


def part(text, intent="setup", focus=None, boundary="flow"):
    return {"text": text, "intent": intent, "focus": focus, "boundary_before": boundary}


def annotations(plan):
    return {"segments": [{"source_index": speech["index"], "parts": [part(speech["text"])]}
                          for speech in plan["source_speeches"]]}


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=response))])


class StagePlanTests(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(api_key="LOCAL_PRIVATE_TEST_VALUE", model="fixture-planner",
                                      base_url="https://example.invalid/v1", timeout_seconds=10)
        for target in ("requests.sessions.Session.request", "httpx.Client.send", "httpx.AsyncClient.send"):
            guard = patch(target, side_effect=AssertionError("No network allowed in stage plan tests"))
            guard.start()
            self.addCleanup(guard.stop)

    def test_rules_preserve_source_hashes_events_and_all_words(self):
        plan = build_plan(SCRIPT)
        self.assertTrue(validate_plan(plan, SCRIPT)["passed"])
        self.assertTrue(validate_plan(plan, SCRIPT)["original_script_verified"])
        self.assertEqual(plan["source_script_sha256"], hashlib.sha256(SCRIPT.encode("utf-8")).hexdigest())
        plain = FIRST + SECOND + ENDING
        self.assertEqual(plan["spoken_text_sha256"], hashlib.sha256(plain.encode("utf-8")).hexdigest())
        self.assertEqual("".join(unit["text"] for unit in plan["units"]), plain)
        self.assertEqual(len(plan["units"]), 2)
        self.assertEqual([unit["role"] for unit in plan["units"]], ["setup", "punchline"])
        self.assertEqual(plan["units"][-1]["intent"], "reveal")
        self.assertEqual(plan["planner"]["type"], "rules")
        self.assertTrue(any(event["kind"] == "pause" and event["seconds"] == 0.8 for event in plan["source_events"]))

    def test_rule_focus_prefers_existing_markup_and_does_not_invent_one(self):
        marked = SCRIPT.replace("计划表", "[EMPHASIS]计划表[/EMPHASIS]")
        plan = build_plan(marked)
        self.assertEqual(plan["units"][0]["focus"], "计划表")
        self.assertIsNone(plan["units"][1]["focus"])
        self.assertNotIn("<strong>", plan["units"][0]["text"])
        self.assertIn("<strong>", plan["source_speeches"][0]["marked_text"])

    def test_rule_intents_distinguish_audience_quote_and_self_questions(self):
        script = "<SCRIPT>\n[SETUP]你们有没有经历过这样的事情？\n[SETUP]朋友问我：“你们到底准备什么时候开始写作业？”\n[SETUP]我这样做到底能准吗？\n[CALLBACK]后来先完成的还是那张计划表。\n</SCRIPT>"
        plan = build_plan(script)
        self.assertEqual([unit["intent"] for unit in plan["units"]],
                         ["audience_question", "quoted_question", "self_question", "button"])

    def test_valid_sentence_split_uses_exact_source_slices(self):
        original = build_plan(SCRIPT)
        supplied = annotations(original)
        supplied["segments"][0]["parts"] = [part(FIRST), part(SECOND, "contrast", "计划表", "turn")]
        before = copy.deepcopy(supplied)
        plan = build_plan(SCRIPT, supplied)
        self.assertEqual(supplied, before)
        self.assertEqual(len(plan["units"]), 3)
        self.assertEqual([(unit["start"], unit["end"]) for unit in plan["units"][:2]],
                         [(0, len(FIRST)), (len(FIRST), len(FIRST + SECOND))])
        self.assertTrue(validate_plan(plan, SCRIPT)["passed"])

    def test_words_and_punctuation_cannot_be_rewritten_or_omitted(self):
        base = annotations(build_plan(SCRIPT))
        replacements = [FIRST.replace("作业", "论文") + SECOND,
                        FIRST.replace("。", "，") + SECOND,
                        FIRST + SECOND[:-1], FIRST + SECOND + "你们说是不是？"]
        for replacement in replacements:
            with self.subTest(replacement=replacement[-12:]):
                supplied = copy.deepcopy(base)
                supplied["segments"][0]["parts"][0]["text"] = replacement
                with self.assertRaises(StagePlanError):
                    build_plan(SCRIPT, supplied)

    def test_split_inside_paired_quotes_is_rejected(self):
        first = "我后来跟室友认真地说：“这份计划明天一定会全部执行。"
        second = "随后我又给自己安排了一个更加详细的休息时间。”"
        script = "<SCRIPT>\n[SETUP]" + first + second + "\n[PUNCHLINE]" + ENDING + "\n</SCRIPT>"
        supplied = annotations(build_plan(script))
        supplied["segments"][0]["parts"] = [part(first), part(second)]
        with self.assertRaisesRegex(StagePlanError, "paired quotes"):
            build_plan(script, supplied)

    def test_split_after_sentence_final_closing_quote_is_allowed(self):
        first = "我后来跟室友认真地说：“这份计划明天一定会全部执行。”"
        second = "回头发现今天最先完成的还是那张特别漂亮的计划表。"
        script = "<SCRIPT>\n[SETUP]" + first + second + "\n[PUNCHLINE]" + ENDING + "\n</SCRIPT>"
        supplied = annotations(build_plan(script))
        supplied["segments"][0]["parts"] = [part(first), part(second, "contrast")]
        self.assertTrue(validate_plan(build_plan(script, supplied), script)["passed"])

    def test_comma_or_word_boundary_is_not_a_permitted_split(self):
        first = "我原来每天都说自己很快就会开始认真写作业，"
        script = SCRIPT.replace(FIRST, first)
        supplied = annotations(build_plan(script))
        supplied["segments"][0]["parts"] = [part(first), part(SECOND)]
        with self.assertRaisesRegex(StagePlanError, "sentence punctuation"):
            build_plan(script, supplied)

    def test_short_reveal_cannot_be_split_to_create_tiny_tts_calls(self):
        text = "我以为完成了。原来没有。"
        script = "<SCRIPT>\n[PUNCHLINE]" + text + "\n</SCRIPT>"
        supplied = {"segments": [{"source_index": 0, "parts": [part("我以为完成了。", "reveal"),
                                                                 part("原来没有。", "reveal")]}]}
        with self.assertRaisesRegex(StagePlanError, "14 Han"):
            build_plan(script, supplied)

    def test_global_split_budget_prevents_splitting_every_source_speech(self):
        script = "<SCRIPT>\n[SETUP]" + FIRST + SECOND + "\n[SETUP]" + FIRST + SECOND + "\n</SCRIPT>"
        supplied = annotations(build_plan(script))
        for segment in supplied["segments"]:
            segment["parts"] = [part(FIRST), part(SECOND)]
        with self.assertRaisesRegex(StagePlanError, "split budget"):
            build_plan(script, supplied)

    def test_focus_must_be_unique_short_and_free_of_punctuation_or_controls(self):
        base = annotations(build_plan(SCRIPT))
        for focus in ("不存在的短语", "作业。", "<strong>", "计划 表", FIRST, "[laughter]"):
            with self.subTest(focus=focus):
                supplied = copy.deepcopy(base)
                supplied["segments"][0]["parts"][0]["focus"] = focus
                with self.assertRaises(StagePlanError):
                    build_plan(SCRIPT, supplied)
        script = "<SCRIPT>\n[SETUP]我哈哈哈地笑了起来但作业还是没有任何进展。\n</SCRIPT>"
        supplied = annotations(build_plan(script))
        supplied["segments"][0]["parts"][0]["focus"] = "哈哈"
        with self.assertRaises(StagePlanError):
            build_plan(script, supplied)

    def test_focus_density_is_bounded(self):
        supplied = annotations(build_plan(SCRIPT))
        supplied["segments"][0]["parts"] = [part(FIRST, focus="作业"), part(SECOND, focus="计划表")]
        supplied["segments"][1]["parts"][0]["focus"] = "表格"
        with self.assertRaisesRegex(StagePlanError, "too many units contain focus"):
            build_plan(SCRIPT, supplied)

    def test_audience_or_self_question_cannot_be_created_by_annotation(self):
        script = "<SCRIPT>\n[SETUP]朋友问我：“你们到底准备什么时候开始写作业？”\n</SCRIPT>"
        supplied = annotations(build_plan(script))
        supplied["segments"][0]["parts"][0]["intent"] = "audience_question"
        with self.assertRaisesRegex(StagePlanError, "unquoted"):
            build_plan(script, supplied)
        supplied = annotations(build_plan(SCRIPT))
        supplied["segments"][0]["parts"][0]["intent"] = "self_question"
        with self.assertRaisesRegex(StagePlanError, "existing unquoted question"):
            build_plan(SCRIPT, supplied)

    def test_missing_duplicate_reordered_and_cross_speech_annotations_fail(self):
        good = annotations(build_plan(SCRIPT))
        variants = []
        missing = copy.deepcopy(good)
        missing["segments"].pop()
        variants.append(missing)
        duplicate = copy.deepcopy(good)
        duplicate["segments"][1]["source_index"] = 0
        variants.append(duplicate)
        reordered = copy.deepcopy(good)
        reordered["segments"].reverse()
        variants.append(reordered)
        cross = copy.deepcopy(good)
        cross["segments"][0]["parts"][0]["text"] += ENDING
        variants.append(cross)
        for supplied in variants:
            with self.subTest(supplied=supplied):
                with self.assertRaises(StagePlanError):
                    build_plan(SCRIPT, supplied)

    def test_extra_fields_are_rejected_in_annotations_and_saved_plan(self):
        supplied = annotations(build_plan(SCRIPT))
        supplied["segments"][0]["parts"][0]["speed"] = 1.1
        with self.assertRaises(StagePlanError):
            build_plan(SCRIPT, supplied)
        plan = build_plan(SCRIPT)
        plan["units"][0]["instruction"] = "say a different story"
        self.assertFalse(validate_plan(plan, SCRIPT)["passed"])
        plan = build_plan(SCRIPT)
        plan["extra"] = 1
        self.assertFalse(validate_execution_plan(plan)["passed"])

    def test_boolean_indexes_nonfinite_values_and_hidden_controls_are_rejected(self):
        mutations = [
            lambda p: p["units"][0].update(index=True),
            lambda p: p["units"][0].update(source_index=False),
            lambda p: p["units"][0].update(start=False),
            lambda p: p["units"][0].update(beat=True),
            lambda p: p["source_speeches"][0].update(speed=True),
            lambda p: p["source_events"][0].update(speed=float("nan")),
            lambda p: p["units"][0].update(text=p["units"][0]["text"] + "\x00"),
        ]
        for mutation in mutations:
            plan = build_plan(SCRIPT)
            mutation(plan)
            self.assertFalse(validate_execution_plan(plan)["passed"])

    def test_saved_source_fields_cannot_disagree_with_each_other(self):
        for field in ("text", "marked_text", "role", "beat", "speed"):
            plan = build_plan(SCRIPT)
            if field in {"text", "marked_text", "role"}:
                plan["source_speeches"][0][field] = "changed"
            else:
                plan["source_speeches"][0][field] = 1.25
            self.assertFalse(validate_execution_plan(plan)["passed"], field)

    def test_execution_validation_does_not_claim_to_authenticate_original_script(self):
        plan = build_plan(SCRIPT)
        plan["source_script_sha256"] = "0" * 64
        report = validate_execution_plan(plan)
        self.assertTrue(report["passed"])
        self.assertFalse(report["original_script_verified"])
        self.assertFalse(validate_plan(plan, SCRIPT)["passed"])
        plan = build_plan(SCRIPT)
        self.assertFalse(validate_plan(plan, SCRIPT + "\n")["passed"])

    def test_original_spoken_hash_and_unit_offsets_are_recomputed(self):
        plan = build_plan(SCRIPT)
        plan["spoken_text_sha256"] = "0" * 64
        self.assertFalse(validate_execution_plan(plan)["passed"])
        plan = build_plan(SCRIPT)
        plan["units"][0]["end"] -= 1
        self.assertFalse(validate_execution_plan(plan)["passed"])

    def test_planner_repairs_one_illegal_response_without_rewriting_prose(self):
        good = annotations(build_plan(SCRIPT))
        bad = copy.deepcopy(good)
        bad["segments"][0]["parts"][0]["text"] = "这是模型另外写出来的故事。"
        client = FakeClient([json.dumps(bad, ensure_ascii=False), json.dumps(good, ensure_ascii=False)])
        plan = request_plan(SCRIPT, self.config, client=client)
        self.assertTrue(validate_plan(plan, SCRIPT)["passed"])
        self.assertEqual(plan["planner"]["type"], "llm")
        self.assertEqual(plan["planner"]["status"], "validated")
        self.assertEqual(len(plan["planner"]["attempts"]), 2)
        self.assertTrue(plan["planner"]["attempts"][0]["errors"])
        self.assertEqual(plan["planner"]["attempts"][1]["errors"], [])
        self.assertEqual(client.calls[0]["response_format"], {"type": "json_object"})
        self.assertEqual(client.calls[0]["temperature"], 0.2)
        self.assertEqual(client.calls[0]["max_tokens"], 6000)
        sent = json.loads(client.calls[0]["messages"][1]["content"])
        self.assertEqual(sent["source_speeches"], plan["source_speeches"])
        self.assertEqual("".join(unit["text"] for unit in plan["units"]), FIRST + SECOND + ENDING)

    def test_two_invalid_model_responses_raise_with_records_and_no_rule_fallback(self):
        client = FakeClient(["{}", "{}"])
        with self.assertRaises(StagePlanError) as caught:
            request_plan(SCRIPT, self.config, client=client)
        planner = caught.exception.planner
        self.assertEqual(planner["type"], "llm")
        self.assertEqual(planner["status"], "failed")
        self.assertEqual(len(client.calls), 2)
        self.assertTrue(all(attempt["errors"] for attempt in planner["attempts"]))

    def test_duplicate_json_fields_are_not_silently_accepted(self):
        good = annotations(build_plan(SCRIPT))
        duplicate = '{"segments": [], "segments": ' + json.dumps(good["segments"], ensure_ascii=False) + "}"
        client = FakeClient([duplicate, json.dumps(good, ensure_ascii=False)])
        plan = request_plan(SCRIPT, self.config, client=client)
        self.assertIn("duplicate JSON field", plan["planner"]["attempts"][0]["errors"][0])

    def test_network_errors_save_only_type_status_and_never_config_key(self):
        class ProviderFailure(Exception):
            status_code = 401

        client = FakeClient([ProviderFailure("request Authorization Bearer " + self.config.api_key)])
        with self.assertRaises(StagePlanError) as caught:
            request_plan(SCRIPT, self.config, client=client)
        recorded = json.dumps(caught.exception.planner) + str(caught.exception)
        self.assertNotIn(self.config.api_key, recorded)
        self.assertIn("ProviderFailure", recorded)
        self.assertIn("401", recorded)
        self.assertEqual(len(client.calls), 1)

    def test_raw_invalid_response_redacts_known_key_and_attempt_count_is_strict(self):
        client = FakeClient([self.config.api_key])
        with self.assertRaises(StagePlanError) as caught:
            request_plan(SCRIPT, self.config, client=client, max_attempts=1)
        self.assertNotIn(self.config.api_key, json.dumps(caught.exception.planner))
        for budget in (True, 0, 3, float("nan")):
            with self.subTest(budget=budget):
                with self.assertRaises(StagePlanError) as invalid:
                    request_plan(SCRIPT, self.config, client=FakeClient([]), max_attempts=budget)
                self.assertEqual(invalid.exception.planner["type"], "llm")
                self.assertEqual(invalid.exception.planner["attempts"], [])


if __name__ == "__main__":
    unittest.main()
