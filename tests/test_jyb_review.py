import copy
import unittest

from openmic.tts_review import normalized_spoken_text, parse_quality, review_script


class ReviewTests(unittest.TestCase):
    def make_case(self):
        setup = "我以为今天可以从容完成作业，结果我又花了很久给桌面分类。" * 6
        punch = "后来我发现，文件夹越来越整齐，真正的工作依然没有开始。" * 3
        lines = ["[INTRO]大家好，我们今天聊聊自己的拖延日常。"]
        for _ in range(5):
            lines.extend(["[SETUP]" + setup, "[PUNCHLINE]" + punch])
        lines.extend(["[SETUP]" + setup, "[CALLBACK]所以我决定先把待办清单重新排个版，至少看起来已经完成一半。"])
        writer = "<SCRIPT>\n" + "\n".join(lines) + "\n</SCRIPT>"
        coach = writer.replace("[PUNCHLINE]", "[PUNCHLINE][EMOTION=calm]")
        coach = coach.replace("真正的工作", "[EMPHASIS]真正的工作[/EMPHASIS]")
        qc = "**HUMOR_SCORE**: 8.5\n**CULTURE_SCORE**: 9\n**STRUCTURE_SCORE**: 8\nDECISION: APPROVED"
        trace = {"approved": True, "messages": [
            {"agent": "JokeWriter", "content": writer},
            {"agent": "PerformanceCoach", "content": coach},
            {"agent": "QualityController", "content": qc},
        ]}
        return coach, trace

    def test_valid_coach_keeps_words_and_returns_full_evidence(self):
        script, trace = self.make_case()
        result = review_script(script, trace)
        self.assertTrue(result["passed"], result["errors"])
        self.assertTrue(result["writer_coach_text_match"])
        self.assertEqual(result["punchlines"], 5)
        self.assertEqual(result["callbacks"], 1)
        self.assertEqual(result["quality_mean"], 8.5)
        self.assertTrue(result["human_review_pending"])

    def test_added_or_missing_words_fail_despite_valid_tags_and_qc(self):
        script, trace = self.make_case()
        changed = script.replace("大家好", "各位好", 1)
        trace["messages"][1]["content"] = changed
        result = review_script(changed, trace)
        self.assertFalse(result["passed"])
        self.assertFalse(result["writer_coach_text_match"])

    def test_saved_script_cannot_borrow_another_approved_trace(self):
        script, trace = self.make_case()
        result = review_script(script.replace("大家好", "各位好", 1), trace)
        self.assertFalse(result["passed"])
        self.assertTrue(any("最近Coach" in item for item in result["errors"]))

    def test_latest_revision_is_used(self):
        script, trace = self.make_case()
        trace["messages"].insert(0, {"agent": "JokeWriter", "content": "旧稿"})
        self.assertTrue(review_script(script, trace)["passed"])

    def test_missing_or_invalid_scores_do_not_get_approved(self):
        for qc in (
            "HUMOR_SCORE:8\nCULTURE_SCORE:9\nDECISION: APPROVED",
            "HUMOR_SCORE:11\nCULTURE_SCORE:9\nSTRUCTURE_SCORE:8\nDECISION: APPROVED",
            "HUMOR_SCORE:6\nCULTURE_SCORE:6\nSTRUCTURE_SCORE:6\nDECISION: APPROVED",
            "HUMOR_SCORE:8\nHUMOR_SCORE:9\nCULTURE_SCORE:9\nSTRUCTURE_SCORE:8\nDECISION: APPROVED",
        ):
            with self.subTest(qc=qc):
                self.assertTrue(parse_quality(qc)["errors"])

    def test_final_decision_is_unique_and_must_really_be_final(self):
        valid = "HUMOR_SCORE:8\nCULTURE_SCORE:9\nSTRUCTURE_SCORE:8\nDECISION: APPROVED"
        self.assertTrue(parse_quality(valid)["approved"])
        for qc in (valid + "\n补充说明", valid + "\nDECISION: APPROVED",
                   valid + "\nDECISION: REVISION_REQUIRED", valid.replace("APPROVED", "REVISION_REQUIRED")):
            with self.subTest(qc=qc):
                self.assertFalse(parse_quality(qc)["approved"])

    def test_missing_structure_short_text_and_protocol_errors_fail(self):
        script, trace = self.make_case()
        for broken in (script.replace("[CALLBACK]", "[SETUP]"), "[SETUP]太短了。", "[UNKNOWN]正文"):
            with self.subTest(broken=broken[:40]):
                self.assertFalse(review_script(broken, trace)["passed"])

    def test_punctuation_can_change_but_filler_is_spoken_content(self):
        self.assertEqual(normalized_spoken_text("[SETUP]你好，世界！"), normalized_spoken_text("[SETUP]你好世界。"))
        self.assertEqual(normalized_spoken_text("[SETUP][FILLER=那个]正文。"), "那个正文")
        self.assertNotEqual(normalized_spoken_text("[SETUP][FILLER=那个]正文。"), normalized_spoken_text("[SETUP]正文。"))


if __name__ == "__main__":
    unittest.main()
