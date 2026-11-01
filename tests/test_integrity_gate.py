import unittest

from openmic.tts_integrity_gate import assess_content, normalize_content


class IntegrityGateTests(unittest.TestCase):
    def test_punctuation_and_nfkc_do_not_rewrite_spoken_characters(self):
        report = assess_content("ＡＢＣ，你好！", "abc你好")
        self.assertTrue(report["passed"])
        self.assertEqual(report["cer"], 0)
        self.assertEqual(report["reference_normalized"], "abc你好")
        self.assertIn("not a listening guarantee", report["scope"])

    def test_required_explicit_numeral_equivalences_and_raw_cer(self):
        for source, transcript in (("五", "5"), ("二十", "20"), ("双十一", "双11"), ("两千元", "2000元"), ("二十六个小时", "26个小时")):
            with self.subTest(source=source):
                report = assess_content(source, transcript)
                self.assertTrue(report["passed"])
                self.assertEqual(report["cer"], 0)
                self.assertGreater(report["raw_cer"], 0)

    def test_decimal_distinction_and_leading_zeros_are_preserved(self):
        self.assertTrue(assess_content("二点五", "2.5")["passed"])
        self.assertFalse(assess_content("2.5", "25")["passed"])
        self.assertTrue(assess_content("零一", "01")["passed"])
        self.assertFalse(assess_content("零一", "1")["passed"])

    def test_ambiguous_numerals_are_not_silently_guessed(self):
        self.assertEqual(normalize_content("万一"), "万一")
        self.assertEqual(normalize_content("两三"), "两三")
        self.assertFalse(assess_content("两三", "23")["passed"])

    def test_short_sentence_missing_or_changed_character_fails(self):
        for transcript in ("abcdefg", "abcxefgh", "abcdefghx"):
            with self.subTest(transcript=transcript):
                self.assertFalse(assess_content("abcdefgh", transcript)["passed"])

    def test_empty_reference_or_transcript_does_not_pass(self):
        self.assertFalse(assess_content("。", "！")["passed"])
        self.assertFalse(assess_content("有正文", "😊")["passed"])

    def test_twelve_percent_boundary_and_rejection_above_it(self):
        source = "abcdefghijklmnopqrstuvwxy"
        within = list(source)
        for index in (5,10,15):
            within[index] = "z"
        report = assess_content(source, "".join(within))
        self.assertEqual(report["cer"], .12)
        self.assertTrue(report["passed"])
        within[19] = "z"
        self.assertFalse(assess_content(source, "".join(within))["passed"])

    def test_one_middle_deletion_uses_three_percent_floor(self):
        source = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMN"
        report = assess_content(source, source[:20]+source[21:])
        self.assertEqual(report["deletions"], 1)
        self.assertEqual(report["deletion_limit"], 1)
        self.assertEqual(report["longest_deletion_run"], 1)
        self.assertTrue(report["passed"])

    def test_two_adjacent_deletions_fail_even_when_global_rate_is_low(self):
        source = "abcdefghijklmnopqrstuvwxyz"*4
        report = assess_content(source, source[:20]+source[22:])
        self.assertLess(report["cer"], .12)
        self.assertLessEqual(report["deletions"], report["deletion_limit"])
        self.assertEqual(report["longest_deletion_run"], 2)
        self.assertFalse(report["passed"])

    def test_nonadjacent_deletions_fail_absolute_count_budget(self):
        source = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMN"
        transcript = "".join(char for index,char in enumerate(source) if index not in {10,20})
        report = assess_content(source, transcript)
        self.assertEqual(report["longest_deletion_run"], 1)
        self.assertEqual(report["deletions"], 2)
        self.assertFalse(report["passed"])

    def test_head_and_tail_deletions_fail_even_with_ample_total_budget(self):
        source = "abcdefghijklmnopqrstuvwxyz"*4
        head = assess_content(source, source[1:])
        tail = assess_content(source, source[:-1])
        self.assertLess(head["cer"], .03)
        self.assertFalse(head["head_passed"])
        self.assertFalse(head["passed"])
        self.assertFalse(tail["tail_passed"])
        self.assertFalse(tail["passed"])

    def test_two_head_or_tail_substitutions_fail_without_deletion(self):
        source = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMN"
        head = assess_content(source, "zz"+source[2:])
        tail = assess_content(source, source[:-2]+"zz")
        self.assertLess(head["cer"], .12)
        self.assertEqual(head["deletions"], 0)
        self.assertEqual(head["head_check"]["substitutions"], 2)
        self.assertFalse(head["passed"])
        self.assertFalse(tail["passed"])

    def test_invented_continuation_is_counted_as_insertions(self):
        source = "abcdefghijklmnopqrst"
        report = assess_content(source, source+"inventedwords")
        self.assertGreater(report["insertions"], 0)
        self.assertFalse(report["passed"])

    def test_edit_counts_equal_normalized_distance(self):
        report = assess_content("abcdefghijklmnopqrst", "abcdezzhijklmnopqrs")
        total = report["deletions"]+report["insertions"]+report["substitutions"]
        self.assertAlmostEqual(total, report["cer"]*report["reference_length"])
        self.assertTrue(all("not audio timestamps" in report[key]["positions_scope"] for key in ("head_check","tail_check")))

    def test_no_homophone_or_paraphrase_auto_correction(self):
        self.assertFalse(assess_content("犹豫", "有")["passed"])
        self.assertFalse(assess_content("退货", "推")["passed"])

    def test_invalid_types_and_unbounded_alignment_are_rejected(self):
        with self.assertRaises(TypeError):
            assess_content("正文", None)
        with self.assertRaises(ValueError):
            assess_content("甲"*2000, "乙"*2000)


if __name__ == "__main__":
    unittest.main()
