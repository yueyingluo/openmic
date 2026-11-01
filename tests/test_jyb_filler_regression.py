import unittest

from openmic.tts_plan import Speech, compile_script


class FillerRegressionTests(unittest.TestCase):
    def test_explicit_filler_is_not_duplicated_by_automatic_insertion(self):
        script = "[SETUP]第一段。[SETUP]第二段。[SETUP][FILLER=那个]" + "这是足够长的一段铺垫内容，" * 4
        events = [item for item in compile_script(script, add_fillers=True) if isinstance(item, Speech)]
        self.assertTrue(events[-1].text.startswith("那个，"))
        self.assertFalse(events[-1].text.startswith("那个，那个，"))

    def test_existing_natural_filler_is_not_duplicated(self):
        script = "[SETUP]第一段。[SETUP]第二段。[SETUP]呃，" + "这是足够长的一段铺垫内容，" * 4
        event = [item for item in compile_script(script, add_fillers=True) if isinstance(item, Speech)][-1]
        self.assertTrue(event.text.startswith("呃，"))


if __name__ == "__main__":
    unittest.main()
