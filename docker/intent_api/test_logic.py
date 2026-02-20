import unittest

from intent_api.logic import classify_intent


class IntentLogicTests(unittest.TestCase):
    def test_macro_pib(self):
        result = classify_intent("ultimo pib")
        self.assertEqual(result["macro"], 1)

    def test_macro_other(self):
        result = classify_intent("hola como estas")
        self.assertEqual(result["macro"], 0)

    def test_context_followup(self):
        result = classify_intent("y eso mismo?")
        self.assertEqual(result["context"], "followup")

    def test_method_intent(self):
        result = classify_intent("metodologia del imacec")
        self.assertEqual(result["intent"], "method")


if __name__ == "__main__":
    unittest.main()
