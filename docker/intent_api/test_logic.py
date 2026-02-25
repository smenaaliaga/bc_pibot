import unittest

from intent_api.logic import classify_intent


class IntentLogicTests(unittest.TestCase):
    def test_macro_pib(self):
        result = classify_intent("ultimo pib")
        self.assertEqual(result["macro"]["label"], 1)

    def test_macro_other(self):
        result = classify_intent("hola como estas")
        self.assertEqual(result["macro"]["label"], 0)

    def test_context_followup(self):
        result = classify_intent("y eso mismo?")
        self.assertEqual(result["context"]["label"], "followup")

    def test_method_intent(self):
        result = classify_intent("metodologia del imacec")
        self.assertEqual(result["intent"]["label"], "method")

    def test_method_intent_with_explicame_detalles(self):
        result = classify_intent("explícame detalles del imacec")
        self.assertEqual(result["intent"]["label"], "method")

    def test_value_intent_with_accented_economy(self):
        result = classify_intent("cuanto aceleró la economía el último mes valor?")
        self.assertEqual(result["macro"]["label"], 1)
        self.assertEqual(result["intent"]["label"], "value")
        self.assertEqual(result["context"]["label"], "standalone")


if __name__ == "__main__":
    unittest.main()
