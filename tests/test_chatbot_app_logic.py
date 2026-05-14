import json
import types
import unittest
from pathlib import Path

from chatbot.core import auto_convert_turkish_scale


ROOT = Path(__file__).resolve().parent.parent


class ChatbotAppLogicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(ROOT / "chatbot/feature_config.json", "r", encoding="utf-8") as f:
            cls.feature_config = json.load(f)
        cls.feature_order = list(cls.feature_config.keys())

    def test_default_values_are_scaled_for_model_input(self):
        defaults = {}
        for feat in self.feature_order:
            raw_val = float(self.feature_config[feat]["default"])
            defaults[feat] = auto_convert_turkish_scale(feat, raw_val)

        self.assertEqual(defaults["Admission grade"], 152.0)
        self.assertEqual(defaults["Previous qualification (grade)"], 161.5)


if __name__ == "__main__":
    unittest.main()
