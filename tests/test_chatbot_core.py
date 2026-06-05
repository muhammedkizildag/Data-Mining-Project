import json
import unittest
from pathlib import Path

from chatbot.core import (
    auto_convert_grade,
    auto_convert_turkish_scale,
    build_categorical_value_map,
    build_tr_to_en_map,
    clean_extracted_data,
    extract_data_from_response,
    fix_feature_name,
)


ROOT = Path(__file__).resolve().parent.parent


class ChatbotCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(ROOT / "chatbot/feature_config.json", "r", encoding="utf-8") as f:
            cls.feature_config = json.load(f)
        cls.feature_order = list(cls.feature_config.keys())
        cls.tr_to_en_map = build_tr_to_en_map(cls.feature_config)
        cls.categorical_value_map = build_categorical_value_map(cls.feature_config)

    def test_fix_feature_name_handles_typo(self):
        fixed = fix_feature_name(
            "Curicular units 1st sem (grade)",
            self.feature_order,
            self.tr_to_en_map,
        )
        self.assertEqual(fixed, "Curricular units 1st sem (grade)")

    def test_auto_convert_grade_handles_four_and_hundred_scale(self):
        self.assertEqual(auto_convert_grade("Curricular units 1st sem (grade)", 14), 14)
        self.assertAlmostEqual(
            auto_convert_grade("Curricular units 1st sem (grade)", 2.8), 14.0
        )
        self.assertAlmostEqual(
            auto_convert_grade("Curricular units 2nd sem (grade)", 85), 17.0
        )

    def test_auto_convert_turkish_scale_maps_to_model_range(self):
        self.assertAlmostEqual(auto_convert_turkish_scale("Admission grade", 500), 190.0)
        self.assertAlmostEqual(
            auto_convert_turkish_scale("Previous qualification (grade)", 100), 190.0
        )

    def test_clean_extracted_data_clamps_invalid_numeric_values(self):
        raw_data = {
            "Age at enrollment": 120,
            "Curricular units 1st sem (grade)": 120,
            "Scholarship holder": 5,
            "Gender": "kadın",
        }

        cleaned = clean_extracted_data(
            raw_data,
            self.feature_order,
            self.feature_config,
            self.tr_to_en_map,
            self.categorical_value_map,
        )

        self.assertEqual(cleaned["Age at enrollment"], 70.0)
        self.assertEqual(cleaned["Curricular units 1st sem (grade)"], 20.0)
        self.assertEqual(cleaned["Gender"], 0.0)
        self.assertNotIn("Scholarship holder", cleaned)

    def test_clean_extracted_data_accepts_ascii_turkish_category_value(self):
        raw_data = {"Gender": "kadin"}

        cleaned = clean_extracted_data(
            raw_data,
            self.feature_order,
            self.feature_config,
            self.tr_to_en_map,
            self.categorical_value_map,
        )

        self.assertEqual(cleaned["Gender"], 0.0)

    def test_extract_data_from_response_parses_and_cleans_payload(self):
        response = (
            'Bilgileri aldim. [DATA: {"yas": 22, "1. dönem not": 2.4, '
            '"Scholarship holder": "evet"}]'
        )

        extracted, clean_text = extract_data_from_response(
            response,
            self.feature_order,
            self.feature_config,
            self.tr_to_en_map,
            self.categorical_value_map,
        )

        self.assertEqual(clean_text, "Bilgileri aldim.")
        self.assertEqual(extracted["Age at enrollment"], 22.0)
        self.assertAlmostEqual(extracted["Curricular units 1st sem (grade)"], 12.0)
        self.assertEqual(extracted["Scholarship holder"], 1.0)


if __name__ == "__main__":
    unittest.main()
