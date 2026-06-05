import unittest

import pandas as pd

from preprocessing.common import (
    OULAD_TARGET_MAP,
    apply_oulad_target_mapping,
    build_dropout_processed_frame,
    encode_oulad_categories,
    fill_oulad_missing_values,
)


class PreprocessingCommonTests(unittest.TestCase):
    def test_build_dropout_processed_frame_encodes_target(self):
        df = pd.DataFrame(
            {
                "Feature A": [1, 2, 3],
                "Target": ["Dropout", "Enrolled", "Graduate"],
            }
        )

        processed, mapping = build_dropout_processed_frame(df)

        self.assertEqual(processed.columns.tolist(), ["Feature A", "Target"])
        self.assertEqual(mapping["Dropout"], 0)
        self.assertEqual(mapping["Enrolled"], 1)
        self.assertEqual(mapping["Graduate"], 2)

    def test_apply_oulad_target_mapping_maps_distinction_to_pass(self):
        student_info = pd.DataFrame({"final_result": ["Withdrawn", "Fail", "Pass", "Distinction"]})

        mapped = apply_oulad_target_mapping(student_info)

        self.assertEqual(mapped["target"].tolist(), [0, 1, 2, 2])

    def test_fill_oulad_missing_values_fills_numeric_and_mode(self):
        df = pd.DataFrame(
            {
                "target": [0, 1],
                "imd_band": ["0-10%", None],
                "avg_score": [None, 55.0],
                "num_assessments": [1, None],
            }
        )

        filled = fill_oulad_missing_values(df)

        self.assertEqual(filled.loc[0, "avg_score"], 0)
        self.assertEqual(filled.loc[1, "num_assessments"], 0)
        self.assertEqual(filled.loc[1, "imd_band"], "0-10%")

    def test_encode_oulad_categories_drops_identifier_columns(self):
        df = pd.DataFrame(
            {
                "code_module": ["AAA", "BBB"],
                "code_presentation": ["2013J", "2014J"],
                "id_student": [1, 2],
                "final_result": ["Pass", "Fail"],
                "gender": ["M", "F"],
                "target": [2, 1],
            }
        )

        encoded, encodings = encode_oulad_categories(df)

        self.assertNotIn("code_module", encoded.columns)
        self.assertIn("gender", encodings)
        self.assertEqual(sorted(encoded["gender"].unique().tolist()), [0, 1])


if __name__ == "__main__":
    unittest.main()
