import unittest

import numpy as np

from modeling.common import build_cv_summary, select_best_model


class ModelingCommonTests(unittest.TestCase):
    def test_select_best_model_returns_highest_macro_mean(self):
        cv_results_macro = {
            "A": np.array([0.50, 0.55]),
            "B": np.array([0.61, 0.60]),
            "C": np.array([0.58, 0.57]),
        }

        best_name, best_score = select_best_model(cv_results_macro)

        self.assertEqual(best_name, "B")
        self.assertAlmostEqual(best_score, 0.605)

    def test_build_cv_summary_formats_expected_columns(self):
        cv_results_macro = {"A": np.array([0.50, 0.60])}
        cv_results_weighted = {"A": np.array([0.70, 0.80])}

        summary = build_cv_summary(cv_results_macro, cv_results_weighted)

        self.assertEqual(summary.columns.tolist(), ["Model", "CV Macro F1", "CV Weighted F1"])
        self.assertEqual(summary.iloc[0]["Model"], "A")
        self.assertEqual(summary.iloc[0]["CV Macro F1"], "%55.00")
        self.assertEqual(summary.iloc[0]["CV Weighted F1"], "%75.00")


if __name__ == "__main__":
    unittest.main()
