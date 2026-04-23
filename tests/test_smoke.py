import json
import unittest
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent.parent


def add_oulad_engineered_features(frame):
    X = frame.copy()
    X["score_per_assessment"] = X["avg_score"] * X["num_assessments"]
    X["click_per_day"] = X["total_clicks"] / (X["total_vle_days"] + 0.001)
    X["assessment_completion_rate"] = X["num_assessments"] / (
        X["num_TMA"] + X["num_CMA"] + X["num_Exam"] + 0.001
    )
    X["forum_ratio"] = X["clicks_forumng"] / (X["total_clicks"] + 0.001)
    X["quiz_ratio"] = X["clicks_quiz"] / (X["total_clicks"] + 0.001)
    X["resource_ratio"] = X["clicks_resource"] / (X["total_clicks"] + 0.001)
    X["score_consistency"] = X["avg_score"] / (X["std_score"] + 0.001)
    X["early_late_ratio"] = X["early_submissions"] / (X["late_submissions"] + 0.001)
    X["tma_cma_score_diff"] = X["avg_score_TMA"] - X["avg_score_CMA"]
    X["engagement_score"] = (
        X["total_clicks"] * X["total_vle_days"] * X["num_distinct_activities"]
    )
    return X


class SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dropout_model = joblib.load(ROOT / "models/best_model_dropout_localized.pkl")
        cls.oulad_model = joblib.load(ROOT / "models/best_model_oulad.pkl")
        cls.dropout_features = joblib.load(ROOT / "models/dropout_localized_features.pkl")
        with open(ROOT / "chatbot/feature_config.json", "r", encoding="utf-8") as f:
            cls.feature_config = json.load(f)

    def test_dropout_model_predicts_on_processed_sample(self):
        df = pd.read_csv(ROOT / "preprocessing/dropout_processed.csv")
        X = df[self.dropout_features].head(3)

        preds = self.dropout_model.predict(X)
        probs = self.dropout_model.predict_proba(X)

        self.assertEqual(len(preds), 3)
        self.assertEqual(probs.shape, (3, 3))

    def test_chatbot_feature_config_matches_model_features(self):
        self.assertEqual(list(self.feature_config.keys()), self.dropout_features)

    def test_oulad_model_predicts_on_processed_sample(self):
        df = pd.read_csv(ROOT / "preprocessing/oulad_processed.csv")
        X = add_oulad_engineered_features(df.drop(columns=["target"]))
        y = df["target"]

        X_train, X_test, y_train, _ = train_test_split(
            X, y, test_size=0.30, random_state=42, stratify=y
        )
        X_test = X_test.head(3).copy()

        mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
        mi_df = pd.DataFrame({"feature": X_train.columns, "mi": mi_scores})
        low_mi = mi_df.loc[mi_df["mi"] < 0.01, "feature"].tolist()
        if low_mi:
            X_test = X_test.drop(columns=low_mi)

        preds = self.oulad_model.predict(X_test)
        probs = self.oulad_model.predict_proba(X_test)

        self.assertEqual(len(preds), 3)
        self.assertEqual(probs.shape, (3, 3))

    def test_processed_schema_contains_expected_targets(self):
        dropout_df = pd.read_csv(ROOT / "preprocessing/dropout_processed.csv", nrows=1)
        oulad_df = pd.read_csv(ROOT / "preprocessing/oulad_processed.csv", nrows=1)

        self.assertIn("Target", dropout_df.columns)
        self.assertIn("target", oulad_df.columns)
        self.assertTrue(set(self.dropout_features).issubset(set(dropout_df.columns)))


if __name__ == "__main__":
    unittest.main()
