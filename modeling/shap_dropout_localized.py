import os
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


OUTPUT_DIR = "modeling/plots_shap_dropout_localized"
MODEL_PATH = "models/best_model_dropout_localized.pkl"
FEATURES_PATH = "models/dropout_localized_features.pkl"
DATA_PATH = "preprocessing/dropout_processed.csv"

TARGET_NAMES = ["Dropout", "Enrolled", "Graduate"]


def normalize_xgboost_contribs(values, n_features):
    """Return XGBoost SHAP contributions as (samples, features, classes)."""
    arr = np.asarray(values)
    if arr.ndim == 2:
        width = n_features + 1
        if arr.shape[1] == width:
            return arr[:, :-1, np.newaxis]
        if arr.shape[1] % width == 0:
            n_classes = arr.shape[1] // width
            arr = arr.reshape(arr.shape[0], n_classes, width)
            return np.transpose(arr[:, :, :-1], (0, 2, 1))

    if arr.ndim == 3:
        if arr.shape[1] == n_features + 1:
            return arr[:, :-1, :]
        if arr.shape[2] == n_features + 1:
            return np.transpose(arr[:, :, :-1], (0, 2, 1))

    raise ValueError(f"Unsupported XGBoost contribution shape: {arr.shape}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  SHAP ANALIZI - Dropout UCI Yerellestirilmis Model")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)
    X = df.drop("Target", axis=1)
    y = df["Target"]

    feature_list = joblib.load(FEATURES_PATH)
    X = X[feature_list]

    _, X_test, _, _ = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    pipeline = joblib.load(MODEL_PATH)
    scaler = pipeline.named_steps["scaler"]
    model = pipeline.named_steps["model"]

    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_list,
        index=X_test.index,
    )

    dmatrix = xgb.DMatrix(X_test_scaled, feature_names=feature_list)
    raw_shap_values = model.get_booster().predict(dmatrix, pred_contribs=True)
    shap_values = normalize_xgboost_contribs(raw_shap_values, len(feature_list))

    mean_abs_by_class = np.abs(shap_values).mean(axis=0)
    if mean_abs_by_class.shape[1] == 1:
        mean_abs_global = mean_abs_by_class[:, 0]
    else:
        mean_abs_global = mean_abs_by_class.mean(axis=1)

    importance_df = pd.DataFrame({
        "feature": feature_list,
        "mean_abs_shap": mean_abs_global,
    })

    for class_idx in range(mean_abs_by_class.shape[1]):
        class_name = TARGET_NAMES[class_idx] if class_idx < len(TARGET_NAMES) else f"class_{class_idx}"
        importance_df[f"mean_abs_shap_{class_name}"] = mean_abs_by_class[:, class_idx]

    importance_df = importance_df.sort_values("mean_abs_shap", ascending=False)

    csv_path = f"{OUTPUT_DIR}/shap_feature_importance.csv"
    importance_df.to_csv(csv_path, index=False)
    print(f"  SHAP tablo: {csv_path}")

    top_df = importance_df.head(15).sort_values("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top_df["feature"], top_df["mean_abs_shap"], color="steelblue", edgecolor="black")
    ax.set_title("SHAP Global Feature Importance - Dropout Localized", fontsize=13, fontweight="bold")
    ax.set_xlabel("Ortalama mutlak SHAP etkisi")
    plt.tight_layout()
    bar_path = f"{OUTPUT_DIR}/01_shap_global_importance.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Grafik: {bar_path}")

    if mean_abs_by_class.shape[1] > 1:
        class_rows = []
        for class_idx, class_name in enumerate(TARGET_NAMES[:mean_abs_by_class.shape[1]]):
            values = mean_abs_by_class[:, class_idx]
            order = np.argsort(values)[::-1][:10]
            for rank, feature_idx in enumerate(order, start=1):
                class_rows.append({
                    "class": class_name,
                    "rank": rank,
                    "feature": feature_list[feature_idx],
                    "mean_abs_shap": values[feature_idx],
                })

        class_df = pd.DataFrame(class_rows)
        class_csv_path = f"{OUTPUT_DIR}/shap_feature_importance_by_class.csv"
        class_df.to_csv(class_csv_path, index=False)
        print(f"  Sinif bazli SHAP tablo: {class_csv_path}")

        fig, axes = plt.subplots(1, mean_abs_by_class.shape[1], figsize=(18, 7), sharex=False)
        if mean_abs_by_class.shape[1] == 1:
            axes = [axes]

        for class_idx, ax in enumerate(axes):
            class_name = TARGET_NAMES[class_idx] if class_idx < len(TARGET_NAMES) else f"class_{class_idx}"
            values = mean_abs_by_class[:, class_idx]
            order = np.argsort(values)[::-1][:10]
            plot_features = [feature_list[i] for i in order][::-1]
            plot_values = values[order][::-1]
            ax.barh(plot_features, plot_values, color="darkorange", edgecolor="black")
            ax.set_title(class_name, fontsize=12, fontweight="bold")
            ax.set_xlabel("Mean |SHAP|")

        plt.suptitle("Class-wise SHAP Feature Importance", fontsize=14, fontweight="bold")
        plt.tight_layout()
        class_plot_path = f"{OUTPUT_DIR}/02_shap_class_importance.png"
        plt.savefig(class_plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Grafik: {class_plot_path}")

    print("\n  SHAP ANALIZI TAMAMLANDI")


if __name__ == "__main__":
    main()
