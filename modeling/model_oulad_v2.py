import joblib
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

from modeling.common import (
    build_cv_summary,
    ensure_dir,
    evaluate_models_with_cv,
    print_holdout_comparison,
    plot_learning_curve_custom,
    plot_multiclass_roc_pr,
    save_confusion_matrix,
    save_cv_boxplot,
    save_feature_importance,
    select_best_model,
)

OUTPUT_DIR = "modeling/plots_oulad_v2"
CV_N_JOBS = 1
SAMPLE_WEIGHT_MODELS = {"XGBoost", "LightGBM", "AdaBoost"}
ensure_dir(OUTPUT_DIR)

TARGET_NAMES = ["Withdrawn", "Fail", "Pass"]
NEW_COLS = [
    "score_per_assessment",
    "click_per_day",
    "assessment_completion_rate",
    "forum_ratio",
    "quiz_ratio",
    "resource_ratio",
    "score_consistency",
    "early_late_ratio",
    "tma_cma_score_diff",
    "engagement_score",
]


def add_engineered_features(frame):
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


def build_model_searches(X_train, y_train):
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    sample_weights = compute_sample_weight("balanced", y_train)

    print("\n  [kNN]")
    knn_pipe = Pipeline([("scaler", MinMaxScaler()), ("model", KNeighborsClassifier())])
    knn_params = {
        "model__n_neighbors": [1, 3, 5, 7, 9, 11, 13, 15],
        "model__metric": ["euclidean", "manhattan"],
    }
    knn_grid = GridSearchCV(knn_pipe, knn_params, cv=kf, scoring="f1_macro", n_jobs=CV_N_JOBS)
    knn_grid.fit(X_train, y_train)
    print(f"  En iyi: {knn_grid.best_params_} → CV F1 (macro): %{knn_grid.best_score_ * 100:.2f}")

    print("\n  [Naive Bayes]")
    nb_pipe = Pipeline([("scaler", MinMaxScaler()), ("model", GaussianNB())])
    nb_pipe.fit(X_train, y_train)
    nb_cv = cross_val_score(nb_pipe, X_train, y_train, cv=kf, scoring="f1_macro")
    print(f"  CV F1 (macro): %{nb_cv.mean() * 100:.2f}")

    print("\n  [Decision Tree]")
    dt_pipe = Pipeline(
        [("scaler", MinMaxScaler()), ("model", DecisionTreeClassifier(random_state=42, class_weight="balanced"))]
    )
    dt_params = {
        "model__max_depth": [3, 5, 7, 10, 15, None],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 5],
    }
    dt_grid = GridSearchCV(dt_pipe, dt_params, cv=kf, scoring="f1_macro", n_jobs=CV_N_JOBS)
    dt_grid.fit(X_train, y_train)
    print(f"  En iyi: {dt_grid.best_params_} → CV F1 (macro): %{dt_grid.best_score_ * 100:.2f}")

    print("\n  [Random Forest]")
    rf_pipe = Pipeline(
        [("scaler", MinMaxScaler()), ("model", RandomForestClassifier(random_state=42, class_weight="balanced"))]
    )
    rf_params = {
        "model__n_estimators": [50, 100, 200, 300],
        "model__max_depth": [5, 10, 15, 20, None],
        "model__min_samples_split": [2, 5, 10],
    }
    rf_grid = GridSearchCV(rf_pipe, rf_params, cv=kf, scoring="f1_macro", n_jobs=CV_N_JOBS)
    rf_grid.fit(X_train, y_train)
    print(f"  En iyi: {rf_grid.best_params_} → CV F1 (macro): %{rf_grid.best_score_ * 100:.2f}")

    print("\n  [ExtraTrees]")
    et_pipe = Pipeline(
        [("scaler", MinMaxScaler()), ("model", ExtraTreesClassifier(random_state=42, class_weight="balanced"))]
    )
    et_params = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [10, 20, None],
        "model__min_samples_split": [2, 5, 10],
    }
    et_grid = GridSearchCV(et_pipe, et_params, cv=kf, scoring="f1_macro", n_jobs=CV_N_JOBS)
    et_grid.fit(X_train, y_train)
    print(f"  En iyi: {et_grid.best_params_} → CV F1 (macro): %{et_grid.best_score_ * 100:.2f}")

    print("\n  [AdaBoost]")
    ada_pipe = Pipeline([("scaler", MinMaxScaler()), ("model", AdaBoostClassifier(random_state=42))])
    ada_params = {
        "model__n_estimators": [50, 100, 200],
        "model__learning_rate": [0.05, 0.1, 0.5, 1.0],
    }
    ada_grid = GridSearchCV(ada_pipe, ada_params, cv=kf, scoring="f1_macro", n_jobs=CV_N_JOBS)
    ada_grid.fit(X_train, y_train, model__sample_weight=sample_weights)
    print(f"  En iyi: {ada_grid.best_params_} → CV F1 (macro): %{ada_grid.best_score_ * 100:.2f}")

    print("\n  [XGBoost]")
    xgb_pipe = Pipeline(
        [("scaler", MinMaxScaler()), ("model", XGBClassifier(random_state=42, eval_metric="mlogloss", verbosity=0))]
    )
    xgb_params = {
        "model__n_estimators": [50, 100, 200, 300],
        "model__max_depth": [3, 5, 7, 10],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "model__subsample": [0.8, 1.0],
    }
    xgb_grid = GridSearchCV(xgb_pipe, xgb_params, cv=kf, scoring="f1_macro", n_jobs=CV_N_JOBS)
    xgb_grid.fit(X_train, y_train, model__sample_weight=sample_weights)
    print(f"  En iyi: {xgb_grid.best_params_} → CV F1 (macro): %{xgb_grid.best_score_ * 100:.2f}")

    lgbm_estimator = None
    if LGBMClassifier is None:
        print("\n  [LightGBM] Atlandı — paket kurulu değil. requirements.txt güncellendi.")
    else:
        print("\n  [LightGBM]")
        lgbm_pipe = Pipeline(
            [
                ("scaler", MinMaxScaler()),
                ("model", LGBMClassifier(random_state=42, verbose=-1, objective="multiclass")),
            ]
        )
        lgbm_params = {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [-1, 10, 20],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__num_leaves": [31, 63],
        }
        lgbm_grid = GridSearchCV(lgbm_pipe, lgbm_params, cv=kf, scoring="f1_macro", n_jobs=CV_N_JOBS)
        lgbm_grid.fit(X_train, y_train, model__sample_weight=sample_weights)
        print(f"  En iyi: {lgbm_grid.best_params_} → CV F1 (macro): %{lgbm_grid.best_score_ * 100:.2f}")
        lgbm_estimator = lgbm_grid.best_estimator_

    print("\n  [SVM]")
    svm_pipe = Pipeline(
        [("scaler", MinMaxScaler()), ("model", SVC(random_state=42, class_weight="balanced", probability=True))]
    )
    svm_params = {
        "model__C": [0.1, 1, 10],
        "model__kernel": ["rbf", "linear"],
        "model__gamma": ["scale", "auto"],
    }
    svm_grid = GridSearchCV(svm_pipe, svm_params, cv=kf, scoring="f1_macro", n_jobs=CV_N_JOBS)
    svm_grid.fit(X_train, y_train)
    print(f"  En iyi: {svm_grid.best_params_} → CV F1 (macro): %{svm_grid.best_score_ * 100:.2f}")

    optimized_models = {
        "kNN": knn_grid.best_estimator_,
        "Naive Bayes": nb_pipe,
        "Decision Tree": dt_grid.best_estimator_,
        "Random Forest": rf_grid.best_estimator_,
        "ExtraTrees": et_grid.best_estimator_,
        "AdaBoost": ada_grid.best_estimator_,
        "XGBoost": xgb_grid.best_estimator_,
        "SVM": svm_grid.best_estimator_,
    }
    if lgbm_estimator is not None:
        optimized_models["LightGBM"] = lgbm_estimator
    return optimized_models


df = pd.read_csv("preprocessing/oulad_processed.csv")
X = df.drop("target", axis=1)
y = df["target"]

print("=" * 70)
print("  MODELLEME v2 — OULAD (Feature Eng. + XGBoost)")
print("  Pipeline yapısı: MinMaxScaler + Model birlikte")
print("=" * 70)

print("\n" + "=" * 70)
print("  FEATURE ENGINEERING")
print("=" * 70)
print(f"  Orijinal özellik sayısı: {X.shape[1]}")
X = add_engineered_features(X)
print(f"  Yeni özellik sayısı: {X.shape[1]} (+{len(NEW_COLS)} türetilmiş)")
for col in NEW_COLS:
    print(f"    - {col}")

print("\n" + "=" * 70)
print("  TRAIN/TEST SPLIT")
print("=" * 70)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
print(f"  Train: {X_train.shape[0]} satır")
print(f"  Test:  {X_test.shape[0]} satır")

print("\n" + "=" * 70)
print("  FEATURE SELECTION — Mutual Information (raw train, Pipeline dışında)")
print("=" * 70)
mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
mi_df = pd.DataFrame({"Özellik": X_train.columns, "MI": mi_scores}).sort_values("MI", ascending=False)
print("\n  MI Skorları (Top 15):")
for _, row in mi_df.head(15).iterrows():
    bar = "█" * int(row["MI"] * 50)
    print(f"    {row['Özellik']:35s} MI={row['MI']:.4f} {bar}")

low_mi = mi_df[mi_df["MI"] < 0.01]["Özellik"].tolist()
if low_mi:
    print(f"\n  MI < 0.01 olan {len(low_mi)} özellik çıkarılıyor: {low_mi}")
    X_train = X_train.drop(columns=low_mi)
    X_test = X_test.drop(columns=low_mi)
    print(f"  Kalan özellik sayısı: {X_train.shape[1]}")
else:
    print("\n  Tüm özellikler MI >= 0.01, çıkarılan yok.")

print("\n" + "=" * 70)
print("  HİPERPARAMETRE OPTİMİZASYONU (Pipeline + GridSearchCV)")
print("  Scoring: f1_macro — her sınıfa eşit ağırlık")
print("  Scaler her CV fold'unda sadece training kısmından fit edilir")
print("=" * 70)
optimized_models = build_model_searches(X_train, y_train)

print("\n" + "=" * 70)
print("  10-FOLD CROSS VALIDATION")
print("=" * 70)
kf10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_results_macro, cv_results_weighted = evaluate_models_with_cv(
    optimized_models, X_train, y_train, kf10
)
for name in optimized_models:
    scores_macro = cv_results_macro[name]
    scores_weighted = cv_results_weighted[name]
    print(
        f"  {name:20s} → Macro F1: %{scores_macro.mean()*100:.2f} (±{scores_macro.std()*100:.2f})  |  Weighted F1: %{scores_weighted.mean()*100:.2f}"
    )

print("\n" + "=" * 70)
print("  MODEL SEÇİMİ (10-Fold CV Macro F1 skoruna göre)")
print("=" * 70)
best_model_name, best_cv_macro = select_best_model(cv_results_macro)
best_cv_weighted = cv_results_weighted[best_model_name].mean()
print(f"\n  En iyi model (CV Macro F1): {best_model_name}")
print(f"  CV Macro F1:    %{best_cv_macro*100:.2f}")
print(f"  CV Weighted F1: %{best_cv_weighted*100:.2f}")
for name, scores in cv_results_macro.items():
    marker = "  ← seçildi" if name == best_model_name else ""
    print(f"    {name:20s} CV Macro: %{scores.mean()*100:.2f}{marker}")

save_cv_boxplot(
    cv_results_macro,
    f"{OUTPUT_DIR}/01_cv_karsilastirma.png",
    "10-Fold CV (Macro F1) — OULAD v2",
)

print("\n" + "=" * 70)
print("  TEST SETİ — FİNAL DEĞERLENDİRME")
print(f"  Model: {best_model_name} (CV ile seçildi, test seti sadece raporlama)")
print("=" * 70)
best_pipeline = optimized_models[best_model_name]
y_pred_best = best_pipeline.predict(X_test)

test_acc = accuracy_score(y_test, y_pred_best)
test_prec = precision_score(y_test, y_pred_best, average="weighted", zero_division=0)
test_rec = recall_score(y_test, y_pred_best, average="weighted", zero_division=0)
best_f1_weighted = f1_score(y_test, y_pred_best, average="weighted", zero_division=0)
best_f1_macro = f1_score(y_test, y_pred_best, average="macro", zero_division=0)
print(f"\n  Accuracy:    %{test_acc*100:.2f}")
print(f"  Precision:   %{test_prec*100:.2f}")
print(f"  Recall:      %{test_rec*100:.2f}")
print(f"  F1-Weighted: %{best_f1_weighted*100:.2f}")
print(f"  F1-Macro:    %{best_f1_macro*100:.2f}")

print("\n" + "=" * 70)
print("  SINIF BAZLI METRİKLER")
print("=" * 70)
print("\n  Classification Report:")
print(classification_report(y_test, y_pred_best, target_names=TARGET_NAMES, digits=4))
withdrawn_recall = recall_score(y_test, y_pred_best, average=None)[0]
print(f"\n  ⚠️  Withdrawn Recall: %{withdrawn_recall*100:.2f}")
print(f"      → Gerçek bırakma öğrencilerinin %{withdrawn_recall*100:.1f}'ini yakalıyoruz")

best_proba = best_pipeline.predict_proba(X_test)
plot_multiclass_roc_pr(
    y_test,
    best_proba,
    TARGET_NAMES,
    f"{OUTPUT_DIR}/03_roc_pr_curves.png",
    f"{best_model_name} (OULAD v2)",
)
print(f"\n  ROC/PR grafiği kaydedildi: {OUTPUT_DIR}/03_roc_pr_curves.png")

plot_learning_curve_custom(
    best_pipeline,
    X_train,
    y_train,
    f"{OUTPUT_DIR}/05_learning_curve.png",
    f"Learning Curve — {best_model_name} (OULAD v2)",
    TARGET_NAMES,
    use_sample_weight=best_model_name in SAMPLE_WEIGHT_MODELS,
)
print(f"  Learning curve kaydedildi: {OUTPUT_DIR}/05_learning_curve.png")

save_confusion_matrix(
    y_test,
    y_pred_best,
    TARGET_NAMES,
    f"{OUTPUT_DIR}/02_confusion_matrix.png",
    f"Confusion Matrix — {best_model_name} (OULAD v2)",
)

print("\n" + "=" * 70)
print("  CV KARŞILAŞTIRMA TABLOSU")
print("=" * 70)
cv_summary_df = build_cv_summary(cv_results_macro, cv_results_weighted)
print(f"\n{cv_summary_df.to_string(index=False)}")
print(f"\n  Seçilen model: {best_model_name} (CV Macro F1: %{best_cv_macro*100:.2f})")
print(f"  Test Macro F1:    %{best_f1_macro*100:.2f}")
print(f"  Test Weighted F1: %{best_f1_weighted*100:.2f}")

print("\n" + "=" * 70)
print("  V2-ÖNCEKI vs V2-GÜNCEL KARŞILAŞTIRMA")
print("=" * 70)
print("\n  Önceki (f1_weighted, class_weight yok): XGBoost → Weighted F1: %80.48")
print(
    f"  Güncel (f1_macro, class_weight balanced): {best_model_name} → Weighted F1: %{best_f1_weighted*100:.2f}, Macro F1: %{best_f1_macro*100:.2f}"
)

print("\n" + "=" * 70)
print("  EN İYİ MODELİ KAYDET (Pipeline)")
print("=" * 70)
joblib.dump(best_pipeline, "models/best_model_oulad.pkl")
print(f"  Model: {best_model_name} (Pipeline: MinMaxScaler + {best_model_name})")
print(f"  F1-Macro:    %{best_f1_macro*100:.2f}")
print(f"  F1-Weighted: %{best_f1_weighted*100:.2f}")
print("  Kaydedildi: models/best_model_oulad.pkl")

save_feature_importance(
    best_pipeline.named_steps["model"],
    X_train.columns,
    f"{OUTPUT_DIR}/04_feature_importance.png",
    f"Feature Importance — {best_model_name}",
    figsize=(10, 12),
)

print("\n" + "=" * 70)
print("  %80/%20 SPLIT KARŞILAŞTIRMASI")
print("=" * 70)
X_train80, X_test20, y_train80, y_test20 = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"  Train: {X_train80.shape[0]} satır | Test: {X_test20.shape[0]} satır")
print("  Tüm modeller %80/%20 üzerinde değerlendiriliyor...")
print_holdout_comparison(
    optimized_models,
    X_train80,
    X_test20,
    y_train80,
    y_test20,
    weighted_fit_models=SAMPLE_WEIGHT_MODELS,
)

print("\n" + "=" * 70)
print("  MODELLEME v2 TAMAMLANDI — OULAD")
print("=" * 70)
