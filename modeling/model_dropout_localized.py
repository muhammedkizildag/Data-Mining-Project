import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from modeling.common import (
    build_cv_summary,
    ensure_dir,
    evaluate_models_with_cv,
    plot_learning_curve_custom,
    plot_multiclass_roc_pr,
    save_confusion_matrix,
    save_cv_boxplot,
    save_feature_importance,
    select_best_model,
)

OUTPUT_DIR = "modeling/plots_dropout_localized"
MODEL_DIR = "models"
CV_N_JOBS = -1
ensure_dir(OUTPUT_DIR)
ensure_dir(MODEL_DIR)

TARGET_NAMES = ["Dropout", "Enrolled", "Graduate"]
REMOVE_COLS = ["Tuition fees up to date", "Debtor", "Inflation rate"]
FEATURE_MAP = {
    "Marital status": "Medeni durum",
    "Application mode": "Başvuru türü (YKS, DGS, Yatay Geçiş)",
    "Application order": "Tercih sırası",
    "Course": "Bölüm",
    "Previous qualification": "Önceki eğitim düzeyi (Lise, Önlisans...)",
    "Previous qualification (grade)": "Önceki eğitim not ortalaması",
    "Mother's qualification": "Anne eğitim düzeyi",
    "Father's qualification": "Baba eğitim düzeyi",
    "Mother's occupation": "Anne mesleği",
    "Father's occupation": "Baba mesleği",
    "Admission grade": "Üniversite giriş puanı (YKS)",
    "Gender": "Cinsiyet",
    "Scholarship holder": "Burs durumu",
    "Age at enrollment": "Kayıt yaşı",
    "Curricular units 1st sem (enrolled)": "1. dönem alınan ders sayısı",
    "Curricular units 1st sem (evaluations)": "1. dönem girilen sınav sayısı",
    "Curricular units 1st sem (approved)": "1. dönem geçilen ders sayısı",
    "Curricular units 1st sem (grade)": "1. dönem not ortalaması",
    "Curricular units 2nd sem (enrolled)": "2. dönem alınan ders sayısı",
    "Curricular units 2nd sem (evaluations)": "2. dönem girilen sınav sayısı",
    "Curricular units 2nd sem (approved)": "2. dönem geçilen ders sayısı",
    "Curricular units 2nd sem (grade)": "2. dönem not ortalaması",
}


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

    return {
        "kNN": knn_grid.best_estimator_,
        "Naive Bayes": nb_pipe,
        "Decision Tree": dt_grid.best_estimator_,
        "Random Forest": rf_grid.best_estimator_,
        "XGBoost": xgb_grid.best_estimator_,
    }


print("=" * 70)
print("  MODELLEME — Dropout UCI (Yerelleştirilmiş / Localized)")
print("  Pipeline yapısı: MinMaxScaler + Model birlikte")
print("=" * 70)

df = pd.read_csv("preprocessing/dropout_processed.csv")

print("\n" + "=" * 70)
print("  TÜRKİYE YERELLEŞTİRME — Özellik Çıkarma")
print("=" * 70)
print(f"\n  Çıkarılan özellikler ({len(REMOVE_COLS)} adet):")
for col in REMOVE_COLS:
    print(f"    ✗ {col}")

df = df.drop(columns=REMOVE_COLS)
X = df.drop("Target", axis=1)
y = df["Target"]

print(f"\n  Kalan özellik sayısı: {X.shape[1]}")
print(f"  Toplam satır: {X.shape[0]}")
print("\n  Kalan özellikler ve Türkiye karşılıkları:")
for orig, tr in FEATURE_MAP.items():
    if orig in X.columns:
        print(f"    {orig:<45s} → {tr}")

print("\n" + "=" * 70)
print("  TRAIN/TEST SPLIT")
print("=" * 70)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
print(f"  Train: {X_train.shape[0]} satır")
print(f"  Test:  {X_test.shape[0]} satır")

print("\n" + "=" * 70)
print("  CHATBOT ÖZELLİK FİLTRESİ")
print("=" * 70)
chatbot_features = list(FEATURE_MAP.keys())
extra_in_train = [c for c in X_train.columns if c not in chatbot_features]
if extra_in_train:
    print(f"  Chatbot'ta olmayan {len(extra_in_train)} özellik çıkarılıyor:")
    for col in extra_in_train:
        print(f"    ✗ {col}")
    X_train = X_train[chatbot_features]
    X_test = X_test[chatbot_features]
print(f"  Kalan özellik sayısı: {X_train.shape[1]} (chatbot ile uyumlu)")

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
    "10-Fold CV (Macro F1) — Dropout UCI (Yerelleştirilmiş)",
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
dropout_recall = recall_score(y_test, y_pred_best, average=None)[0]
print(f"\n  ⚠️  Dropout Recall: %{dropout_recall*100:.2f} (en kritik metrik)")
print(f"      → Gerçek dropout öğrencilerinin %{dropout_recall*100:.1f}'ini yakalıyoruz")

best_proba = best_pipeline.predict_proba(X_test)
plot_multiclass_roc_pr(
    y_test,
    best_proba,
    TARGET_NAMES,
    f"{OUTPUT_DIR}/03_roc_pr_curves.png",
    f"{best_model_name} (Yerelleştirilmiş)",
)
print(f"\n  ROC/PR grafiği kaydedildi: {OUTPUT_DIR}/03_roc_pr_curves.png")

plot_learning_curve_custom(
    best_pipeline,
    X_train,
    y_train,
    f"{OUTPUT_DIR}/05_learning_curve.png",
    f"Learning Curve — {best_model_name} (Yerelleştirilmiş)",
    TARGET_NAMES,
    use_sample_weight=best_model_name == "XGBoost",
)
print(f"  Learning curve kaydedildi: {OUTPUT_DIR}/05_learning_curve.png")

save_confusion_matrix(
    y_test,
    y_pred_best,
    TARGET_NAMES,
    f"{OUTPUT_DIR}/02_confusion_matrix.png",
    f"Confusion Matrix — {best_model_name} (Yerelleştirilmiş)",
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
print("  YERELLEŞTİRME ETKİSİ")
print("=" * 70)
print("\n  Önceki (f1_weighted, class_weight yok): XGBoost → Weighted F1: %75.10, Macro F1: %69.06")
print(
    f"  Güncel (f1_macro, class_weight balanced): {best_model_name} → Weighted F1: %{best_f1_weighted*100:.2f}, Macro F1: %{best_f1_macro*100:.2f}"
)
print("\n  Çıkarılan 3 özellik: Tuition fees, Debtor, Inflation rate")

print("\n" + "=" * 70)
print("  EN İYİ MODELİ KAYDET (Pipeline)")
print("=" * 70)
model_path = "models/best_model_dropout_localized.pkl"
joblib.dump(best_pipeline, model_path)
print(f"  Model: {best_model_name} (Pipeline: MinMaxScaler + {best_model_name})")
print(f"  F1-Macro:    %{best_f1_macro*100:.2f}")
print(f"  F1-Weighted: %{best_f1_weighted*100:.2f}")
print(f"  Özellik sayısı: {X_train.shape[1]}")
print(f"  Kaydedildi: {model_path}")

feature_list = list(X_train.columns)
joblib.dump(feature_list, "models/dropout_localized_features.pkl")
print("  Özellik listesi: models/dropout_localized_features.pkl")

trained_scaler = best_pipeline.named_steps["scaler"]
scaler_params = {}
for i, col in enumerate(X_train.columns):
    scaler_params[col] = {"min": float(trained_scaler.data_min_[i]), "max": float(trained_scaler.data_max_[i])}
with open("models/dropout_localized_scaler_params.json", "w", encoding="utf-8") as f:
    json.dump(scaler_params, f, ensure_ascii=False, indent=2)
print("  Scaler parametreleri (dokümantasyon): models/dropout_localized_scaler_params.json")

save_feature_importance(
    best_pipeline.named_steps["model"],
    X_train.columns,
    f"{OUTPUT_DIR}/04_feature_importance.png",
    f"Feature Importance — {best_model_name} (Yerelleştirilmiş)",
    figsize=(10, 8),
)

print("\n" + "=" * 70)
print("  YERELLEŞTİRME TAMAMLANDI")
print("=" * 70)
