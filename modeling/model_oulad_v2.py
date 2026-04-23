import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = (12, 6)
sns.set_style("whitegrid")

output_dir = "modeling/plots_oulad_v2"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv("preprocessing/oulad_processed.csv")
X = df.drop('target', axis=1)
y = df['target']

target_names = ['Withdrawn', 'Fail', 'Pass']

print("=" * 70)
print("  MODELLEME v2 — OULAD (Feature Eng. + XGBoost)")
print("  Pipeline yapısı: MinMaxScaler + Model birlikte")
print("=" * 70)

# ============================================================
# FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 70)
print("  FEATURE ENGINEERING")
print("=" * 70)

print(f"  Orijinal özellik sayısı: {X.shape[1]}")

X['score_per_assessment'] = X['avg_score'] * X['num_assessments']
X['click_per_day'] = X['total_clicks'] / (X['total_vle_days'] + 0.001)
X['assessment_completion_rate'] = X['num_assessments'] / (X['num_TMA'] + X['num_CMA'] + X['num_Exam'] + 0.001)
X['forum_ratio'] = X['clicks_forumng'] / (X['total_clicks'] + 0.001)
X['quiz_ratio'] = X['clicks_quiz'] / (X['total_clicks'] + 0.001)
X['resource_ratio'] = X['clicks_resource'] / (X['total_clicks'] + 0.001)
X['score_consistency'] = X['avg_score'] / (X['std_score'] + 0.001)
X['early_late_ratio'] = X['early_submissions'] / (X['late_submissions'] + 0.001)
X['tma_cma_score_diff'] = X['avg_score_TMA'] - X['avg_score_CMA']
X['engagement_score'] = X['total_clicks'] * X['total_vle_days'] * X['num_distinct_activities']

new_cols = ['score_per_assessment', 'click_per_day', 'assessment_completion_rate',
            'forum_ratio', 'quiz_ratio', 'resource_ratio', 'score_consistency',
            'early_late_ratio', 'tma_cma_score_diff', 'engagement_score']

print(f"  Yeni özellik sayısı: {X.shape[1]} (+{len(new_cols)} türetilmiş)")
for col in new_cols:
    print(f"    - {col}")

# ============================================================
# TRAIN/TEST SPLIT
# ============================================================
print("\n" + "=" * 70)
print("  TRAIN/TEST SPLIT")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

print(f"  Train: {X_train.shape[0]} satır")
print(f"  Test:  {X_test.shape[0]} satır")

# ============================================================
# FEATURE SELECTION — MI (raw train verisi üzerinde, Pipeline dışında)
# ============================================================
print("\n" + "=" * 70)
print("  FEATURE SELECTION — Mutual Information (raw train, Pipeline dışında)")
print("=" * 70)

mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
mi_df = pd.DataFrame({'Özellik': X_train.columns, 'MI': mi_scores}).sort_values('MI', ascending=False)

print(f"\n  MI Skorları (Top 15):")
for _, row in mi_df.head(15).iterrows():
    bar = "█" * int(row['MI'] * 50)
    print(f"    {row['Özellik']:35s} MI={row['MI']:.4f} {bar}")

low_mi = mi_df[mi_df['MI'] < 0.01]['Özellik'].tolist()
if low_mi:
    print(f"\n  MI < 0.01 olan {len(low_mi)} özellik çıkarılıyor: {low_mi}")
    X_train = X_train.drop(columns=low_mi)
    X_test = X_test.drop(columns=low_mi)
    print(f"  Kalan özellik sayısı: {X_train.shape[1]}")
else:
    print(f"\n  Tüm özellikler MI >= 0.01, çıkarılan yok.")

# ============================================================
# PIPELINE TABANLI HİPERPARAMETRE OPTİMİZASYONU
# ============================================================
print("\n" + "=" * 70)
print("  HİPERPARAMETRE OPTİMİZASYONU (Pipeline + GridSearchCV)")
print("  Scoring: f1_macro — her sınıfa eşit ağırlık")
print("  Scaler her CV fold'unda sadece training kısmından fit edilir")
print("=" * 70)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
sample_weights = compute_sample_weight('balanced', y_train)

print("\n  [kNN]")
knn_pipe = Pipeline([('scaler', MinMaxScaler()), ('model', KNeighborsClassifier())])
knn_params = {
    'model__n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15],
    'model__metric': ['euclidean', 'manhattan']
}
knn_grid = GridSearchCV(knn_pipe, knn_params, cv=kf, scoring='f1_macro', n_jobs=-1)
knn_grid.fit(X_train, y_train)
print(f"  En iyi: {knn_grid.best_params_} → CV F1 (macro): %{knn_grid.best_score_*100:.2f}")

print("\n  [Naive Bayes]")
nb_pipe = Pipeline([('scaler', MinMaxScaler()), ('model', GaussianNB())])
nb_pipe.fit(X_train, y_train)
nb_cv = cross_val_score(nb_pipe, X_train, y_train, cv=kf, scoring='f1_macro')
print(f"  CV F1 (macro): %{nb_cv.mean()*100:.2f}")

print("\n  [Decision Tree]")
dt_pipe = Pipeline([('scaler', MinMaxScaler()), ('model', DecisionTreeClassifier(random_state=42, class_weight='balanced'))])
dt_params = {
    'model__max_depth': [3, 5, 7, 10, 15, None],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 5]
}
dt_grid = GridSearchCV(dt_pipe, dt_params, cv=kf, scoring='f1_macro', n_jobs=-1)
dt_grid.fit(X_train, y_train)
print(f"  En iyi: {dt_grid.best_params_} → CV F1 (macro): %{dt_grid.best_score_*100:.2f}")

print("\n  [Random Forest]")
rf_pipe = Pipeline([('scaler', MinMaxScaler()), ('model', RandomForestClassifier(random_state=42, class_weight='balanced'))])
rf_params = {
    'model__n_estimators': [50, 100, 200, 300],
    'model__max_depth': [5, 10, 15, 20, None],
    'model__min_samples_split': [2, 5, 10]
}
rf_grid = GridSearchCV(rf_pipe, rf_params, cv=kf, scoring='f1_macro', n_jobs=-1)
rf_grid.fit(X_train, y_train)
print(f"  En iyi: {rf_grid.best_params_} → CV F1 (macro): %{rf_grid.best_score_*100:.2f}")

print("\n  [XGBoost]")
xgb_pipe = Pipeline([('scaler', MinMaxScaler()), ('model', XGBClassifier(random_state=42, eval_metric='mlogloss', verbosity=0))])
xgb_params = {
    'model__n_estimators': [50, 100, 200, 300],
    'model__max_depth': [3, 5, 7, 10],
    'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'model__subsample': [0.8, 1.0]
}
xgb_grid = GridSearchCV(xgb_pipe, xgb_params, cv=kf, scoring='f1_macro', n_jobs=-1)
xgb_grid.fit(X_train, y_train, model__sample_weight=sample_weights)
print(f"  En iyi: {xgb_grid.best_params_} → CV F1 (macro): %{xgb_grid.best_score_*100:.2f}")

# ============================================================
# 10-FOLD CROSS VALIDATION
# ============================================================
print("\n" + "=" * 70)
print("  10-FOLD CROSS VALIDATION")
print("=" * 70)

kf10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

optimized_models = {
    "kNN": knn_grid.best_estimator_,
    "Naive Bayes": nb_pipe,
    "Decision Tree": dt_grid.best_estimator_,
    "Random Forest": rf_grid.best_estimator_,
    "XGBoost": xgb_grid.best_estimator_
}

cv_results_macro = {}
cv_results_weighted = {}
for name, pipe in optimized_models.items():
    if name == "XGBoost":
        macro_scores = []
        weighted_scores = []
        for train_idx, val_idx in kf10.split(X_train, y_train):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            fold_weights = compute_sample_weight('balanced', y_fold_train)
            pipe_clone = clone(pipe)
            pipe_clone.fit(X_fold_train, y_fold_train, model__sample_weight=fold_weights)
            y_fold_pred = pipe_clone.predict(X_fold_val)
            macro_scores.append(f1_score(y_fold_val, y_fold_pred, average='macro', zero_division=0))
            weighted_scores.append(f1_score(y_fold_val, y_fold_pred, average='weighted', zero_division=0))
        scores_macro = np.array(macro_scores)
        scores_weighted = np.array(weighted_scores)
    else:
        scores_macro = cross_val_score(pipe, X_train, y_train, cv=kf10, scoring='f1_macro')
        scores_weighted = cross_val_score(pipe, X_train, y_train, cv=kf10, scoring='f1_weighted')
    cv_results_macro[name] = scores_macro
    cv_results_weighted[name] = scores_weighted
    print(f"  {name:20s} → Macro F1: %{scores_macro.mean()*100:.2f} (±{scores_macro.std()*100:.2f})  |  Weighted F1: %{scores_weighted.mean()*100:.2f}")

# ============================================================
# MODEL SEÇİMİ (CV skoruna göre — test seti kullanılmaz)
# ============================================================
print("\n" + "=" * 70)
print("  MODEL SEÇİMİ (10-Fold CV Macro F1 skoruna göre)")
print("=" * 70)

cv_mean_macro = {name: scores.mean() for name, scores in cv_results_macro.items()}
best_model_name = max(cv_mean_macro, key=cv_mean_macro.get)
best_cv_macro = cv_mean_macro[best_model_name]
best_cv_weighted = cv_results_weighted[best_model_name].mean()

print(f"\n  En iyi model (CV Macro F1): {best_model_name}")
print(f"  CV Macro F1:    %{best_cv_macro*100:.2f}")
print(f"  CV Weighted F1: %{best_cv_weighted*100:.2f}")

for name, scores in cv_results_macro.items():
    marker = "  ← seçildi" if name == best_model_name else ""
    print(f"    {name:20s} CV Macro: %{scores.mean()*100:.2f}{marker}")

fig, ax = plt.subplots(figsize=(10, 6))
cv_data = pd.DataFrame(cv_results_macro)
cv_data.boxplot(ax=ax)
ax.set_title('10-Fold CV (Macro F1) — OULAD v2', fontsize=13, fontweight='bold')
ax.set_ylabel('F1-Macro')
plt.tight_layout()
plt.savefig(f"{output_dir}/01_cv_karsilastirma.png", dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# TEST SETİ — FİNAL DEĞERLENDİRME (sadece seçilen model, bir kez)
# ============================================================
print("\n" + "=" * 70)
print("  TEST SETİ — FİNAL DEĞERLENDİRME")
print(f"  Model: {best_model_name} (CV ile seçildi, test seti sadece raporlama)")
print("=" * 70)

best_pipeline = optimized_models[best_model_name]
y_pred_best = best_pipeline.predict(X_test)

test_acc = accuracy_score(y_test, y_pred_best)
test_prec = precision_score(y_test, y_pred_best, average='weighted', zero_division=0)
test_rec = recall_score(y_test, y_pred_best, average='weighted', zero_division=0)
best_f1_weighted = f1_score(y_test, y_pred_best, average='weighted', zero_division=0)
best_f1_macro = f1_score(y_test, y_pred_best, average='macro', zero_division=0)

print(f"\n  Accuracy:    %{test_acc*100:.2f}")
print(f"  Precision:   %{test_prec*100:.2f}")
print(f"  Recall:      %{test_rec*100:.2f}")
print(f"  F1-Weighted: %{best_f1_weighted*100:.2f}")
print(f"  F1-Macro:    %{best_f1_macro*100:.2f}")

# ============================================================
# SINIF BAZLI METRİKLER (Class-wise)
# ============================================================
print("\n" + "=" * 70)
print("  SINIF BAZLI METRİKLER")
print("=" * 70)

print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred_best, target_names=target_names, digits=4))

withdrawn_recall = recall_score(y_test, y_pred_best, average=None)[0]
print(f"\n  ⚠️  Withdrawn Recall: %{withdrawn_recall*100:.2f}")
print(f"      → Gerçek bırakma öğrencilerinin %{withdrawn_recall*100:.1f}'ini yakalıyoruz")

# ============================================================
# CONFUSION MATRIX (sadece seçilen model)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=target_names, yticklabels=target_names)
ax.set_title(f'Confusion Matrix — {best_model_name} (OULAD v2)', fontsize=13, fontweight='bold')
ax.set_xlabel('Tahmin')
ax.set_ylabel('Gerçek')
plt.tight_layout()
plt.savefig(f"{output_dir}/02_confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# CV KARŞILAŞTIRMA TABLOSU
# ============================================================
print("\n" + "=" * 70)
print("  CV KARŞILAŞTIRMA TABLOSU")
print("=" * 70)

cv_summary = []
for name in optimized_models:
    cv_summary.append({
        "Model": name,
        "CV Macro F1": f"%{cv_results_macro[name].mean()*100:.2f}",
        "CV Weighted F1": f"%{cv_results_weighted[name].mean()*100:.2f}",
    })
cv_summary_df = pd.DataFrame(cv_summary)
print(f"\n{cv_summary_df.to_string(index=False)}")

print(f"\n  Seçilen model: {best_model_name} (CV Macro F1: %{best_cv_macro*100:.2f})")
print(f"  Test Macro F1:    %{best_f1_macro*100:.2f}")
print(f"  Test Weighted F1: %{best_f1_weighted*100:.2f}")

print("\n" + "=" * 70)
print("  V2-ÖNCEKI vs V2-GÜNCEL KARŞILAŞTIRMA")
print("=" * 70)
print(f"\n  Önceki (f1_weighted, class_weight yok): XGBoost → Weighted F1: %80.48")
print(f"  Güncel (f1_macro, class_weight balanced): {best_model_name} → Weighted F1: %{best_f1_weighted*100:.2f}, Macro F1: %{best_f1_macro*100:.2f}")

# ============================================================
# EN İYİ MODELİ KAYDET (Pipeline: scaler + model birlikte)
# ============================================================
print("\n" + "=" * 70)
print("  EN İYİ MODELİ KAYDET (Pipeline)")
print("=" * 70)

best_pipeline = optimized_models[best_model_name]
model_path = "models/best_model_oulad.pkl"
joblib.dump(best_pipeline, model_path)
print(f"  Model: {best_model_name} (Pipeline: MinMaxScaler + {best_model_name})")
print(f"  F1-Macro:    %{best_f1_macro*100:.2f}")
print(f"  F1-Weighted: %{best_f1_weighted*100:.2f}")
print(f"  Kaydedildi: {model_path}")

actual_model = best_pipeline.named_steps['model']
if hasattr(actual_model, 'feature_importances_'):
    fi = pd.DataFrame({
        'Özellik': X_train.columns,
        'Önem': actual_model.feature_importances_
    }).sort_values('Önem', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 12))
    ax.barh(fi['Özellik'], fi['Önem'], color='steelblue', edgecolor='black')
    ax.set_title(f'Feature Importance — {best_model_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Önem')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/04_feature_importance.png", dpi=150, bbox_inches='tight')
    plt.close()

print("\n" + "=" * 70)
print("  MODELLEME v2 TAMAMLANDI — OULAD")
print("=" * 70)
