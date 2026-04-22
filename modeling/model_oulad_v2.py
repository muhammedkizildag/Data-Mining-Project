import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
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
print("  Scaler her CV fold'unda sadece training kısmından fit edilir")
print("=" * 70)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n  [kNN]")
knn_pipe = Pipeline([('scaler', MinMaxScaler()), ('model', KNeighborsClassifier())])
knn_params = {
    'model__n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15],
    'model__metric': ['euclidean', 'manhattan']
}
knn_grid = GridSearchCV(knn_pipe, knn_params, cv=kf, scoring='f1_weighted', n_jobs=-1)
knn_grid.fit(X_train, y_train)
print(f"  En iyi: {knn_grid.best_params_} → CV F1: %{knn_grid.best_score_*100:.2f}")

print("\n  [Naive Bayes]")
nb_pipe = Pipeline([('scaler', MinMaxScaler()), ('model', GaussianNB())])
nb_pipe.fit(X_train, y_train)
nb_cv = cross_val_score(nb_pipe, X_train, y_train, cv=kf, scoring='f1_weighted')
print(f"  CV F1: %{nb_cv.mean()*100:.2f}")

print("\n  [Decision Tree]")
dt_pipe = Pipeline([('scaler', MinMaxScaler()), ('model', DecisionTreeClassifier(random_state=42))])
dt_params = {
    'model__max_depth': [3, 5, 7, 10, 15, None],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 5]
}
dt_grid = GridSearchCV(dt_pipe, dt_params, cv=kf, scoring='f1_weighted', n_jobs=-1)
dt_grid.fit(X_train, y_train)
print(f"  En iyi: {dt_grid.best_params_} → CV F1: %{dt_grid.best_score_*100:.2f}")

print("\n  [Random Forest]")
rf_pipe = Pipeline([('scaler', MinMaxScaler()), ('model', RandomForestClassifier(random_state=42))])
rf_params = {
    'model__n_estimators': [50, 100, 200, 300],
    'model__max_depth': [5, 10, 15, 20, None],
    'model__min_samples_split': [2, 5, 10]
}
rf_grid = GridSearchCV(rf_pipe, rf_params, cv=kf, scoring='f1_weighted', n_jobs=-1)
rf_grid.fit(X_train, y_train)
print(f"  En iyi: {rf_grid.best_params_} → CV F1: %{rf_grid.best_score_*100:.2f}")

print("\n  [XGBoost]")
xgb_pipe = Pipeline([('scaler', MinMaxScaler()), ('model', XGBClassifier(random_state=42, eval_metric='mlogloss', verbosity=0))])
xgb_params = {
    'model__n_estimators': [50, 100, 200, 300],
    'model__max_depth': [3, 5, 7, 10],
    'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'model__subsample': [0.8, 1.0]
}
xgb_grid = GridSearchCV(xgb_pipe, xgb_params, cv=kf, scoring='f1_weighted', n_jobs=-1)
xgb_grid.fit(X_train, y_train)
print(f"  En iyi: {xgb_grid.best_params_} → CV F1: %{xgb_grid.best_score_*100:.2f}")

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

cv_results = {}
for name, pipe in optimized_models.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=kf10, scoring='f1_weighted')
    cv_results[name] = scores
    print(f"  {name:20s} → Ort F1: %{scores.mean()*100:.2f} (±{scores.std()*100:.2f})")

fig, ax = plt.subplots(figsize=(10, 6))
cv_data = pd.DataFrame(cv_results)
cv_data.boxplot(ax=ax)
ax.set_title('10-Fold CV Sonuçları — OULAD v2 (FE + XGBoost)', fontsize=13, fontweight='bold')
ax.set_ylabel('F1-Score')
plt.tight_layout()
plt.savefig(f"{output_dir}/01_cv_karsilastirma.png", dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# TEST SETİ DEĞERLENDİRMESİ
# ============================================================
print("\n" + "=" * 70)
print("  TEST SETİ DEĞERLENDİRMESİ")
print("=" * 70)

final_results = []

for name, pipe in optimized_models.items():
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    final_results.append({
        "Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1
    })

    print(f"\n  --- {name} ---")
    print(f"  Accuracy:  %{acc*100:.2f}")
    print(f"  Precision: %{prec*100:.2f}")
    print(f"  Recall:    %{rec*100:.2f}")
    print(f"  F1-Score:  %{f1*100:.2f}")

# ============================================================
# SINIF BAZLI METRİKLER (Class-wise)
# ============================================================
print("\n" + "=" * 70)
print("  SINIF BAZLI METRİKLER")
print("=" * 70)

best_temp_name = pd.DataFrame(final_results).loc[pd.DataFrame(final_results)['F1-Score'].idxmax(), 'Model']
best_temp_pipe = optimized_models[best_temp_name]
y_pred_best = best_temp_pipe.predict(X_test)

print(f"\n  En iyi model: {best_temp_name}")
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred_best, target_names=target_names, digits=4))

f1_macro = f1_score(y_test, y_pred_best, average='macro')
print(f"  Weighted F1: %{f1_score(y_test, y_pred_best, average='weighted')*100:.2f}")
print(f"  Macro F1:    %{f1_macro*100:.2f}")

withdrawn_recall = recall_score(y_test, y_pred_best, average=None)[0]
print(f"\n  ⚠️  Withdrawn Recall: %{withdrawn_recall*100:.2f}")
print(f"      → Gerçek bırakma öğrencilerinin %{withdrawn_recall*100:.1f}'ini yakalıyoruz")

# ============================================================
# CONFUSION MATRIX
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, (name, pipe) in enumerate(optimized_models.items()):
    y_pred = pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=target_names, yticklabels=target_names)
    axes[i].set_title(f'{name}', fontsize=13, fontweight='bold')
    axes[i].set_xlabel('Tahmin')
    axes[i].set_ylabel('Gerçek')
axes[5].axis('off')

plt.suptitle('Confusion Matrix — OULAD v2', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{output_dir}/02_confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# KARŞILAŞTIRMA
# ============================================================
print("\n" + "=" * 70)
print("  KARŞILAŞTIRMA TABLOSU")
print("=" * 70)

results_df = pd.DataFrame(final_results)
results_df_display = results_df.copy()
for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
    results_df_display[col] = results_df_display[col].apply(lambda x: f"%{x*100:.2f}")
print(f"\n{results_df_display.to_string(index=False)}")

best_model_name = results_df.loc[results_df['F1-Score'].idxmax(), 'Model']
best_f1 = results_df['F1-Score'].max()
print(f"\n  En iyi model (F1-Score): {best_model_name} (%{best_f1*100:.2f})")

print("\n" + "=" * 70)
print("  V1 vs V2 KARŞILAŞTIRMA")
print("=" * 70)
print(f"\n  V1 en iyi: Random Forest → F1: %94.56")
print(f"  V2 en iyi: {best_model_name} → F1: %{best_f1*100:.2f}")
print(f"  Fark: {(best_f1*100 - 94.56):+.2f} puan")

fig, ax = plt.subplots(figsize=(14, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(results_df))
width = 0.18
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

for i, metric in enumerate(metrics):
    bars = ax.bar(x + i*width, results_df[metric]*100, width, label=metric, color=colors[i], edgecolor='black')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{height:.1f}', ha='center', va='bottom', fontsize=7)

ax.set_xticks(x + width*1.5)
ax.set_xticklabels(results_df['Model'])
ax.set_ylabel('Skor (%)')
ax.set_title('Model Karşılaştırması — OULAD v2 (FE + XGBoost)', fontsize=13, fontweight='bold')
ax.legend()
ax.set_ylim(0, 110)
plt.tight_layout()
plt.savefig(f"{output_dir}/03_model_karsilastirma.png", dpi=150, bbox_inches='tight')
plt.close()

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
print(f"  F1-Score: %{best_f1*100:.2f}")
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
