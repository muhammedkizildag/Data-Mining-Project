# ⚠️ UYARI: Bu dosya artık aktif değildir.
# Student Habits veri seti OULAD ile değiştirilmiştir.
# Güncel OULAD modeli: model_oulad_v2.py
# Sadece geçmiş karşılaştırma için saklanmaktadır.

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = (12, 6)
sns.set_style("whitegrid")

output_dir = "modeling/plots_habits_v2"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv("preprocessing/habits_processed.csv")
X = df.drop('risk_level', axis=1)
y = df['risk_level']

target_names = ['Düşük', 'Orta', 'Yüksek']

print("=" * 70)
print("  MODELLEME v2 — Student Habits (SMOTE + Feature Eng. + XGBoost)")
print("=" * 70)

# ============================================================
# FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 70)
print("  FEATURE ENGINEERING")
print("=" * 70)

print(f"  Orijinal özellik sayısı: {X.shape[1]}")

X['study_social_ratio'] = X['study_hours_per_day'] / (X['social_media_hours'] + 0.01)
X['study_netflix_ratio'] = X['study_hours_per_day'] / (X['netflix_hours'] + 0.01)
X['screen_time_total'] = X['social_media_hours'] + X['netflix_hours']
X['sleep_mental_interaction'] = X['sleep_hours'] * X['mental_health_rating']
X['study_mental_interaction'] = X['study_hours_per_day'] * X['mental_health_rating']
X['healthy_lifestyle'] = X['sleep_hours'] + X['exercise_frequency'] + X['diet_quality']
X['study_attendance_interaction'] = X['study_hours_per_day'] * X['attendance_percentage']

# Normalize new features (0-1)
from sklearn.preprocessing import MinMaxScaler
new_cols = ['study_social_ratio', 'study_netflix_ratio', 'screen_time_total',
            'sleep_mental_interaction', 'study_mental_interaction',
            'healthy_lifestyle', 'study_attendance_interaction']
scaler = MinMaxScaler()
X[new_cols] = scaler.fit_transform(X[new_cols])

print(f"  Yeni özellik sayısı: {X.shape[1]} (+{len(new_cols)} türetilmiş)")
print(f"  Eklenen özellikler:")
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
# SMOTE
# ============================================================
print("\n" + "=" * 70)
print("  SMOTE (Sentetik Azınlık Örnekleme)")
print("=" * 70)

print(f"\n  SMOTE öncesi sınıf dağılımı (Train):")
for val, name in enumerate(target_names):
    cnt = sum(y_train == val)
    print(f"    {name}: {cnt}")

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"\n  SMOTE sonrası sınıf dağılımı (Train):")
for val, name in enumerate(target_names):
    cnt = sum(y_train_smote == val)
    print(f"    {name}: {cnt}")

print(f"\n  Train boyutu: {X_train.shape[0]} → {X_train_smote.shape[0]}")

# ============================================================
# SMOTE'SIZ SONUÇLAR (KARŞILAŞTIRMA İÇİN)
# ============================================================
print("\n" + "=" * 70)
print("  KARŞILAŞTIRMA: SMOTE'siz vs SMOTE'lu (Önceki parametrelerle)")
print("=" * 70)

# Basit karşılaştırma
for label, X_tr, y_tr in [("SMOTE'siz", X_train, y_train), ("SMOTE'lu", X_train_smote, y_train_smote)]:
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"  {label:12s} → Random Forest F1: %{f1*100:.2f}")

# ============================================================
# HİPERPARAMETRE OPTİMİZASYONU (SMOTE'lu veri ile)
# ============================================================
print("\n" + "=" * 70)
print("  HİPERPARAMETRE OPTİMİZASYONU (GridSearchCV)")
print("=" * 70)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# --- kNN ---
print("\n  [kNN]")
knn_params = {
    'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15],
    'metric': ['euclidean', 'manhattan']
}
knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=kf, scoring='f1_weighted', n_jobs=-1)
knn_grid.fit(X_train_smote, y_train_smote)
print(f"  En iyi: {knn_grid.best_params_} → CV F1: %{knn_grid.best_score_*100:.2f}")

# --- Naive Bayes ---
print("\n  [Naive Bayes]")
nb_model = GaussianNB()
nb_model.fit(X_train_smote, y_train_smote)
nb_cv = cross_val_score(nb_model, X_train_smote, y_train_smote, cv=kf, scoring='f1_weighted')
print(f"  CV F1: %{nb_cv.mean()*100:.2f}")

# --- Decision Tree ---
print("\n  [Decision Tree]")
dt_params = {
    'max_depth': [3, 5, 7, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}
dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_params, cv=kf, scoring='f1_weighted', n_jobs=-1)
dt_grid.fit(X_train_smote, y_train_smote)
print(f"  En iyi: {dt_grid.best_params_} → CV F1: %{dt_grid.best_score_*100:.2f}")

# --- Random Forest ---
print("\n  [Random Forest]")
rf_params = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10]
}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=kf, scoring='f1_weighted', n_jobs=-1)
rf_grid.fit(X_train_smote, y_train_smote)
print(f"  En iyi: {rf_grid.best_params_} → CV F1: %{rf_grid.best_score_*100:.2f}")

# --- XGBoost ---
print("\n  [XGBoost]")
xgb_params = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}
xgb_grid = GridSearchCV(
    XGBClassifier(random_state=42, eval_metric='mlogloss', verbosity=0),
    xgb_params, cv=kf, scoring='f1_weighted', n_jobs=-1
)
xgb_grid.fit(X_train_smote, y_train_smote)
print(f"  En iyi: {xgb_grid.best_params_} → CV F1: %{xgb_grid.best_score_*100:.2f}")

# ============================================================
# 10-FOLD CROSS VALIDATION
# ============================================================
print("\n" + "=" * 70)
print("  10-FOLD CROSS VALIDATION")
print("=" * 70)

kf10 = KFold(n_splits=10, shuffle=True, random_state=42)

optimized_models = {
    "kNN": knn_grid.best_estimator_,
    "Naive Bayes": nb_model,
    "Decision Tree": dt_grid.best_estimator_,
    "Random Forest": rf_grid.best_estimator_,
    "XGBoost": xgb_grid.best_estimator_
}

cv_results = {}
for name, model in optimized_models.items():
    scores = cross_val_score(model, X_train_smote, y_train_smote, cv=kf10, scoring='f1_weighted')
    cv_results[name] = scores
    print(f"  {name:20s} → Ort F1: %{scores.mean()*100:.2f} (±{scores.std()*100:.2f})")

fig, ax = plt.subplots(figsize=(10, 6))
cv_data = pd.DataFrame(cv_results)
cv_data.boxplot(ax=ax)
ax.set_title('10-Fold CV Sonuçları — Student Habits v2 (SMOTE + FE + XGBoost)', fontsize=13, fontweight='bold')
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

for name, model in optimized_models.items():
    y_pred = model.predict(X_test)
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
# CONFUSION MATRIX
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, (name, model) in enumerate(optimized_models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=target_names, yticklabels=target_names)
    axes[i].set_title(f'{name}', fontsize=13, fontweight='bold')
    axes[i].set_xlabel('Tahmin')
    axes[i].set_ylabel('Gerçek')
axes[5].axis('off')

plt.suptitle('Confusion Matrix — Student Habits v2', fontsize=15, fontweight='bold')
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

# V1 vs V2 karşılaştırma
print("\n" + "=" * 70)
print("  V1 vs V2 KARŞILAŞTIRMA")
print("=" * 70)
print(f"\n  V1 (önceki) en iyi: Naive Bayes → F1: %75.19")
print(f"  V2 (şimdi)  en iyi: {best_model_name} → F1: %{best_f1*100:.2f}")
print(f"  İyileşme: +{(best_f1*100 - 75.19):.2f} puan")

# Grafik
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
ax.set_title('Model Karşılaştırması — Student Habits v2 (SMOTE + FE + XGBoost)', fontsize=13, fontweight='bold')
ax.legend()
ax.set_ylim(0, 110)
plt.tight_layout()
plt.savefig(f"{output_dir}/03_model_karsilastirma.png", dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# EN İYİ MODELİ KAYDET
# ============================================================
print("\n" + "=" * 70)
print("  EN İYİ MODELİ KAYDET")
print("=" * 70)

best_model = optimized_models[best_model_name]
model_path = "models/best_model_habits.pkl"
joblib.dump(best_model, model_path)
print(f"  Model: {best_model_name}")
print(f"  F1-Score: %{best_f1*100:.2f}")
print(f"  Kaydedildi: {model_path}")

if hasattr(best_model, 'feature_importances_'):
    fi = pd.DataFrame({
        'Özellik': X.columns,
        'Önem': best_model.feature_importances_
    }).sort_values('Önem', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(fi['Özellik'], fi['Önem'], color='steelblue', edgecolor='black')
    ax.set_title(f'Feature Importance — {best_model_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Önem')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/04_feature_importance.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Grafik: {output_dir}/04_feature_importance.png]")

print("\n" + "=" * 70)
print("  MODELLEME v2 TAMAMLANDI — Student Habits")
print("=" * 70)
