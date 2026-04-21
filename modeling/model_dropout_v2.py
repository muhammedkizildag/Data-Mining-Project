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
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = (12, 6)
sns.set_style("whitegrid")

output_dir = "modeling/plots_dropout_v2"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv("preprocessing/dropout_processed.csv")
X = df.drop('Target', axis=1)
y = df['Target']

target_names = ['Dropout', 'Enrolled', 'Graduate']

print("=" * 70)
print("  MODELLEME v2 — Dropout UCI (Feature Eng. + XGBoost)")
print("=" * 70)

# ============================================================
# FEATURE ENGINEERING
# ============================================================
print("\n" + "=" * 70)
print("  FEATURE ENGINEERING")
print("=" * 70)

print(f"  Orijinal özellik sayısı: {X.shape[1]}")

X['sem1_success_rate'] = X['Curricular units 1st sem (approved)'] / (X['Curricular units 1st sem (enrolled)'] + 0.001)
X['sem2_success_rate'] = X['Curricular units 2nd sem (approved)'] / (X['Curricular units 2nd sem (enrolled)'] + 0.001)
X['total_approved'] = X['Curricular units 1st sem (approved)'] + X['Curricular units 2nd sem (approved)']
X['total_grade'] = X['Curricular units 1st sem (grade)'] + X['Curricular units 2nd sem (grade)']
X['grade_improvement'] = X['Curricular units 2nd sem (grade)'] - X['Curricular units 1st sem (grade)']
X['approved_improvement'] = X['Curricular units 2nd sem (approved)'] - X['Curricular units 1st sem (approved)']
X['eval_approved_ratio_1'] = X['Curricular units 1st sem (approved)'] / (X['Curricular units 1st sem (evaluations)'] + 0.001)
X['eval_approved_ratio_2'] = X['Curricular units 2nd sem (approved)'] / (X['Curricular units 2nd sem (evaluations)'] + 0.001)

from sklearn.preprocessing import MinMaxScaler
new_cols = ['sem1_success_rate', 'sem2_success_rate', 'total_approved', 'total_grade',
            'grade_improvement', 'approved_improvement', 'eval_approved_ratio_1', 'eval_approved_ratio_2']
scaler = MinMaxScaler()
X[new_cols] = scaler.fit_transform(X[new_cols])

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
# HİPERPARAMETRE OPTİMİZASYONU
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
knn_grid.fit(X_train, y_train)
print(f"  En iyi: {knn_grid.best_params_} → CV F1: %{knn_grid.best_score_*100:.2f}")

# --- Naive Bayes ---
print("\n  [Naive Bayes]")
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_cv = cross_val_score(nb_model, X_train, y_train, cv=kf, scoring='f1_weighted')
print(f"  CV F1: %{nb_cv.mean()*100:.2f}")

# --- Decision Tree ---
print("\n  [Decision Tree]")
dt_params = {
    'max_depth': [3, 5, 7, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}
dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_params, cv=kf, scoring='f1_weighted', n_jobs=-1)
dt_grid.fit(X_train, y_train)
print(f"  En iyi: {dt_grid.best_params_} → CV F1: %{dt_grid.best_score_*100:.2f}")

# --- Random Forest ---
print("\n  [Random Forest]")
rf_params = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10]
}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=kf, scoring='f1_weighted', n_jobs=-1)
rf_grid.fit(X_train, y_train)
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
xgb_grid.fit(X_train, y_train)
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
    scores = cross_val_score(model, X_train, y_train, cv=kf10, scoring='f1_weighted')
    cv_results[name] = scores
    print(f"  {name:20s} → Ort F1: %{scores.mean()*100:.2f} (±{scores.std()*100:.2f})")

fig, ax = plt.subplots(figsize=(10, 6))
cv_data = pd.DataFrame(cv_results)
cv_data.boxplot(ax=ax)
ax.set_title('10-Fold CV Sonuçları — Dropout UCI v2 (FE + XGBoost)', fontsize=13, fontweight='bold')
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

plt.suptitle('Confusion Matrix — Dropout UCI v2', fontsize=15, fontweight='bold')
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
print(f"\n  V1 (önceki) en iyi: Random Forest → F1: %76.23")
print(f"  V2 (şimdi)  en iyi: {best_model_name} → F1: %{best_f1*100:.2f}")
print(f"  İyileşme: +{(best_f1*100 - 76.23):.2f} puan")

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
ax.set_title('Model Karşılaştırması — Dropout UCI v2 (FE + XGBoost)', fontsize=13, fontweight='bold')
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
model_path = "models/best_model_dropout.pkl"
joblib.dump(best_model, model_path)
print(f"  Model: {best_model_name}")
print(f"  F1-Score: %{best_f1*100:.2f}")
print(f"  Kaydedildi: {model_path}")

if hasattr(best_model, 'feature_importances_'):
    fi = pd.DataFrame({
        'Özellik': X.columns,
        'Önem': best_model.feature_importances_
    }).sort_values('Önem', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.barh(fi['Özellik'], fi['Önem'], color='steelblue', edgecolor='black')
    ax.set_title(f'Feature Importance — {best_model_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Önem')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/04_feature_importance.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Grafik: {output_dir}/04_feature_importance.png]")

print("\n" + "=" * 70)
print("  MODELLEME v2 TAMAMLANDI — Dropout UCI")
print("=" * 70)
