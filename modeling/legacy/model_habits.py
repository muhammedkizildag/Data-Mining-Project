"""Legacy Student Habits modeli.

Aktif final modelleme akışı için bu dosya yerine güncel README ve OULAD/Dropout
script'leri kullanılmalıdır.
"""

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
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

from modeling.legacy_notice import print_legacy_notice

plt.rcParams['figure.figsize'] = (12, 6)
sns.set_style("whitegrid")

print_legacy_notice(
    script_name="model_habits.py",
    current_script="modeling/model_oulad_v2.py",
    reason="Student Habits hattı tarihsel karşılaştırma için tutulur; aktif final çıktı değildir.",
)

output_dir = "modeling/plots_habits"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv("preprocessing/habits_processed.csv")
X = df.drop('risk_level', axis=1)
y = df['risk_level']

target_names = ['Düşük', 'Orta', 'Yüksek']

print("=" * 70)
print("  MODELLEME — Student Habits")
print("=" * 70)
print(f"\n  Veri: {X.shape[0]} satır × {X.shape[1]} özellik")
print(f"  Sınıflar: {dict(zip(target_names, [sum(y==i) for i in range(3)]))}")

# ============================================================
# ADIM 1: TRAIN/TEST SPLIT
# ============================================================
print("\n" + "=" * 70)
print("  ADIM 1: Train/Test Split")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

print(f"  Train: {X_train.shape[0]} satır ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"  Test:  {X_test.shape[0]} satır ({X_test.shape[0]/len(X)*100:.0f}%)")
print(f"  Stratify: Evet (sınıf oranları korundu)")

# ============================================================
# ADIM 2: BASELINE MODELLER
# ============================================================
print("\n" + "=" * 70)
print("  ADIM 2: Baseline Modeller (Varsayılan Parametreler)")
print("=" * 70)

baseline_models = {
    "kNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

baseline_results = []
for name, model in baseline_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    baseline_results.append({"Model": name, "Baseline Accuracy": f"{acc*100:.2f}%"})
    print(f"  {name:25s} → Accuracy: %{acc*100:.2f}")

print(f"\n  (Bunlar referans değerler, şimdi optimize edeceğiz)")

# ============================================================
# ADIM 3: HİPERPARAMETRE OPTİMİZASYONU (GridSearchCV)
# ============================================================
print("\n" + "=" * 70)
print("  ADIM 3: Hiperparametre Optimizasyonu (GridSearchCV)")
print("=" * 70)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# --- kNN ---
print("\n  [kNN]")
knn_params = {
    'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15],
    'metric': ['euclidean', 'manhattan']
}
knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=kf, scoring='accuracy', n_jobs=-1)
knn_grid.fit(X_train, y_train)
print(f"  En iyi parametreler: {knn_grid.best_params_}")
print(f"  En iyi CV skoru: %{knn_grid.best_score_*100:.2f}")

# --- Naive Bayes ---
print("\n  [Naive Bayes]")
print(f"  Parametre yok — varsayılan kullanılacak")
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# --- Decision Tree ---
print("\n  [Decision Tree]")
dt_params = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5]
}
dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_params, cv=kf, scoring='accuracy', n_jobs=-1)
dt_grid.fit(X_train, y_train)
print(f"  En iyi parametreler: {dt_grid.best_params_}")
print(f"  En iyi CV skoru: %{dt_grid.best_score_*100:.2f}")

# --- Random Forest ---
print("\n  [Random Forest]")
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10]
}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=kf, scoring='accuracy', n_jobs=-1)
rf_grid.fit(X_train, y_train)
print(f"  En iyi parametreler: {rf_grid.best_params_}")
print(f"  En iyi CV skoru: %{rf_grid.best_score_*100:.2f}")

# ============================================================
# ADIM 4: 10-FOLD CROSS VALIDATION
# ============================================================
print("\n" + "=" * 70)
print("  ADIM 4: 10-Fold Cross Validation (Optimize Edilmiş Modeller)")
print("=" * 70)

kf10 = KFold(n_splits=10, shuffle=True, random_state=42)

optimized_models = {
    "kNN": knn_grid.best_estimator_,
    "Naive Bayes": nb_model,
    "Decision Tree": dt_grid.best_estimator_,
    "Random Forest": rf_grid.best_estimator_
}

cv_results = {}
for name, model in optimized_models.items():
    scores = cross_val_score(model, X_train, y_train, cv=kf10, scoring='accuracy')
    cv_results[name] = scores
    print(f"  {name:20s} → Ort: %{scores.mean()*100:.2f} (±{scores.std()*100:.2f})")

# CV sonuçları boxplot
fig, ax = plt.subplots(figsize=(10, 6))
cv_data = pd.DataFrame(cv_results)
cv_data.boxplot(ax=ax)
ax.set_title('10-Fold Cross Validation Sonuçları — Student Habits', fontsize=14, fontweight='bold')
ax.set_ylabel('Accuracy')
plt.tight_layout()
plt.savefig(f"{output_dir}/01_cv_karsilastirma.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  [Grafik: {output_dir}/01_cv_karsilastirma.png]")

# ============================================================
# ADIM 5: TEST SETİ DEĞERLENDİRMESİ
# ============================================================
print("\n" + "=" * 70)
print("  ADIM 5: Test Seti Değerlendirmesi")
print("=" * 70)

final_results = []

for name, model in optimized_models.items():
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    final_results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1
    })

    print(f"\n  --- {name} ---")
    print(f"  Accuracy:  %{acc*100:.2f}")
    print(f"  Precision: %{prec*100:.2f}")
    print(f"  Recall:    %{rec*100:.2f}")
    print(f"  F1-Score:  %{f1*100:.2f}")

# ============================================================
# ADIM 6: CONFUSION MATRIX
# ============================================================
print("\n" + "=" * 70)
print("  ADIM 6: Confusion Matrix")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for i, (name, model) in enumerate(optimized_models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=target_names, yticklabels=target_names)
    axes[i].set_title(f'{name}', fontsize=13, fontweight='bold')
    axes[i].set_xlabel('Tahmin')
    axes[i].set_ylabel('Gerçek')

plt.suptitle('Confusion Matrix — Student Habits', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{output_dir}/02_confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  [Grafik: {output_dir}/02_confusion_matrix.png]")

# ============================================================
# ADIM 6B: KARŞILAŞTIRMA TABLOSU VE GRAFİK
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

fig, ax = plt.subplots(figsize=(12, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(results_df))
width = 0.2
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

for i, metric in enumerate(metrics):
    bars = ax.bar(x + i*width, results_df[metric]*100, width, label=metric, color=colors[i], edgecolor='black')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{height:.1f}', ha='center', va='bottom', fontsize=8)

ax.set_xticks(x + width*1.5)
ax.set_xticklabels(results_df['Model'])
ax.set_ylabel('Skor (%)')
ax.set_title('Model Karşılaştırması — Student Habits', fontsize=14, fontweight='bold')
ax.legend()
ax.set_ylim(0, 110)
plt.tight_layout()
plt.savefig(f"{output_dir}/03_model_karsilastirma.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  [Grafik: {output_dir}/03_model_karsilastirma.png]")

# ============================================================
# ADIM 7: EN İYİ MODELİ KAYDET
# ============================================================
print("\n" + "=" * 70)
print("  ADIM 7: En İyi Modeli Kaydet")
print("=" * 70)

best_model = optimized_models[best_model_name]
model_path = "models/best_model_habits.pkl"
joblib.dump(best_model, model_path)
print(f"  Model: {best_model_name}")
print(f"  F1-Score: %{best_f1*100:.2f}")
print(f"  Kaydedildi: {model_path}")

# Feature importance (ağaç tabanlı modeller için)
if hasattr(best_model, 'feature_importances_'):
    fi = pd.DataFrame({
        'Özellik': X.columns,
        'Önem': best_model.feature_importances_
    }).sort_values('Önem', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(fi['Özellik'], fi['Önem'], color='steelblue', edgecolor='black')
    ax.set_title(f'Feature Importance — {best_model_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Önem')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/04_feature_importance.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Grafik: {output_dir}/04_feature_importance.png]")

print("\n" + "=" * 70)
print("  MODELLEME TAMAMLANDI — Student Habits")
print("=" * 70)
