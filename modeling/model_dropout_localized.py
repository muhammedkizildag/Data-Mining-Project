import pandas as pd
import numpy as np
import joblib
import json
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
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc, precision_recall_curve,
                             average_precision_score)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = (12, 6)
sns.set_style("whitegrid")


def plot_multiclass_roc_pr(y_true, y_proba, target_names, output_path, title_prefix):
    classes = np.arange(len(target_names))
    y_bin = label_binarize(y_true, classes=classes)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for class_idx, class_name in enumerate(target_names):
        fpr, tpr, _ = roc_curve(y_bin[:, class_idx], y_proba[:, class_idx])
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y_bin[:, class_idx], y_proba[:, class_idx])
        avg_precision = average_precision_score(y_bin[:, class_idx], y_proba[:, class_idx])

        axes[0].plot(fpr, tpr, linewidth=2, label=f"{class_name} (AUC={roc_auc:.3f})")
        axes[1].plot(recall, precision, linewidth=2, label=f"{class_name} (AP={avg_precision:.3f})")

    micro_fpr, micro_tpr, _ = roc_curve(y_bin.ravel(), y_proba.ravel())
    micro_auc = auc(micro_fpr, micro_tpr)
    axes[0].plot(micro_fpr, micro_tpr, linestyle="--", color="black", linewidth=2,
                 label=f"Micro-average (AUC={micro_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], linestyle=":", color="gray")
    axes[0].set_title(f"ROC Curves — {title_prefix}", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(fontsize=9)

    micro_precision, micro_recall, _ = precision_recall_curve(y_bin.ravel(), y_proba.ravel())
    micro_ap = average_precision_score(y_bin, y_proba, average="micro")
    axes[1].plot(micro_recall, micro_precision, linestyle="--", color="black", linewidth=2,
                 label=f"Micro-average (AP={micro_ap:.3f})")
    axes[1].set_title(f"Precision-Recall Curves — {title_prefix}", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_learning_curve_custom(best_pipeline, X_data, y_data, output_path, title, target_names, use_sample_weight=False):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_fracs = np.linspace(0.2, 1.0, 5)

    train_means, train_stds = [], []
    val_means, val_stds = [], []
    sample_counts = []

    for frac_idx, frac in enumerate(train_fracs):
        fold_train_scores = []
        fold_val_scores = []
        fold_sizes = []

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_data, y_data)):
            X_fold_train_full = X_data.iloc[train_idx]
            y_fold_train_full = y_data.iloc[train_idx]
            X_fold_val = X_data.iloc[val_idx]
            y_fold_val = y_data.iloc[val_idx]

            if frac < 1.0:
                subset_size = max(len(target_names) * 2, int(len(X_fold_train_full) * frac))
                X_fold_train, _, y_fold_train, _ = train_test_split(
                    X_fold_train_full,
                    y_fold_train_full,
                    train_size=subset_size,
                    stratify=y_fold_train_full,
                    random_state=42 + frac_idx + fold_idx,
                )
            else:
                X_fold_train = X_fold_train_full
                y_fold_train = y_fold_train_full

            fold_sizes.append(len(X_fold_train))
            pipe_clone = clone(best_pipeline)

            if use_sample_weight:
                fold_weights = compute_sample_weight('balanced', y_fold_train)
                pipe_clone.fit(X_fold_train, y_fold_train, model__sample_weight=fold_weights)
            else:
                pipe_clone.fit(X_fold_train, y_fold_train)

            y_train_pred = pipe_clone.predict(X_fold_train)
            y_val_pred = pipe_clone.predict(X_fold_val)

            fold_train_scores.append(f1_score(y_fold_train, y_train_pred, average='macro', zero_division=0))
            fold_val_scores.append(f1_score(y_fold_val, y_val_pred, average='macro', zero_division=0))

        sample_counts.append(int(np.mean(fold_sizes)))
        train_means.append(np.mean(fold_train_scores))
        train_stds.append(np.std(fold_train_scores))
        val_means.append(np.mean(fold_val_scores))
        val_stds.append(np.std(fold_val_scores))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(sample_counts, train_means, marker='o', linewidth=2, label='Train Macro F1', color='tab:blue')
    ax.plot(sample_counts, val_means, marker='o', linewidth=2, label='Validation Macro F1', color='tab:orange')
    ax.fill_between(sample_counts, np.array(train_means) - np.array(train_stds),
                    np.array(train_means) + np.array(train_stds), alpha=0.15, color='tab:blue')
    ax.fill_between(sample_counts, np.array(val_means) - np.array(val_stds),
                    np.array(val_means) + np.array(val_stds), alpha=0.15, color='tab:orange')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Eğitim Örnek Sayısı')
    ax.set_ylabel('Macro F1')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

output_dir = "modeling/plots_dropout_localized"
os.makedirs(output_dir, exist_ok=True)

print("=" * 70)
print("  MODELLEME — Dropout UCI (Yerelleştirilmiş / Localized)")
print("  Pipeline yapısı: MinMaxScaler + Model birlikte")
print("=" * 70)

df = pd.read_csv("preprocessing/dropout_processed.csv")

# ============================================================
# ÖZELLİK ÇIKARMA (Türkiye'ye uymayanlar)
# ============================================================
print("\n" + "=" * 70)
print("  TÜRKİYE YERELLEŞTİRME — Özellik Çıkarma")
print("=" * 70)

remove_cols = ['Tuition fees up to date', 'Debtor', 'Inflation rate']

print(f"\n  Çıkarılan özellikler ({len(remove_cols)} adet):")
for col in remove_cols:
    print(f"    ✗ {col}")

df = df.drop(columns=remove_cols)

X = df.drop('Target', axis=1)
y = df['Target']

print(f"\n  Kalan özellik sayısı: {X.shape[1]}")
print(f"  Toplam satır: {X.shape[0]}")

print(f"\n  Kalan özellikler ve Türkiye karşılıkları:")
feature_map = {
    'Marital status': 'Medeni durum',
    'Application mode': 'Başvuru türü (YKS, DGS, Yatay Geçiş)',
    'Application order': 'Tercih sırası',
    'Course': 'Bölüm',
    'Previous qualification': 'Önceki eğitim düzeyi (Lise, Önlisans...)',
    'Previous qualification (grade)': 'Önceki eğitim not ortalaması',
    'Mother\'s qualification': 'Anne eğitim düzeyi',
    'Father\'s qualification': 'Baba eğitim düzeyi',
    'Mother\'s occupation': 'Anne mesleği',
    'Father\'s occupation': 'Baba mesleği',
    'Admission grade': 'Üniversite giriş puanı (YKS)',
    'Gender': 'Cinsiyet',
    'Scholarship holder': 'Burs durumu',
    'Age at enrollment': 'Kayıt yaşı',
    'Curricular units 1st sem (enrolled)': '1. dönem alınan ders sayısı',
    'Curricular units 1st sem (evaluations)': '1. dönem girilen sınav sayısı',
    'Curricular units 1st sem (approved)': '1. dönem geçilen ders sayısı',
    'Curricular units 1st sem (grade)': '1. dönem not ortalaması',
    'Curricular units 2nd sem (enrolled)': '2. dönem alınan ders sayısı',
    'Curricular units 2nd sem (evaluations)': '2. dönem girilen sınav sayısı',
    'Curricular units 2nd sem (approved)': '2. dönem geçilen ders sayısı',
    'Curricular units 2nd sem (grade)': '2. dönem not ortalaması',
}
for orig, tr in feature_map.items():
    if orig in X.columns:
        print(f"    {orig:<45s} → {tr}")

target_names = ['Dropout', 'Enrolled', 'Graduate']

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
# CHATBOT ÖZELLİK FİLTRESİ (sadece chatbot'un topladığı 22 özellik)
# ============================================================
print("\n" + "=" * 70)
print("  CHATBOT ÖZELLİK FİLTRESİ")
print("=" * 70)

chatbot_features = list(feature_map.keys())
extra_in_train = [c for c in X_train.columns if c not in chatbot_features]
if extra_in_train:
    print(f"  Chatbot'ta olmayan {len(extra_in_train)} özellik çıkarılıyor:")
    for col in extra_in_train:
        print(f"    ✗ {col}")
    X_train = X_train[chatbot_features]
    X_test = X_test[chatbot_features]

print(f"  Kalan özellik sayısı: {X_train.shape[1]} (chatbot ile uyumlu)")

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
ax.set_title('10-Fold CV (Macro F1) — Dropout UCI (Yerelleştirilmiş)', fontsize=13, fontweight='bold')
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

dropout_recall = recall_score(y_test, y_pred_best, average=None)[0]
print(f"\n  ⚠️  Dropout Recall: %{dropout_recall*100:.2f} (en kritik metrik)")
print(f"      → Gerçek dropout öğrencilerinin %{dropout_recall*100:.1f}'ini yakalıyoruz")

best_proba = best_pipeline.predict_proba(X_test)
plot_multiclass_roc_pr(
    y_test,
    best_proba,
    target_names,
    f"{output_dir}/03_roc_pr_curves.png",
    f"{best_model_name} (Yerelleştirilmiş)"
)
print(f"\n  ROC/PR grafiği kaydedildi: {output_dir}/03_roc_pr_curves.png")

use_weighted_fit = best_model_name == "XGBoost"
plot_learning_curve_custom(
    best_pipeline,
    X_train,
    y_train,
    f"{output_dir}/05_learning_curve.png",
    f"Learning Curve — {best_model_name} (Yerelleştirilmiş)",
    target_names,
    use_sample_weight=use_weighted_fit,
)
print(f"  Learning curve kaydedildi: {output_dir}/05_learning_curve.png")

# ============================================================
# CONFUSION MATRIX (sadece seçilen model)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=target_names, yticklabels=target_names)
ax.set_title(f'Confusion Matrix — {best_model_name} (Yerelleştirilmiş)', fontsize=13, fontweight='bold')
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
print("  YERELLEŞTİRME ETKİSİ")
print("=" * 70)
print(f"\n  Önceki (f1_weighted, class_weight yok): XGBoost → Weighted F1: %75.10, Macro F1: %69.06")
print(f"  Güncel (f1_macro, class_weight balanced): {best_model_name} → Weighted F1: %{best_f1_weighted*100:.2f}, Macro F1: %{best_f1_macro*100:.2f}")
print(f"\n  Çıkarılan 3 özellik: Tuition fees, Debtor, Inflation rate")

# ============================================================
# EN İYİ MODELİ KAYDET (Pipeline: scaler + model birlikte)
# ============================================================
print("\n" + "=" * 70)
print("  EN İYİ MODELİ KAYDET (Pipeline)")
print("=" * 70)
model_path = "models/best_model_dropout_localized.pkl"
os.makedirs("models", exist_ok=True)
joblib.dump(best_pipeline, model_path)
print(f"  Model: {best_model_name} (Pipeline: MinMaxScaler + {best_model_name})")
print(f"  F1-Macro:    %{best_f1_macro*100:.2f}")
print(f"  F1-Weighted: %{best_f1_weighted*100:.2f}")
print(f"  Özellik sayısı: {X_train.shape[1]}")
print(f"  Kaydedildi: {model_path}")

feature_list = list(X_train.columns)
joblib.dump(feature_list, "models/dropout_localized_features.pkl")
print(f"  Özellik listesi: models/dropout_localized_features.pkl")

trained_scaler = best_pipeline.named_steps['scaler']
scaler_params = {}
for i, col in enumerate(X_train.columns):
    scaler_params[col] = {
        'min': float(trained_scaler.data_min_[i]),
        'max': float(trained_scaler.data_max_[i])
    }
scaler_path = "models/dropout_localized_scaler_params.json"
with open(scaler_path, "w", encoding="utf-8") as f:
    json.dump(scaler_params, f, ensure_ascii=False, indent=2)
print(f"  Scaler parametreleri (dokümantasyon): {scaler_path}")

actual_model = best_pipeline.named_steps['model']
if hasattr(actual_model, 'feature_importances_'):
    fi = pd.DataFrame({
        'Özellik': X_train.columns,
        'Önem': actual_model.feature_importances_
    }).sort_values('Önem', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(fi['Özellik'], fi['Önem'], color='steelblue', edgecolor='black')
    ax.set_title(f'Feature Importance — {best_model_name} (Yerelleştirilmiş)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Önem')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/04_feature_importance.png", dpi=150, bbox_inches='tight')
    plt.close()

print("\n" + "=" * 70)
print("  YERELLEŞTİRME TAMAMLANDI")
print("=" * 70)
