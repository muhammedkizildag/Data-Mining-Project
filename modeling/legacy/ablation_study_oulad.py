# ⚠️ UYARI: Bu dosya eski preprocessing çıktısını bekliyordu.
# Preprocessing güncellendiğinden sonuçları değişebilir.
# Güncel modelleme: model_oulad_v2.py (leakage düzeltilmiş)

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler

from modeling.legacy_notice import print_legacy_notice

print_legacy_notice(
    script_name="ablation_study_oulad.py",
    current_script="modeling/model_oulad_v2.py",
    reason="bu ablation dosyası eski preprocessing varsayımlarını taşıyabilir; aktif final raporu değildir.",
)

print("=" * 70)
print("  ABLATION STUDY — OULAD")
print("  (Her iyileştirmenin bireysel etkisi)")
print("=" * 70)

df = pd.read_csv("preprocessing/oulad_processed.csv")
X_orig = df.drop('target', axis=1)
y = df['target']

def add_features_oulad(X):
    X = X.copy()
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
    scaler = MinMaxScaler()
    X[new_cols] = scaler.fit_transform(X[new_cols])
    return X

X_fe = add_features_oulad(X_orig)

models_4 = {
    "kNN": KNeighborsClassifier(n_neighbors=9, metric='manhattan'),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=None, min_samples_split=5, random_state=42),
}

models_5 = {
    **models_4,
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=10, learning_rate=0.05, subsample=0.8, random_state=42, eval_metric='mlogloss', verbosity=0)
}

def evaluate(X_data, y_data, models):
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.30, random_state=42, stratify=y_data)
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        results[name] = f1
    best_name = max(results, key=results.get)
    best_f1 = results[best_name]
    return results, best_name, best_f1

# Test 1: Orijinal + 4 Algo (v1)
r1, b1_name, b1_f1 = evaluate(X_orig, y, models_4)

# Test 2: Orijinal + 5 Algo (+XGBoost)
r2, b2_name, b2_f1 = evaluate(X_orig, y, models_5)

# Test 3: FE + 4 Algo
r3, b3_name, b3_f1 = evaluate(X_fe, y, models_4)

# Test 4: FE + 5 Algo (v2)
r4, b4_name, b4_f1 = evaluate(X_fe, y, models_5)

print("\n" + "-" * 70)
print("  SONUÇLAR (En iyi model F1-Score)")
print("-" * 70)

print(f"\n  {'Senaryo':<40s} {'Kazanan':<18s} {'F1':>8s} {'Fark':>8s}  Açıklama")
print(f"  {'─'*40} {'─'*18} {'─'*8} {'─'*8}  {'─'*25}")

ref_f1 = b1_f1
scenarios = [
    ("1. Orijinal + 4 Algo (v1)", r1, b1_name, b1_f1, "Referans"),
    ("2. Orijinal + 5 Algo (+XGB)", r2, b2_name, b2_f1, "Sadece XGBoost etkisi"),
    ("3. FE + 4 Algo", r3, b3_name, b3_f1, "Sadece Feature Eng. etkisi"),
    ("4. FE + 5 Algo (v2)", r4, b4_name, b4_f1, "FE + XGBoost birlikte"),
]

for label, results, best_name, best_f1, desc in scenarios:
    diff = best_f1 - ref_f1
    sign = "+" if diff >= 0 else ""
    print(f"  {label:<40s} {best_name:<18s} {best_f1*100:>7.2f}% {sign}{diff*100:>6.2f}  {desc}")

# Detaylı tablo
print("\n" + "-" * 70)
print("  DETAYLI TABLO (Tüm modeller, tüm senaryolar)")
print("-" * 70)

all_models = ["kNN", "Naive Bayes", "Decision Tree", "Random Forest", "XGBoost"]
all_results = [r1, r2, r3, r4]

print(f"\n  {'Model':<18s}", end="")
for label in ["v1(Orig+4)", "+XGB", "+FE", "v2(FE+XGB)"]:
    print(f" {label:>12s}", end="")
print()
print(f"  {'─'*18}", end="")
for _ in range(4):
    print(f" {'─'*12}", end="")
print()

for model_name in all_models:
    print(f"  {model_name:<18s}", end="")
    for r in all_results:
        if model_name in r:
            print(f" {r[model_name]*100:>11.2f}%", end="")
        else:
            print(f" {'—':>12s}", end="")
    print()

# Bireysel katkılar
print("\n" + "-" * 70)
print("  BİREYSEL KATKILAR (v1 referans alınarak)")
print("-" * 70)
print(f"\n  v1 Referans (Orijinal + 4 Algo): {b1_name} → F1: %{b1_f1*100:.2f}")
print(f"\n  Sadece XGBoost eklemek:   {b2_name} → F1: %{b2_f1*100:.2f}  (fark: {(b2_f1-b1_f1)*100:+.2f})")
print(f"  Sadece Feature Eng.:     {b3_name} → F1: %{b3_f1*100:.2f}  (fark: {(b3_f1-b1_f1)*100:+.2f})")
print(f"  FE + XGBoost birlikte:   {b4_name} → F1: %{b4_f1*100:.2f}  (fark: {(b4_f1-b1_f1)*100:+.2f})")

print("\n" + "=" * 70)
print("  ABLATION STUDY TAMAMLANDI — OULAD")
print("=" * 70)
