import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

# ============================================================
# STUDENT HABITS — ABLATION STUDY
# ============================================================
print("=" * 70)
print("  ABLATION STUDY — Student Habits")
print("  (Her iyileştirmenin bireysel etkisi)")
print("=" * 70)

df = pd.read_csv("preprocessing/habits_processed.csv")
X_orig = df.drop('risk_level', axis=1)
y = df['risk_level']

# Feature Engineering fonksiyonu
def add_features_habits(X):
    X = X.copy()
    X['study_social_ratio'] = X['study_hours_per_day'] / (X['social_media_hours'] + 0.01)
    X['study_netflix_ratio'] = X['study_hours_per_day'] / (X['netflix_hours'] + 0.01)
    X['screen_time_total'] = X['social_media_hours'] + X['netflix_hours']
    X['sleep_mental_interaction'] = X['sleep_hours'] * X['mental_health_rating']
    X['study_mental_interaction'] = X['study_hours_per_day'] * X['mental_health_rating']
    X['healthy_lifestyle'] = X['sleep_hours'] + X['exercise_frequency'] + X['diet_quality']
    X['study_attendance_interaction'] = X['study_hours_per_day'] * X['attendance_percentage']
    new_cols = ['study_social_ratio', 'study_netflix_ratio', 'screen_time_total',
                'sleep_mental_interaction', 'study_mental_interaction',
                'healthy_lifestyle', 'study_attendance_interaction']
    scaler = MinMaxScaler()
    X[new_cols] = scaler.fit_transform(X[new_cols])
    return X

X_fe = add_features_habits(X_orig)

models_4 = {
    "kNN": KNeighborsClassifier(n_neighbors=15, metric='manhattan'),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(max_depth=7, min_samples_leaf=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

models_5 = {
    **models_4,
    "XGBoost": XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.1, subsample=0.8, random_state=42, eval_metric='mlogloss', verbosity=0)
}

def evaluate(X_data, y_data, models, use_smote=False):
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.30, random_state=42, stratify=y_data)

    if use_smote:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        results[name] = f1

    best_name = max(results, key=results.get)
    best_f1 = results[best_name]
    return results, best_name, best_f1

# Test 1: Orijinal özellikler + 4 algoritma (v1)
r1, b1_name, b1_f1 = evaluate(X_orig, y, models_4, use_smote=False)

# Test 2: Orijinal özellikler + XGBoost eklendi (sadece algoritma etkisi)
r2, b2_name, b2_f1 = evaluate(X_orig, y, models_5, use_smote=False)

# Test 3: Feature Engineering + 4 algoritma (sadece FE etkisi)
r3, b3_name, b3_f1 = evaluate(X_fe, y, models_4, use_smote=False)

# Test 4: SMOTE + orijinal özellikler + 4 algoritma (sadece SMOTE etkisi)
r4, b4_name, b4_f1 = evaluate(X_orig, y, models_4, use_smote=True)

# Test 5: Feature Engineering + XGBoost (FE + algoritma, SMOTE yok)
r5, b5_name, b5_f1 = evaluate(X_fe, y, models_5, use_smote=False)

# Test 6: SMOTE + Feature Engineering + 5 algoritma (v2 - hepsi)
r6, b6_name, b6_f1 = evaluate(X_fe, y, models_5, use_smote=True)

# Test 7: SMOTE + orijinal özellikler + 5 algoritma
r7, b7_name, b7_f1 = evaluate(X_orig, y, models_5, use_smote=True)

print("\n" + "-" * 70)
print("  SONUÇLAR (En iyi model F1-Score)")
print("-" * 70)

scenarios = [
    ("1. Orijinal + 4 Algo (v1)", r1, b1_name, b1_f1, "Referans"),
    ("2. Orijinal + 5 Algo (+XGB)", r2, b2_name, b2_f1, "Sadece XGBoost etkisi"),
    ("3. FE + 4 Algo", r3, b3_name, b3_f1, "Sadece Feature Eng. etkisi"),
    ("4. SMOTE + Orijinal + 4 Algo", r4, b4_name, b4_f1, "Sadece SMOTE etkisi"),
    ("5. FE + 5 Algo (SMOTE yok)", r5, b5_name, b5_f1, "FE + XGBoost birlikte"),
    ("6. SMOTE + Orijinal + 5 Algo", r7, b7_name, b7_f1, "SMOTE + XGBoost birlikte"),
    ("7. SMOTE + FE + 5 Algo (v2)", r6, b6_name, b6_f1, "Hepsi birlikte"),
]

print(f"\n  {'Senaryo':<40s} {'Kazanan':<18s} {'F1':>8s} {'Fark':>8s}  Açıklama")
print(f"  {'─'*40} {'─'*18} {'─'*8} {'─'*8}  {'─'*25}")

ref_f1 = b1_f1
for label, results, best_name, best_f1, desc in scenarios:
    diff = best_f1 - ref_f1
    sign = "+" if diff >= 0 else ""
    print(f"  {label:<40s} {best_name:<18s} {best_f1*100:>7.2f}% {sign}{diff*100:>6.2f}  {desc}")

# Detaylı tablo
print("\n" + "-" * 70)
print("  DETAYLI TABLO (Tüm modeller, tüm senaryolar)")
print("-" * 70)

all_models = ["kNN", "Naive Bayes", "Decision Tree", "Random Forest", "XGBoost"]
all_results = [r1, r2, r3, r4, r5, r7, r6]
scenario_labels = ["v1\n(Orig+4)", "+XGB\n(Orig+5)", "+FE\n(FE+4)", "+SMOTE\n(SM+4)", "FE+XGB\n(FE+5)", "SM+XGB\n(SM+5)", "v2\n(All)"]

print(f"\n  {'Model':<18s}", end="")
for label in ["v1(Orig+4)", "+XGB", "+FE", "+SMOTE", "FE+XGB", "SM+XGB", "v2(All)"]:
    print(f" {label:>10s}", end="")
print()
print(f"  {'─'*18}", end="")
for _ in range(7):
    print(f" {'─'*10}", end="")
print()

for model_name in all_models:
    print(f"  {model_name:<18s}", end="")
    for r in all_results:
        if model_name in r:
            print(f" {r[model_name]*100:>9.2f}%", end="")
        else:
            print(f" {'—':>10s}", end="")
    print()

# Bireysel katkılar
print("\n" + "-" * 70)
print("  BİREYSEL KATKILAR (v1 referans alınarak)")
print("-" * 70)
print(f"\n  v1 Referans (Orijinal + 4 Algo): {b1_name} → F1: %{b1_f1*100:.2f}")
print(f"\n  Sadece XGBoost eklemek:           {b2_name} → F1: %{b2_f1*100:.2f}  (fark: {(b2_f1-b1_f1)*100:+.2f})")
print(f"  Sadece Feature Engineering:       {b3_name} → F1: %{b3_f1*100:.2f}  (fark: {(b3_f1-b1_f1)*100:+.2f})")
print(f"  Sadece SMOTE:                     {b4_name} → F1: %{b4_f1*100:.2f}  (fark: {(b4_f1-b1_f1)*100:+.2f})")
print(f"  FE + XGBoost (SMOTE yok):         {b5_name} → F1: %{b5_f1*100:.2f}  (fark: {(b5_f1-b1_f1)*100:+.2f})")
print(f"  Hepsi birlikte (v2):              {b6_name} → F1: %{b6_f1*100:.2f}  (fark: {(b6_f1-b1_f1)*100:+.2f})")

# ============================================================
# DROPOUT UCI — ABLATION STUDY
# ============================================================
print("\n\n" + "=" * 70)
print("  ABLATION STUDY — Dropout UCI")
print("  (Her iyileştirmenin bireysel etkisi)")
print("=" * 70)

df2 = pd.read_csv("preprocessing/dropout_processed.csv")
X_orig2 = df2.drop('Target', axis=1)
y2 = df2['Target']

def add_features_dropout(X):
    X = X.copy()
    X['sem1_success_rate'] = X['Curricular units 1st sem (approved)'] / (X['Curricular units 1st sem (enrolled)'] + 0.001)
    X['sem2_success_rate'] = X['Curricular units 2nd sem (approved)'] / (X['Curricular units 2nd sem (enrolled)'] + 0.001)
    X['total_approved'] = X['Curricular units 1st sem (approved)'] + X['Curricular units 2nd sem (approved)']
    X['total_grade'] = X['Curricular units 1st sem (grade)'] + X['Curricular units 2nd sem (grade)']
    X['grade_improvement'] = X['Curricular units 2nd sem (grade)'] - X['Curricular units 1st sem (grade)']
    X['approved_improvement'] = X['Curricular units 2nd sem (approved)'] - X['Curricular units 1st sem (approved)']
    X['eval_approved_ratio_1'] = X['Curricular units 1st sem (approved)'] / (X['Curricular units 1st sem (evaluations)'] + 0.001)
    X['eval_approved_ratio_2'] = X['Curricular units 2nd sem (approved)'] / (X['Curricular units 2nd sem (evaluations)'] + 0.001)
    new_cols = ['sem1_success_rate', 'sem2_success_rate', 'total_approved', 'total_grade',
                'grade_improvement', 'approved_improvement', 'eval_approved_ratio_1', 'eval_approved_ratio_2']
    scaler = MinMaxScaler()
    X[new_cols] = scaler.fit_transform(X[new_cols])
    return X

X_fe2 = add_features_dropout(X_orig2)

models_4b = {
    "kNN": KNeighborsClassifier(n_neighbors=15, metric='manhattan'),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42),
}

models_5b = {
    **models_4b,
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, subsample=1.0, random_state=42, eval_metric='mlogloss', verbosity=0)
}

d1_r, d1_name, d1_f1 = evaluate(X_orig2, y2, models_4b, use_smote=False)
d2_r, d2_name, d2_f1 = evaluate(X_orig2, y2, models_5b, use_smote=False)
d3_r, d3_name, d3_f1 = evaluate(X_fe2, y2, models_4b, use_smote=False)
d4_r, d4_name, d4_f1 = evaluate(X_fe2, y2, models_5b, use_smote=False)

print(f"\n  {'Senaryo':<40s} {'Kazanan':<18s} {'F1':>8s} {'Fark':>8s}  Açıklama")
print(f"  {'─'*40} {'─'*18} {'─'*8} {'─'*8}  {'─'*25}")

d_ref = d1_f1
d_scenarios = [
    ("1. Orijinal + 4 Algo (v1)", d1_r, d1_name, d1_f1, "Referans"),
    ("2. Orijinal + 5 Algo (+XGB)", d2_r, d2_name, d2_f1, "Sadece XGBoost etkisi"),
    ("3. FE + 4 Algo", d3_r, d3_name, d3_f1, "Sadece Feature Eng. etkisi"),
    ("4. FE + 5 Algo (v2)", d4_r, d4_name, d4_f1, "FE + XGBoost birlikte"),
]

for label, results, best_name, best_f1, desc in d_scenarios:
    diff = best_f1 - d_ref
    sign = "+" if diff >= 0 else ""
    print(f"  {label:<40s} {best_name:<18s} {best_f1*100:>7.2f}% {sign}{diff*100:>6.2f}  {desc}")

print(f"\n  BİREYSEL KATKILAR:")
print(f"  v1 Referans:              {d1_name} → F1: %{d1_f1*100:.2f}")
print(f"  Sadece XGBoost eklemek:   {d2_name} → F1: %{d2_f1*100:.2f}  (fark: {(d2_f1-d1_f1)*100:+.2f})")
print(f"  Sadece Feature Eng.:      {d3_name} → F1: %{d3_f1*100:.2f}  (fark: {(d3_f1-d1_f1)*100:+.2f})")
print(f"  FE + XGBoost birlikte:    {d4_name} → F1: %{d4_f1*100:.2f}  (fark: {(d4_f1-d1_f1)*100:+.2f})")

print("\n" + "=" * 70)
print("  ABLATION STUDY TAMAMLANDI")
print("=" * 70)
