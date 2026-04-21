import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
import os
import warnings
warnings.filterwarnings('ignore')

output_dir = "preprocessing"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv("datasets/student_habits/student_habits_performance.csv")

print("=" * 60)
print("  VERİ ÖN İŞLEME — Student Habits")
print("=" * 60)

# ============================================================
# 1. GEREKSIZ SÜTUNLARI ÇIKAR
# ============================================================
print("\n--- 1. Gereksiz Sütunlar ---")
print(f"  student_id çıkarıldı (benzersiz ID, modele katkısı yok)")
df = df.drop('student_id', axis=1)

# ============================================================
# 2. HEDEF DEĞİŞKENİ OLUŞTUR
# ============================================================
print("\n--- 2. Hedef Değişken ---")
df['risk_level'] = pd.cut(df['exam_score'], bins=[0, 50, 75, 100], labels=['Düşük', 'Orta', 'Yüksek'])
print(f"  exam_score → risk_level (Düşük <50 / Orta 50-75 / Yüksek >75)")
print(f"  Dağılım:")
for val, cnt in df['risk_level'].value_counts().sort_index().items():
    print(f"    {val}: {cnt} ({cnt/len(df)*100:.1f}%)")

df = df.drop('exam_score', axis=1)

# ============================================================
# 3. EKSİK VERİ İŞLEME
# ============================================================
print("\n--- 3. Eksik Veri ---")
missing = df.isnull().sum()
missing = missing[missing > 0]
for col, cnt in missing.items():
    print(f"  {col}: {cnt} eksik ({cnt/len(df)*100:.1f}%)")

print(f"  → parental_education_level: mod (en sık değer) ile dolduruldu")
df['parental_education_level'].fillna(df['parental_education_level'].mode()[0], inplace=True)

print(f"  Eksik veri kaldı mı: {df.isnull().sum().sum()}")

# ============================================================
# 4. KATEGORİK DEĞİŞKENLERİ ENCODE ET
# ============================================================
print("\n--- 4. Kategorik Değişken Encoding ---")

cat_cols = df.select_dtypes(include='object').columns.tolist()
print(f"  Kategorik sütunlar: {cat_cols}")

le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"  {col}: {le_dict[col]}")

# risk_level da encode
le_target = LabelEncoder()
df['risk_level'] = le_target.fit_transform(df['risk_level'])
target_mapping = dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))
print(f"  risk_level: {target_mapping}")

# ============================================================
# 5. NORMALİZASYON
# ============================================================
print("\n--- 5. Normalizasyon (MinMaxScaler 0-1) ---")

X = df.drop('risk_level', axis=1)
y = df['risk_level']

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
print(f"  Normalize edilecek sütunlar: {numeric_cols}")

scaler = MinMaxScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

print(f"  Tüm değerler 0-1 arasına getirildi")
print(f"\n  Normalize sonrası istatistikler:")
print(X.describe().round(3).to_string())

# ============================================================
# 6. FEATURE SELECTION (Mutual Information)
# ============================================================
print("\n--- 6. Feature Selection (Mutual Information) ---")

mi_scores = mutual_info_classif(X, y, random_state=42)
mi_df = pd.DataFrame({'Özellik': X.columns, 'MI_Score': mi_scores})
mi_df = mi_df.sort_values('MI_Score', ascending=False)

print(f"\n  Özellik önem sıralaması:")
for _, row in mi_df.iterrows():
    bar = "█" * int(row['MI_Score'] * 50)
    print(f"    {row['Özellik']:35s} | {row['MI_Score']:.4f} | {bar}")

threshold = 0.01
low_features = mi_df[mi_df['MI_Score'] < threshold]['Özellik'].tolist()
if low_features:
    print(f"\n  Düşük önemli özellikler (MI < {threshold}): {low_features}")
    print(f"  → Bu özellikler çıkarılabilir ama şimdilik tutuyoruz")

# ============================================================
# 7. SONUÇ
# ============================================================
print("\n" + "=" * 60)
print("  SONUÇ")
print("=" * 60)

final_df = pd.concat([X, y], axis=1)
print(f"\n  Final veri seti: {final_df.shape[0]} satır × {final_df.shape[1]} sütun")
print(f"  Özellik sayısı: {X.shape[1]}")
print(f"  Hedef değişken: risk_level ({target_mapping})")

final_df.to_csv(f"{output_dir}/habits_processed.csv", index=False)
print(f"\n  Kaydedildi: {output_dir}/habits_processed.csv")
