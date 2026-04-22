import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
import os
import warnings
warnings.filterwarnings('ignore')

output_dir = "preprocessing"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv("datasets/dropout_academic_success/data.csv", sep=";")
df.columns = df.columns.str.strip()

print("=" * 60)
print("  VERİ ÖN İŞLEME — Dropout UCI")
print("=" * 60)

# ============================================================
# 1. GENEL KONTROL
# ============================================================
print("\n--- 1. Genel Kontrol ---")
print(f"  Boyut: {df.shape[0]} satır × {df.shape[1]} sütun")
print(f"  Eksik veri: {df.isnull().sum().sum()} (yok)")

# ============================================================
# 2. HEDEF DEĞİŞKEN ENCODE
# ============================================================
print("\n--- 2. Hedef Değişken ---")
print(f"  Orijinal dağılım:")
for val, cnt in df['Target'].value_counts().items():
    print(f"    {val}: {cnt} ({cnt/len(df)*100:.1f}%)")

le_target = LabelEncoder()
df['Target_encoded'] = le_target.fit_transform(df['Target'])
target_mapping = dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))
print(f"\n  Encoding: {target_mapping}")

# ============================================================
# 3. ÖZELLİKLERİ AYIR
# ============================================================
X = df.drop(['Target', 'Target_encoded'], axis=1)
y = df['Target_encoded']

print(f"\n--- 3. Özellikler ---")
print(f"  Toplam özellik sayısı: {X.shape[1]}")

# ============================================================
# 4. SONUÇ (normalizasyon ve feature selection modeling aşamasında yapılacak)
# ============================================================
print("\n" + "=" * 60)
print("  SONUÇ")
print("=" * 60)
print("  NOT: Normalizasyon ve MI feature selection modeling aşamasında")
print("       train/test split sonrası yapılacak (data leakage önlemi).")

final_df = pd.concat([X, y.reset_index(drop=True)], axis=1)
final_df = final_df.rename(columns={'Target_encoded': 'Target'})

print(f"\n  Final veri seti: {final_df.shape[0]} satır × {final_df.shape[1]} sütun")
print(f"  Özellik sayısı: {X.shape[1]}")
print(f"  Hedef değişken: Target ({target_mapping})")

final_df.to_csv(f"{output_dir}/dropout_processed.csv", index=False)
print(f"\n  Kaydedildi: {output_dir}/dropout_processed.csv")
