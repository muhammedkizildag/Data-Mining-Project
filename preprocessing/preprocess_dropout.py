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
# 4. NORMALİZASYON
# ============================================================
print("\n--- 4. Normalizasyon (MinMaxScaler 0-1) ---")

scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

print(f"  Tüm değerler 0-1 arasına getirildi")

# ============================================================
# 5. FEATURE SELECTION (Mutual Information)
# ============================================================
print("\n--- 5. Feature Selection (Mutual Information) ---")

mi_scores = mutual_info_classif(X_scaled, y, random_state=42)
mi_df = pd.DataFrame({'Özellik': X_scaled.columns, 'MI_Score': mi_scores})
mi_df = mi_df.sort_values('MI_Score', ascending=False)

print(f"\n  Özellik önem sıralaması (Top 15):")
for _, row in mi_df.head(15).iterrows():
    bar = "█" * int(row['MI_Score'] * 30)
    print(f"    {row['Özellik']:50s} | {row['MI_Score']:.4f} | {bar}")

print(f"\n  Düşük önemli özellikler (Bottom 10):")
for _, row in mi_df.tail(10).iterrows():
    bar = "█" * int(row['MI_Score'] * 30) if row['MI_Score'] > 0 else "·"
    print(f"    {row['Özellik']:50s} | {row['MI_Score']:.4f} | {bar}")

# Düşük MI skorlu özellikleri çıkar
threshold = 0.01
low_features = mi_df[mi_df['MI_Score'] < threshold]['Özellik'].tolist()
if low_features:
    print(f"\n  MI < {threshold} olan özellikler ({len(low_features)} adet): {low_features}")
    X_scaled = X_scaled.drop(columns=low_features)
    print(f"  → Bu özellikler çıkarıldı")
    print(f"  Kalan özellik sayısı: {X_scaled.shape[1]}")

# ============================================================
# 6. SONUÇ
# ============================================================
print("\n" + "=" * 60)
print("  SONUÇ")
print("=" * 60)

final_df = pd.concat([X_scaled, y.reset_index(drop=True)], axis=1)
final_df = final_df.rename(columns={'Target_encoded': 'Target'})

print(f"\n  Final veri seti: {final_df.shape[0]} satır × {final_df.shape[1]} sütun")
print(f"  Özellik sayısı: {X_scaled.shape[1]}")
print(f"  Hedef değişken: Target ({target_mapping})")

final_df.to_csv(f"{output_dir}/dropout_processed.csv", index=False)
print(f"\n  Kaydedildi: {output_dir}/dropout_processed.csv")
