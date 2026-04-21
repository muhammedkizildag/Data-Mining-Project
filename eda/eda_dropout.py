import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
sns.set_style("whitegrid")

output_dir = "eda/plots_dropout"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv("datasets/dropout_academic_success/data.csv", sep=";")
df.columns = df.columns.str.strip()

# ============================================================
# 1. GENEL BAKIŞ
# ============================================================
print("=" * 60)
print("  1. GENEL BAKIŞ")
print("=" * 60)
print(f"\nBoyut: {df.shape[0]} satır × {df.shape[1]} sütun")

print(f"\n--- Sütunlar ---")
for i, col in enumerate(df.columns):
    dtype = df[col].dtype
    nunique = df[col].nunique()
    print(f"  {i+1:2d}. {col:50s} | {str(dtype):8s} | {nunique:4d} benzersiz")

print(f"\n--- Eksik Veri ---")
missing = df.isnull().sum()
missing = missing[missing > 0]
if len(missing) > 0:
    for col, cnt in missing.items():
        print(f"  {col}: {cnt} eksik ({cnt/len(df)*100:.1f}%)")
else:
    print("  Eksik veri yok ✓")

# ============================================================
# 2. HEDEF DEĞİŞKEN ANALİZİ
# ============================================================
print("\n" + "=" * 60)
print("  2. HEDEF DEĞİŞKEN ANALİZİ")
print("=" * 60)

print(f"\n--- Target dağılımı ---")
for val, cnt in df['Target'].value_counts().items():
    print(f"  {val}: {cnt} ({cnt/len(df)*100:.1f}%)")

fig, ax = plt.subplots(figsize=(8, 5))
colors = {'Dropout': '#e74c3c', 'Enrolled': '#f39c12', 'Graduate': '#27ae60'}
counts = df['Target'].value_counts()
counts.plot(kind='bar', ax=ax, color=[colors[x] for x in counts.index], edgecolor='black')
ax.set_title('Hedef Değişken Dağılımı (Target)')
ax.set_xlabel('Durum')
ax.set_ylabel('Öğrenci Sayısı')
ax.tick_params(axis='x', rotation=0)
for i, (val, cnt) in enumerate(counts.items()):
    ax.text(i, cnt + 30, f'{cnt}\n(%{cnt/len(df)*100:.1f})', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(f"{output_dir}/01_hedef_degisken.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  [Grafik: {output_dir}/01_hedef_degisken.png]")

# ============================================================
# 3. ÖNEMLİ SAYISAL DEĞİŞKENLER
# ============================================================
print("\n" + "=" * 60)
print("  3. ÖNEMLİ SAYISAL DEĞİŞKENLER")
print("=" * 60)

key_numeric = [
    'Age at enrollment', 'Admission grade',
    'Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)',
    'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)',
    'Unemployment rate', 'Inflation rate', 'GDP'
]

print(f"\n--- İstatistikler ---")
print(df[key_numeric].describe().round(2).to_string())

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()

for i, col in enumerate(key_numeric):
    axes[i].hist(df[col], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    axes[i].set_title(col, fontsize=10)
    axes[i].axvline(x=df[col].mean(), color='red', linestyle='--', alpha=0.7)

plt.suptitle('Önemli Sayısal Değişkenlerin Dağılımları', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{output_dir}/02_sayisal_dagilimlar.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  [Grafik: {output_dir}/02_sayisal_dagilimlar.png]")

# ============================================================
# 4. TARGET'A GÖRE KARŞILAŞTIRMALAR
# ============================================================
print("\n" + "=" * 60)
print("  4. TARGET'A GÖRE KARŞILAŞTIRMALAR")
print("=" * 60)

print(f"\n--- Target'a göre ortalamalar ---")
group_means = df.groupby('Target')[key_numeric].mean()
print(group_means.round(2).to_string())

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()
order = ['Dropout', 'Enrolled', 'Graduate']
palette = {'Dropout': '#e74c3c', 'Enrolled': '#f39c12', 'Graduate': '#27ae60'}

for i, col in enumerate(key_numeric):
    sns.boxplot(data=df, x='Target', y=col, ax=axes[i], order=order,
                hue='Target', palette=palette, legend=False)
    axes[i].set_title(col, fontsize=10)
    axes[i].set_xlabel('')

plt.suptitle('Target Durumuna Göre Sayısal Değişken Dağılımları', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{output_dir}/03_target_karsilastirma.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  [Grafik: {output_dir}/03_target_karsilastirma.png]")

# ============================================================
# 5. KORELASYON ANALİZİ
# ============================================================
print("\n" + "=" * 60)
print("  5. KORELASYON ANALİZİ")
print("=" * 60)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Target_num'] = le.fit_transform(df['Target'])

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove('Target_num')

corrs = df[numeric_cols + ['Target_num']].corr()['Target_num'].drop('Target_num').sort_values(ascending=False)

print(f"\n--- Target ile en güçlü korelasyonlar ---")
print("  Pozitif (Mezuniyetle ilişkili):")
for col, val in corrs.head(5).items():
    print(f"    {col:50s} | r={val:+.3f}")
print("  Negatif (Bırakmayla ilişkili):")
for col, val in corrs.tail(5).items():
    print(f"    {col:50s} | r={val:+.3f}")

top_features = list(corrs.head(5).index) + list(corrs.tail(5).index)
corr_subset = df[top_features + ['Target_num']].corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_subset, dtype=bool))
sns.heatmap(corr_subset, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=True, ax=ax, linewidths=0.5)
ax.set_title('Korelasyon Matrisi (En Etkili 10 Özellik + Target)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{output_dir}/04_korelasyon_matrisi.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  [Grafik: {output_dir}/04_korelasyon_matrisi.png]")

# ============================================================
# 6. KATEGORİK/ORDINAL DEĞİŞKENLER VS TARGET
# ============================================================
print("\n" + "=" * 60)
print("  6. KATEGORİK DEĞİŞKENLER VS TARGET")
print("=" * 60)

cat_features = ['Marital status', 'Gender', 'Scholarship holder',
                'Debtor', 'Tuition fees up to date', 'Displaced']

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, col in enumerate(cat_features):
    ct = pd.crosstab(df[col], df['Target'], normalize='index')[order] * 100
    ct.plot(kind='bar', ax=axes[i], color=['#e74c3c', '#f39c12', '#27ae60'], edgecolor='black')
    axes[i].set_title(f'{col} vs Target')
    axes[i].set_ylabel('Yüzde (%)')
    axes[i].tick_params(axis='x', rotation=0)
    axes[i].legend(title='Target', fontsize=8)

plt.suptitle('Kategorik Değişkenler vs Target', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{output_dir}/05_kategorik_vs_target.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  [Grafik: {output_dir}/05_kategorik_vs_target.png]")

# ============================================================
# 7. EN ETKİLİ ÖZELLİKLER - SCATTER
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

scatter_features = [
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)',
    'Curricular units 1st sem (approved)'
]

for i, col in enumerate(scatter_features):
    for target, color in palette.items():
        mask = df['Target'] == target
        axes[i].scatter(df.loc[mask, col], df.loc[mask, 'Admission grade'],
                       c=color, label=target, alpha=0.3, s=15)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Admission Grade')
    axes[i].set_title(f'{col}\nvs Admission Grade')
    axes[i].legend(fontsize=8)

plt.suptitle('En Etkili Özellikler - Scatter Plot', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{output_dir}/06_scatter_iliskiler.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  [Grafik: {output_dir}/06_scatter_iliskiler.png]")

# ============================================================
# 8. YAŞ DAĞILIMI TARGET'A GÖRE
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))
for target, color in palette.items():
    subset = df[df['Target'] == target]
    ax.hist(subset['Age at enrollment'], bins=20, alpha=0.5, color=color, label=target, edgecolor='black')
ax.set_title('Kayıt Yaşı Dağılımı - Target Durumuna Göre')
ax.set_xlabel('Kayıt Yaşı')
ax.set_ylabel('Frekans')
ax.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/07_yas_dagilimi.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  [Grafik: {output_dir}/07_yas_dagilimi.png]")

print("\n" + "=" * 60)
print("  EDA TAMAMLANDI - Dropout UCI")
print("=" * 60)
print(f"\n  Toplam {len(os.listdir(output_dir))} grafik kaydedildi: {output_dir}/")
