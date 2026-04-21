import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
sns.set_style("whitegrid")

output_dir = "eda/plots_habits"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv("datasets/student_habits/student_habits_performance.csv")
df = df.drop('student_id', axis=1)

# Hedef değişkeni oluştur
df['risk_level'] = pd.cut(df['exam_score'], bins=[0, 50, 75, 100], labels=['Düşük', 'Orta', 'Yüksek'])

# ============================================================
# 1. GENEL BAKIŞ
# ============================================================
print("=" * 60)
print("  1. GENEL BAKIŞ")
print("=" * 60)
print(f"\nBoyut: {df.shape[0]} satır × {df.shape[1]} sütun")
print(f"\n--- Veri Tipleri ---")
print(df.dtypes)
print(f"\n--- Eksik Veri ---")
missing = df.isnull().sum()
missing = missing[missing > 0]
if len(missing) > 0:
    for col, cnt in missing.items():
        print(f"  {col}: {cnt} eksik ({cnt/len(df)*100:.1f}%)")
else:
    print("  Eksik veri yok")

print(f"\n--- Sayısal Değişkenler İstatistikleri ---")
print(df.describe().round(2).to_string())

# ============================================================
# 2. HEDEF DEĞİŞKEN ANALİZİ
# ============================================================
print("\n" + "=" * 60)
print("  2. HEDEF DEĞİŞKEN ANALİZİ")
print("=" * 60)

print(f"\n--- exam_score dağılımı ---")
print(df['exam_score'].describe())

print(f"\n--- risk_level sınıf dağılımı ---")
for val, cnt in df['risk_level'].value_counts().sort_index().items():
    print(f"  {val}: {cnt} ({cnt/len(df)*100:.1f}%)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['exam_score'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
axes[0].axvline(x=50, color='red', linestyle='--', label='Düşük/Orta sınırı (50)')
axes[0].axvline(x=75, color='green', linestyle='--', label='Orta/Yüksek sınırı (75)')
axes[0].set_title('Sınav Notu Dağılımı')
axes[0].set_xlabel('Exam Score')
axes[0].set_ylabel('Frekans')
axes[0].legend()

colors = ['#e74c3c', '#f39c12', '#27ae60']
df['risk_level'].value_counts().sort_index().plot(kind='bar', ax=axes[1], color=colors, edgecolor='black')
axes[1].set_title('Risk Seviyesi Dağılımı')
axes[1].set_xlabel('Risk Seviyesi')
axes[1].set_ylabel('Öğrenci Sayısı')
axes[1].tick_params(axis='x', rotation=0)

for i, (val, cnt) in enumerate(df['risk_level'].value_counts().sort_index().items()):
    axes[1].text(i, cnt + 5, f'{cnt}\n(%{cnt/len(df)*100:.1f})', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(f"{output_dir}/01_hedef_degisken.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  [Grafik kaydedildi: {output_dir}/01_hedef_degisken.png]")

# ============================================================
# 3. SAYISAL DEĞİŞKENLERİN DAĞILIMI
# ============================================================
print("\n" + "=" * 60)
print("  3. SAYISAL DEĞİŞKENLERİN DAĞILIMI")
print("=" * 60)

numeric_cols = ['age', 'study_hours_per_day', 'social_media_hours', 'netflix_hours',
                'attendance_percentage', 'sleep_hours', 'exercise_frequency', 'mental_health_rating']

fig, axes = plt.subplots(2, 4, figsize=(18, 10))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    axes[i].hist(df[col], bins=25, color='steelblue', edgecolor='black', alpha=0.7)
    axes[i].set_title(col)
    axes[i].axvline(x=df[col].mean(), color='red', linestyle='--', alpha=0.7, label=f'Ort: {df[col].mean():.1f}')
    axes[i].legend(fontsize=8)

plt.suptitle('Sayısal Değişkenlerin Dağılımları', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{output_dir}/02_sayisal_dagilimlar.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  [Grafik kaydedildi: {output_dir}/02_sayisal_dagilimlar.png]")

# Boxplot - Aykırı değer kontrolü
fig, axes = plt.subplots(2, 4, figsize=(18, 10))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    sns.boxplot(data=df, y=col, ax=axes[i], color='steelblue')
    axes[i].set_title(col)

plt.suptitle('Sayısal Değişkenler - Boxplot (Aykırı Değer Kontrolü)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{output_dir}/03_boxplot_aykiri_deger.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  [Grafik kaydedildi: {output_dir}/03_boxplot_aykiri_deger.png]")

# ============================================================
# 4. KATEGORİK DEĞİŞKENLERİN DAĞILIMI
# ============================================================
print("\n" + "=" * 60)
print("  4. KATEGORİK DEĞİŞKENLERİN DAĞILIMI")
print("=" * 60)

cat_cols = ['gender', 'part_time_job', 'diet_quality', 'parental_education_level',
            'internet_quality', 'extracurricular_participation']

for col in cat_cols:
    print(f"\n  {col}:")
    for val, cnt in df[col].value_counts().items():
        print(f"    {val}: {cnt} ({cnt/len(df)*100:.1f}%)")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    counts = df[col].value_counts()
    counts.plot(kind='bar', ax=axes[i], color='steelblue', edgecolor='black')
    axes[i].set_title(col)
    axes[i].tick_params(axis='x', rotation=45)
    for j, (val, cnt) in enumerate(counts.items()):
        axes[i].text(j, cnt + 3, str(cnt), ha='center', fontsize=9)

plt.suptitle('Kategorik Değişkenlerin Dağılımları', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{output_dir}/04_kategorik_dagilimlar.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  [Grafik kaydedildi: {output_dir}/04_kategorik_dagilimlar.png]")

# ============================================================
# 5. KORELASYON ANALİZİ
# ============================================================
print("\n" + "=" * 60)
print("  5. KORELASYON ANALİZİ")
print("=" * 60)

numeric_df = df[numeric_cols + ['exam_score']]
corr_matrix = numeric_df.corr()

print(f"\n--- exam_score ile korelasyonlar ---")
corrs = corr_matrix['exam_score'].drop('exam_score').sort_values(ascending=False)
for col, val in corrs.items():
    direction = "↑ Pozitif" if val > 0 else "↓ Negatif"
    strength = "GÜÇLÜ" if abs(val) > 0.5 else "ORTA" if abs(val) > 0.3 else "Zayıf"
    print(f"  {col:30s} | r={val:+.3f} | {direction} | {strength}")

fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=True, ax=ax,
            linewidths=0.5, cbar_kws={"shrink": 0.8})
ax.set_title('Korelasyon Matrisi (Sayısal Değişkenler)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{output_dir}/05_korelasyon_matrisi.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  [Grafik kaydedildi: {output_dir}/05_korelasyon_matrisi.png]")

# ============================================================
# 6. RİSK SEVİYESİNE GÖRE KARŞILAŞTIRMALAR
# ============================================================
print("\n" + "=" * 60)
print("  6. RİSK SEVİYESİNE GÖRE KARŞILAŞTIRMALAR")
print("=" * 60)

print(f"\n--- Risk seviyesine göre ortalamalar ---")
group_means = df.groupby('risk_level')[numeric_cols].mean()
print(group_means.round(2).to_string())

fig, axes = plt.subplots(2, 4, figsize=(18, 10))
axes = axes.flatten()
order = ['Düşük', 'Orta', 'Yüksek']
palette = {'Düşük': '#e74c3c', 'Orta': '#f39c12', 'Yüksek': '#27ae60'}

for i, col in enumerate(numeric_cols):
    sns.boxplot(data=df, x='risk_level', y=col, ax=axes[i], order=order, palette=palette)
    axes[i].set_title(col)
    axes[i].set_xlabel('')

plt.suptitle('Risk Seviyesine Göre Sayısal Değişken Dağılımları', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{output_dir}/06_risk_karsilastirma.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  [Grafik kaydedildi: {output_dir}/06_risk_karsilastirma.png]")

# Kategorik değişkenler vs risk seviyesi
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, col in enumerate(cat_cols):
    ct = pd.crosstab(df[col], df['risk_level'], normalize='index')[order] * 100
    ct.plot(kind='bar', ax=axes[i], color=['#e74c3c', '#f39c12', '#27ae60'], edgecolor='black')
    axes[i].set_title(f'{col} vs Risk Seviyesi')
    axes[i].set_ylabel('Yüzde (%)')
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].legend(title='Risk', fontsize=8)

plt.suptitle('Kategorik Değişkenler vs Risk Seviyesi', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{output_dir}/07_kategorik_vs_risk.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  [Grafik kaydedildi: {output_dir}/07_kategorik_vs_risk.png]")

# ============================================================
# 7. EN ÖNEMLİ İLİŞKİLER (Scatter)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

scatter_cols = ['study_hours_per_day', 'mental_health_rating', 'social_media_hours']
for i, col in enumerate(scatter_cols):
    for level, color in palette.items():
        mask = df['risk_level'] == level
        axes[i].scatter(df.loc[mask, col], df.loc[mask, 'exam_score'],
                       c=color, label=level, alpha=0.5, s=20)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Exam Score')
    axes[i].set_title(f'{col} vs Exam Score')
    axes[i].legend()

plt.suptitle('En Güçlü İlişkiler - Scatter Plot', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{output_dir}/08_scatter_iliskiler.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  [Grafik kaydedildi: {output_dir}/08_scatter_iliskiler.png]")

print("\n" + "=" * 60)
print("  EDA TAMAMLANDI - Student Habits")
print("=" * 60)
print(f"\n  Toplam {len(os.listdir(output_dir))} grafik kaydedildi: {output_dir}/")
