import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (12, 6)
sns.set_style("whitegrid")

output_dir = "eda/plots_oulad"
os.makedirs(output_dir, exist_ok=True)

print("=" * 70)
print("  EDA — OULAD (Open University Learning Analytics)")
print("=" * 70)

base = "datasets/oulad"
si = pd.read_csv(f"{base}/studentInfo.csv")
sa = pd.read_csv(f"{base}/studentAssessment.csv")
assessments = pd.read_csv(f"{base}/assessments.csv")
sv = pd.read_csv(f"{base}/studentVle.csv")
vle = pd.read_csv(f"{base}/vle.csv")

target_map = {'Withdrawn': 0, 'Fail': 1, 'Pass': 2, 'Distinction': 2}
si['target'] = si['final_result'].map(target_map)
target_names = {0: 'Withdrawn', 1: 'Fail', 2: 'Pass+Dist'}

# ============================================================
# 1. HEDEF DEĞİŞKEN DAĞILIMI
# ============================================================
print("\n" + "-" * 70)
print("  1. Hedef Değişken Dağılımı")
print("-" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

counts = si['final_result'].value_counts()
colors_4 = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db']
axes[0].bar(counts.index, counts.values, color=colors_4, edgecolor='black')
axes[0].set_title('Orijinal 4 Sınıf', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Öğrenci Sayısı')
for i, (idx, val) in enumerate(counts.items()):
    axes[0].text(i, val + 200, f'{val}\n(%{val/len(si)*100:.1f})', ha='center', fontsize=9)

target_counts = si['target'].value_counts().sort_index()
colors_3 = ['#e74c3c', '#f39c12', '#2ecc71']
labels_3 = [target_names[i] for i in target_counts.index]
axes[1].bar(labels_3, target_counts.values, color=colors_3, edgecolor='black')
axes[1].set_title('3 Sınıfa Daraltılmış', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Öğrenci Sayısı')
for i, val in enumerate(target_counts.values):
    axes[1].text(i, val + 200, f'{val}\n(%{val/len(si)*100:.1f})', ha='center', fontsize=9)

plt.suptitle('Hedef Değişken Dağılımı — OULAD', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{output_dir}/01_target_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  [Grafik: {output_dir}/01_target_distribution.png]")

for val in [0, 1, 2]:
    cnt = (si['target'] == val).sum()
    print(f"  {target_names[val]:15s}: {cnt:>6d} (%{cnt/len(si)*100:.1f})")

# ============================================================
# 2. DEMOGRAFİK ANALİZ
# ============================================================
print("\n" + "-" * 70)
print("  2. Demografik Analiz")
print("-" * 70)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

for ax, col, title in zip(axes.flatten(),
    ['gender', 'age_band', 'highest_education', 'disability', 'imd_band', 'region'],
    ['Cinsiyet', 'Yaş Grubu', 'Eğitim Düzeyi', 'Engellilik', 'Yoksunluk Bandı', 'Bölge']):
    ct = pd.crosstab(si[col], si['final_result'], normalize='index') * 100
    ct[['Distinction', 'Pass', 'Fail', 'Withdrawn']].plot(kind='bar', stacked=True, ax=ax,
        color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'], edgecolor='black')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('Oran (%)')
    ax.set_xlabel('')
    ax.legend(fontsize=7, loc='upper right')
    ax.tick_params(axis='x', rotation=45)

plt.suptitle('Demografik Dağılımlar ve Sonuç — OULAD', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{output_dir}/02_demographics.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  [Grafik: {output_dir}/02_demographics.png]")

print(f"\n  Cinsiyet: {si['gender'].value_counts().to_dict()}")
print(f"  Yaş: {si['age_band'].value_counts().to_dict()}")
print(f"  Eğitim: {si['highest_education'].value_counts().to_dict()}")

# ============================================================
# 3. AKADEMİK PERFORMANS
# ============================================================
print("\n" + "-" * 70)
print("  3. Akademik Performans (Assessment Notları)")
print("-" * 70)

sa_full = sa.merge(assessments, on='id_assessment', how='left')
sa_full = sa_full.merge(si[['code_module', 'code_presentation', 'id_student', 'target']],
                         on=['code_module', 'code_presentation', 'id_student'], how='left')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (target_val, name) in enumerate(target_names.items()):
    subset = sa_full[sa_full['target'] == target_val]['score'].dropna()
    axes[0].hist(subset, bins=50, alpha=0.5, label=f'{name} (ort:{subset.mean():.1f})', density=True)
axes[0].set_title('Not Dağılımı (Tüm Assessments)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Skor')
axes[0].legend()

for atype in ['TMA', 'CMA', 'Exam']:
    means = []
    for target_val in [0, 1, 2]:
        subset = sa_full[(sa_full['assessment_type'] == atype) & (sa_full['target'] == target_val)]['score'].dropna()
        means.append(subset.mean())
    axes[1].bar([x + 0.25 * (['TMA', 'CMA', 'Exam'].index(atype)) for x in range(3)],
                means, width=0.25, label=atype, edgecolor='black')
axes[1].set_xticks(range(3))
axes[1].set_xticklabels([target_names[i] for i in range(3)])
axes[1].set_title('Ortalama Not (Tip Bazlı)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Ortalama Skor')
axes[1].legend()

assess_count = sa_full.groupby(['code_module', 'code_presentation', 'id_student', 'target']).size().reset_index(name='count')
for target_val, name in target_names.items():
    subset = assess_count[assess_count['target'] == target_val]['count']
    axes[2].hist(subset, bins=30, alpha=0.5, label=f'{name} (ort:{subset.mean():.1f})')
axes[2].set_title('Tamamlanan Assessment Sayısı', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Assessment Sayısı')
axes[2].legend()

plt.suptitle('Akademik Performans — OULAD', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{output_dir}/03_academic_performance.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  [Grafik: {output_dir}/03_academic_performance.png]")

for target_val, name in target_names.items():
    subset = sa_full[sa_full['target'] == target_val]['score'].dropna()
    print(f"  {name:15s}: ort={subset.mean():.1f}, std={subset.std():.1f}")

# ============================================================
# 4. VLE ETKİLEŞİM ANALİZİ
# ============================================================
print("\n" + "-" * 70)
print("  4. VLE Etkileşim Analizi (Platform Kullanımı)")
print("-" * 70)

key_cols = ['code_module', 'code_presentation', 'id_student']
sv_agg = sv.groupby(key_cols).agg(
    total_clicks=('sum_click', 'sum'),
    active_days=('date', 'nunique')
).reset_index()
sv_agg = sv_agg.merge(si[key_cols + ['target']], on=key_cols, how='left')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for target_val, name in target_names.items():
    subset = sv_agg[sv_agg['target'] == target_val]['total_clicks']
    subset_clipped = subset[subset < subset.quantile(0.95)]
    axes[0].hist(subset_clipped, bins=50, alpha=0.5, label=f'{name} (ort:{subset.mean():.0f})')
axes[0].set_title('Toplam Tıklama (VLE)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Toplam Tıklama')
axes[0].legend()

click_means = []
day_means = []
for target_val in [0, 1, 2]:
    subset = sv_agg[sv_agg['target'] == target_val]
    click_means.append(subset['total_clicks'].mean())
    day_means.append(subset['active_days'].mean())

x = np.arange(3)
labels = [target_names[i] for i in range(3)]
bars1 = axes[1].bar(x - 0.15, click_means, 0.3, label='Ort. Tıklama', color='#3498db', edgecolor='black')
ax2 = axes[1].twinx()
bars2 = ax2.bar(x + 0.15, day_means, 0.3, label='Ort. Aktif Gün', color='#e74c3c', edgecolor='black')
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels)
axes[1].set_ylabel('Ort. Tıklama')
ax2.set_ylabel('Ort. Aktif Gün')
axes[1].set_title('Ortalama VLE Kullanımı', fontsize=12, fontweight='bold')
axes[1].legend(loc='upper left')
ax2.legend(loc='upper right')

plt.suptitle('VLE Etkileşim Analizi — OULAD', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{output_dir}/04_vle_engagement.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  [Grafik: {output_dir}/04_vle_engagement.png]")

for target_val, name in target_names.items():
    subset = sv_agg[sv_agg['target'] == target_val]
    print(f"  {name:15s}: ort tıklama={subset['total_clicks'].mean():.0f}, ort aktif gün={subset['active_days'].mean():.0f}")

# ============================================================
# 5. KORELASYON MATRİSİ
# ============================================================
print("\n" + "-" * 70)
print("  5. Korelasyon Matrisi")
print("-" * 70)

df = pd.read_csv("preprocessing/oulad_processed.csv")

fig, ax = plt.subplots(figsize=(16, 14))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            ax=ax, square=True, linewidths=0.5, annot_kws={'size': 7})
ax.set_title('Korelasyon Matrisi — OULAD', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{output_dir}/05_correlation_matrix.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  [Grafik: {output_dir}/05_correlation_matrix.png]")

target_corr = corr['target'].drop('target').sort_values(ascending=False)
print(f"\n  Target ile korelasyonlar:")
for feat, val in target_corr.items():
    bar = "+" * int(abs(val) * 30) if val > 0 else "-" * int(abs(val) * 30)
    print(f"    {feat:30s} r={val:+.3f} {bar}")

# ============================================================
# 6. MUTUAL INFORMATION BAR CHART
# ============================================================
print("\n" + "-" * 70)
print("  6. Mutual Information (Feature Importance)")
print("-" * 70)

from sklearn.feature_selection import mutual_info_classif

X = df.drop('target', axis=1)
y = df['target']
mi = mutual_info_classif(X, y, random_state=42)
mi_df = pd.DataFrame({'Feature': X.columns, 'MI': mi}).sort_values('MI', ascending=True)

fig, ax = plt.subplots(figsize=(10, 10))
colors = ['#e74c3c' if v < 0.05 else '#f39c12' if v < 0.15 else '#2ecc71' for v in mi_df['MI']]
ax.barh(mi_df['Feature'], mi_df['MI'], color=colors, edgecolor='black')
ax.set_title('Mutual Information — OULAD', fontsize=14, fontweight='bold')
ax.set_xlabel('MI Score')
plt.tight_layout()
plt.savefig(f"{output_dir}/06_mutual_information.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"  [Grafik: {output_dir}/06_mutual_information.png]")

print(f"\n  Top 10 özellik:")
for _, row in mi_df.tail(10).iloc[::-1].iterrows():
    print(f"    {row['Feature']:30s} MI={row['MI']:.4f}")

print("\n" + "=" * 70)
print("  EDA TAMAMLANDI — OULAD")
print("=" * 70)
