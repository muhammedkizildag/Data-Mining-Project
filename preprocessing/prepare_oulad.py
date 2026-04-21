import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("  OULAD — VERİ HAZIRLAMA (7 Tablo → 1 Dataset)")
print("=" * 70)

base = "datasets/oulad"

studentInfo = pd.read_csv(f"{base}/studentInfo.csv")
assessments = pd.read_csv(f"{base}/assessments.csv")
studentAssessment = pd.read_csv(f"{base}/studentAssessment.csv")
vle = pd.read_csv(f"{base}/vle.csv")
studentVle = pd.read_csv(f"{base}/studentVle.csv")
studentRegistration = pd.read_csv(f"{base}/studentRegistration.csv")
courses = pd.read_csv(f"{base}/courses.csv")

print(f"\n  studentInfo:         {studentInfo.shape}")
print(f"  assessments:         {assessments.shape}")
print(f"  studentAssessment:   {studentAssessment.shape}")
print(f"  vle:                 {vle.shape}")
print(f"  studentVle:          {studentVle.shape}")
print(f"  studentRegistration: {studentRegistration.shape}")
print(f"  courses:             {courses.shape}")

# ============================================================
# HEDEF DEĞİŞKEN: 4 sınıf → 3 sınıf
# ============================================================
print("\n" + "=" * 70)
print("  HEDEF DEĞİŞKEN DÖNÜŞÜMÜ")
print("=" * 70)

print(f"\n  Orijinal (4 sınıf):")
print(f"  {studentInfo['final_result'].value_counts().to_dict()}")

target_map = {
    'Withdrawn': 0,
    'Fail': 1,
    'Pass': 2,
    'Distinction': 2
}
studentInfo['target'] = studentInfo['final_result'].map(target_map)

target_names = {0: 'Withdrawn', 1: 'Fail', 2: 'Pass'}
print(f"\n  Yeni (3 sınıf):")
for val in [0, 1, 2]:
    cnt = (studentInfo['target'] == val).sum()
    pct = cnt / len(studentInfo) * 100
    print(f"    {target_names[val]}: {cnt} (%{pct:.1f})")

# ============================================================
# ÖĞRENCİ BAZLI UNIQUE KEY
# ============================================================
key_cols = ['code_module', 'code_presentation', 'id_student']

# ============================================================
# ASSESSMENT ÖZELLİKLERİ
# ============================================================
print("\n" + "=" * 70)
print("  ASSESSMENT ÖZELLİKLERİ")
print("=" * 70)

sa = studentAssessment.merge(assessments, on='id_assessment', how='left')

sa_agg = sa.groupby(key_cols).agg(
    avg_score=('score', 'mean'),
    std_score=('score', 'std'),
    min_score=('score', 'min'),
    max_score=('score', 'max'),
    num_assessments=('score', 'count'),
    num_missing_score=('score', lambda x: x.isnull().sum()),
).reset_index()
sa_agg['std_score'] = sa_agg['std_score'].fillna(0)

for atype in ['TMA', 'CMA', 'Exam']:
    subset = sa[sa['assessment_type'] == atype]
    type_agg = subset.groupby(key_cols).agg(
        **{f'avg_score_{atype}': ('score', 'mean'),
           f'num_{atype}': ('score', 'count')}
    ).reset_index()
    sa_agg = sa_agg.merge(type_agg, on=key_cols, how='left')

sa_agg = sa_agg.fillna(0)

submit = sa.dropna(subset=['date_submitted', 'date']).copy()
submit['submit_delay'] = submit['date_submitted'] - submit['date']
delay_agg = submit.groupby(key_cols).agg(
    avg_submit_delay=('submit_delay', 'mean'),
    late_submissions=('submit_delay', lambda x: (x > 0).sum()),
    early_submissions=('submit_delay', lambda x: (x <= 0).sum()),
).reset_index()
sa_agg = sa_agg.merge(delay_agg, on=key_cols, how='left')
sa_agg = sa_agg.fillna(0)

print(f"  Assessment özellikleri: {sa_agg.shape[1] - 3} özellik")
print(f"  Özellikler: {[c for c in sa_agg.columns if c not in key_cols]}")

# ============================================================
# VLE (SANAL ÖĞRENME ORTAMI) ÖZELLİKLERİ
# ============================================================
print("\n" + "=" * 70)
print("  VLE ÖZELLİKLERİ (10.6M tıklama → öğrenci bazlı)")
print("=" * 70)

sv = studentVle.merge(vle[['id_site', 'activity_type']], on='id_site', how='left')

vle_total = sv.groupby(key_cols).agg(
    total_clicks=('sum_click', 'sum'),
    total_vle_days=('date', 'nunique'),
    avg_daily_clicks=('sum_click', 'mean'),
    num_distinct_activities=('id_site', 'nunique'),
).reset_index()

top_activities = ['resource', 'oucontent', 'url', 'forumng', 'quiz',
                  'subpage', 'homepage', 'questionnaire', 'page']
for act in top_activities:
    act_sub = sv[sv['activity_type'] == act]
    act_agg = act_sub.groupby(key_cols).agg(
        **{f'clicks_{act}': ('sum_click', 'sum')}
    ).reset_index()
    vle_total = vle_total.merge(act_agg, on=key_cols, how='left')

vle_total = vle_total.fillna(0)

print(f"  VLE özellikleri: {vle_total.shape[1] - 3} özellik")
print(f"  Özellikler: {[c for c in vle_total.columns if c not in key_cols]}")

# ============================================================
# KAYIT ÖZELLİKLERİ
# ============================================================
print("\n" + "=" * 70)
print("  KAYIT ÖZELLİKLERİ")
print("=" * 70)

reg = studentRegistration.copy()
reg['unregistered'] = reg['date_unregistration'].notna().astype(int)
reg_features = reg[key_cols + ['date_registration', 'unregistered']]

print(f"  Kayıt özellikleri: date_registration, unregistered")

# ============================================================
# DERS ÖZELLİKLERİ
# ============================================================
course_features = courses.rename(columns={'module_presentation_length': 'course_length'})

# ============================================================
# HEPSİNİ BİRLEŞTİR
# ============================================================
print("\n" + "=" * 70)
print("  TABLOLARI BİRLEŞTİR")
print("=" * 70)

df = studentInfo.copy()
df = df.merge(sa_agg, on=key_cols, how='left')
df = df.merge(vle_total, on=key_cols, how='left')
df = df.merge(reg_features, on=key_cols, how='left')
df = df.merge(course_features, on=['code_module', 'code_presentation'], how='left')

print(f"  Birleştirme sonrası: {df.shape[0]} satır × {df.shape[1]} sütun")

numeric_fill_cols = [c for c in df.columns if df[c].dtype in ['float64', 'int64'] and c != 'target']
for c in numeric_fill_cols:
    if df[c].isnull().sum() > 0:
        df[c] = df[c].fillna(0)

df['imd_band'] = df['imd_band'].fillna(df['imd_band'].mode()[0])

print(f"  Eksik veri sonrası: {df.isnull().sum().sum()} eksik")

# ============================================================
# ENCODING
# ============================================================
print("\n" + "=" * 70)
print("  ENCODING")
print("=" * 70)

from sklearn.preprocessing import LabelEncoder

drop_cols = ['code_module', 'code_presentation', 'id_student', 'final_result']
df = df.drop(columns=drop_cols)

cat_cols = df.select_dtypes(include='object').columns.tolist()
print(f"  Kategorik sütunlar: {cat_cols}")

le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"    {col}: {le_dict[col]}")

# ============================================================
# NORMALİZASYON
# ============================================================
print("\n" + "=" * 70)
print("  NORMALİZASYON")
print("=" * 70)

from sklearn.preprocessing import MinMaxScaler

feature_cols = [c for c in df.columns if c != 'target']
scaler = MinMaxScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

print(f"  {len(feature_cols)} özellik MinMaxScaler ile 0-1 aralığına normalize edildi")

# ============================================================
# FEATURE SELECTION (Mutual Information)
# ============================================================
print("\n" + "=" * 70)
print("  FEATURE SELECTION (Mutual Information)")
print("=" * 70)

from sklearn.feature_selection import mutual_info_classif

X = df.drop('target', axis=1)
y = df['target']

mi = mutual_info_classif(X, y, random_state=42)
mi_df = pd.DataFrame({'Özellik': X.columns, 'MI': mi}).sort_values('MI', ascending=False)

print(f"\n  Mutual Information Skorları:")
for _, row in mi_df.iterrows():
    bar = "█" * int(row['MI'] * 50)
    flag = " ← DÜŞ" if row['MI'] < 0.01 else ""
    print(f"    {row['Özellik']:30s} MI={row['MI']:.4f} {bar}{flag}")

low_mi = mi_df[mi_df['MI'] < 0.01]['Özellik'].tolist()
if low_mi:
    print(f"\n  MI < 0.01 olan {len(low_mi)} özellik çıkarılıyor: {low_mi}")
    df = df.drop(columns=low_mi)

# ============================================================
# KAYDET
# ============================================================
print("\n" + "=" * 70)
print("  SONUÇ")
print("=" * 70)

output_path = "preprocessing/oulad_processed.csv"
df.to_csv(output_path, index=False)

print(f"  Final veri seti: {df.shape[0]} satır × {df.shape[1]} sütun")
print(f"  Özellik sayısı: {df.shape[1] - 1}")
print(f"  Hedef: target (0=Withdrawn, 1=Fail, 2=Pass)")
print(f"\n  Sınıf dağılımı:")
for val in [0, 1, 2]:
    cnt = (df['target'] == val).sum()
    pct = cnt / len(df) * 100
    print(f"    {target_names[val]}: {cnt} (%{pct:.1f})")
print(f"\n  Kaydedildi: {output_path}")

print("\n" + "=" * 70)
print("  VERİ HAZIRLAMA TAMAMLANDI")
print("=" * 70)
