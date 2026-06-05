import pandas as pd
import joblib
import json

print("=" * 70)
print("  CHATBOT HAZIRLIK — Normalizasyon ve Konfigürasyon")
print("=" * 70)

df_raw = pd.read_csv("datasets/dropout_academic_success/data.csv", sep=";")

target_map = {'Dropout': 0, 'Enrolled': 1, 'Graduate': 2}
df_raw['Target'] = df_raw['Target'].map(target_map)

features = [
    'Marital status', 'Application mode', 'Application order', 'Course',
    'Previous qualification', 'Previous qualification (grade)',
    "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation",
    'Admission grade', 'Gender', 'Scholarship holder', 'Age at enrollment',
    'Curricular units 1st sem (enrolled)', 'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (approved)', 'Curricular units 1st sem (grade)',
    'Curricular units 2nd sem (enrolled)', 'Curricular units 2nd sem (evaluations)',
    'Curricular units 2nd sem (approved)', 'Curricular units 2nd sem (grade)'
]

feature_config = {
    "Marital status": {
        "tr": "Medeni durum",
        "question": "Medeni durumunuz nedir?",
        "type": "categorical",
        "options": {"Bekar": 1, "Evli": 2, "Dul": 3, "Boşanmış": 4, "Birlikte yaşıyor": 5, "Ayrılmış": 6},
        "default": 1,
        "priority": "low"
    },
    "Application mode": {
        "tr": "Başvuru türü",
        "question": "Üniversiteye nasıl başvurdunuz?",
        "type": "categorical",
        "options": {"YKS (normal sınav)": 1, "DGS (dikey geçiş)": 15, "Yatay geçiş": 17, "Uluslararası": 39, "Özel yetenek": 42, "25 yaş üstü": 26, "Diğer": 7},
        "default": 1,
        "priority": "low"
    },
    "Application order": {
        "tr": "Tercih sırası",
        "question": "Bu bölüm kaçıncı tercihinizdi?",
        "type": "numeric",
        "range": [0, 9],
        "default": 1,
        "priority": "medium"
    },
    "Course": {
        "tr": "Bölüm",
        "question": "Hangi bölümde okuyorsunuz?",
        "type": "categorical",
        "options": {"Hemşirelik": 9500, "Sosyal Hizmet": 9556, "Gazetecilik": 9070, "Yönetim": 9147, "Eğitim": 9853, "Mühendislik": 9119, "Bilişim": 9130, "Tasarım": 9670, "Tarım": 9085, "Veterinerlik": 8014, "Diğer": 9003},
        "default": 9003,
        "priority": "medium"
    },
    "Previous qualification": {
        "tr": "Önceki eğitim",
        "question": "Üniversiteye gelmeden önceki eğitiminiz nedir?",
        "type": "categorical",
        "options": {"Lise": 1, "Lisans": 2, "Önlisans": 5, "Yüksek Lisans": 3, "Doktora": 4, "Diğer": 12},
        "default": 1,
        "priority": "low"
    },
    "Previous qualification (grade)": {
        "tr": "Önceki eğitim notu",
        "question": "Lise/önceki eğitim not ortalamanız neydi? (100 üzerinden)",
        "type": "numeric",
        "range": [0, 100],
        "default": 70,
        "priority": "medium"
    },
    "Mother's qualification": {
        "tr": "Anne eğitim düzeyi",
        "question": "Annenizin eğitim düzeyi nedir?",
        "type": "categorical",
        "options": {"İlkokul": 4, "Ortaokul": 9, "Lise": 1, "Üniversite": 2, "Yüksek lisans": 3, "Okuryazar değil": 34, "Bilinmiyor": 29},
        "default": 1,
        "priority": "low"
    },
    "Father's qualification": {
        "tr": "Baba eğitim düzeyi",
        "question": "Babanızın eğitim düzeyi nedir?",
        "type": "categorical",
        "options": {"İlkokul": 4, "Ortaokul": 9, "Lise": 1, "Üniversite": 2, "Yüksek lisans": 3, "Okuryazar değil": 34, "Bilinmiyor": 29},
        "default": 1,
        "priority": "low"
    },
    "Mother's occupation": {
        "tr": "Anne mesleği",
        "question": "Anneniz ne iş yapıyor?",
        "type": "categorical",
        "options": {"Çalışmıyor/Ev hanımı": 0, "Memur": 1, "İşçi": 9, "Serbest meslek": 5, "Emekli": 10, "Öğretmen": 3, "Sağlık": 2, "Diğer": 7},
        "default": 0,
        "priority": "low"
    },
    "Father's occupation": {
        "tr": "Baba mesleği",
        "question": "Babanız ne iş yapıyor?",
        "type": "categorical",
        "options": {"Çalışmıyor": 0, "Memur": 1, "İşçi": 9, "Serbest meslek": 5, "Emekli": 10, "Mühendis/Teknik": 4, "Esnaf": 7, "Diğer": 3},
        "default": 9,
        "priority": "low"
    },
    "Admission grade": {
        "tr": "Üniversite giriş puanı",
        "question": "YKS puanınız neydi? (500 üzerinden)",
        "type": "numeric",
        "range": [100, 500],
        "default": 300,
        "priority": "medium"
    },
    "Gender": {
        "tr": "Cinsiyet",
        "question": "Cinsiyetiniz?",
        "type": "categorical",
        "options": {"Kadın": 0, "Erkek": 1},
        "default": 1,
        "priority": "high"
    },
    "Scholarship holder": {
        "tr": "Burs durumu",
        "question": "Burs alıyor musunuz?",
        "type": "categorical",
        "options": {"Hayır": 0, "Evet": 1},
        "default": 0,
        "priority": "high"
    },
    "Age at enrollment": {
        "tr": "Kayıt yaşı",
        "question": "Üniversiteye kaç yaşında başladınız?",
        "type": "numeric",
        "range": [17, 70],
        "default": 19,
        "priority": "high"
    },
    "Curricular units 1st sem (enrolled)": {
        "tr": "1. dönem alınan ders",
        "question": "1. dönem kaç ders aldınız?",
        "type": "numeric",
        "range": [0, 26],
        "default": 6,
        "priority": "essential"
    },
    "Curricular units 1st sem (evaluations)": {
        "tr": "1. dönem girilen sınav",
        "question": "1. dönem kaç sınava girdiniz?",
        "type": "numeric",
        "range": [0, 45],
        "default": 6,
        "priority": "essential"
    },
    "Curricular units 1st sem (approved)": {
        "tr": "1. dönem geçilen ders",
        "question": "1. dönem kaç ders geçtiniz?",
        "type": "numeric",
        "range": [0, 26],
        "default": 5,
        "priority": "essential"
    },
    "Curricular units 1st sem (grade)": {
        "tr": "1. dönem not ortalaması",
        "question": "1. dönem not ortalamanız? (0-20 arası)",
        "type": "numeric",
        "range": [0, 18.875],
        "default": 12,
        "priority": "essential"
    },
    "Curricular units 2nd sem (enrolled)": {
        "tr": "2. dönem alınan ders",
        "question": "2. dönem kaç ders aldınız?",
        "type": "numeric",
        "range": [0, 23],
        "default": 6,
        "priority": "essential"
    },
    "Curricular units 2nd sem (evaluations)": {
        "tr": "2. dönem girilen sınav",
        "question": "2. dönem kaç sınava girdiniz?",
        "type": "numeric",
        "range": [0, 33],
        "default": 6,
        "priority": "essential"
    },
    "Curricular units 2nd sem (approved)": {
        "tr": "2. dönem geçilen ders",
        "question": "2. dönem kaç ders geçtiniz?",
        "type": "numeric",
        "range": [0, 20],
        "default": 5,
        "priority": "essential"
    },
    "Curricular units 2nd sem (grade)": {
        "tr": "2. dönem not ortalaması",
        "question": "2. dönem not ortalamanız? (0-20 arası)",
        "type": "numeric",
        "range": [0, 18.571428571428573],
        "default": 12,
        "priority": "essential"
    }
}

with open("chatbot/feature_config.json", "w", encoding="utf-8") as f:
    json.dump(feature_config, f, ensure_ascii=False, indent=2)

print(f"  Feature config kaydedildi: chatbot/feature_config.json")

ref_stats = {}
for target_val, target_name in {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}.items():
    subset = df_raw[df_raw['Target'] == target_val]
    stats = {}
    for col in features:
        stats[col] = {
            'mean': round(float(subset[col].mean()), 2),
            'median': round(float(subset[col].median()), 2)
        }
    ref_stats[target_name] = stats

with open("chatbot/reference_stats.json", "w", encoding="utf-8") as f:
    json.dump(ref_stats, f, ensure_ascii=False, indent=2)

print(f"  Referans istatistikler kaydedildi: chatbot/reference_stats.json")

print(f"\n  Özet:")
print(f"  - Essential (mutlaka sorulacak): {sum(1 for v in feature_config.values() if v['priority'] == 'essential')} özellik")
print(f"  - High (önemli): {sum(1 for v in feature_config.values() if v['priority'] == 'high')} özellik")
print(f"  - Medium (sorulabilir): {sum(1 for v in feature_config.values() if v['priority'] == 'medium')} özellik")
print(f"  - Low (varsayılan kullanılabilir): {sum(1 for v in feature_config.values() if v['priority'] == 'low')} özellik")

print("\n" + "=" * 70)
print("  CHATBOT HAZIRLIK TAMAMLANDI")
print("=" * 70)
