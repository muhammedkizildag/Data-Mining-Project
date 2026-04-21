# Öğrenci Başarı ve Terk Riski Tahmini

Bu proje, veri madenciliği dersi kapsamında hazırlanmış bir öğrenci başarı/terk riski tahmin çalışmasıdır. Proje; veri seti araştırması, keşifsel veri analizi, ön işleme, sınıflandırma modelleri, ablation study, Türkiye'ye uyarlanmış modelleme ve Streamlit tabanlı akademik danışman chatbotundan oluşur.

Ana hedef yalnızca yüksek doğruluklu bir model üretmek değil, eğitilmiş modeli öğrencinin doğal dille etkileşime geçebileceği bir danışman arayüzüne bağlamaktır.

## Proje Özeti

Projede üç ana veri hattı bulunur:

| Veri seti | Rol | Satır | Hedef | Durum |
|---|---:|---:|---|---|
| UCI Predict Students' Dropout and Academic Success | Ana akademik başarı/terk veri seti | 4.424 | Dropout / Enrolled / Graduate | Kullanılıyor |
| OULAD | Büyük ölçekli gerçek öğrenme analitiği veri seti | 32.593 | Withdrawn / Fail / Pass | Kullanılıyor |
| Student Habits vs Academic Performance | İlk yaşam alışkanlığı denemesi | 1.000 | Düşük / Orta / Yüksek | Karşılaştırma/geçmiş çalışma |

Son uygulama olan chatbot, Türkiye'ye uyarlanmış Dropout UCI modeliyle çalışır. Bu model `models/best_model_dropout_localized.pkl` dosyasındadır ve 22 özellik kullanır.

## Klasör Yapısı

```text
Data-Mining-Project/
├── datasets/
│   ├── dropout_academic_success/     # UCI Dropout veri seti
│   ├── oulad/                        # OULAD 7 ilişkisel CSV tablosu
│   ├── student_habits/               # Student Habits veri seti
│   ├── student_depression/           # Yedek/elenen veri seti
│   └── student_stress_factors/       # Yedek/elenen veri seti
├── student/                          # UCI Student Performance eski veri seti
├── eda/
│   ├── eda_dropout.py
│   ├── eda_habits.py
│   ├── eda_oulad.py
│   ├── plots_dropout/
│   ├── plots_habits/
│   └── plots_oulad/
├── preprocessing/
│   ├── preprocess_dropout.py
│   ├── preprocess_habits.py
│   ├── prepare_oulad.py
│   ├── dropout_processed.csv
│   ├── habits_processed.csv
│   └── oulad_processed.csv
├── modeling/
│   ├── model_dropout.py
│   ├── model_dropout_v2.py
│   ├── model_dropout_localized.py
│   ├── model_habits.py
│   ├── model_habits_v2.py
│   ├── model_oulad.py
│   ├── model_oulad_v2.py
│   ├── ablation_study.py
│   ├── ablation_study_oulad.py
│   └── plots_*/
├── models/
│   ├── best_model_dropout.pkl
│   ├── best_model_dropout_localized.pkl
│   ├── best_model_habits.pkl
│   ├── best_model_oulad.pkl
│   └── dropout_localized_features.pkl
├── chatbot/
│   ├── app.py
│   ├── prepare_chatbot.py
│   ├── feature_config.json
│   ├── scaler_params.json
│   └── reference_stats.json
├── gunluk.md                         # Proje karar günlüğü
├── main.py                           # Eski/bağlayıcı olmayan başlangıç kodu
├── dropout.py                        # Eski/bağlayıcı olmayan başlangıç kodu
└── requirements.txt
```

`main.py` ve `dropout.py` ilk deneme dosyalarıdır. Güncel akış için `eda/`, `preprocessing/`, `modeling/` ve `chatbot/` klasörleri esas alınmalıdır.

## Kurulum

Python 3.9+ önerilir.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Mevcut `requirements.txt` UTF-16 formatındadır. Bazı ortamlarda doğrudan `pip install -r requirements.txt` sorun çıkarabilir. Gerekli temel paketler şunlardır:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn joblib streamlit groq
```

## Çalıştırma Sırası

Tüm komutlar proje kök dizininden çalıştırılmalıdır.

### 1. EDA grafikleri

```bash
python eda/eda_dropout.py
python eda/eda_oulad.py
python eda/eda_habits.py
```

Çıktılar:

- `eda/plots_dropout/`
- `eda/plots_oulad/`
- `eda/plots_habits/`

### 2. Ön işleme

OULAD veri setindeki `studentVle.csv` dosyası GitHub dosya limitini aştığı için doğrudan commit edilmez. Bu dosya `datasets/oulad/oulad.zip` içinde bulunur. Temiz klonda OULAD ön işleme çalıştırılmadan önce zip dosyası açılmalıdır:

```bash
unzip datasets/oulad/oulad.zip -d datasets/oulad
```

```bash
python preprocessing/preprocess_dropout.py
python preprocessing/prepare_oulad.py
python preprocessing/preprocess_habits.py
```

Çıktılar:

| Dosya | Boyut | Hedef dağılımı |
|---|---:|---|
| `preprocessing/dropout_processed.csv` | 4.424 x 26 | 0: 1421, 1: 794, 2: 2209 |
| `preprocessing/oulad_processed.csv` | 32.593 x 31 | 0: 10156, 1: 7052, 2: 15385 |
| `preprocessing/habits_processed.csv` | 1.000 x 15 | 0: 131, 1: 492, 2: 377 |

### 3. Modelleme

Ana modeller:

```bash
python modeling/model_dropout.py
python modeling/model_dropout_v2.py
python modeling/model_oulad.py
python modeling/model_oulad_v2.py
```

Student Habits geçmiş/karşılaştırma modelleri:

```bash
python modeling/model_habits.py
python modeling/model_habits_v2.py
```

Türkiye'ye uyarlanmış chatbot modeli:

```bash
python modeling/model_dropout_localized.py
```

Model çıktıları `models/` klasörüne, grafikler `modeling/plots_*` klasörlerine kaydedilir.

### 4. Ablation study

```bash
python modeling/ablation_study.py
python modeling/ablation_study_oulad.py
```

Bu dosyalar XGBoost, feature engineering ve SMOTE gibi iyileştirmelerin tekil katkılarını karşılaştırır.

### 5. Chatbot hazırlığı

```bash
python chatbot/prepare_chatbot.py
```

Bu adım chatbotun ihtiyaç duyduğu şu dosyaları üretir/günceller:

- `chatbot/scaler_params.json`
- `chatbot/feature_config.json`
- `chatbot/reference_stats.json`

## Chatbot

Chatbot Streamlit ile çalışır ve Groq API üzerinden `llama-3.3-70b-versatile` modelini kullanır. LLM sohbetten yapılandırılmış öğrenci verisi çıkarır, bu veri normalize edilir ve `models/best_model_dropout_localized.pkl` XGBoost modeline gönderilir.

### API anahtarı

Önerilen yöntem:

```bash
export GROQ_API_KEY="groq_api_anahtariniz"
```

Alternatif olarak `chatbot/api_key.txt` dosyası oluşturulabilir. Bu dosya `.gitignore` kapsamındadır ve repoya eklenmemelidir.

### Uygulamayı başlatma

```bash
streamlit run chatbot/app.py
```

Chatbot akışı:

1. Öğrenci doğal dille akademik durumunu anlatır.
2. LLM mesajlardan model özelliklerini JSON formatında çıkarır.
3. Eksik düşük öncelikli alanlar varsayılan değerlerle tamamlanır.
4. XGBoost modeli Dropout / Enrolled / Graduate olasılıklarını üretir.
5. LLM sonucu yapıcı önerilerle öğrenciye açıklar.

## Model Sonuçları

### Student Habits v2

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| kNN | %56.33 | %57.40 | %56.33 | %56.74 |
| Naive Bayes | %67.33 | %71.47 | %67.33 | %68.20 |
| Decision Tree | %65.00 | %65.23 | %65.00 | %65.03 |
| Random Forest | %80.33 | %80.35 | %80.33 | %80.32 |
| XGBoost | %80.67 | %80.67 | %80.67 | %80.62 |

### Dropout UCI v2

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| kNN | %74.77 | %73.43 | %74.77 | %73.72 |
| Naive Bayes | %74.32 | %74.65 | %74.32 | %74.10 |
| Decision Tree | %75.98 | %74.32 | %75.98 | %74.62 |
| Random Forest | %77.71 | %76.46 | %77.71 | %76.64 |
| XGBoost | %78.01 | %77.04 | %78.01 | %77.26 |

### OULAD v2

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| kNN | %93.81 | %93.85 | %93.81 | %93.68 |
| Naive Bayes | %90.61 | %90.50 | %90.61 | %90.38 |
| Decision Tree | %92.80 | %92.75 | %92.80 | %92.68 |
| Random Forest | %94.72 | %94.70 | %94.72 | %94.66 |
| XGBoost | %94.88 | %94.86 | %94.88 | %94.82 |

### Türkiye'ye uyarlanmış Dropout modeli

Chatbotta kullanılan modeldir. Türkiye bağlamında doğrudan karşılığı zayıf olan üç özellik çıkarılmıştır:

- `Tuition fees up to date`
- `Debtor`
- `Inflation rate`

| Model | F1 |
|---|---:|
| kNN | %68.91 |
| Naive Bayes | %66.76 |
| Decision Tree | %71.54 |
| Random Forest | %73.85 |
| XGBoost | %75.27 |

Kaydedilen model:

```text
models/best_model_dropout_localized.pkl
```

## Veri Seti Kararları

Projede veri setleri doğrudan birleştirilmemiştir. Bunun nedeni farklı kaynaklardaki satırların aynı öğrencilere ait olmaması ve ortak bir kimlik alanı bulunmamasıdır. Rastgele birleştirme sahte korelasyon üreteceği için bilimsel olarak geçersiz kabul edilmiştir.

Student Habits veri seti ilk aşamada kullanılmış, fakat sentetik olması ve performans sınırı nedeniyle ana gerçek veri seti olarak OULAD'a geçilmiştir. Student Habits çalışmaları repoda karşılaştırma ve ablation geçmişi olarak korunur.

## Önemli Notlar

- OULAD modelinde `unregistered` gibi hedefe çok yakın sinyaller bulunduğu için performans oldukça yüksektir. Bu model dönem sonu/durum sınıflandırması için güçlüdür; erken uyarı sistemi olarak kullanılacaksa zaman kesiti ve özellik seçimi yeniden tasarlanmalıdır.
- Ön işleme dosyaları mevcut haliyle tüm veri üzerinde scaler/feature selection uygular. Raporlamada bu tercih açıklanmalı; daha katı üretim standardı için `Pipeline` ve train-only fit yaklaşımı tercih edilmelidir.
- `main.py` ve `dropout.py` eski başlangıç kodlarıdır; güncel deney sonuçları için `modeling/` altındaki dosyalar kullanılmalıdır.
- API anahtarı hiçbir koşulda repoya commit edilmemelidir.

## Proje Günlüğü

Ayrıntılı karar süreci, veri seti araştırması, modelleme gerekçeleri ve sonuç yorumları `gunluk.md` dosyasında tutulmuştur. README güncel çalıştırma ve proje haritası için, `gunluk.md` ise ayrıntılı rapor/karar geçmişi için kullanılmalıdır.
