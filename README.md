# Öğrenci Başarı ve Terk Riski Tahmini

Bu proje, veri madenciliği dersi kapsamında hazırlanmış bir öğrenci başarı/terk riski tahmin çalışmasıdır. Proje; veri seti araştırması, keşifsel veri analizi, ön işleme, sınıflandırma modelleri, metodolojik hata düzeltmeleri, Türkiye'ye uyarlanmış modelleme ve Streamlit tabanlı akademik danışman chatbotundan oluşur.

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
│   ├── model_dropout_localized.py     # Güncel chatbot/Dropout modeli
│   ├── model_oulad_v2.py              # Güncel OULAD modeli
│   ├── model_dropout.py               # Legacy baseline
│   ├── model_dropout_v2.py            # Legacy, aktif değil
│   ├── model_habits.py                # Legacy
│   ├── model_habits_v2.py             # Legacy, aktif değil
│   ├── model_oulad.py                 # Legacy baseline
│   ├── ablation_study.py              # Legacy
│   ├── ablation_study_oulad.py        # Legacy
│   └── plots_*/
├── models/
│   ├── best_model_dropout_localized.pkl
│   ├── best_model_oulad.pkl
│   ├── best_model_habits.pkl
│   ├── dropout_localized_features.pkl
│   └── dropout_localized_scaler_params.json
├── chatbot/
│   ├── app.py
│   ├── prepare_chatbot.py
│   ├── feature_config.json
│   └── reference_stats.json
├── GUNLUK.md                         # Proje karar günlüğü
└── requirements.txt
```

Güncel sonuçlar için esas dosyalar `preprocessing/preprocess_dropout.py`, `preprocessing/prepare_oulad.py`, `modeling/model_dropout_localized.py`, `modeling/model_oulad_v2.py` ve `chatbot/app.py` dosyalarıdır. Bazı eski model/ablation dosyaları geçmiş karşılaştırmaları korumak için repoda tutulur; başlarında legacy uyarısı bulunan bu dosyalar güncel metodolojik sonuç olarak kullanılmamalıdır.

## Kurulum

Python 3.9+ önerilir.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Bağımlılıklar `requirements.txt` dosyasında listelenmiştir:

```bash
pip install -r requirements.txt
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
```

Çıktılar:

| Dosya | Boyut | Hedef dağılımı |
|---|---:|---|
| `preprocessing/dropout_processed.csv` | 4.424 x 37 | 0: 1421, 1: 794, 2: 2209 |
| `preprocessing/oulad_processed.csv` | 32.593 x 39 | 0: 10156, 1: 7052, 2: 15385 |

Bu aşamada scaler ve mutual information feature selection uygulanmaz. Bu işlemler data leakage oluşmaması için modelleme dosyalarında train/test ayrımından sonra yapılır.

### 3. Güncel modelleme

Güncel ve raporlanabilir ana modeller:

```bash
python modeling/model_dropout_localized.py
python modeling/model_oulad_v2.py
```

`model_dropout_localized.py`, chatbotta kullanılan Türkiye'ye uyarlanmış Dropout UCI modelini üretir. `model_oulad_v2.py`, `unregistered` target leakage özelliği çıkarılmış OULAD modelidir.

Legacy/geçmiş karşılaştırma dosyaları:

```bash
python modeling/model_dropout.py
python modeling/model_dropout_v2.py
python modeling/model_habits.py
python modeling/model_habits_v2.py
python modeling/model_oulad.py
python modeling/ablation_study.py
python modeling/ablation_study_oulad.py
```

Bu legacy dosyalar güncel preprocessing çıktılarıyla birebir uyumlu olmayabilir ve bazı eski dosyalarda split öncesi scaler/feature engineering kullanımı bulunduğu için güncel final sonuç olarak raporlanmamalıdır.

Model çıktıları `models/` klasörüne, grafikler `modeling/plots_*` klasörlerine kaydedilir.

### 4. Chatbot hazırlığı

```bash
python chatbot/prepare_chatbot.py
```

Bu adım chatbotun ihtiyaç duyduğu şu dosyaları üretir/günceller:

- `chatbot/feature_config.json`
- `chatbot/reference_stats.json`

## Chatbot

Chatbot Streamlit ile çalışır ve Groq API üzerinden `llama-3.3-70b-versatile` modelini kullanır. LLM sohbetten yapılandırılmış öğrenci verisi çıkarır, bu veri `models/best_model_dropout_localized.pkl` dosyasındaki Pipeline'a (MinMaxScaler + XGBoost) ham olarak gönderilir. Normalizasyon Pipeline içinde otomatik yapılır.

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

### Türkiye'ye uyarlanmış Dropout modeli

Chatbotta kullanılan modeldir. Türkiye bağlamında doğrudan karşılığı zayıf olan üç özellik çıkarılmıştır:

- `Tuition fees up to date`
- `Debtor`
- `Inflation rate`

| Model | F1 |
|---|---:|
| kNN | %67.27 |
| Naive Bayes | %66.76 |
| Decision Tree | %70.57 |
| Random Forest | %73.44 |
| XGBoost | %75.10 |

XGBoost sınıf bazlı sonuçları:

| Sınıf | Precision | Recall | F1 |
|---|---:|---:|---:|
| Dropout | %78.88 | %72.60 | %75.61 |
| Enrolled | %51.53 | %42.44 | %46.54 |
| Graduate | %80.65 | %89.89 | %85.02 |

Kaydedilen model ve yardımcı dosyalar:

```text
models/best_model_dropout_localized.pkl    # Pipeline: MinMaxScaler + XGBoost
models/dropout_localized_features.pkl      # Özellik listesi
models/dropout_localized_scaler_params.json # Scaler parametreleri (dokümantasyon)
```

### OULAD v2

| Model | F1 |
|---|---:|
| kNN | %77.18 |
| Naive Bayes | %69.05 |
| Decision Tree | %77.55 |
| Random Forest | %80.03 |
| XGBoost | %80.48 |

XGBoost sınıf bazlı sonuçları:

| Sınıf | Precision | Recall | F1 |
|---|---:|---:|---:|
| Withdrawn | %75.26 | %82.87 | %78.88 |
| Fail | %60.82 | %45.84 | %52.28 |
| Pass | %92.38 | %96.64 | %94.46 |

OULAD modelinde eski `%94` civarı skorların ana nedeni `unregistered` özelliğinin hedef değişkene çok yakın bilgi taşımasıydı. Bu özellik çıkarıldıktan sonra temiz OULAD v2 XGBoost F1 skoru `%80.48` olarak raporlanır.

### Legacy Sonuçlar

Student Habits, Dropout UCI v1/v2 ve eski ablation çalışmaları proje gelişim sürecini göstermek için korunur. Ancak final metodolojik değerlendirmede yukarıdaki iki güncel model esas alınmalıdır.

## Veri Seti Kararları

Projede veri setleri doğrudan birleştirilmemiştir. Bunun nedeni farklı kaynaklardaki satırların aynı öğrencilere ait olmaması ve ortak bir kimlik alanı bulunmamasıdır. Rastgele birleştirme sahte korelasyon üreteceği için bilimsel olarak geçersiz kabul edilmiştir.

Student Habits veri seti ilk aşamada kullanılmış, fakat sentetik olması ve performans sınırı nedeniyle ana gerçek veri seti olarak OULAD'a geçilmiştir. Student Habits çalışmaları repoda karşılaştırma ve ablation geçmişi olarak korunur.

## Önemli Notlar

- OULAD tarafında `unregistered` target leakage özelliği çıkarılmıştır. Eski `%94` civarı skorlar metodolojik olarak temiz final skor kabul edilmemelidir.
- OULAD modeli dönem sonu/durum sınıflandırması olarak konumlandırılmalıdır. Erken uyarı sistemi olarak kullanılacaksa yalnızca belirli bir zaman kesitine kadar oluşan VLE ve assessment verileriyle yeniden tasarlanmalıdır.
- Güncel modelleme dosyalarında scaler ve model `sklearn.pipeline.Pipeline` içinde birlikte çalışır. CV her fold'unda scaler sadece o fold'un training kısmından fit edilir. OULAD v2'de MI feature selection Pipeline dışında raw train verisi üzerinde yapılır; bu bilinçli bir tercih olup pratikte etkisi yoktur (çıkarılan 8 özelliğin MI skorları sıfıra yakındır).
- Başında legacy uyarısı bulunan eski model ve ablation dosyaları final sonuç üretmek için değil, proje geçmişini göstermek için saklanır.
- API anahtarı hiçbir koşulda repoya commit edilmemelidir.

## Proje Günlüğü

Ayrıntılı karar süreci, veri seti araştırması, modelleme gerekçeleri ve sonuç yorumları `GUNLUK.md` dosyasında tutulmuştur. README güncel çalıştırma ve proje haritası için, `GUNLUK.md` ise ayrıntılı rapor/karar geçmişi için kullanılmalıdır.
