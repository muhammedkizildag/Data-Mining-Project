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
│   ├── shap_oulad.py                  # OULAD için SHAP analizi
│   ├── shap_dropout_localized.py       # SHAP açıklanabilirlik analizi
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

SHAP açıklanabilirlik grafikleri için:

```bash
python modeling/shap_dropout_localized.py
python modeling/shap_oulad.py
```

Bu adımlar ilgili Pipeline modellerini kullanır ve çıktıları sırasıyla `modeling/plots_shap_dropout_localized/` ve `modeling/plots_shap_oulad/` klasörlerine kaydeder.

Güncel modelleme script'leri ayrıca seçilen final model için şu ek grafikleri de üretir:
- `03_roc_pr_curves.png` — çok sınıflı ROC ve Precision-Recall eğrileri
- `05_learning_curve.png` — Macro F1 tabanlı learning curve

### 4. Chatbot hazırlığı

```bash
python chatbot/prepare_chatbot.py
```

Bu adım chatbotun ihtiyaç duyduğu şu dosyaları üretir/günceller:

- `chatbot/feature_config.json`
- `chatbot/reference_stats.json`

### 5. Smoke testleri

Temel artifact ve şema kontrollerini çalıştırmak için:

```bash
python -m unittest discover -s tests -v
```

Bu testler şunları doğrular:
- kaydedilmiş Dropout Localized modelinin yüklenip örnek tahmin üretebildiğini,
- chatbot feature sırasının model feature listesiyle eşleştiğini,
- kaydedilmiş OULAD modelinin örnek tahmin üretebildiğini,
- işlenmiş veri dosyalarında beklenen target/feature şemasının bulunduğunu.

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

Optimizasyon `f1_macro` ile yapılır (her sınıfa eşit ağırlık). Decision Tree ve Random Forest `class_weight='balanced'`, XGBoost `sample_weight` kullanır. Model seçimi 10-fold CV Macro F1 skoruna göre yapılır, test seti sadece final raporlama için bir kez kullanılır.

10-Fold CV sonuçları:

| Model | CV Macro F1 | CV Weighted F1 |
|---|---:|---:|
| kNN | %61.13 | %68.15 |
| Naive Bayes | %60.04 | %67.03 |
| Decision Tree | %63.83 | %68.72 |
| Random Forest | %69.04 | %74.47 |
| XGBoost | %68.88 | %73.96 |

Seçilen model: **Random Forest** (CV Macro F1: %69.04). Test seti sınıf bazlı sonuçları:

| Sınıf | Precision | Recall | F1 |
|---|---:|---:|---:|
| Dropout | %83.38 | %65.81 | %73.56 |
| Enrolled | %44.09 | %57.98 | %50.09 |
| Graduate | %82.60 | %84.46 | %83.52 |

Test Macro F1: %69.06, Test Weighted F1: %74.33

Kaydedilen model ve yardımcı dosyalar:

```text
models/best_model_dropout_localized.pkl    # Pipeline: MinMaxScaler + Random Forest
models/dropout_localized_features.pkl      # Özellik listesi
models/dropout_localized_scaler_params.json # Scaler parametreleri (dokümantasyon)
```

### OULAD v2

Aynı şekilde `f1_macro` optimizasyonu, `class_weight='balanced'` / `sample_weight` ve CV tabanlı model seçimi kullanılır.

10-Fold CV sonuçları:

| Model | CV Macro F1 | CV Weighted F1 |
|---|---:|---:|
| kNN | %72.03 | %77.98 |
| Naive Bayes | %61.74 | %69.51 |
| Decision Tree | %72.82 | %77.75 |
| Random Forest | %75.15 | %79.98 |
| XGBoost | %76.33 | %80.99 |

Seçilen model: **XGBoost** (CV Macro F1: %76.33). Test seti sınıf bazlı sonuçları:

| Sınıf | Precision | Recall | F1 |
|---|---:|---:|---:|
| Withdrawn | %78.64 | %74.89 | %76.72 |
| Fail | %55.20 | %58.65 | %56.87 |
| Pass | %93.56 | %93.82 | %93.69 |

Test Macro F1: %75.76, Test Weighted F1: %80.44

OULAD modelinde eski `%94` civarı skorların ana nedeni `unregistered` özelliğinin hedef değişkene çok yakın bilgi taşımasıydı. Bu özellik çıkarıldıktan sonra temiz OULAD v2 XGBoost Weighted F1 skoru `%80.44`, Macro F1 `%75.76` olarak raporlanır.

### Legacy Sonuçlar

Student Habits, Dropout UCI v1/v2 ve eski ablation çalışmaları proje gelişim sürecini göstermek için korunur. Ancak final metodolojik değerlendirmede yukarıdaki iki güncel model esas alınmalıdır.

### Model Açıklanabilirliği

Dropout Localized modeli için SHAP analizi `modeling/shap_dropout_localized.py` ile, OULAD modeli için ise `modeling/shap_oulad.py` ile üretilir. SHAP sonuçları modelin tahmininde hangi değişkenlerin daha etkili göründüğünü açıklar; bu sonuçlar nedensellik iddiası olarak yorumlanmamalıdır.

### Ek Değerlendirme Grafikleri

Dropout Localized ve OULAD v2 script'leri seçilen final model için çok sınıflı ROC/PR eğrileri ile learning curve grafikleri de üretir. ROC/PR tarafında her sınıf ayrı ayrı ve micro-average olarak raporlanır; learning curve tarafında ise eğitim örnek sayısı arttıkça train/validation Macro F1 davranışı gözlemlenir.

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
