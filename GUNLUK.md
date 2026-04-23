# Proje Günlüğü — Öğrenci Başarı Tahmini (Veri Madenciliği)

## Proje Tanımı

Bu proje, veri madenciliği dersi kapsamında geliştirilen bir ödev projesidir. Amaç, öğrencilerin akademik verilerine, yaşam alışkanlıklarına ve demografik bilgilerine dayalı olarak başarı durumlarını tahmin eden bir sınıflandırma modeli oluşturmaktır.

Projenin nihai hedefi sadece bir model değil, aynı zamanda öğrencilerin etkileşime geçebileceği bir **chatbot** arayüzü sunmaktır.

---

## Başlangıç Durumu

Projede daha önce bir ödev kapsamında yazılmış iki Python dosyası mevcuttu:

- **main.py** — UCI Student Performance veri seti üzerinde kNN ve Naive Bayes ile geçti/kaldı (binary) sınıflandırması
- **dropout.py** — Öğrenci terk veri seti üzerinde 3 sınıflı (Dropout/Enrolled/Graduate) sınıflandırma

Bu dosyalar öncül çalışmalardı ve bağlayıcı değildi. Proje sıfırdan şekillendirildi.

---

## Veri Seti Araştırması

### Araştırma Süreci

Kaggle, UCI ML Repository ve diğer açık veri kaynaklarında "student dropout prediction", "student academic success", "student performance prediction" gibi terimlerle kapsamlı bir araştırma yapıldı. Toplamda 9 aday veri seti belirlendi.

### Değerlendirilen Veri Setleri

| # | Veri Seti | Satır | Sütun | Hedef Değişken | Kaynak |
|---|-----------|-------|-------|----------------|--------|
| 1 | UCI Predict Students' Dropout and Academic Success | 4.424 | 37 | Target (Dropout/Enrolled/Graduate) | UCI + Kaggle |
| 2 | Open University Learning Analytics Dataset (OULAD) | 32.593 | 7 tablo | final_result (Pass/Fail/Withdrawn/Distinction) | Open University |
| 3 | UCI Student Performance | 649 | 33 | G3 (sınav notu) | UCI + Kaggle |
| 4 | Student Performance Factors | 6.607 | 20 | Exam_Score (sürekli) | Kaggle |
| 5 | Student Performance & Behavior | 14.003 | 16 | ExamScore/FinalGrade | Zenodo + Kaggle |
| 6 | Student Habits vs Academic Performance | 1.000 | 16 | exam_score (sürekli) | Kaggle |
| 7 | xAPI-Edu-Data | 480 | 17 | Class (L/M/H) | Kaggle |
| 8 | Student Depression Dataset | 27.901 | 13 | Depression (binary) | Kaggle |
| 9 | Student Stress Factors | 1.100 | 21 | stress_level (3 sınıf) | Kaggle |

### Tüm Veri Setleri Repoya İndirildi

Değerlendirme yapabilmek için uygun görülen tüm veri setleri `datasets/` klasörüne indirildi. UCI veri setleri doğrudan, Kaggle veri setleri API üzerinden çekildi.

---

## Veri Seti Seçim Süreci

### İlk Tercih: Performance Factors (6.607 satır)

Başlangıçta **Student Performance Factors** veri seti birincil olarak seçildi. Nedenleri:
- 6.607 satır ile yeterli büyüklük
- Çalışma saati, motivasyon, uyku, fiziksel aktivite gibi zengin özellikler
- Hem sayısal hem kategorik değişkenler
- Az eksik veri (%1 civarı)

### Kritik Sorun: Not Dağılımı

Veri seti incelendiğinde ciddi bir sorun ortaya çıktı: `Exam_Score` değişkeninin dağılımı **çok dar bir aralıkta yoğunlaşmıştı**. Notların büyük çoğunluğu 65-69 arasında sıkışmıştı (standart sapma sadece 3.9).

Bu durum 3 sınıflı sınıflandırmayı anlamsız kıldı:

| Eşik Denemesi | Düşük | Orta | Yüksek | Sonuç |
|---------------|-------|------|--------|-------|
| <60 / 60-79 / ≥80 | %2.2 | %97.1 | %0.7 | Kullanılamaz — tek sınıf |
| <65 / 65-79 / ≥80 | %32.3 | %67.0 | %0.7 | Yüksek sınıf boş |
| Quantile (eşit dağılım) | %43.6 | %31.8 | %24.6 | En dengeli ama Orta sınıf sadece 67-69 arası (3 puanlık aralık) |

**Karar:** 67-69 gibi 3 puanlık bir aralığa "Orta" demek gerçek hayatta anlamsızdı. Veri seti sınıflandırma problemine uygun değildi.

### Alternatif Arayışı

Diğer veri setlerinin hedef değişken dağılımları karşılaştırıldı:

| Veri Seti | Dağılım | Uygunluk |
|-----------|---------|----------|
| Performance Factors | Std: 3.9 — çok dar | ❌ Kötü |
| Student Habits | Std: 16.9 — geniş yayılım (18-100) | ✅ Çok iyi |
| Dropout UCI | Zaten 3 sınıf hazır | ✅ Çok iyi |
| Stress Factors | Mükemmel denge (33/33/33) | ✅ İyi |

### Nihai Seçim: İki Veri Seti

**1. Student Habits vs Academic Performance (1.000 satır, 16 sütun)**
- Geniş not yayılımı (18.4 - 100)
- Psikolojik özellikler: mental_health_rating, sleep_hours
- Modern yaşam alışkanlıkları: social_media_hours, netflix_hours
- Projenin yaşam tarzı/psikolojik boyutunu temsil ediyor

**2. UCI Predict Students' Dropout and Academic Success (4.424 satır, 37 sütun)**
- Hedef değişken zaten 3 sınıf (Dropout/Enrolled/Graduate)
- Büyük ve temiz veri (eksik veri yok)
- Akademik performans verileri zengin (dönemlik ders notları, kredi sayıları)
- Projenin akademik boyutunu temsil ediyor

**İki veri seti eşit ağırlıkta** kullanılacak. Biri yaşam tarzı odaklı, diğeri akademik odaklı — "iki farklı bakış açısından öğrenci başarısı" hikayesi anlatılacak.

### Birden Fazla Veri Setini Birleştirme Fikri

Tüm veri setlerini birleştirip tek bir zengin veri seti oluşturma fikri değerlendirildi. **Reddedildi** çünkü:
- Veri setlerindeki satırlar farklı öğrencilere ait
- Aralarında eşleştirme yapacak ortak kimlik (ID) yok
- Farklı ülkelerden, farklı zamanlarda toplanmış
- Rastgele birleştirme sahte korelasyonlar üretir — bilimsel olarak geçersiz

### Elenen Veri Setleri

Repoya indirildikten sonra şu veri setleri silindi:

| Veri Seti | Eleme Sebebi |
|-----------|-------------|
| **Performance Factors (6.607)** | Not dağılımı çok dar (std=3.9), sınıflandırmaya uygun değil |
| **xAPI-Edu-Data (480)** | Sadece 480 satır, proje için çok küçük |
| **Student Perf & Behavior (5.000)** | Grade dağılımı çok dengesiz (A: %0.3), Parent_Education_Level %20.5 eksik veri |

### Belirsiz (Yedek) Olarak Tutulanlar

| Veri Seti | Tutulma Sebebi |
|-----------|---------------|
| Student Stress Factors (1.100) | Mükemmel sınıf dengesi, psikolojik özellikler zengin |
| Student Depression (27.901) | Büyük veri seti, ileride işe yarayabilir |
| student/ klasörü (UCI orijinal) | Zaten mevcut, küçük yer kaplıyor |

---

## Problem Tanımlama

### Hedef Değişken Kararları

**Student Habits — 3 sınıf:**
- Düşük: exam_score < 50
- Orta: exam_score 50-75
- Yüksek: exam_score > 75
- Dağılım: Düşük %13.1 / Orta %49.2 / Yüksek %37.7

Bu eşikler, not dağılımının geniş yayılımı sayesinde doğal ve anlamlı sınıflar oluşturuyor.

**Dropout UCI — 3 sınıf (hazır):**
- Dropout (Terk): 1.421 (%32.1)
- Enrolled (Hâlâ kayıtlı): 794 (%17.9)
- Graduate (Mezun): 2.209 (%49.9)

### Enrolled Sınıfı Tartışması

Enrolled öğrencilerin durumu belirsiz — ne mezun olmuşlar ne terk etmişler. Üç seçenek değerlendirildi:

1. **3 sınıf olarak bırak** — Veri kaybı yok, iki veri seti de 3 sınıflı olur
2. **Enrolled'u çıkar, binary yap** — Net sonuç ama 794 satır kaybı
3. **Enrolled'u Dropout ile birleştir** — "Risk var" mantığı ama haksızlık olabilir

**Karar: 3 sınıf olarak bırakıldı.** Hem Student Habits ile paralel olması hem veri kaybı olmaması hem de "devam eden risk grubu" olarak yorumlanabilmesi sebebiyle.

---

## EDA (Keşifsel Veri Analizi)

### Student Habits — Bulgular

**Korelasyon analizi (exam_score ile):**
- `study_hours_per_day`: r=+0.83 — **çok güçlü pozitif** (baskın özellik)
- `mental_health_rating`: r=+0.32 — orta pozitif
- `exercise_frequency`: r=+0.16 — zayıf pozitif
- `sleep_hours`: r=+0.12 — zayıf pozitif
- `social_media_hours`: r=-0.17 — zayıf negatif
- `netflix_hours`: r=-0.17 — zayıf negatif
- `age`: r=-0.01 — ilişki yok

**Risk grupları arası farklar:**
- Düşük risk grubu: ortalama 1.62 saat/gün çalışma
- Yüksek risk grubu: ortalama 4.75 saat/gün çalışma
- Mental health: Düşük grupta 3.85, Yüksek grupta 6.41

**Eksik veri:** `parental_education_level` — 91 satır (%9.1)

**Kategorik değişken dağılımları:**
- Cinsiyet: Female %48.1 / Male %47.7 / Other %4.2
- Yarı zamanlı iş: Hayır %78.5 / Evet %21.5
- Ders dışı aktivite: Hayır %68.2 / Evet %31.8

### Dropout UCI — Bulgular

**En güçlü korelasyonlar (Target ile):**
- 2. dönem onaylanan dersler: r=+0.62 (en güçlü)
- 2. dönem not ortalaması: r=+0.57
- 1. dönem onaylanan dersler: r=+0.53
- 1. dönem not ortalaması: r=+0.49
- Harç ödeme durumu: r=+0.41

**Target grupları arası farklar:**
- Dropout öğrencileri: daha yaşlı (ort 26), çok az ders geçmiş (1. dönem ort 2.55), düşük notlar
- Graduate öğrencileri: daha genç (ort 22), çok ders geçmiş (1. dönem ort 6.23), yüksek notlar
- Ekonomik göstergeler (işsizlik, enflasyon, GDP): hedefle zayıf ilişkili

**Eksik veri:** Yok

EDA grafikleri `eda/plots_habits/` (8 grafik) ve `eda/plots_dropout/` (7 grafik) klasörlerine kaydedildi.

---

## Veri Ön İşleme

### Student Habits

| Adım | Yapılan İşlem |
|------|--------------|
| Gereksiz sütun | `student_id` çıkarıldı |
| Hedef değişken | exam_score → risk_level (3 sınıf: 0=Düşük, 1=Orta, 2=Yüksek) |
| Eksik veri | `parental_education_level` → mod (en sık değer: "High School") ile dolduruldu |
| Encoding | 6 kategorik sütun LabelEncoder ile sayısallaştırıldı |
| Normalizasyon | MinMaxScaler (tüm değerler 0-1 arasına) |
| Feature Selection | Mutual Information analizi yapıldı. study_hours_per_day baskın (MI=0.37), çoğu özellik çok düşük MI skorlu. Tüm özellikler tutuldu |
| Çıktı | 1.000 satır × 15 sütun → `preprocessing/habits_processed.csv` |

### Dropout UCI

| Adım | Yapılan İşlem |
|------|--------------|
| Eksik veri | Yok |
| Hedef değişken | Target → LabelEncoder (Dropout=0, Enrolled=1, Graduate=2) |
| Normalizasyon | MinMaxScaler (tüm değerler 0-1 arasına) |
| Feature Selection | 36 özellikten MI < 0.01 olan 11 tanesi çıkarıldı → 25 özellik kaldı |
| Çıkarılan özellikler | Curricular units 1st sem (credited), Unemployment rate, Educational special needs, Daytime/evening attendance, International, Curricular units 1st/2nd sem (without evaluations), Curricular units 2nd sem (credited), Displaced, Nacionality, GDP |
| Çıktı | 4.424 satır × 26 sütun → `preprocessing/dropout_processed.csv` |

---

## Algoritma Seçimi

### Seçim Kriterleri

1. **Problem tipi:** Sınıflandırma → regresyon algoritmaları elendi
2. **Veri boyutu:** 1.000-4.424 satır, orta ölçek → deep learning gereksiz
3. **Özellik yapısı:** Hem sayısal hem kategorik → her ikisini de işleyebilen algoritmalar
4. **Farklı yaklaşımlar:** Ders gereği farklı paradigmalardan algoritmalar seçilmeli

### Seçilen 4 Algoritma

| Algoritma | Yaklaşım | Seçilme Sebebi |
|-----------|----------|---------------|
| **kNN** | Mesafe tabanlı | En basit ve sezgisel. Baseline model. Normalize veriyle iyi çalışır |
| **Naive Bayes** | Olasılık tabanlı | Farklı bir mantık. Küçük veri setlerinde iyi. Hızlı |
| **Decision Tree** | Kural tabanlı | Yorumlanabilirlik yüksek. Feature importance verir. Görselleştirme güçlü |
| **Random Forest** | Ensemble (topluluk) | Genelde en yüksek doğruluk. Overfitting'e dayanıklı. Decision Tree'nin güçlendirilmiş hali |

### Değerlendirilip Seçilmeyen Algoritmalar

| Algoritma | Seçilmeme Sebebi |
|-----------|-----------------|
| SVM | Güçlü ama "black box" — yorumlama zor, projede yorumlama önemli |
| Logistic Regression | İkili sınıflandırmada daha doğal, 3 sınıf için biraz daha az uygun |
| Neural Network / Deep Learning | 1.000-4.000 satır için overkill |
| XGBoost / LightGBM | Çok güçlü ama ders seviyesinin üstünde, açıklaması zor |

---

## Chatbot Kararı

Projenin son aşaması olarak bir chatbot planlandı. İki farklı yaklaşım değerlendirildi:

**Reddedilen:** Genel sohbet chatbotu (ChatGPT wrapper). Veri madenciliğiyle bağlantısı yok.

**Kabul edilen:** Modelle entegre tahmin chatbotu. Öğrenci sohbet ederek kendi verilerini girer, eğitilmiş model arka planda çalışır, kişiselleştirilmiş sonuç döner.

Chatbot akışı:
1. Soru sorarak model için gerekli özellikleri toplar
2. Eğitilmiş model tahmin yapar
3. Feature importance bilgisiyle kişiselleştirilmiş öneri üretir
4. What-if analizi yapabilir ("çalışma saatimi artırsam?")

Chatbot, projenin son aşaması olarak modelleme bittikten sonra gelecek.

---

## Proje Yapısı (Mevcut)

```
Data-Mining-Project/
├── datasets/
│   ├── dropout_academic_success/    → data.csv (4.424 satır, kullanılacak)
│   ├── student_habits/              → student_habits_performance.csv (1.000 satır, kullanılacak)
│   ├── student_stress_factors/      → StressLevelDataset.csv (yedek)
│   └── student_depression/          → student_depression.csv (yedek)
├── student/                         → UCI orijinal (student-mat.csv, student-por.csv)
├── eda/
│   ├── eda_habits.py                → Student Habits EDA kodu
│   ├── eda_dropout.py               → Dropout UCI EDA kodu
│   ├── plots_habits/                → 8 grafik
│   └── plots_dropout/               → 7 grafik
├── preprocessing/
│   ├── preprocess_habits.py         → Student Habits ön işleme kodu
│   ├── preprocess_dropout.py        → Dropout UCI ön işleme kodu
│   ├── habits_processed.csv         → İşlenmiş veri
│   └── dropout_processed.csv        → İşlenmiş veri
├── modeling/
│   ├── model_habits.py              → v1 modelleme (4 algoritma)
│   ├── model_habits_v2.py           → v2 modelleme (SMOTE + FE + XGBoost)
│   ├── model_dropout.py             → v1 modelleme (4 algoritma)
│   ├── model_dropout_v2.py          → v2 modelleme (FE + XGBoost)
│   ├── ablation_study.py            → Bireysel katkı analizi
│   ├── plots_habits/                → v1 grafikleri
│   ├── plots_habits_v2/             → v2 grafikleri
│   ├── plots_dropout/               → v1 grafikleri
│   └── plots_dropout_v2/            → v2 grafikleri
├── models/
│   ├── best_model_habits.pkl        → XGBoost (F1: %80.62)
│   └── best_model_dropout.pkl       → XGBoost (F1: %77.26)
├── main.py                          → Eski ödev kodu
├── dropout.py                       → Eski ödev kodu
├── requirements.txt                 → Bağımlılıklar
└── .gitignore
```

---

## Modelleme Planı

### Modelleme Akışı (7 adım)

Modelleme aşaması her iki veri seti için ayrı ayrı uygulanacak.

**Adım 1 — Train/Test Split:**
Veri %70 eğitim / %30 test olarak ayrılacak. Student Habits'te 1.000 satır olduğu için %30 test = 300 satır, yeterli güvenilirlik sağlar.

**Adım 2 — Baseline Model (Varsayılan parametreler):**
4 algoritma hiçbir ayar yapılmadan varsayılan parametrelerle çalıştırılacak. Amaç: referans nokta oluşturmak.
- kNN: k=5
- Naive Bayes: parametresiz
- Decision Tree: sınırsız derinlik
- Random Forest: 100 ağaç

**Adım 3 — Hiperparametre Optimizasyonu (GridSearchCV):**
Her modelin en iyi parametre kombinasyonu aranacak:
- kNN → k değeri (1,3,5,7,9,11,13,15), mesafe metriği (euclidean, manhattan)
- Naive Bayes → parametre yok
- Decision Tree → max_depth (3,5,7,10,None), min_samples_split, min_samples_leaf
- Random Forest → n_estimators (50,100,200), max_depth, min_samples_split

**Adım 4 — 10-Fold Cross Validation:**
En iyi parametrelerle bulunan modeller 10-Fold CV ile doğrulanacak. Eğitim verisini 10 parçaya böl, her seferinde 9'uyla eğit 1'iyle test et, ortalamasını al.

**Adım 5 — Test Seti Değerlendirmesi:**
%30 test verisi üzerinde final metrikleri: Accuracy, Precision, Recall, F1-Score, Confusion Matrix.

**Adım 6 — Model Karşılaştırması:**
4 modelin sonuçları yan yana tablo ve grafik ile karşılaştırılacak. En iyi model seçilip gerekçelendirilecek.

**Adım 7 — En İyi Modeli Kaydet:**
En yüksek performanslı model `.pkl` olarak kaydedilecek. Chatbot bu dosyayı kullanacak.

### Akış Şeması

```
İşlenmiş Veri (.csv)
        │
        ▼
  Train/Test Split (%70/%30)
        │
        ├──── Train (%70) → Baseline → GridSearchCV → 10-Fold CV
        │
        ├──── Test (%30) → Final Değerlendirme → Confusion Matrix
        │
        ▼
  Karşılaştırma Tablosu → En İyi Model → .pkl kaydet
```

Bu süreç Student Habits ve Dropout UCI için ayrı ayrı uygulanacak. Sonunda iki veri setinin sonuçları da birbiriyle karşılaştırılacak.

---

## Mimari Kararı

**Karar: Modüler monolitik yapı.** Mikroservis mimarisi değerlendirildi ve reddedildi.

**Neden mikroservis değil:**
- Proje tek makine, tek kullanıcı, tek amaç
- Veri madenciliği ders ödevi ölçeğinde mimari karmaşıklık gereksiz
- Servisler arası API iletişimi bu projede overhead

**Neden monolitik çorba da değil:**
- Her aşama kendi klasöründe ve dosyasında (eda/, preprocessing/, modeling/, chatbot/)
- Her modül bağımsız çalışabilir (kendi csv'sini okur, kendi çıktısını üretir)
- Tek bağlantı noktası: modeling → .pkl → chatbot
- Herhangi bir aşama değiştirilse diğerleri bozulmaz

---

## Modelleme Sonuçları

### Student Habits

**Baseline (varsayılan parametreler):**

| Model | Accuracy |
|-------|----------|
| kNN (k=5) | %53.67 |
| Naive Bayes | %76.00 |
| Decision Tree | %69.33 |
| Random Forest | %76.33 |

**Hiperparametre Optimizasyonu (GridSearchCV):**
- kNN → En iyi: k=15, metric=manhattan (CV: %61.14)
- Naive Bayes → Parametre yok
- Decision Tree → En iyi: max_depth=7, min_samples_leaf=5 (CV: %73.57)
- Random Forest → En iyi: n_estimators=200, max_depth=None (CV: %80.14)

**10-Fold Cross Validation (optimize sonrası):**
- kNN: %61.00 (±4.38)
- Naive Bayes: %78.14 (±4.04)
- Decision Tree: %72.57 (±4.51)
- Random Forest: %79.00 (±4.83)

**Test Seti Final Sonuçları:**

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| kNN | %63.00 | %66.53 | %63.00 | %59.66 |
| **Naive Bayes** | **%76.00** | **%77.99** | **%76.00** | **%75.19** |
| Decision Tree | %68.33 | %68.23 | %68.33 | %68.22 |
| Random Forest | %75.00 | %75.41 | %75.00 | %73.79 |

**En iyi model: Naive Bayes (F1: %75.19)** → `models/best_model_habits.pkl`

Dikkat çekici bulgu: Naive Bayes'in Random Forest'ı geçmesi. Bunun sebebi muhtemelen study_hours_per_day'in baskın özellik olması (r=0.83). Naive Bayes bu tür tek baskın özellikli veri setlerinde iyi çalışır çünkü özelliklerin bağımsız dağılımını iyi modeller.

### Dropout UCI

**Baseline (varsayılan parametreler):**

| Model | Accuracy |
|-------|----------|
| kNN (k=5) | %66.94 |
| Naive Bayes | %68.83 |
| Decision Tree | %66.64 |
| Random Forest | %77.64 |

**Hiperparametre Optimizasyonu (GridSearchCV):**
- kNN → En iyi: k=15, metric=manhattan (CV: %72.32)
- Naive Bayes → Parametre yok
- Decision Tree → En iyi: max_depth=5, min_samples_leaf=5 (CV: %74.26)
- Random Forest → En iyi: n_estimators=50, max_depth=15, min_samples_split=10 (CV: %76.71)

**10-Fold Cross Validation (optimize sonrası):**
- kNN: %71.99 (±3.51)
- Naive Bayes: %69.31 (±3.87)
- Decision Tree: %74.13 (±3.25)
- Random Forest: %77.03 (±2.53)

**Test Seti Final Sonuçları:**

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| kNN | %72.06 | %71.00 | %72.06 | %69.83 |
| Naive Bayes | %68.83 | %66.33 | %68.83 | %66.91 |
| Decision Tree | %74.40 | %72.10 | %74.40 | %71.57 |
| **Random Forest** | **%77.79** | **%76.40** | **%77.79** | **%76.23** |

**En iyi model: Random Forest (F1: %76.23)** → `models/best_model_dropout.pkl`

Feature Importance (Random Forest - Dropout):
- En önemli: 2. dönem onaylanan dersler, 2. dönem notları, 1. dönem onaylanan dersler
- Orta önem: Harç ödeme, giriş notu, bölüm
- Düşük önem: Medeni durum, cinsiyet, önceki yeterlilik

### İki Veri Seti Karşılaştırması (v1)

| | Student Habits | Dropout UCI |
|---|---|---|
| En iyi model | Naive Bayes | Random Forest |
| En iyi F1 | %75.19 | %76.23 |
| Veri boyutu | 1.000 satır | 4.424 satır |
| Özellik sayısı | 14 | 25 |
| Baskın özellik | study_hours_per_day (r=0.83) | Curricular units 2nd sem (approved) (r=0.62) |

Farklı veri setlerinde farklı algoritmaların öne çıkması beklenen bir sonuç. Habits'te tek baskın özellik olduğu için Naive Bayes yeterli, Dropout'ta daha karmaşık ilişkiler olduğu için ensemble yöntemi (Random Forest) avantajlı.

---

## Modelleme v2 — İyileştirme (SMOTE + Feature Engineering + XGBoost)

### Neden İyileştirme Yapıldı?

v1 sonuçları (%75-76 F1) "iyi" kategorisindeydi ama daha yukarıya çıkma potansiyeli vardı. Üç iyileştirme uygulandı:

### 1. SMOTE (Sadece Student Habits)

Student Habits'te Düşük sınıf sadece %13 (92 eğitim örneği). Model bu sınıfı yeterince öğrenemiyordu.

| | Düşük | Orta | Yüksek |
|---|---|---|---|
| SMOTE öncesi | 92 | 344 | 264 |
| SMOTE sonrası | 344 | 344 | 344 |

Dropout UCI'da uygulanmadı çünkü en küçük sınıf bile 794 öğrenci (%18) — yeterli denge.

### 2. Feature Engineering

**Student Habits — 7 yeni özellik:**
- `study_social_ratio`: çalışma/sosyal medya oranı
- `study_netflix_ratio`: çalışma/Netflix oranı
- `screen_time_total`: toplam ekran süresi
- `sleep_mental_interaction`: uyku × zihinsel sağlık
- `study_mental_interaction`: çalışma × zihinsel sağlık
- `healthy_lifestyle`: uyku + egzersiz + diyet
- `study_attendance_interaction`: çalışma × devam

**Dropout UCI — 8 yeni özellik:**
- `sem1_success_rate`: 1. dönem ders geçme oranı
- `sem2_success_rate`: 2. dönem ders geçme oranı
- `total_approved`: toplam geçilen ders
- `total_grade`: toplam not
- `grade_improvement`: 2. dönem - 1. dönem not farkı
- `approved_improvement`: 2. dönem - 1. dönem geçilen ders farkı
- `eval_approved_ratio_1`: 1. dönem sınav/geçme oranı
- `eval_approved_ratio_2`: 2. dönem sınav/geçme oranı

### 3. XGBoost (5. Algoritma)

4 algoritmanın yanına XGBoost eklendi. Daha önce "ders seviyesinin üstünde" diye elenmişti, ancak doğruluk artırmak için dahil edildi. XGBoost gradient boosting tabanlı bir ensemble yöntem, genelde Random Forest'ı geçer.

### v2 Sonuçları — Student Habits

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| kNN | %56.33 | %57.40 | %56.33 | %56.74 |
| Naive Bayes | %67.33 | %71.47 | %67.33 | %68.20 |
| Decision Tree | %65.00 | %65.23 | %65.00 | %65.03 |
| Random Forest | %80.33 | %80.35 | %80.33 | %80.32 |
| **XGBoost** | **%80.67** | **%80.67** | **%80.67** | **%80.62** |

**En iyi: XGBoost (F1: %80.62)** — v1'e göre +5.43 puan iyileşme

Hiperparametre: learning_rate=0.1, max_depth=3, n_estimators=300, subsample=0.8

Not: kNN ve Naive Bayes SMOTE sonrası düştü çünkü SMOTE yapay verileri bu mesafe/olasılık tabanlı modelleri karıştırdı. Ama ağaç tabanlı modeller (DT, RF, XGB) büyük sıçrama yaptı.

### v2 Sonuçları — Dropout UCI

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| kNN | %74.77 | %73.43 | %74.77 | %73.72 |
| Naive Bayes | %74.32 | %74.65 | %74.32 | %74.10 |
| Decision Tree | %75.98 | %74.32 | %75.98 | %74.62 |
| Random Forest | %77.71 | %76.46 | %77.71 | %76.64 |
| **XGBoost** | **%78.01** | **%77.04** | **%78.01** | **%77.26** |

**En iyi: XGBoost (F1: %77.26)** — v1'e göre +1.03 puan iyileşme

Hiperparametre: learning_rate=0.1, max_depth=3, n_estimators=200, subsample=1.0

### v1 vs v2 Genel Karşılaştırma

| | v1 En İyi Model | v1 F1 | v2 En İyi Model | v2 F1 | İyileşme |
|---|---|---|---|---|---|
| Student Habits | Naive Bayes | %75.19 | XGBoost | %80.62 | **+5.43** |
| Dropout UCI | Random Forest | %76.23 | XGBoost | %77.26 | **+1.03** |

Habits'te büyük sıçrama oldu çünkü hem SMOTE (sınıf dengesi) hem feature engineering (yeni sinyaller) hem XGBoost birlikte etki etti. Dropout'ta artış daha mütevazı çünkü veri zaten dengeliydi ve v1'de Random Forest iyi performans gösteriyordu.

Her iki veri setinde de XGBoost kazandı — bu beklenen bir sonuç, gradient boosting genelde diğer yöntemleri geçer.

---

## Ablation Study — Bireysel Katkı Analizi

### Neden Yapıldı?

v2'de %5.43 (Habits) ve %1.03 (Dropout) iyileşme sağlandı ama bu iyileşmenin ne kadarı XGBoost'tan, ne kadarı Feature Engineering'den, ne kadarı SMOTE'tan geldiği bilinmiyordu. Her iyileştirmenin bireysel etkisini ölçmek için ablation study yapıldı.

### Yöntem

Her iyileştirme tek tek açılıp kapatılarak 7 farklı senaryo (Habits) ve 4 farklı senaryo (Dropout) test edildi. Tüm senaryolarda aynı train/test split (70/30, random_state=42, stratified) kullanıldı. Ablation study'de GridSearchCV yapılmadı — sabit parametrelerle çalışıldı, böylece karşılaştırma adil oldu.

### Student Habits — Sonuçlar

| # | Senaryo | Kazanan Model | F1 | Fark (v1'e göre) |
|---|---------|--------------|-----|-------------------|
| 1 | Orijinal + 4 Algo (v1 referans) | Naive Bayes | %75.19 | — |
| 2 | Orijinal + 5 Algo (+XGBoost) | XGBoost | %79.61 | **+4.42** |
| 3 | FE + 4 Algo | Random Forest | %79.73 | **+4.54** |
| 4 | SMOTE + Orijinal + 4 Algo | Random Forest | %77.68 | +2.49 |
| 5 | FE + 5 Algo (SMOTE yok) | XGBoost | %79.89 | +4.70 |
| 6 | SMOTE + Orijinal + 5 Algo | XGBoost | %79.66 | +4.47 |
| 7 | SMOTE + FE + 5 Algo (v2) | XGBoost | %80.62 | **+5.43** |

**Detaylı Tablo (Tüm modeller):**

| Model | v1(Orig+4) | +XGB | +FE | +SMOTE | FE+XGB | SM+XGB | v2(All) |
|-------|-----------|------|-----|--------|--------|--------|---------|
| kNN | %59.66 | %59.66 | %67.57 | %45.81 | %67.57 | %45.81 | %59.81 |
| Naive Bayes | %75.19 | %75.19 | %69.01 | %74.80 | %69.01 | %74.80 | %68.20 |
| Decision Tree | %68.22 | %68.22 | %72.79 | %68.70 | %72.79 | %68.70 | %67.97 |
| Random Forest | %73.79 | %73.79 | %79.73 | %77.68 | %79.73 | %77.68 | %79.69 |
| XGBoost | — | %79.61 | — | — | %79.89 | %79.66 | %80.62 |

**Bireysel Katkı Sıralaması:**
1. **Feature Engineering: +4.54** — En yüksek bireysel katkı. Türetilmiş özellikler (çalışma/sosyal medya oranı vb.) modelin sınıfları ayırma gücünü artırdı.
2. **XGBoost: +4.42** — Neredeyse FE kadar etkili. Gradient boosting diğer algoritmalardan belirgin şekilde güçlü.
3. **SMOTE: +2.49** — En düşük bireysel katkı. Sınıf dengesi sağlandı ama etkisi diğer ikisinin yarısı kadar.

**Önemli Bulgular:**
- FE + XGBoost (SMOTE olmadan) %79.89 veriyor. SMOTE sadece son +0.73 puanı ekliyor (79.89 → 80.62).
- SMOTE, kNN'i ciddi şekilde bozmuş (%59.66 → %45.81). Yapay veriler mesafe tabanlı modeli yanıltıyor.
- Naive Bayes hem FE'den hem SMOTE'tan zarar görmüş. Tek baskın özellik (study_hours) ortamında ek özellikler gürültü yaratmış.
- Ağaç tabanlı modeller (DT, RF, XGB) her iyileştirmeden fayda sağlamış.

### Dropout UCI — Sonuçlar

| # | Senaryo | Kazanan Model | F1 | Fark (v1'e göre) |
|---|---------|--------------|-----|-------------------|
| 1 | Orijinal + 4 Algo (v1 referans) | Random Forest | %76.38 | — |
| 2 | Orijinal + 5 Algo (+XGBoost) | XGBoost | %77.94 | **+1.57** |
| 3 | FE + 4 Algo | Random Forest | %75.70 | **-0.68** |
| 4 | FE + 5 Algo (v2) | XGBoost | %77.26 | +0.88 |

**Bireysel Katkı Sıralaması:**
1. **XGBoost: +1.57** — Tek başına en iyi sonuç. Dropout verisinde ana iyileşme kaynağı.
2. **Feature Engineering: -0.68** — Olumsuz etki! Türetilmiş özellikler gürültü eklemiş.

**Önemli Bulgular:**
- Dropout'ta Feature Engineering tek başına zararlı. Orijinal özellikler zaten güçlü sinyaller içeriyordu (2. dönem onaylanan dersler r=0.62). Ek özellikler fazla korelasyonlu ve gürültülü.
- XGBoost tek başına (%77.94), FE + XGBoost'tan (%77.26) bile daha iyi. FE, XGBoost'un performansını da düşürmüş.
- SMOTE zaten uygulanmadı çünkü sınıf dengesi yeterliydi (en küçük sınıf %18).

### Genel Çıkarımlar

| İyileştirme | Student Habits | Dropout UCI |
|-------------|---------------|-------------|
| XGBoost | +4.42 ✅ | +1.57 ✅ |
| Feature Engineering | +4.54 ✅ | -0.68 ❌ |
| SMOTE | +2.49 ✅ | Uygulanmadı |
| Hepsi birlikte | +5.43 ✅ | +0.88 ✅ |

- **XGBoost her iki veri setinde de pozitif katkı sağladı** — en güvenilir iyileştirme.
- **Feature Engineering veri setine bağlı** — Habits'te çok faydalı, Dropout'ta zararlı. Her zaman "daha fazla özellik = daha iyi" değil.
- **SMOTE marjinal katkı sağladı** — Düşük sınıf dengesini düzeltti ama bazı modelleri bozdu.
- Üç iyileştirmenin toplamı, bireysel katkılarının toplamından düşük (sinerjik ama kısmen örtüşen etkiler).

---

## Veri Seti Değişikliği: Student Habits → OULAD

### Neden Değiştirildi?

Student Habits veri seti sentetik (yapay üretilmiş) veritiydi ve en iyi sonuç v2 ile %80.62 F1'de kaldı. Daha yüksek performans için gerçek bir veri seti arandı.

### Araştırma Süreci

10 farklı eğitim veri seti incelendi:

| Veri Seti | Satır | Gerçek? | Uygun? | Neden? |
|-----------|-------|---------|--------|--------|
| **OULAD (Open University UK)** | **32.593** | **Evet** | **Seçildi** | Büyük, gerçek, Nature'da yayınlandı |
| UCI Student Perf (Portekiz) | 395-649 | Evet | Hayır | Çok küçük |
| xAPI-Edu-Data | 480 | Evet | Hayır | Çok küçük |
| KDD Cup 2010 | Milyonlarca | Evet | Hayır | Farklı problem tipi |
| Kıbrıs Higher Ed | 145 | Evet | Hayır | Çok küçük |
| Oman Moodle | 326 | Evet | Hayır | Çok küçük |
| Student Depression | 27.901 | Muhtemelen sentetik | Hayır | Binary, konu farklı |
| Student Stress | 1.100 | Muhtemelen sentetik | Hayır | Sentetik |

### OULAD (Open University Learning Analytics Dataset)

- **Kaynak:** The Open University (İngiltere'nin en büyük üniversitesi)
- **Yayın:** Kuzilek et al., Nature Scientific Data, 2017
- **Boyut:** 32.593 öğrenci + 10.6M VLE etkileşim logu
- **Yapı:** 7 ilişkisel CSV tablosu (studentInfo, assessments, studentAssessment, vle, studentVle, studentRegistration, courses)
- **Orijinal hedef:** 4 sınıf (Pass %37.9, Withdrawn %31.2, Fail %21.6, Distinction %9.3)

**3 sınıfa daraltma:**
- Withdrawn: 10.156 (%31.2) — Terk edenler
- Fail: 7.052 (%21.6) — Başarısız olanlar
- Pass (+ Distinction): 15.385 (%47.2) — Başarılı olanlar

Bu yapı Dropout UCI ile paralel: orada Dropout/Enrolled/Graduate, burada Withdrawn/Fail/Pass.

### Veri Hazırlama (7 Tablo → 1 Dataset)

7 ilişkisel tablo join edilerek öğrenci bazlı tek bir veri seti oluşturuldu:

**Assessment özellikleri (15):**
avg_score, std_score, min_score, max_score, num_assessments, num_missing_score, avg_score_TMA, num_TMA, avg_score_CMA, num_CMA, avg_score_Exam, num_Exam, avg_submit_delay, late_submissions, early_submissions

**VLE davranış özellikleri (13):**
total_clicks, total_vle_days, avg_daily_clicks, num_distinct_activities, clicks_resource, clicks_oucontent, clicks_url, clicks_forumng, clicks_quiz, clicks_subpage, clicks_homepage, clicks_questionnaire, clicks_page

**Kayıt/ders özellikleri:**
date_registration, unregistered, course_length, studied_credits

**Demografik:**
gender, region, highest_education, imd_band, age_band, disability, num_of_prev_attempts

**Feature Selection (Mutual Information):**
MI < 0.01 olan 9 özellik çıkarıldı: highest_education, imd_band, gender, num_of_prev_attempts, region, num_missing_score, course_length, disability, age_band

**Final:** 32.593 satır × 30 özellik → `preprocessing/oulad_processed.csv`

### EDA Bulguları

**En güçlü korelasyonlar (target ile):**
- unregistered: r=-0.89 (kayıt sildiren = withdrawn, çok güçlü)
- num_TMA: r=+0.76
- num_assessments: r=+0.75
- max_score: r=+0.68
- early_submissions: r=+0.67
- avg_score: r=+0.65
- total_vle_days: r=+0.62

**VLE kullanım farkları:**
- Pass öğrencileri: ort 2.069 tıklama, 92 aktif gün
- Fail öğrencileri: ort 688 tıklama, 35 aktif gün
- Withdrawn öğrencileri: ort 445 tıklama, 23 aktif gün

Başarılı öğrenciler platformu 4.6 kat daha fazla kullanıyor — çok güçlü ayrıştırıcı sinyal.

**Assessment not farkları:**
- Pass: ort 79.1
- Withdrawn: ort 66.1
- Fail: ort 64.7

EDA grafikleri: `eda/plots_oulad/` (6 grafik)

---

## OULAD Modelleme Sonuçları

### v1 — 4 Algoritma

**Hiperparametre Optimizasyonu (GridSearchCV):**
- kNN → k=9, manhattan (CV F1: %93.84)
- Naive Bayes → parametre yok (CV F1: %89.67)
- Decision Tree → max_depth=10, min_samples_leaf=5 (CV F1: %92.92)
- Random Forest → n_estimators=50, max_depth=None, min_samples_split=5 (CV F1: %94.41)

**10-Fold CV:**
- kNN: %93.89 (±0.54)
- Naive Bayes: %89.73 (±0.75)
- Decision Tree: %93.11 (±0.52)
- Random Forest: %94.31 (±0.43)

**Test Seti Sonuçları:**

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| kNN | %93.59 | %93.60 | %93.59 | %93.46 |
| Naive Bayes | %90.48 | %90.34 | %90.48 | %90.31 |
| Decision Tree | %93.77 | %93.74 | %93.77 | %93.68 |
| **Random Forest** | **%94.63** | **%94.62** | **%94.63** | **%94.56** |

**En iyi: Random Forest (F1: %94.56)**

### v2 — Feature Engineering + XGBoost

**10 yeni türetilmiş özellik:**
score_per_assessment, click_per_day, assessment_completion_rate, forum_ratio, quiz_ratio, resource_ratio, score_consistency, early_late_ratio, tma_cma_score_diff, engagement_score

**Hiperparametre Optimizasyonu:**
- XGBoost → learning_rate=0.05, max_depth=10, n_estimators=200, subsample=0.8 (CV F1: %94.85)

**Test Seti Sonuçları:**

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| kNN | %93.81 | %93.85 | %93.81 | %93.68 |
| Naive Bayes | %90.61 | %90.50 | %90.61 | %90.38 |
| Decision Tree | %92.80 | %92.75 | %92.80 | %92.68 |
| Random Forest | %94.72 | %94.70 | %94.72 | %94.66 |
| **XGBoost** | **%94.88** | **%94.86** | **%94.88** | **%94.82** |

**En iyi: XGBoost (F1: %94.82)** → `models/best_model_oulad.pkl`

### v1 vs v2:
- v1: Random Forest F1: %94.56
- v2: XGBoost F1: %94.82
- İyileşme: +0.26 puan (v1 zaten çok yüksekti, artış marjinal)

### OULAD Ablation Study

| # | Senaryo | Kazanan | F1 | Fark |
|---|---------|---------|-----|------|
| 1 | Orijinal + 4 Algo (v1) | Random Forest | %94.56 | — |
| 2 | Orijinal + 5 Algo (+XGB) | XGBoost | %94.76 | +0.20 |
| 3 | FE + 4 Algo | Random Forest | %94.47 | -0.09 |
| 4 | FE + 5 Algo (v2) | XGBoost | %94.82 | +0.26 |

**Bulgular:**
- **XGBoost: +0.20** — Marjinal ama pozitif katkı.
- **Feature Engineering: -0.09** — Dropout UCI'da olduğu gibi burada da hafif negatif. Orijinal özellikler zaten çok güçlü (num_TMA r=0.76, unregistered r=-0.89), ek özellikler gürültü eklemiş.
- **FE + XGBoost birlikte: +0.26** — XGBoost, FE'nin eklediği gürültüyü tolere edebiliyor.
- v1 zaten %94.56 olduğu için iyileştirme marjı çok dar. Bu veri setinde ham özellikler yeterince güçlü.

### Student Habits vs OULAD Karşılaştırma

| | Student Habits (sentetik) | OULAD (gerçek) |
|---|---|---|
| v1 en iyi | Naive Bayes %75.19 | Random Forest %94.56 |
| v2 en iyi | XGBoost %80.62 | XGBoost %94.82 |
| Fark | — | **+14.20 puan** |

OULAD ile dramatik bir performans artışı sağlandı. Gerçek veri, güçlü sinyaller ve 32x daha büyük veri seti farkı yarattı.

---

## Kalan Adımlar

| # | Aşama | Durum |
|---|-------|-------|
| 1 | Veri Seti Seçimi | ✅ Tamamlandı |
| 2 | Veri Seti Değişikliği (Habits → OULAD) | ✅ Tamamlandı |
| 3 | Problem Tanımlama | ✅ Tamamlandı |
| 4 | EDA (Dropout + OULAD) | ✅ Tamamlandı |
| 5 | Veri Ön İşleme | ✅ Tamamlandı |
| 6 | Modelleme v1 (4 algoritma) | ✅ Tamamlandı |
| 7 | Modelleme v2 (FE + XGBoost) | ✅ Tamamlandı |
| 8 | En iyi modelleri kaydet (.pkl) | ✅ Tamamlandı |
| 9 | Ablation Study (Dropout + OULAD) | ✅ Tamamlandı |
| 10 | Yerelleştirme (Dropout UCI → Türkiye) | ✅ Tamamlandı |
| 11 | Chatbot | ✅ Tamamlandı |
| 12 | Raporlama | ⬜ |

---

## Chatbot — Akademik Danışman Asistanı

### Mimari

```
Öğrenci ↔ Streamlit UI ↔ Google Gemini (LLM) ↔ XGBoost Modeli (.pkl)
          (chat arayüz)   (doğal sohbet +        (tahmin +
                           veri çıkarma +          olasılıklar)
                           sonuç yorumlama)
```

### Teknoloji

- **Frontend:** Streamlit (Python tabanlı web arayüzü)
- **LLM Katmanı:** Google Gemini 2.0 Flash (ücretsiz API)
- **ML Model:** XGBoost (yerelleştirilmiş, 22 özellik, F1: %75.27)

### Chatbot Akışı

1. Öğrenci doğal dilde sohbet eder
2. Gemini LLM sohbetten yapılandırılmış veri çıkarır (JSON)
3. Yeterli veri toplandığında XGBoost modeli tahmin yapar
4. Gemini sonuçları yapıcı ve destekleyici dille yorumlar
5. Öğrenciye kişisel öneriler ve what-if analizi sunulur

### Özellik Öncelikleri

| Öncelik | Sayı | Özellikler |
|---------|------|------------|
| Essential (mutlaka) | 8 | Dönem ders sayıları, notlar, sınav sayıları |
| High (önemli) | 3 | Yaş, cinsiyet, burs durumu |
| Medium (sorulabilir) | 4 | Tercih sırası, bölüm, giriş puanı, önceki not |
| Low (varsayılan) | 7 | Medeni durum, başvuru türü, anne/baba bilgileri |

### Tasarım Kararları

1. **"Kalırsın" demek yerine yapıcı dil** — Öğrenciye risk seviyesi gösterilir ama her zaman iyileştirme önerileri verilir
2. **Doğal sohbet** — Anket formatı değil, sohbet akışında bilgi toplanır
3. **Referans karşılaştırma** — Öğrencinin verileri mezun/terk öğrenci ortalamalarıyla karşılaştırılır
4. **Eksik veri toleransı** — Öğrenci bazı bilgileri bilmezse varsayılan değerler kullanılır
5. **LLM + ML hibrit** — Gemini doğal dil işler, XGBoost tahmin yapar. İki farklı AI birlikte çalışır.

### Dosya Yapısı

```
chatbot/
├── app.py                → Ana Streamlit uygulaması
├── prepare_chatbot.py    → Normalizasyon ve config hazırlığı
├── scaler_params.json    → MinMaxScaler min/max değerleri
├── feature_config.json   → 22 özellik: Türkçe ad, tip, seçenekler, varsayılan, öncelik
└── reference_stats.json  → Dropout/Enrolled/Graduate grup ortalamaları
```

### Çalıştırma

```bash
streamlit run chatbot/app.py
```

Groq API key gerekli (ücretsiz): https://console.groq.com/keys

---

## Yerelleştirme (Dropout UCI → Türkiye)

### Neden Yapıldı?

Dropout UCI Portekiz üniversitesinden. Chatbot Türk öğrencilere yönelik olacağı için Türkiye'de karşılığı olmayan özellikler çıkarılıp model yeniden eğitildi.

### Çıkarılan Özellikler (3 adet)

| Özellik | Çıkarılma Sebebi |
|---|---|
| Tuition fees up to date | Türkiye'de devlet üniversitelerinde harç yok |
| Debtor | Borçlu kavramı Türk eğitim sisteminde farklı |
| Inflation rate | Portekiz'e ait makroekonomik veri |

### Uyarlanan Özellikler (soru metni değişti, özellik aynı)

| Orijinal | Türkiye Karşılığı |
|---|---|
| Admission grade | YKS puanı |
| Previous qualification | Önceki eğitim (Lise, Önlisans...) |
| Application mode | Başvuru türü (YKS, DGS, Yatay Geçiş) |
| Mother/Father qualification | Anne/Baba eğitim düzeyi |
| Mother/Father occupation | Anne/Baba mesleği |

### Yerelleştirilmiş Model Sonuçları

| Model | F1-Score |
|---|---|
| kNN | %68.91 |
| Naive Bayes | %66.76 |
| Decision Tree | %71.54 |
| Random Forest | %73.85 |
| **XGBoost** | **%75.27** |

### Performans Etkisi

| Versiyon | Özellik | F1 |
|---|---|---|
| Orijinal (v2) | 25 özellik | %77.26 |
| Yerelleştirilmiş | 22 özellik | %75.27 |
| Fark | -3 özellik | **-1.99 puan** |

3 özellik çıkardık, sadece ~2 puan kaybettik. Tuition fees (r=+0.41) en güçlü çıkarılan özellikti, bu kayıbın çoğu oradan geliyor. Kabul edilebilir bir kayıp — chatbot tutarlılığı için doğru karar.

### Chatbot İçin Kullanılacak Model

`models/best_model_dropout_localized.pkl` — XGBoost, 22 özellik, F1: %75.27

---

## LLM Sağlayıcı Değişikliği: Gemini → Groq

### Sorun

Google Gemini API (ücretsiz tier) Türkiye'den kullanılamadı. API key alınıp chatbot'a girilmesine rağmen `429 You exceeded your current quota` hatası alındı. Hata detayı incelendiğinde tüm ücretsiz metrikler için `limit: 0` olduğu görüldü — bu, Türkiye'den bölgesel kısıtlama olduğuna işaret ediyor.

### Çözüm: Groq API

Groq, ücretsiz LLM API hizmeti sunuyor ve Türkiye'den erişilebilir durumda. `llama-3.3-70b-versatile` modeli kullanılıyor.

| Özellik | Gemini | Groq |
|---|---|---|
| Model | Gemini 2.0 Flash | Llama 3.3 70B |
| Maliyet | Ücretsiz (bölge kısıtlı) | Ücretsiz |
| Türkiye erişimi | ❌ (quota: 0) | ✅ |
| API stili | Google genai SDK | OpenAI-uyumlu |

### Yapılan Değişiklikler

1. `groq` Python paketi kuruldu (`pip install groq`)
2. `chatbot/api_key.txt` → Groq API key ile güncellendi
3. `chatbot/app.py` tamamen yeniden yazıldı:
   - `google.generativeai` → `from groq import Groq`
   - Gemini'nin stateful chat API'si → Groq'un stateless `chat.completions.create` API'si
   - Her istekte sistem prompt + sohbet geçmişi gönderiliyor
   - `chat_with_llm()` fonksiyonu eklendi
   - Sohbet geçmişi `st.session_state.chat_history` ile manuel takip ediliyor

### Test Sonucu

Groq API Türkçe konuşma, veri çıkarımı ve analiz akışı başarıyla çalıştı.

---

## Chatbot — Son Durum

### Mimari

```
Öğrenci ↔ Streamlit UI ↔ Groq (Llama 3.3 70B) ↔ XGBoost Modeli
                                    ↓
                           [DATA: {...}] çıkarımı
                                    ↓
                          MinMaxScaler normalizasyon
                                    ↓
                          XGBoost tahmin + olasılıklar
                                    ↓
                          LLM yapıcı yorum + öneriler
```

### Özellikler

- Doğal Türkçe sohbet (anket değil, danışman havası)
- 22 özellik sohbet içinde toplanıyor (4 öncelik seviyesi)
- Otomatik veri çıkarımı (`[DATA: {...}]` formatı, öğrenciye görünmez)
- 3 sınıflı tahmin: Terk Riski / Devam Ediyor / Mezuniyet
- Olasılık dağılımı çubuk grafik
- Mezun vs terk ortalamalarıyla karşılaştırma
- Yapıcı ve destekleyici dil (asla "kalırsın" demez)
- API key gizli (`api_key.txt`, gitignore'da)
- Streamlit UI temizlenmiş (deploy butonu, menü, footer gizli)

### Dosya Yapısı (Güncel)

```
chatbot/
├── app.py                → Ana uygulama (Groq + Llama 3.3 70B)
├── prepare_chatbot.py    → Normalizasyon ve config hazırlığı
├── api_key.txt           → Groq API key (gitignore'da)
├── scaler_params.json    → MinMaxScaler min/max değerleri
├── feature_config.json   → 22 özellik: Türkçe ad, tip, seçenekler, varsayılan, öncelik
└── reference_stats.json  → Dropout/Enrolled/Graduate grup ortalamaları
```

### Çalıştırma

```bash
streamlit run chatbot/app.py
```

Groq API key gerekli (ücretsiz): https://console.groq.com/keys

---

## Chatbot İlk Test ve Düzeltmeler (22 Nisan 2026)

### Tespit Edilen Sorunlar

| # | Sorun | Detay |
|---|-------|-------|
| 1 | **Type Error** | `unsupported operand type(s) for -: 'str' and 'float'` — LLM değerleri string olarak gönderince `normalize_value` patladı |
| 2 | **Türkçe-İngilizce karışımı** | LLM cevaplarında "sometimes", "information" gibi İngilizce kelimeler karışıyordu |
| 3 | **Not sistemi uyumsuzluğu** | Öğrenci "4 üzerinden 2.4" dedi, model 0-20 arası bekliyor, dönüşüm yoktu |
| 4 | **Soru tekrarı** | Dolaylı cevapları anlamıyordu, aynı soruyu tekrar soruyordu |
| 5 | **Uzun cevaplar** | LLM gereksiz uzun ve "ders veren" tonda yazıyordu |

### Yapılan Düzeltmeler

1. **Bug fix:** `normalize_value()` fonksiyonuna `float()` dönüşümü eklendi — string değerler artık otomatik sayıya çevriliyor
2. **System prompt güçlendirildi:**
   - Kesin Türkçe kuralı: İngilizce kelime kullanımı yasaklandı
   - Not sistemi dönüşüm talimatları: 4'lük → 20'lik, 100'lük → 20'lik otomatik çevirme
   - Ders sayısı hesaplama: "alttan X dersim var" → alınan - geçilen mantığı
   - Dolaylı cevap anlama talimatı eklendi
   - Cevap uzunluğu sınırı: 2-3 cümle
   - DATA değerlerinin her zaman sayısal (int/float) olması gerektiği vurgulandı

---

## Chatbot İkinci Test Turu ve Kapsamlı İyileştirmeler (22 Nisan 2026)

### Test Yöntemi

5 farklı senaryo ile Groq API üzerinden otomatik test yapıldı:
1. Başarılı öğrenci (yaş + not dönüşümü)
2. Zorlanan öğrenci (dil + typo)
3. Dolaylı cevap veren öğrenci
4. İngilizce konuşma denemesi
5. Çok az bilgi veren öğrenci

### İkinci Turda Tespit Edilen Sorunlar

| # | Sorun | Ciddiyet | Çözüm |
|---|-------|----------|-------|
| 1 | LLM Türkçe feature adları kullanıyor ("Cinsiyet" vs "Gender") | Kritik | Kod tarafında TR→EN eşleme tablosu eklendi |
| 2 | LLM kategorik değerleri metin gönderiyor ("Kadın" vs 0) | Kritik | Kategorik değer→sayı dönüşüm tablosu eklendi |
| 3 | Feature adı yazım hatası ("Curicular" vs "Curricular") | Kritik | Typo düzeltme sözlüğü + fuzzy match iyileştirmesi |
| 4 | 4'lük not sistemi dönüştürülmüyor (1.2 kalıyor, 6.0 olmalı) | Yüksek | Kod tarafında otomatik dönüşüm: grade ≤ 4.0 → (val/4)*20 |
| 5 | null/None değerler gönderiliyor | Yüksek | clean_extracted_data'da null filtreleme |
| 6 | Çince karakter sızması ("改善") | Orta | clean_non_turkish() fonksiyonu eklendi |
| 7 | Yaş vs kayıt yaşı karışıklığı | Orta | Prompt'ta yaş hesaplama kuralı eklendi |
| 8 | Not dönüşümünü öğrenciye gösterme | Orta | Prompt'ta "GİZLİ" dönüşüm kuralı |
| 9 | Fuzzy match tek kelimelik feature'lara yanlış eşleşme | Düşük | Minimum 2 ortak kelime zorunlu |

### Uygulanan Kod Düzeltmeleri

1. **TR→EN Feature Eşleme Tablosu**: 30+ Türkçe feature adı → İngilizce karşılık
2. **Kategorik Değer Dönüşüm Tablosu**: "Kadın"→0, "Erkek"→1, "Evet"→1, "Hayır"→0 vb.
3. **Typo Düzeltme Sözlüğü**: "curicular"→"curricular" vb.
4. **Otomatik Not Dönüşümü**: Grade alanlarında değer ≤ 4.0 ise (4'lük/4.0)×20 uygulanır
5. **Null Filtreleme**: None, "null", "" değerler otomatik atlanır
6. **Yabancı Karakter Temizleme**: Çince/Japonca/Korece karakterler siliniyor
7. **Fuzzy Match İyileştirmesi**: Tek kelimelik feature adlarına yanlış eşleşme önlendi
8. **System Prompt v3**: Dil, yaş, not, DATA kuralları güçlendirildi

---

## Metodolojik Düzeltmeler (22 Nisan 2026)

### Yapılan Düzeltmeler

#### 1. Data Leakage Düzeltmesi
**Sorun:** MinMaxScaler ve MI feature selection tüm veri üzerinde (split öncesi) yapılıyordu.
**Düzeltme:**
- `preprocess_dropout.py` ve `prepare_oulad.py`'den scaler ve MI çıkarıldı
- Preprocessing artık sadece temizlik ve encoding yapıyor
- `model_dropout_localized.py` ve `model_oulad_v2.py`'de:
  - Önce train/test split
  - Scaler sadece train'e fit, test'e transform
  - MI sadece train üzerinde hesaplanıyor

#### 2. OULAD Target Leakage Düzeltmesi
**Sorun:** `unregistered` özelliği (kaydını sildirdi mi?) hedef değişkene çok yakındı — Withdrawn sınıfıyla neredeyse aynı bilgiyi taşıyordu.
**Düzeltme:** `prepare_oulad.py`'den `unregistered` özelliği tamamen çıkarıldı.

**Etki:** OULAD F1 skoru %94.56 → %80.40'a düştü. Bu düşüş, eski skorun ne kadar şişirilmiş olduğunu kanıtlıyor.

#### 3. Class-wise Metrikler Eklendi
Her iki modele de eklenen yeni metrikler:
- Classification Report (sınıf bazlı precision/recall/f1)
- Macro F1 (weighted F1'a ek olarak)
- Dropout/Withdrawn Recall (en kritik metrik — risk öğrencilerini yakalama oranı)

### Güncel Model Sonuçları

#### Dropout UCI (Yerelleştirilmiş) — Chatbot Modeli
| Model | F1-Score |
|---|---|
| kNN | %68.91 |
| Naive Bayes | %66.76 |
| Decision Tree | %71.54 |
| Random Forest | %73.85 |
| **XGBoost** | **%75.27** |

Sınıf bazlı (XGBoost):
| Sınıf | Precision | Recall | F1 |
|---|---|---|---|
| Dropout | %80.09 | %72.37 | %76.04 |
| Enrolled | %51.24 | %42.86 | %46.67 |
| Graduate | %81.03 | %90.50 | %85.51 |

Dropout Recall: %72.37 — gerçek terk öğrencilerinin %72.4'ünü yakalıyor.

#### OULAD v2 (FE + XGBoost)
| Model | F1-Score |
|---|---|
| kNN | %76.96 |
| Naive Bayes | %69.05 |
| Decision Tree | %77.74 |
| Random Forest | %80.13 |
| **XGBoost** | **%80.40** |

Sınıf bazlı (XGBoost):
| Sınıf | Precision | Recall | F1 |
|---|---|---|---|
| Withdrawn | %74.87 | %83.13 | %78.79 |
| Fail | %61.24 | %45.32 | %52.09 |
| Pass | %92.36 | %96.64 | %94.45 |

Withdrawn Recall: %83.13 — risk öğrencilerinin %83.1'ini yakalıyor.

### Önceki vs Güncel Skorlar

| Model | Eski F1 | Yeni F1 | Fark | Neden |
|---|---|---|---|---|
| Dropout Localized | %75.27 | %75.27 | 0 | Scaler leakage etkisi minimal |
| OULAD v2 | %94.56 | %80.40 | **-14.16** | unregistered target leakage çıkarıldı |

---

## README Güncellemesi ve Akış Netleştirmesi (22 Nisan 2026)

Metodolojik düzeltmelerden sonra README dosyası güncel kod akışıyla uyumlu hale getirildi.

### Yapılan Güncellemeler

1. **Güncel ana dosyalar netleştirildi**
   - `preprocessing/preprocess_dropout.py`
   - `preprocessing/prepare_oulad.py`
   - `modeling/model_dropout_localized.py`
   - `modeling/model_oulad_v2.py`
   - `chatbot/app.py`

2. **Legacy dosyalar ayrıldı**
   Eski model ve ablation dosyaları proje geçmişini göstermek için korunuyor, ancak final metodolojik sonuç olarak kullanılmaması gerektiği README'de belirtildi.

3. **Eski OULAD skorları düzeltilmiş bağlama alındı**
   `%94` civarı OULAD skorlarının `unregistered` target leakage nedeniyle şiştiği, temiz OULAD v2 skorunun `%80.40` olduğu README'ye işlendi.

4. **Ön işleme çıktıları güncellendi**
   - `dropout_processed.csv`: 4.424 x 37
   - `oulad_processed.csv`: 32.593 x 39

5. **Data leakage açıklaması README'ye eklendi**
   Scaler ve mutual information işlemlerinin preprocessing aşamasında değil, train/test split sonrası modelleme aşamasında yapıldığı belirtildi.

6. **OULAD konumlandırması netleştirildi**
   OULAD modeli dönem sonu/durum sınıflandırması olarak konumlandırıldı. Erken uyarı sistemi için zaman kesitiyle yeniden tasarım gerektiği not edildi.

### Commit

Bu düzenlemeler `Fix leakage and update docs` commit'i ile `feature/student-success-prediction` branch'ine pushlandı.

---

## KFold → StratifiedKFold Düzeltmesi (22 Nisan 2026)

### Sorun

Her iki güncel modelleme dosyasında (`model_dropout_localized.py`, `model_oulad_v2.py`) cross-validation için `KFold` kullanılıyordu. Sınıflandırma problemlerinde `KFold`, her fold'da sınıf dağılımını garanti etmez. Özellikle Dropout UCI'da Enrolled sınıfı sadece %18 — bazı fold'larda bu sınıftan çok az örnek düşebilir ve CV skorları yanıltıcı olabilir.

Aynı zamanda `train_test_split`'te zaten `stratify=y` kullanılıyordu ama CV'de kullanılmıyordu — bu bir tutarsızlıktı.

### Düzeltme

Her iki dosyada `KFold` → `StratifiedKFold` ile değiştirildi:
- 5-Fold (GridSearchCV için): `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- 10-Fold (final CV için): `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`

`StratifiedKFold` her fold'da orijinal sınıf oranlarını korur, böylece tüm fold'larda Dropout/Enrolled/Graduate (veya Withdrawn/Fail/Pass) dağılımı aynı kalır.

### Etkilenen Dosyalar

- `modeling/model_dropout_localized.py`
- `modeling/model_oulad_v2.py`

### Yeni Sonuçlar

| Model | Eski F1 (KFold) | Yeni F1 (StratifiedKFold) | Fark |
|---|---|---|---|
| Dropout Localized (XGBoost) | %75.27 | %75.10 | -0.17 |
| OULAD v2 (XGBoost) | %80.40 | %80.48 | +0.08 |

Farklar minimal — bu beklenen bir sonuç çünkü veri setleri yeterince büyük olduğunda KFold ve StratifiedKFold yakın sonuçlar üretir. Ancak metodolojik olarak StratifiedKFold daha doğru ve train_test_split ile tutarlı.

### Sınıf Bazlı Güncel Sonuçlar

**Dropout Localized (XGBoost):**

| Sınıf | Precision | Recall | F1 |
|---|---|---|---|
| Dropout | %78.88 | %72.60 | %75.61 |
| Enrolled | %51.53 | %42.44 | %46.54 |
| Graduate | %80.65 | %89.89 | %85.02 |

Dropout Recall: %72.60

**OULAD v2 (XGBoost):**

| Sınıf | Precision | Recall | F1 |
|---|---|---|---|
| Withdrawn | %75.26 | %82.87 | %78.88 |
| Fail | %60.82 | %45.84 | %52.28 |
| Pass | %92.38 | %96.64 | %94.46 |

Withdrawn Recall: %82.87

---

## sklearn Pipeline Entegrasyonu (22 Nisan 2026)

### Sorun

Modelleme dosyalarında MinMaxScaler, GridSearchCV'den **önce** tüm training verisi üzerinde fit ediliyordu. Bu, CV fold'larında hafif bir leakage yaratıyordu — her validation fold'u, tüm training setinden fit edilmiş scaler istatistiklerini görüyordu.

Ayrıca chatbot tarafında model ve scaler ayrı artifact'lardı (`.pkl` + `scaler_params.json`). `prepare_chatbot.py` scaler parametrelerini tüm veriden hesaplıyordu (training-only olması gerekirken) — bu da ayrı bir leakage kaynağıydı.

### Çözüm: sklearn Pipeline

`Pipeline([('scaler', MinMaxScaler()), ('model', Algo())])` yapısı ile scaler ve model birlikte sarıldı:

1. **CV'de temiz scaler:** GridSearchCV her fold'unda scaler'ı sadece o fold'un training kısmından fit eder, validation kısmına sadece transform uygular.
2. **Tek artifact:** Model kaydedilirken scaler + model birlikte `.pkl` olarak kaydedilir.
3. **Chatbot basitleşti:** Manuel `normalize_value()` fonksiyonu kaldırıldı, `scaler_params.json` yüklemesi kaldırıldı. Chatbot artık raw değerlerle `pipeline.predict()` çağırır, Pipeline kendi scaler'ını uygular.
4. **Tutarsızlık giderildi:** Chatbot artık training-only scaler kullanır (Pipeline içinde gömülü), daha önce tüm veriden hesaplanan scaler yerine.

### Etkilenen Dosyalar

| Dosya | Değişiklik |
|---|---|
| `modeling/model_dropout_localized.py` | Pipeline yapısı, `model__param` formatı, raw veri kullanımı |
| `modeling/model_oulad_v2.py` | Aynı Pipeline yapısı, MI raw train üzerinde (Pipeline dışında) |
| `chatbot/app.py` | `normalize_value()` kaldırıldı, `scaler_params` kaldırıldı, `pipeline.predict()` |
| `chatbot/prepare_chatbot.py` | Scaler params hesaplama kaldırıldı |

### OULAD MI Feature Selection Notu

OULAD'da MI feature selection Pipeline dışında raw training verisi üzerinde yapılıyor. MI önceden scaled veri üzerinde hesaplanıyordu. Raw veride de aynı 8 özellik (MI < 0.01) çıkarıldı — sonuçlar etkilenmedi.

### Sonuçlar

| Model | Önceki F1 | Pipeline F1 | Fark |
|---|---|---|---|
| Dropout Localized (XGBoost) | %75.10 | %75.10 | 0 |
| OULAD v2 (XGBoost) | %80.48 | %80.48 | 0 |

Skorlarda değişiklik yok — Pipeline metodolojik düzeltme, performans etkisi yok.

---

## Pipeline Sonrası Temizlik (22 Nisan 2026)

### Yapılan Düzeltmeler

1. **Chatbot feature names warning düzeltildi**
   Chatbot `np.array` ile tahmin gönderiyordu, Pipeline ise feature isimleriyle fit edilmişti. Bu `X does not have valid feature names` uyarısı veriyordu. `np.array` yerine `pd.DataFrame(columns=FEATURE_ORDER)` kullanıldı.

2. **`chatbot/scaler_params.json` silindi**
   Pipeline entegrasyonundan sonra chatbot artık bu dosyayı yüklemiyordu. Kullanılmayan dosya repodan çıkarıldı.

3. **README güncellendi**
   - Klasör yapısından `scaler_params.json` çıkarıldı
   - Chatbot hazırlık adımından `scaler_params.json` çıkarıldı
   - Chatbot açıklaması Pipeline yapısını yansıtacak şekilde güncellendi
   - "Pipeline gelecekte kullanılabilir" notu → "Pipeline kullanılıyor" olarak güncellendi
   - Model dosyaları açıklamalarına Pipeline notu eklendi

4. **MI Pipeline dışında bırakıldı (bilinçli tercih)**
   OULAD'daki MI feature selection Pipeline dışında raw train verisi üzerinde yapılıyor. MI'yı Pipeline içine almak custom transformer veya sabit `k` gerektirir. Çıkarılan 8 özelliğin MI skorları sıfıra yakın olduğundan hangi fold'da hesaplansa da aynı sonucu verir — pratikte CV optimizm riski yok.

5. **README skorları güncellendi**
   README'deki model sonuçları Pipeline + StratifiedKFold sonrası güncel değerlerle güncellendi: Dropout XGBoost %75.27 → %75.10, OULAD XGBoost %80.40 → %80.48. Tekrar eden iki Pipeline notu tek maddede birleştirildi.

6. **Gereksiz numpy importları kaldırıldı**
   `chatbot/app.py` ve `chatbot/prepare_chatbot.py`'den kullanılmayan `import numpy as np` satırları çıkarıldı.

---

## Metodoloji İncelemesi ve Chatbot İyileştirmeleri (22 Nisan 2026)

### YKS Puanı ve Lise Ortalaması Ölçek Dönüşümü

**Sorun:** Model Portekiz eğitim sisteminin 95-190 ölçeğinde eğitilmiş. Türk öğrenci "YKS'den 380 aldım" veya "lise ortalamam 85" derse, bu değerler dönüştürülmeden modele gidiyordu. MinMaxScaler 0-1 aralığının dışına çıkıyor, tahmin anlamsızlaşıyordu.

**Çözüm:**
- `app.py`'ye `auto_convert_turkish_scale()` fonksiyonu eklendi
- Admission grade > 190 ise YKS puanı kabul edilip `(value/500)*(190-95)+95` formülüyle Portekiz ölçeğine dönüştürülür
- Previous qualification (grade) için 0 < value ≤ 100 ise 100'lük lise notu kabul edilip `(value/100)*(190-95)+95` formülüyle dönüştürülür
- Admission grade ≤ 190 ise zaten model ölçeğinde kabul edilir, dokunulmaz
- `feature_config.json`'daki range'ler ve default'lar Türk ölçeğine güncellendi (YKS: 100-500 default 300, Lise: 0-100 default 70)
- `predict_student()` fonksiyonunda default değerler de aynı dönüşümden geçirilir
- `prepare_chatbot.py` aynı range/question/default değerleriyle güncellendi (yeniden çalıştırıldığında tutarlılık bozulmaz)
- System prompt'a LLM'in bu değerleri dönüştürmemesi talimatı eklendi

### Not Ortalaması Dönüşümünün Python'a Taşınması

**Sorun:** LLM promptunda "4'lük veya 100'lük notu 20'lik sisteme çevir" talimatı vardı. Ancak LLM'ler matematikte güvenilir değil. Öğrenci "100 üzerinden 85" deyip LLM dönüştürmeyi unutursa, `auto_convert_grade()` sadece ≤4.0 değerleri yakalıyordu — 85 değeri 20'lik ölçekte girilmiş sayılıyor, model anlamsız tahmin üretiyordu.

**Çözüm:**
- `auto_convert_grade()` genişletildi: ≤4.0 → 4'lük sistem, >20 → 100'lük sistem, 4-20 arası → zaten 20'lik
- System prompt'tan dönüşüm formülleri kaldırıldı, LLM'e "ham değeri olduğu gibi yaz" talimatı verildi
- Tüm not dönüşümü artık %100 Python tarafında yapılıyor, LLM'e matematik bırakılmıyor

### 2. Dönem Varsayılan Değer Fallback

**Sorun:** Henüz 2. dönemi görmemiş bir 1. sınıf öğrencisi chatbot'a sadece 1. dönem bilgilerini verdiğinde, 2. dönem için iyimser default değerler (6 ders alınmış, 5 geçilmiş, 12/20 not) kullanılıyordu. Bu, zor durumdaki öğrencinin gerçek riskini gizliyordu.

**Çözüm:**
- `predict_student()` fonksiyonuna `SEM_FALLBACK_MAP` eklendi
- 2. dönem verileri eksikse ve 1. dönem karşılıkları varsa, 1. dönem değerleri 2. döneme kopyalanır
- Her iki dönem de eksikse default değerler kullanılmaya devam eder
- Öğrencinin mevcut trendi korunur: 1. dönemde 3 ders geçmiş öğrencinin 2. dönemi de 3 olarak varsayılır, 5 değil

---

## SHAP Açıklanabilirlik Analizi (22 Nisan 2026)

### Amaç

Model yalnızca olasılık üretmekle kalmasın, hangi değişkenlerin tahmini daha çok etkilediği de raporlanabilsin istendi. SHAP bu amaçla eklendi.

### Uygulama

- Chatbot içine canlı SHAP hesabı eklenmedi; performans ve bağımlılık riskini azaltmak için ayrı analiz script'i yazıldı.
- `modeling/shap_dropout_localized.py` dosyası eklendi.
- Script, `models/best_model_dropout_localized.pkl` içindeki Pipeline'ı yükler.
- Test setini aynı feature sırasıyla hazırlar ve Pipeline içindeki scaler ile dönüştürür.
- Eğer en iyi model XGBoost ise `pred_contribs=True`, ağaç tabanlı sklearn modeli ise `shap.TreeExplainer` kullanarak SHAP katkı değerlerini hesaplar.
- Global ve sınıf bazlı SHAP önem tabloları/grafikleri `modeling/plots_shap_dropout_localized/` altına kaydedilir.

### Not

SHAP çıktıları nedensellik iddiası değildir. Yalnızca modelin mevcut tahmin mekanizmasında hangi özelliklerin daha etkili göründüğünü açıklar.

---

## Chatbot Tahmin Öncesi Kontrol Akışı (23 Nisan 2026)

### Sorun

Chatbot yeterli bilgiyi topladığı anda tahmini otomatik üretiyordu. Kullanıcının hangi verilerin modele girdiğini görme, eksik kalan önemli alanları fark etme veya yanlış toplanmış bir bilgiyi tahmin öncesinde düzeltme şansı yoktu.

### Çözüm

- `chatbot/app.py` içinde tahmin öncesi onay adımı eklendi.
- `ANALIZ_HAZIR` geldikten sonra model artık doğrudan çalıştırılmıyor.
- Önce "Tahmin Öncesi Kontrol" bölümü gösteriliyor.
- Bu bölümde:
  - toplanan veriler kullanıcı dostu Türkçe adlarla listeleniyor,
  - eksik kalan `essential` ve `high` öncelikli alanlar ayrı gösteriliyor.
- Kullanıcı iki aksiyon arasından seçim yapabiliyor:
  - `Tahmini Oluştur`
  - `Bilgileri Düzelt`
- `Tahmini Oluştur` seçilirse mevcut pipeline ile analiz üretiliyor.
- `Bilgileri Düzelt` seçilirse tahmin bekletiliyor ve kullanıcı ek/düzeltme bilgisi girmeye devam edebiliyor.

### Teknik Not

- Bu geliştirme yalnızca chatbot katmanında yapıldı; model eğitimi, preprocessing veya kayıtlı model dosyaları değiştirilmedi.
- `pending_prediction` adında yeni bir session state alanı eklendi.
- Mevcut tahmin üretim metni ayrı bir yardımcı fonksiyona taşınarak akış daha okunur hale getirildi.

---

## OULAD için SHAP Analizi (23 Nisan 2026)

### Amaç

Dropout Localized modeli için eklenen açıklanabilirlik yaklaşımının OULAD tarafında da bulunması istendi. Böylece yalnızca bir veri seti için değil, iki güncel model için de hangi özelliklerin tahmini daha çok etkilediği görselleştirilebilecek.

### Uygulama

- `modeling/shap_oulad.py` dosyası eklendi.
- Script, `models/best_model_oulad.pkl` Pipeline modelini yükler.
- `model_oulad_v2.py` içindeki feature engineering adımları aynı şekilde yeniden uygulanır.
- Train/test split aynı parametrelerle kurulur (`test_size=0.30`, `random_state=42`, `stratify=y`).
- OULAD v2 ile uyumlu olması için Mutual Information filtresi yine yalnızca train tarafında hesaplanır ve `MI < 0.01` olan özellikler çıkarılır.
- Ardından Pipeline içindeki scaler ile test verisi dönüştürülür ve XGBoost'un `pred_contribs=True` mekanizmasıyla SHAP katkı değerleri hesaplanır.
- Global ve sınıf bazlı SHAP tablo/grafikleri `modeling/plots_shap_oulad/` altına kaydedilir.

### Not

Bu script ayrı bir analiz katmanıdır; model eğitimi akışını değiştirmez. SHAP çıktıları OULAD modelinin karar mantığını yorumlamaya yardımcı olur, ancak nedensellik iddiası taşımaz.

---

## Sınıf Dengesizliği Düzeltmesi — f1_macro + class_weight (23 Nisan 2026)

### Problem

Her iki modelde de GridSearchCV `scoring='f1_weighted'` ile optimize ediliyordu. Weighted F1, büyük sınıfları (Graduate, Pass) daha fazla ödüllendirdiği için küçük sınıflar (Enrolled %18, Fail %22) ihmal ediliyordu. Enrolled sınıfı F1 %46.54 ve recall %42.44 ile çok düşüktü.

### Uygulanan Değişiklikler

Her iki modelleme dosyasında (`model_dropout_localized.py`, `model_oulad_v2.py`):

1. **GridSearchCV scoring:** `'f1_weighted'` → `'f1_macro'` — her sınıfa eşit ağırlık
2. **Decision Tree & Random Forest:** `class_weight='balanced'` eklendi — sınıf frekansına ters orantılı ağırlık
3. **XGBoost:** `compute_sample_weight('balanced', y_train)` ile örnek ağırlıkları hesaplanıp `model__sample_weight` olarak GridSearchCV'ye geçildi
4. **Raporlama:** Hem `F1-Weighted` hem `F1-Macro` birlikte raporlanır. En iyi model seçimi `F1-Macro`'ya göre yapılır
5. **10-fold CV:** Her iki metrik de paralel raporlanır

### Dropout Localized Sonuçları

| Metrik | Önceki | Güncel | Fark |
|--------|--------|--------|------|
| Weighted F1 | %75.10 | %74.33 | -0.77 |
| Macro F1 | %69.06 | %69.06 | +0.00 |
| Enrolled F1 | %46.54 | %50.09 | +3.55 |
| Enrolled Recall | %42.44 | %57.98 | +15.54 |
| Dropout Recall | %72.60 | %65.81 | -6.79 |

En iyi model: Random Forest (CV Macro F1: %69.04). Enrolled recall'u %42 → %58'e çıktı. Weighted F1'de küçük kayıp var ama sınıflar arası denge iyileşti. Test seti artık model seçimi için değil, yalnızca final raporlama için kullanılıyor.

### OULAD v2 Sonuçları

| Metrik | Önceki | Güncel | Fark |
|--------|--------|--------|------|
| Weighted F1 | %80.48 | %80.44 | -0.04 |
| Macro F1 | — | %75.76 | (yeni metrik) |
| Fail F1 | %52.28 | %56.87 | +4.59 |
| Fail Recall | %45.84 | %58.65 | +12.81 |
| Withdrawn Recall | %82.87 | %74.89 | -7.98 |

En iyi model: XGBoost (CV Macro F1: %76.33). Fail recall'u %46 → %59'a çıktı. Weighted F1 neredeyse aynı kaldı (sadece 0.04 puan kayıp). Test seti yalnızca final değerlendirme için tutuldu.

### Metodolojik Not

- `f1_macro`, her sınıfa eşit önem verir. Bu, azınlık sınıflarının (Enrolled, Fail) ihmal edilmesini önler.
- `class_weight='balanced'`, sınıf frekansına göre `n_samples / (n_classes * n_samples_per_class)` ağırlığı otomatik hesaplar.
- XGBoost'ta native class_weight parametresi olmadığı için `compute_sample_weight` ile örnek bazlı ağırlıklar hesaplanıp GridSearchCV'ye geçildi.
- Model seçimi 10-fold CV Macro F1 skoruna göre yapılır; test seti tek seferlik final raporlama için ayrılır.
- Weighted F1'deki küçük düşüş beklenen bir trade-off: büyük sınıflardan biraz feda edilerek küçük sınıflar iyileştirildi. Eğitimsel açıdan doğru olan, her üç sınıfın dengeli tahmin edilmesidir.

---

## Temel Smoke Testler (23 Nisan 2026)

### Amaç

Projede artık kayıtlı model dosyaları, chatbot feature sırası ve işlenmiş veri şeması arasında sessiz uyumsuzluk oluşmamasını erken fark etmek için hafif ama gerçek bir test katmanı gerektiği değerlendirildi.

### Eklenenler

- `tests/test_smoke.py` dosyası eklendi.
- Testler `unittest` ile yazıldı; ek test framework bağımlılığı gerekmedi.
- Çalıştırma komutu:

```bash
python -m unittest discover -s tests -v
```

### Kapsam

1. `models/best_model_dropout_localized.pkl` yüklenir ve örnek veri üzerinde `predict / predict_proba` çalıştırılır.
2. `chatbot/feature_config.json` sırası ile `models/dropout_localized_features.pkl` sırası birebir karşılaştırılır.
3. `models/best_model_oulad.pkl` yüklenir; OULAD için feature engineering + MI filtre mantığı uygulanıp örnek tahmin üretilir.
4. `preprocessing/dropout_processed.csv` ve `preprocessing/oulad_processed.csv` dosyalarında beklenen target/feature şeması kontrol edilir.

### Not

Bu testler performans benchmark'ı değildir. Ama model dosyası bozulması, feature sırası kayması, preprocessing şema kırılması gibi pratik hataları erken yakalamak için yeterli temel güvence sağlar.

---

## ROC/PR Eğrileri ve Learning Curve (23 Nisan 2026)

### Amaç

Sadece tek bir F1 skoru raporlamak model davranışını tam göstermiyor. Özellikle çok sınıflı yapıda sınıf bazlı ayrım gücünü ve veri arttıkça modelin nasıl davrandığını ayrıca görmek istendi.

### Eklenenler

Her iki güncel modelleme script'ine (`model_dropout_localized.py`, `model_oulad_v2.py`) seçilen final model için iki yeni değerlendirme grafiği eklendi:

1. **ROC/PR eğrileri**
   - `03_roc_pr_curves.png`
   - Her sınıf için one-vs-rest ROC ve Precision-Recall eğrileri çizilir
   - Micro-average eğrisi de ayrıca gösterilir

2. **Learning curve**
   - `05_learning_curve.png`
   - Eğitim örnek sayısı arttıkça train ve validation Macro F1 değişimi çizilir
   - Amaç overfitting / underfitting davranışını kaba düzeyde gözlemlemektir

### Teknik Not

- Learning curve mevcut metodolojiyle uyumlu kalması için seçilen final model üstünden hesaplanır.
- Seçilen model XGBoost ise fold bazında `sample_weight` korunur; diğer modeller normal `fit()` akışıyla değerlendirilir.
- ROC/PR eğrileri çok sınıflı problem için one-vs-rest yaklaşımıyla hesaplanır.
- Bu grafikler model seçimi için değil, final analizi zenginleştirmek için kullanılır.

---

## Model Seçimi ve XGBoost CV Düzeltmesi (23 Nisan 2025)

### Problem 1 — Test seti ile model seçimi

Daha önce en iyi model test set F1-Macro skoruna göre seçilip aynı test set skoru "final sonuç" olarak raporlanıyordu. Bu metodolojik olarak yanlış: test seti model seçim sürecine girerse raporlanan performans iyimser olur. Doğru yaklaşım: modeli CV skoru ile seçmek, test setini sadece seçilen modelin final değerlendirmesi için bir kez kullanmak.

### Problem 2 — XGBoost 10-fold CV'de sample_weight eksikliği

GridSearchCV'de XGBoost'a `model__sample_weight` geçiriliyordu ama 10-fold `cross_val_score` çağrılarında geçirilmiyordu. DT/RF'nin `class_weight='balanced'` parametresi model objesinin içinde olduğu için otomatik uygulanır, ama XGBoost'un sample_weight'i harici olduğundan elle geçirilmesi gerekir.

### Uygulanan Değişiklikler

Her iki modelleme dosyasında:

1. **Model seçimi:** Test set skorları yerine 10-fold CV Macro F1 ortalamasından en iyi model seçilir
2. **XGBoost 10-fold CV:** `cross_val_score` yerine manuel fold döngüsü yazıldı — her fold'da `compute_sample_weight('balanced', y_fold_train)` hesaplanıp `model__sample_weight` olarak geçilir
3. **Test seti:** Sadece CV ile seçilen model için bir kez kullanılır, tüm modeller için ayrı ayrı değil
4. **Confusion matrix:** 5 modelin 2x3 grid'i yerine sadece seçilen modelin tek confusion matrix'i kaydedilir
5. **CV karşılaştırma tablosu:** Tüm modellerin CV skorları raporlanır (test skorları değil)

### Sonuçlar

**Dropout Localized:** CV ile Random Forest seçildi (CV Macro %69.04 vs XGBoost %68.88). XGBoost'un sample_weight ile 10-fold CV skoru GridSearchCV'deki skordan farklı çıktı çünkü artık her fold'da kendi ağırlıkları hesaplanıyor. Test Macro F1 %69.06 — CV skoru ile çok tutarlı.

**OULAD v2:** CV ile yine XGBoost seçildi (CV Macro %76.33). Test Macro F1 %75.76 — yine CV ile tutarlı.

### Metodolojik Not

- CV skoru ile test skoru arasındaki küçük farklar beklenen varyans. Yakın olmaları modelin generalize ettiğini gösterir.
- Dropout modelinde Random Forest'ın seçilmesi, sample_weight düzeltmesinin etkisi: XGBoost artık her fold'da tutarlı ağırlıklarla eğitilince CV skoru biraz düştü ve RF öne geçti.
- Bu düzeltmeyle raporlanan skorlar artık metodolojik olarak temiz: test seti hiçbir karar sürecine karışmıyor.
