import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix

# 1. Veri Setini Yükleme
df = pd.read_csv(r'dataset\dropout\students_dropout_academic_success.csv', sep=',')
df.columns = df.columns.str.strip()

# 2. Veri Ön İşleme
le = LabelEncoder()
# 'target' sütununu sayısallaştırıyoruz (Dropout: 0, Enrolled: 1, Graduate: 2)
df['target_encoded'] = le.fit_transform(df['target'])

# Özellikleri (X) ve hedef değişkeni (y) ayırma
# Hem orijinal metin sütununu hem de yeni sayısal sütunu X'ten çıkarıyoruz
X = df.drop(['target', 'target_encoded'], axis=1)
y = df['target_encoded']

# Verileri normalleştirme (kNN algoritması için kritiktir)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 3. Modellerin Tanımlanması
knn = KNeighborsClassifier(n_neighbors=5)
nb = GaussianNB()

# 4. 10-Katlı Çapraz Doğrulama (10-Fold Cross Validation) 
kf = KFold(n_splits=10, shuffle=True, random_state=42)
scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

knn_results = cross_validate(knn, X_scaled, y, cv=kf, scoring=scoring)
nb_results = cross_validate(nb, X_scaled, y, cv=kf, scoring=scoring)

# 5. Sonuçların Yazdırılması
def print_metrics(model_name, results):
    print(f"\n--- {model_name} Performans Sonuçları (10-Fold CV) ---")
    print(f"Accuracy (CV Ortalama):  %{results['test_accuracy'].mean()*100:.2f}")
    print(f"Precision: {results['test_precision_macro'].mean():.4f}")
    print(f"Recall:    {results['test_recall_macro'].mean():.4f}")
    print(f"F1-Score:  {results['test_f1_macro'].mean():.4f}")

print_metrics("k-En Yakın Komşu (kNN)", knn_results)
print_metrics("Naive Bayes", nb_results)

# 6. Karmaşıklık Matrisi (Confusion Matrix) Görselleştirme 
# Ödev şartı: %70 eğitim ve %30 test ayrımı (test_size=0.3)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

plot_models = {
    "k-En Yakın Komşu (kNN)": knn,
    "Naive Bayes": nb
}

for name, model in plot_models.items():
    # Modeli eğit ve test seti üzerinde tahmin yap
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Görselleştirme
    plt.figure(figsize=(8,6))
    cm = confusion_matrix(y_test, y_pred)
    
    # Isı haritası çizimi - le.classes_ sayesinde Dropout/Enrolled/Graduate isimleri görünür
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_)
    
    plt.title(f'{name} - Hata Matrisi (Öğrenci Durumu)')
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek Değer')
    plt.tight_layout()
    plt.show()