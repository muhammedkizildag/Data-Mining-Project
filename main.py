import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Veriyi yükle (UCI verisi ';' ile ayrılmıştır)
df = pd.read_csv('dataset/student/student-mat.csv', sep=';')

# Hedef değişkeni oluştur: G3 >= 10 ise Geçti (1), değilse Kaldı (0)
df['pass'] = (df['G3'] >= 10).astype(int)
df = df.drop('G3', axis=1) # Orijinal notu çıkar

# Kategorik verileri sayısallaştır (Label Encoding)
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

# Sayısal verileri normalleştir (0-1 arasına getir)
scaler = MinMaxScaler()
numerical_cols = ['age', 'absences', 'G1', 'G2']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])


X = df.drop('pass', axis=1)
y = df['pass']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)



# Modelleri tanımla
models = {
    "kNN (Euclidean)": KNeighborsClassifier(n_neighbors=5, metric='euclidean'),
    "Naive Bayes": GaussianNB()
}

results = []
kf = KFold(n_splits=10, shuffle=True, random_state=42)

for name, model in models.items():
    # 10-Fold Cross Validation Skoru (Doğruluk)
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf)
    
    # Modeli test seti ile değerlendir
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Metrikleri hesapla
    results.append({
        "Algoritma": name,
        "CV Mean Accuracy": f"%{cv_scores.mean()*100:.2f}",
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred)
    })



    # Sonuç tablosu
results_df = pd.DataFrame(results)
print(results_df)

# Görselleştirme: Hata Matrisi
# Her iki model için ayrı ayrı hata matrisi çizdirmek için döngüyü kullanın
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    plt.figure(figsize=(6,4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Hata Matrisi - {name}') # Dinamik başlık (KNN veya Naive Bayes)
    plt.ylabel('Gerçek Değer')
    plt.xlabel('Tahmin Edilen')
    plt.show()