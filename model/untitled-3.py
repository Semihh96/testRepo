import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# Dosyayı oku
df = pd.read_csv("output_file_cleaned_tagged.csv")

# Özellikler (X) ve hedef (y) değişkenlerini belirleyin
X = df['tag']
y = df['response']

# Metin verilerini sayısal verilere dönüştürme (TF-IDF kullanarak)
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Eğitim ve test verilerini ayırma
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# LinearSVC modelini tanımlama
svc = LinearSVC()

# Modeli eğitme
svc.fit(X_train, y_train)

# Tahmin yapma
y_pred2 = svc.predict(X_test)

# Doğruluğu hesaplama
print("Accuracy: " + str(accuracy_score(y_test, y_pred2)))

# Sınıflandırma raporu
print(classification_report(y_test, y_pred2))


