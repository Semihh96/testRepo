import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib  # Modeli kaydetmek için

# NLTK veri setlerini indirin (ilk seferde bir kez)
nltk.download('punkt')
nltk.download('stopwords')

# Stemming için PorterStemmer'ı tanımlayın
stemmer = PorterStemmer()

# Dosyanızı okuyun (doğru dosya yolu ve formatı kullanın)
df = pd.read_csv("output_file_cleaned_tagged.csv")

# NaN değerlerini boş bir string ile doldurma
df['response'] = df['response'].fillna('')
df['tag'] = df['tag'].fillna('Unknown')

# Veri çerçevesinin ilk birkaç satırını yazdırın
print(df.head())


def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        words = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        cleaned_words = [word for word in words if word not in stop_words and len(word) > 1]
        
        # Stemming işlemi ekleyin
        stemmed_words = [stemmer.stem(word) for word in cleaned_words]
        return ' '.join(stemmed_words)
    else:
        return '' 

# 'response' ve 'tag' sütunlarını temizleme
df['tag'] = df['response'].apply(preprocess_text)


# Temizlenmiş veri çerçevesini yeni bir CSV dosyasına kaydetme
df.to_csv("output_file_cleaned_tagged_updated.csv", index=False)

# Özellikler (X) ve hedef (y) değişkenlerini belirleyin
X = df['tag']
y = df['response']

# Metin verilerini sayısal verilere dönüştürme (TF-IDF kullanarak)
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Eğitim ve test verilerini ayırma
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Logistic Regression modelini tanımlama
lr = LogisticRegression(C=2, max_iter=1000, n_jobs=-1, class_weight='balanced')

# Modeli eğitme
lr.fit(X_train, y_train)

# Tahmin yapma
y_pred1 = lr.predict(X_test)

# Doğruluğu hesaplama
accuracy = accuracy_score(y_test, y_pred1)
print(f"Accuracy: {accuracy:.2f}")

# Sınıflandırma raporu
print(classification_report(y_test, y_pred1))

# Modeli kaydetme
joblib.dump(lr, 'logistic_regression_model.pkl')

# TF-IDF vectorizer'ı kaydetme (modeli tekrar kullanırken gerekli olabilir)
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
