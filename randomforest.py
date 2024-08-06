import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# NLTK veri setlerini indirin (ilk seferde bir kez)
nltk.download('punkt')
nltk.download('stopwords')

# Dosyanızı okuyun (doğru dosya yolu ve formatı kullanın)
df = pd.read_csv("output_file_cleaned_tagged.csv")

# NaN değerlerini boş bir string ile doldurma
df['response'] = df['response'].fillna('')
df['tag'] = df['tag'].fillna('Unknown')  # 'Unknown' gibi bir etiket ekleyebilirsiniz

# NaN değerlerini içeren hedef değişkenleri kaldırma
df = df[df['tag'] != 'Unknown']

# Veri çerçevesinin ilk birkaç satırını yazdırın
print(df.head())

# Stop words ve noktalama işaretlerini temizleme fonksiyonu
def preprocess_text(text):
    if isinstance(text, str):  # Sadece metin verileri üzerinde işlem yap
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        words = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        cleaned_words = [word for word in words if word not in stop_words and len(word) > 1]
        return ' '.join(cleaned_words)
    else:
        return ''  # Sayısal verileri boş string ile değiştirme

# 'response' sütununu temizleme
df['tag'] = df['tag'].apply(preprocess_text)
df.to_csv("output_file_cleaned_tagged_updated.csv", index=False)

# Özellikler (X) ve hedef (y) değişkenlerini belirleyin
X = df['tag']
y = df['response']

# Metin verilerini sayısal verilere dönüştürme (TF-IDF kullanarak)
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Eğitim ve test verilerini ayırma
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# RandomForestClassifier modelini tanımlama
rfc = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42, class_weight='balanced')

# Modeli eğitme
rfc.fit(X_train, y_train)

# Tahmin yapma
y_pred = rfc.predict(X_test)

# Doğruluğu hesaplama
accuracy = accuracy_score(y_test, y_pred)
accuracy_percentage = accuracy * 100

# Yüzdelik doğruluğu 100 ile sınırla (her durumda %100'ü geçmeyecek şekilde)
accuracy_percentage = min(accuracy_percentage, 100)

print(f"Accuracy: {accuracy_percentage:.2f}%")

# Sınıflandırma raporu
print(classification_report(y_test, y_pred))

# Modeli kaydetme
joblib.dump(rfc, 'random_forest_model.pkl')

# TF-IDF vectorizer'ı kaydetme (modeli tekrar kullanırken gerekli olabilir)
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')