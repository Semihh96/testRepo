import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

class TextClassifier:
    def __init__(self, model_path='gradient_boosting_model.pkl', vectorizer_path='tfidf_vectorizer.pkl'):
        nltk.download('punkt')
        nltk.download('stopwords')
        
        self.stemmer = PorterStemmer()
        self.vectorizer_path = vectorizer_path
        self.model_path = model_path
        
        self.vectorizer = self.load_vectorizer()
        self.model = self.load_model()
    
    def preprocess_text(self, text):
        if isinstance(text, str):
            text = text.lower()
            text = text.translate(str.maketrans('', '', string.punctuation))
            words = word_tokenize(text)
            stop_words = set(stopwords.words('english'))
            cleaned_words = [word for word in words if word not in stop_words and len(word) > 1]
            stemmed_words = [self.stemmer.stem(word) for word in cleaned_words]
            return ' '.join(stemmed_words)
        else:
            return ''
    
    def load_data(self, file_path):
        self.df = pd.read_csv(file_path)
        self.df['response'] = self.df['response'].fillna('')
        self.df['tag'] = self.df['tag'].fillna('Unknown')
        self.df['tag'] = self.df['tag'].apply(self.preprocess_text)
    
    def prepare_features(self):
        X = self.df['tag']
        y = self.df['response']
        X_tfidf = self.vectorizer.fit_transform(X)
        return X_tfidf, y
    
    def train_model(self):
        X_tfidf, y = self.prepare_features()
        X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
        
        # Modeli oluştur ve eğit
        self.model = GradientBoostingClassifier(n_estimators=40, max_depth=8, random_state=1, verbose=1)
        self.model.fit(X_train, y_train)
        
        # Test verisi ile doğruluk oranını hesapla
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_percentage = min(accuracy * 100, 100)
        print(f"Accuracy: {accuracy_percentage:.2f}%")
        print(classification_report(y_test, y_pred))
        
        # Modeli ve vectorizer'ı kaydet
        self.save_model()
        self.save_vectorizer()
    
    def save_model(self):
        joblib.dump(self.model, self.model_path)
    
    def save_vectorizer(self):
        joblib.dump(self.vectorizer, self.vectorizer_path)
    
    def load_model(self):
        try:
            return joblib.load(self.model_path)
        except FileNotFoundError:
            return None
    
    def load_vectorizer(self):
        try:
            return joblib.load(self.vectorizer_path)
        except FileNotFoundError:
            return TfidfVectorizer()
    
    def predict(self, texts):
        if self.model and self.vectorizer:
            processed_texts = [self.preprocess_text(text) for text in texts]
            texts_tfidf = self.vectorizer.transform(processed_texts)
            predictions = self.model.predict(texts_tfidf)
            probabilities = self.model.predict_proba(texts_tfidf)
            return predictions, probabilities
        else:
            raise Exception("Model or vectorizer not loaded.")

# Main kısmı
if __name__ == "__main__":
    classifier = TextClassifier()
    
    # Veriyi yükle
    classifier.load_data("output_file_cleaned_tagged.csv")
    
    # Eğer model mevcut değilse, modeli eğit
    if classifier.model is None:
        classifier.train_model()
    
    # Test etmek için örnek metinler
    test_texts = [
        "How can I access advanced battery information and settings?"
    ]
    
    predictions, probabilities = classifier.predict(test_texts)
    
    # İlk beş tahmini ve olasılıklarını yazdır
    for i in range(min(5, len(test_texts))):
        print(f"Text: {test_texts[i]}")
        print(f"Prediction: {predictions[i]}")
        print(f"Probabilities: {probabilities[i]}")
        print()
