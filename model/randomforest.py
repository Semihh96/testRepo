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

class TextClassifier:
    def __init__(self, data_path, model_path='random_forest_model.pkl', vectorizer_path='tfidf_vectorizer.pkl'):
        # NLTK veri setlerini indir
        nltk.download('punkt')
        nltk.download('stopwords')
        
        # Dosya yolunu ayarla ve veriyi oku
        self.data_path = data_path
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        
        # Stop words ve noktalama işaretlerini temizleme fonksiyonunu tanımla
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer()
        self.rfc = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42, class_weight='balanced')
        
        # Model ve vectorizer'ı yükle
        self.load_model()
        
    def preprocess_text(self, text):
        if isinstance(text, str):
            text = text.lower()
            text = text.translate(str.maketrans('', '', string.punctuation))
            words = word_tokenize(text)
            cleaned_words = [word for word in words if word not in self.stop_words and len(word) > 1]
            return ' '.join(cleaned_words)
        else:
            return ''
    
    def clean_data(self):
        df = pd.read_csv(self.data_path)
        df['response'] = df['response'].fillna('')
        df['tag'] = df['tag'].fillna('Unknown')
        df = df[df['tag'] != 'Unknown']
        df['tag'] = df['tag'].apply(self.preprocess_text)
        df.to_csv("output_file_cleaned_tagged_updated.csv", index=False)
        return df
    
    def prepare_data(self, df):
        X = df['tag']
        y = df['response']
        X_tfidf = self.vectorizer.fit_transform(X)
        return train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
    
    def train_model(self):
        df = self.clean_data()
        X_train, X_test, y_train, y_test = self.prepare_data(df)
        self.rfc.fit(X_train, y_train)
        y_pred = self.rfc.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_percentage = accuracy * 100
        accuracy_percentage = min(accuracy_percentage, 100)
        
        print(f"Accuracy: {accuracy_percentage:.2f}%")
        print(classification_report(y_test, y_pred))
        
        joblib.dump(self.rfc, self.model_path)
        joblib.dump(self.vectorizer, self.vectorizer_path)
    
    def load_model(self):
        try:
            self.rfc = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)
        except FileNotFoundError:
            print("Model or vectorizer not found. Please train and save them first.")
            raise
    
    def print_predictions_with_probabilities(self, texts):
        if len(texts) > 5:
            texts = texts[:5]
        
        texts_tfidf = self.vectorizer.transform(texts)
        predictions = self.rfc.predict(texts_tfidf)
        probs = self.rfc.predict_proba(texts_tfidf)
        

        for i in range(len(texts)):
            print(f"Text: {texts[i]}")
            print(f"Prediction: {predictions[i]}")
            probs_percent = [f"{prob * 100:.2f}%" for prob in probs[i]]
            print(f"Probabilities: {', '.join(probs_percent)}")
            print()

        return predictions[0]

def main():
    data_path = "output_file_cleaned_tagged.csv"
    classifier = TextClassifier(data_path)
    
    # Eğitim ve model kaydetme
    if not joblib.os.path.exists(classifier.model_path):
        classifier.train_model()
    
    # Test etmek için örnek metinler
    test_texts = [
        "How do I reset my smartphone and completely return it to the initial factory settings?",
    ]
    
    classifier.print_predictions_with_probabilities(test_texts)

if __name__ == "__main__":
    main()
