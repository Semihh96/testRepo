# Import necessary libraries
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Download necessary NLTK datasets
nltk.download('punkt')
nltk.download('stopwords')

# Initialize PorterStemmer
stemmer = PorterStemmer()

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    cleaned_words = [stemmer.stem(word) for word in words if word not in stop_words and len(word) > 1]
    return ' '.join(cleaned_words)

# Load dataset
df = pd.read_csv("output_file_cleaned_tagged.csv")

# Fill NaN values
df['response'] = df['response'].fillna('')
df['tag'] = df['tag'].fillna('Unknown')

# Apply preprocessing to text data
df['response'] = df['response'].apply(preprocess_text)
df['tag'] = df['tag'].apply(preprocess_text)

# Save cleaned data
df.to_csv("output_file_cleaned_tagged_updated.csv", index=False)

# Split data into features and target
X = df['tag']
y = df['response']

# Convert text data to TF-IDF features
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Define and train logistic regression model
lr = LogisticRegression(C=2, max_iter=1000, n_jobs=-1)
lr.fit(X_train, y_train)

# Make predictions
y_pred = lr.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Classification report
print(classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Save model and vectorizer
joblib.dump(lr, 'logistic_regression_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
