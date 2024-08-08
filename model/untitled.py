import pandas as pd

# Veriyi okuma
df = pd.read_csv("/content/bbc-text.csv", engine='python', encoding='UTF-8')

# İlk birkaç satırı görme
print(df.head())

# Veri setinin genel bilgilerini görme
print(df.info())

# İstatistiksel özet
print(df.describe(include='all'))

# Eksik değerleri kontrol etme
print(df.isnull().sum())

# Eksik değerleri doldurma (örneğin, boş metin ile)
df.fillna('', inplace=True)

# Eksik değerleri silme
df.dropna(inplace=True)

import re
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer

# Küçük harfe çevirme, satır sonlarını ve geri dönüşleri kaldırma
df['text'] = df['text'].apply(lambda x: x.lower().strip().replace('\n', ' ').replace('\r', ' '))

# Harf dışı karakterleri ve Unicode karakterlerini kaldırma
df['text'] = df['text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', ' ', x)).apply(lambda x: re.sub(r'[^\x00-\x7F]+', '', x))

# Linkleri kaldırma
df['text'] = df['text'].apply(lambda x: re.sub(r'http\S+', '', x))

tokenizer = RegexpTokenizer(r'\w+')
df['tokens'] = df['text'].apply(lambda x: tokenizer.tokenize(x))

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

# Stop words ve kısa kelimeleri kaldırma
df['filtered_tokens'] = df['tokens'].apply(lambda x: [word for word in x if word not in stop_words and len(word) > 1])


lemmatizer = WordNetLemmatizer()

# Lemmatizasyon
df['lemmatized_text'] = df['filtered_tokens'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x]))

# CSV formatında kaydetme
df.to_csv("/content/cleaned_bbc-text.csv", index=False, encoding='UTF-8')

# JSON formatında kaydetme
df.to_json("/content/cleaned_bbc-text.json", orient='records', lines=True, force_ascii=False)

import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Word cloud görselleştirme
text = ' '.join(df['lemmatized_text'])
wordcloud = WordCloud(width=800, height=400, background_color ='white').generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
