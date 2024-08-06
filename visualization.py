import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud,ImageColorGenerator

# Örnek DataFrame oluşturma
data = {
    'text': ["Bu bir örnek metindir.", "Başka bir metin örneği.", "Daha kısa bir metin.", "Oldukça uzun bir metin örneği burada yer alıyor."]
}
df = pd.DataFrame(data)

# Figure ve subplot oluşturma
fig = plt.figure(figsize=(14, 7))

# Metin uzunluklarını hesaplama ve yeni bir sütun ekleme
df['length'] = df.text.str.split().apply(len)

# Histogram oluşturma
ax1 = fig.add_subplot(122)
sns.histplot(df['length'], ax=ax1, color='green')

# Metin uzunluklarının istatistiksel özetini hesaplama
describe = df.length.describe().to_frame().round(2)

# İstatistiksel özet tablosunu oluşturma
ax2 = fig.add_subplot(121)
ax2.axis('off')
font_size = 14
bbox = [0, 0, 1, 1]
table = ax2.table(cellText=describe.values, rowLabels=describe.index, bbox=bbox, colLabels=describe.columns)
table.set_fontsize(font_size)

# Başlık ekleme
fig.suptitle('Distribution of text length for text.', fontsize=16)

# Grafiği gösterme
plt.show()

sns.set_theme(style="whitegrid")
sns.countplot(x=df["category"])

normal_words =' '.join([text for text in df['Text']])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

