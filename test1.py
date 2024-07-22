import nltk

# nltk paketini indir
nltk.download('stopwords')

# stopwords listesini kullan
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
print(stop_words)

import nltk
nltk.data.path.append('/path/to/nltk_data')  # Uygun yolu belirtin
nltk.download('stopwords')