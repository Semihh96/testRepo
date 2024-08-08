import pandas as pd
import re

# Excel dosyasının yolu
excel_file = 'testRepo/final-data revize answerseng.xlsx'

# Excel dosyasını pandas DataFrame olarak yükleme
df = pd.read_excel(excel_file)

# DataFrame'in ilk birkaç satırını ve sütun adlarını kontrol etme
print("Orijinal DataFrame:")
print(df.head())
print("Sütun adları:", df.columns)

# İlk sütunu kaldırma
# Öncelikle DataFrame yapısını inceleyin
print("İlk sütunun adı:", df.columns[0])

# İlk sütunu kaldırma
df = df.iloc[:, 1:].reset_index(drop=True)

# Sütun adlarını kontrol etme
print("Güncellenmiş Sütun adları:", df.columns)

# Başındaki sayıları ve virgülleri temizleme ve tag etiketi ekleme fonksiyonu
def clean_and_tag_text(text):
    if isinstance(text, str):  # Metin değilse işleme
        # Sayıları ve virgülleri baştan temizle
        cleaned_text = re.sub(r'^\d+\.\d*,?\s*', '', text).strip()
        return cleaned_text
    return text

# Sütun isimlerini yeniden adlandırma
df.columns = ['tag', 'response']

# DataFrame'deki 'tag' ve 'response' sütunlarını temizleme
df['tag'] = df['tag'].apply(clean_and_tag_text)
df['response'] = df['response'].apply(clean_and_tag_text)

# Temizlenmiş ve etiketlenmiş DataFrame'i CSV formatında kaydetme
output_file = 'output_file_cleaned_tagged.csv'
df.to_csv(output_file, index=False, encoding='UTF-8')

print(f"Temizlenmiş ve etiketlenmiş DataFrame CSV dosyasına kaydedildi: {output_file}")

import pandas as pd
import re

# Excel dosyasının yolu
excel_file = 'testRepo/final-data revize answerseng.xlsx'

# Excel dosyasını pandas DataFrame olarak yükleme
df = pd.read_excel(excel_file)

# DataFrame'in ilk birkaç satırını ve sütun adlarını kontrol etme
print("Orijinal DataFrame:")
print(df.head())
print("Sütun adları:", df.columns)

# İlk sütunu kaldırma
df = df.iloc[:, 1:].reset_index(drop=True)

# Başındaki sayıları ve virgülleri temizleme fonksiyonu
def clean_text(text):
    if isinstance(text, str):  # Metin değilse işleme
        # Sayıları ve virgülleri baştan temizle
        cleaned_text = re.sub(r'^\d+\.\d*,?\s*', '', text).strip()
        return cleaned_text
    return text

# İkinci sütuna 'tag:' etiketi ekleme fonksiyonu
def add_tag_prefix(text):
    if isinstance(text, str):  # Metin değilse işleme
        return f"{text}"
    return text

# İkinci sütuna 'tag:' etiketi ekleme
df.iloc[:, 0] = df.iloc[:, 0].apply(add_tag_prefix)  # İkinci sütun

# Üçüncü sütunun başındaki sayıları ve virgülleri temizleme
df.iloc[:, 1] = df.iloc[:, 1].apply(clean_text)  # Üçüncü sütun

# Sütun isimlerini yeniden adlandırma
df.columns = ['tag', 'response']

# Temizlenmiş ve etiketlenmiş DataFrame'i CSV formatında kaydetme
output_file = 'output_file_tagged_cleaned.csv'
df.to_csv(output_file, index=False, encoding='UTF-8')

print(f"Etiketlenmiş ve temizlenmiş DataFrame CSV dosyasına kaydedildi: {output_file}")




