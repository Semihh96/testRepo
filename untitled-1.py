import pandas as pd

# Excel dosyasının yolu
excel_file = 'testRepo/final-data revize answerseng.xlsx'

# Excel dosyasını pandas DataFrame olarak yükleme
df = pd.read_excel(excel_file)

# DataFrame'in ilk birkaç satırını kontrol etme
print("Orijinal DataFrame:")
print(df.head())

# Sütun isimlerini kontrol etme (gerekirse yeniden adlandırma)
print("Sütun isimleri:", df.columns)

# Eğer sütun isimlerini yeniden adlandırmak gerekiyorsa:
# df = df.rename(columns={'EskiSutunAdi1': 'tag', 'EskiSutunAdi2': 'response'})

# DataFrame'i CSV formatında kaydetme
output_file = 'output_file.csv'
df.to_csv(output_file, index=False, encoding='UTF-8')

print(f"DataFrame CSV dosyasına kaydedildi: {output_file}")
