#https://github.com/xtekky/gpt4free
from g4f.client import Client
from langdetect import detect
from googletrans import Translator

client = Client()
translator = Translator()

def chatbot_response(user_input):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"system", "content":"Sadece teknoloji ile ilgili sorulara yanıt verilecektir. Tüm yanıtlar maksimum 10 cümle olacak. Tüm yanıtlar sadece Türkçe olacak.Yanıtların açık ve kısa olmasına dikkat edin. Lütfen başka bir dilde yanıt vermeyin."},
                  {"role":"user", "content":user_input}]
    )
    content = response.choices[0].message.content
    
    if detect(content) != 'tr':
        translated = translator.translate(content, src='auto', dest='tr').text
        print("translated")
        return translated
    else:
        print("content")
        return content

print(chatbot_response("çay tarifi"))