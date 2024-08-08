import google.generativeai as genai

genai.configure(api_key='AIzaSyBpl0UbDHJ2w8e92pFEmUx_rMXIblS5ni8')
model = genai.GenerativeModel(model_name='gemini-1.5-flash')

def getResponse(text):
    response = model.generate_content(text)
    return response

print(getResponse('Telefonum çok takılıyor, ne yapabilirim?'))
