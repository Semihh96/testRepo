import openai
from config import OPENAI_API_KEY

# API anahtarını ayarlayın
openai.api_key = OPENAI_API_KEY

# API çağrısını yapın
response = openai.Completion.create(
    engine="text-davinci-003",  # veya kullandığınız motor
    prompt="Hello, world!",
    max_tokens=5
)

# Yanıtı yazdırın
print(response.choices[0].text.strip())
