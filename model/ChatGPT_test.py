import requests

# OpenAI API key
OPENAI_API_KEY = ''

def query_chatgpt(prompt):
    url = 'https://api.openai.com/v1/chat/completions'
    headers = {
        'Authorization': f'Bearer {OPENAI_API_KEY}',
        'Content-Type': 'application/json'
    }
    data = {
        'model': 'gpt-3.5-turbo',  # if you want you can change the model
        'messages': [{'role': 'user', 'content': prompt}]
    }
    response = requests.post(url, json=data, headers=headers)
    return response.json()

print(query_chatgpt("merhaba"))