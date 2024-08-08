from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat/message', methods=['POST'])
def chat_message():
    user_message = request.json.get('message')
    response = get_bot_response(user_message)
    return jsonify({"response": response})

def get_bot_response(message):
    # Basit bir chatbot yanıt mantığı
    if 'merhaba' in message.lower():
        return 'Merhaba! Nasılsınız?'
    elif 'nasılsın' in message.lower():
        return 'Ben bir yapay zekayım, bu yüzden hissetmiyorum ama size yardımcı olabilirim!'
    else:
        return 'Anlayamadım, lütfen başka bir şey söyleyin.'


if __name__ == '__main__':
    app.run(debug=True, port=5000)
