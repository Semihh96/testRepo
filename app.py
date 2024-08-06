import openai
from flask import Flask, request, jsonify

# API anahtarınızı buraya ekleyin
openai.api_key = 'YOUR_API_KEY'

app = Flask(__name__)

def generate_response(prompt):
    # OpenAI API çağrısı yaparak yanıt oluşturma
    response = openai.Completion.create(
        engine="gpt-3.5-turbo",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Chatbot</title>
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; }
            #chat-box { max-width: 600px; margin: 0 auto; }
            #chat-log { border: 1px solid #ccc; padding: 10px; height: 300px; overflow-y: scroll; }
            #user-input { width: calc(100% - 90px); }
            #send-btn { width: 80px; }
        </style>
    </head>
    <body>
        <h1>Chatbot</h1>
        <div id="chat-box">
            <div id="chat-log"></div>
            <input type="text" id="user-input" placeholder="Type a message...">
            <button id="send-btn">Send</button>
        </div>

        <script>
            $(document).ready(function() {
                $('#send-btn').click(function() {
                    var user_input = $('#user-input').val();
                    $.post('/get_response', {user_input: user_input}, function(data) {
                        $('#chat-log').append('<div>User: ' + user_input + '</div>');
                        $('#chat-log').append('<div>Bot: ' + data.response + '</div>');
                        $('#user-input').val('');
                        $('#chat-log').scrollTop($('#chat-log')[0].scrollHeight); // Scroll to bottom
                    });
                });
            });
        </script>
    </body>
    </html>
    '''

@app.route('/get_response', methods=['POST'])
def chat_response():
    user_input = request.form['user_input']
    bot_response = generate_response(user_input)
    return jsonify({"response": bot_response})

if __name__ == '__main__':
    app.run(debug=True)
