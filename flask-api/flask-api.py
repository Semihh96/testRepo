from flask import Flask, jsonify, request
from model.randomforest import TextClassifier
import os, sys
sys.path.insert(0,"testRepo")

app = Flask(__name__)
data_path = "output_file_cleaned_tagged.csv"
classifier = TextClassifier(data_path)

@app.route('/')
def hello_world():
    return 'Merhaba API!'

@app.route('/predict', methods=['GET'])
def greet():
    test_texts = [
        "How do I reset my smartphone and completely return it to the initial factory settings?",
    ]
    
    prediction = classifier.print_predictions_with_probabilities(test_texts)
    return jsonify({'prediction': f'{prediction}!'})

if __name__ == '__main__':
    app.run(debug=True)
