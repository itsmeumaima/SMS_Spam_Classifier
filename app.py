from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

# Download necessary resources
nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

# Text preprocessing function
def transform_text(text):
    ps = PorterStemmer()
    text = text.lower()
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric characters
    words = [i for i in text if i.isalnum()]

    # Remove stopwords and punctuation
    words = [i for i in words if i not in stopwords.words('english') and i not in string.punctuation]

    # Apply stemming
    words = [ps.stem(i) for i in words]

    return " ".join(words)

# Load vectorizer and model
tfidf = pickle.load(open('vectorizer (1).pkl', 'rb'))
model = pickle.load(open('model (1).pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sms = request.form['Message']
    transformed_sms = transform_text(sms)
    input_data = tfidf.transform([transformed_sms])
    prediction = model.predict(input_data)[0]

    result = "Spam" if prediction == 1 else "Not Spam"
    return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)
