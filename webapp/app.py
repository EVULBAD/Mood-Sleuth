import pandas as pd
import os
import joblib
import random
import time

from flask import Flask, render_template, request, jsonify, send_file
from datetime import datetime
from sentiment_analysis import preprocess, rating_to_sentiment

# Load the model and vectorizer.
logreg_model = joblib.load('sentiment_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

# Definitions.
UPLOAD_FOLDER = '/home/evulotyh/mood-sleuth/uploads'
TEMP_FOLDER = '/home/evulotyh/mood-sleuth/tmp'
ALLOWED_EXTENSIONS = {'csv'}
global_variables = {}

# Create app.
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TEMP_FOLDER'] = TEMP_FOLDER

# Function to check that upload is csv.
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to create a unique filename if the file already exists in the directory.
def get_unique_filename(filepath):
    base, extension = os.path.splitext(filepath)
    counter = 1
    while os.path.exists(filepath):
        filepath = f"{base}_{counter}{extension}"
        counter += 1
    return filepath

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No selected file. Please upload csv.'})
            if file and allowed_file(file.filename):
                filename = file.filename
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file_path = get_unique_filename(file_path)
                file.save(file_path)
                df = pd.read_csv(file_path)
                df['Preprocessed'] = df['Review'].apply(preprocess)
                X_vec = vectorizer.transform(df['Preprocessed'])
                df['Rating'] = logreg_model.predict(X_vec)
                df['Sentiment'] = df['Rating'].apply(rating_to_sentiment)
                mean_predicted_rating = df['Rating'].mean()
                avg_sentiment = rating_to_sentiment(round(mean_predicted_rating))

                # Generate unique filename using current time and a random 2 digits.
                dt = datetime.now()
                dt = dt.strftime('%H%M%S%f')[:-3]
                rand = random.random()
                rand = round(rand * 100)
                rand = str(rand)
                output_name = 'mood-sleuth_' + dt + rand + '.csv'
                global_variables['output_file'] = output_name

                # Turn dataframe into CSV for downloading.
                sentiment_analysis_report = df[['Review', 'Rating', 'Sentiment']].to_csv(index=False)
                sentiment_analysis_report_path = os.path.join(app.config['TEMP_FOLDER'], output_name)
                df[['Review', 'Rating', 'Sentiment']].to_csv(sentiment_analysis_report_path, index=False)

                # Delete file after processing.
                os.remove(file_path)

                # Polling mechanism to ensure the file is available.
                while not os.path.exists(sentiment_analysis_report_path):
                   time.sleep(1)
                time.sleep(3)

                return jsonify({
                    'avg_sentiment': avg_sentiment,
                    'sentiment_analysis_report': sentiment_analysis_report
                })
            else:
                return jsonify({'error': 'File invalid. Please only upload csv.'})

        elif 'text_input' in request.form:
            input_text = request.form['text_input']
            preprocessed_text = preprocess(input_text)
            input_vector = vectorizer.transform([preprocessed_text])
            predicted_rating_int = int(logreg_model.predict(input_vector)[0])
            predicted_rating_text = rating_to_sentiment(predicted_rating_int)

            return jsonify({
                'input_text': input_text,
                'predicted_rating_int': predicted_rating_int,
                'predicted_rating_text': predicted_rating_text
            })

    return render_template('index.html')

@app.route('/download_temp', methods=['GET'])
def download_temp():
    output_name = global_variables.get('output_file')
    output_path = os.path.join(app.config['TEMP_FOLDER'], output_name)
    if os.path.exists(output_path):
        return send_file(output_path, as_attachment=True)
    else:
        return "Error: File not found.", 404

if __name__ == '__main__':
    app.run()
