# Imports.
import pandas as pd
import os
import joblib

from flask import Flask, render_template, request, redirect, url_for
from sentiment_analysis import preprocess, rating_to_sentiment

# Load the model and vectorizer
logreg_model = joblib.load('sentiment_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

# Define upload folder path.
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

# Create app.
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check that upload is csv.
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to create a unique filename if the file already exists in the directory.
def get_unique_filename(filepath):
    # Split the file path into base and extension.
    base, extension = os.path.splitext(filepath)
    counter = 1

    # Check if file exists, and append a number to the base if it does.
    while os.path.exists(filepath):
        filepath = f"{base}_{counter}{extension}"
        counter += 1

    return filepath

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Check if the post request has the file part.
        if 'file' in request.files:
            file = request.files['file']
            # If user does not select file, browser may also submit an empty part without filename.
            if file.filename == '':
                return 'No selected file. Please upload csv.'
            if file and allowed_file(file.filename):
                filename = file.filename
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                # Check for existing files and create a unique filename.
                file_path = get_unique_filename(file_path)
                # Save the file with the unique filename.
                file.save(file_path)
                # Load the CSV file into a pandas DataFrame.
                df = pd.read_csv(file_path)
                # Remove exact duplicate reviews.
                df = df.drop_duplicates(subset=['Review'])
                # Perform sentiment analysis on reviews.
                df['Preprocessed'] = df['Review'].apply(preprocess)
                X_vec = vectorizer.transform(df['Preprocessed'])
                df['Predicted Rating'] = logreg_model.predict(X_vec)
                # Perform EDA
                summary = df.describe()
                # summary = summary.drop(['top', 'freq', 'unique'])
                # Render the EDA results template with predictions.
                return render_template('results.html', summary=summary.to_html(), predictions=df[['Review', 'Predicted Rating']].to_html())

            else:
                return "File invalid. Please only upload csv."
        elif 'text_input' in request.form:
            # Text input handling
            input_text = request.form['text_input']
            preprocessed_text = preprocess(input_text)
            input_vector = vectorizer.transform([preprocessed_text])
            predicted_rating = logreg_model.predict(input_vector)[0]

            # Render a template to show the result
            return render_template('single_result.html', input_text=input_text, predicted_rating=predicted_rating)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)