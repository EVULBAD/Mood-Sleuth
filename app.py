# Imports.
import pandas as pd
import os
import joblib

from flask import Flask, render_template, request, send_file
from sentiment_analysis import preprocess, rating_to_sentiment
from io import BytesIO

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
        filepath = f'{base}_{counter}{extension}'
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
                file_path = get_unique_filename(file_path)
                file.save(file_path)
                df = pd.read_csv(file_path)
                df = df.drop_duplicates(subset=['Review'])
                
                # Perform sentiment analysis.
                df['Preprocessed'] = df['Review'].apply(preprocess)
                X_vec = vectorizer.transform(df['Preprocessed'])
                df['Rating'] = logreg_model.predict(X_vec)
                mean_predicted_rating = df['Rating'].mean()
                avg_sentiment = rating_to_sentiment(round(mean_predicted_rating))
                example_positive_review = df[df['Rating'] == 5].sample(1)['Review'].values[0] if not df[df['Rating'] == 5].empty else 'No positive reviews found.'
                example_negative_review = df[df['Rating'] == 1].sample(1)['Review'].values[0] if not df[df['Rating'] == 1].empty else 'No negative reviews found.'
                
                # Prepare CSV for download.
                sentiment_analysis_report = df[['Review', 'Rating']].to_csv(index=False)

                # Delete file after analysis.
                os.remove(file_path)
            else:
                return 'File invalid. Please only upload csv.'
        elif 'text_input' in request.form:
            # Text input handling.
            input_text = request.form['text_input']
            preprocessed_text = preprocess(input_text)
            input_vector = vectorizer.transform([preprocessed_text])
            predicted_rating_int = logreg_model.predict(input_vector)[0]
            predicted_rating_text = rating_to_sentiment(predicted_rating_int)

            # Render the result on the same page.
            return render_template('index.html', input_text=input_text, predicted_rating_int=predicted_rating_int, predicted_rating_text=predicted_rating_text)

        return render_template(
            'index.html',
            avg_sentiment = avg_sentiment,
            example_positive_review = example_positive_review,
            example_negative_review = example_negative_review,
            sentiment_analysis_report = sentiment_analysis_report
        )

    return render_template('index.html')

@app.route('/download_predictions', methods=['POST'])
def download_predictions():
    # Retrieve the CSV content from the form.
    csv_content = request.form.get('sentiment_analysis_report')

    # Prepare the CSV file for download.
    output = BytesIO()
    output.write(csv_content.encode('utf-8'))
    output.seek(0)

    return send_file(
        output,
        as_attachment=True,
        download_name='sentiment_analysis_report.csv',
        mimetype='text/csv'
    )

if __name__ == '__main__':
    app.run(debug=True)