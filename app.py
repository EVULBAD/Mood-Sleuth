# Imports.
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import os

# Define upload folder path.
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

# Create app.
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper function to check that upload is csv.
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Check if the post request has the file part.
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        # If user does not select file, browser may also submit an empty part without filename.
        if file.filename == '':
            return 'No selected file'
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Load the CSV file into a pandas DataFrame.
            df = pd.read_csv(file_path)

            # Process the CSV file (replace with your sentiment analysis code).
            return f"Uploaded and read {len(df)} rows from the CSV file."
        if ~allowed_file(file.filename):
            return f"File invalid. Please only upload csv."
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)