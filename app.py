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

def get_unique_filename(filepath):
    """
    Helper function to create a unique filename if the file already exists in the directory.
    """
    # Split the file path into base and extension
    base, extension = os.path.splitext(filepath)
    counter = 1

    # Check if file exists, and append a number to the base if it does
    while os.path.exists(filepath):
        filepath = f"{base}_{counter}{extension}"
        counter += 1

    return filepath

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Check if the post request has the file part.
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        # If user does not select file, browser may also submit an empty part without filename.
        if file.filename == '':
            return 'No selected file. Please upload csv.'
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Check for existing files and create a unique filename
            file_path = get_unique_filename(file_path)

            # Save the file with the unique filename
            file.save(file_path)

            # Load the CSV file into a pandas DataFrame.
            df = pd.read_csv(file_path)

            # Remove exact duplicate reviews.
            df = df.drop_duplicates(subset=['Reviews'])

            # Perform EDA.
            summary = df.describe()
            summary = summary.drop(['top', 'freq', 'unique'])

            # Convert missing values Series to DataFrame.
            missing_values = df.isnull().sum().reset_index()
            missing_values.columns = ['Column', 'Missing Values']

            # Render the EDA results template.
            return render_template('results.html', summary=summary.to_html(), missing_values=missing_values.to_html())
        else:
            return "File invalid. Please only upload csv."
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)