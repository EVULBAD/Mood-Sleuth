import joblib

# Load the model and vectorizer.
model = joblib.load('sentiment_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')