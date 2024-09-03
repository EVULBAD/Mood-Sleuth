# Imports.
import joblib
import pandas as pd
import string
import re
import joblib

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Load the model and vectorizer.
model = joblib.load('sentiment_model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

# Initializations for preprocessing.
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Function to help handle negations.
def handle_negations(tokens):
    # Initialization.
    negation_words = {"not", "no", "never", "n't"}
    new_tokens = []
    negation = False

    # For each token, determine is word is being negated.
    for token in tokens:
        if token in negation_words:
            negation = True
        elif negation:
            new_tokens.append("NOT_" + token)
            negation = False
        else:
            new_tokens.append(token)

    return new_tokens

# Function to preprocess text.
def preprocess(text):
    # Lowercase.
    text = text.lower()
    # Remove punctuation.
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers.
    text = re.sub(r'\d+', '', text)
    # Remove unwanted characters.
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = ' '.join(text.split())
    # Tokenize and ensure negations are handled.
    tokens = word_tokenize(text)
    tokens = handle_negations(tokens)
    # Remove stopwords.
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize and stem words.
    lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens]
    # Join words back into a single string.
    processed_text = ' '.join(lemmatized_words)
    # Return.
    return processed_text

# Function to determine sentiment.
def rating_to_sentiment(rating):
    if rating == 1:
        return "negative"
    elif rating == 2:
        return 'mostly negative'
    elif rating == 3:
        return 'neutral'
    elif rating == 4:
        return 'mostly positive'
    elif rating == 5:
        return 'positive'