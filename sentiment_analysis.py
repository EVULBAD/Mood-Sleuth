# Imports.
import pandas as pd
import numpy as np
import string
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Initializations for preprocessing.
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Preprocessing function.
def preprocess_text(text):
    # Lowercase.
    text = text.lower()
    # Remove punctuation.
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers.
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespaces.
    text = ' '.join(text.split())
    # Tokenize.
    tokens = word_tokenize(text)
    # Remove stopwords.
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize words.
    lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens]
    # Join words back into a single string.
    processed_text = ' '.join(lemmatized_words)
    # Return.
    return processed_text

# Read and preprocess data.
# Training csvs path: ./train/r[csv name].csv
mcdonalds_reviews = pd.read_csv("./train/reviews_mcdonalds_labelled.csv")
hotel_reviews = pd.read_csv("./train/reviews_hotels_labelled.csv")
reviews_df = pd.concat([mcdonalds_reviews, hotel_reviews], ignore_index=True)
reviews_df['Preprocessed'] = reviews_df['Review'].apply(preprocess_text)
print(reviews_df)