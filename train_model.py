# Imports.
import pandas as pd
import string
import re
import random

from Tools.scripts.objgraph import printundef
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

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
    # Remove unwanted characters.
    text = re.sub(r'[^\x00-\x7F]+', '', text)
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
# Training dataframe.
reviews_1 = pd.read_csv("./training/train/reviews_amazon_labelled.csv")
reviews_2 = pd.read_csv("./training/train/reviews_restaurants_labelled.csv")
training_df = pd.concat([reviews_1, reviews_2], ignore_index=True)
training_df = training_df.iloc[5000:]
training_df['Preprocessed'] = training_df['Review'].apply(preprocess_text)

# Testing dataframe.
reviews_3 = pd.read_csv("./training/test/reviews_hotels_labelled.csv")
reviews_4 = pd.read_csv("./training/test/reviews_mcdonalds_labelled.csv")
testing_df = pd.concat([reviews_3, reviews_4], ignore_index=True)
testing_df = testing_df.iloc[5000:]
testing_df['Preprocessed'] = testing_df['Review'].apply(preprocess_text)

# Initializations for training.
model = LogisticRegression(max_iter=200)
X = training_df['Preprocessed']
y = training_df['Sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1, stratify=y)

# Vectorize.
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train.
model.fit(X_train_vec, y_train)

# Evaluate.
y_pred = model.predict(X_test_vec)
features_nd = X_test_vec.toarray()

print(training_df['Sentiment'].value_counts())
print(testing_df['Sentiment'].value_counts())
print()

for i in range(10):
    idx = random.randint(0, len(X_test) - 1)
    print(f"Review: {testing_df['Review'].iloc[idx]}")
    print(f"Preprocessed: {testing_df['Preprocessed'].iloc[idx]}")
    print(f"Predicted Sentiment: {y_pred[idx]}")
    print(f"Actual Sentiment: {testing_df['Sentiment'].iloc[idx]}")
    print()

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, labels=[0, 2], target_names=['Negative', 'Positive']))