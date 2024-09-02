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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score

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
            # Apply negation to the following words
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

print('Reading and preprocessing training data...')

# Read and preprocess data.
# Training dataframe.
reviews_1 = pd.read_csv("./training/train/reviews_disneyland_balanced.csv")
reviews_2 = pd.read_csv("./training/train/reviews_tripadvisor_balanced.csv")
training_df = pd.concat([reviews_1, reviews_2], ignore_index=True)
training_df['Preprocessed'] = training_df['Review'].apply(preprocess)

print('Training data processed.')
print()
print('Reading and preprocessing testing data...')

# Testing dataframe.
reviews_3 = pd.read_csv("./training/test/reviews_mcdonalds_balanced.csv")
reviews_4 = pd.read_csv("./training/test/reviews_amazonfood_balanced.csv")
testing_df = pd.concat([reviews_3, reviews_4], ignore_index=True)
testing_df['Preprocessed'] = testing_df['Review'].apply(preprocess)

print('Testing data processed.')
print()
print('Training in progress...')

# Initializations for training.
model = LogisticRegression(max_iter=1000)
X = training_df['Preprocessed']
y = training_df['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

# Vectorize.
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train.
model.fit(X_train_vec, y_train)

print('Training complete.')
print()

# Evaluate.
y_pred = model.predict(X_test_vec)
features_nd = X_test_vec.toarray()

print('Training value counts:')
print(training_df['Rating'].value_counts())
print()
print('Testing value counts:')
print(testing_df['Rating'].value_counts())
print()

# Map indices from X_test to the corresponding testing_df indices
test_indices = X_test.index.tolist()

for i in range(10):
    idx = random.choice(test_indices)
    print(f"Review: {testing_df['Review'].iloc[idx]}")
    print(f"Preprocessed: {testing_df['Preprocessed'].iloc[idx]}")
    print(f"Predicted Rating: {y_pred[test_indices.index(idx)]}")
    print(f"Actual Rating: {testing_df['Rating'].iloc[idx]}")
    print()

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['Negative', 'Mostly Negative', 'Neutral', 'Mostly Positive', 'Positive']))