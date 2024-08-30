# Imports.
import pandas as pd
import string
import re
import random
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
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
mcdonalds_reviews = pd.read_csv("./train/reviews_mcdonalds_labelled.csv")
hotel_reviews = pd.read_csv("./train/reviews_hotels_labelled.csv")
reviews_df = pd.concat([mcdonalds_reviews, hotel_reviews], ignore_index=True)
reviews_df['Preprocessed'] = reviews_df['Review'].apply(preprocess_text)

# Initializations for training.
model = LogisticRegression()
X = reviews_df['Preprocessed']
y = reviews_df['Sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Vectorize.
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train.
model.fit(X_train_vec, y_train)

# Evaluate.
y_pred = model.predict(X_test_vec)
features_nd = X_test_vec.toarray()

for i in range(5):
    idx = random.randint(0, len(X_test) - 1)
    print(f"Review: {reviews_df['Review'].iloc[idx]}")
    print(f"Predicted Sentiment: {y_pred[idx]}")
    print(f"Actual Sentiment: {y_test.iloc[idx]}")
    print()

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))