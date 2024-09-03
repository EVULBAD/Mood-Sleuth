# Imports.
import pandas as pd
import random
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score

from sentiment_analysis import preprocess

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
logreg_model = LogisticRegression(max_iter=1000)
X = training_df['Preprocessed']
y = training_df['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

# Vectorize.
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train.
logreg_model.fit(X_train_vec, y_train)

print('Training complete.')
print()

# Evaluate.
y_pred = logreg_model.predict(X_test_vec)
features_nd = X_test_vec.toarray()
test_indices = X_test.index.tolist()

print('Training value counts:')
print(training_df['Rating'].value_counts())
print()
print('Testing value counts:')
print(testing_df['Rating'].value_counts())
print()

for i in range(10):
    idx = random.choice(test_indices)
    print(f"Review: {testing_df['Review'].iloc[idx]}")
    print(f"Preprocessed: {testing_df['Preprocessed'].iloc[idx]}")
    print(f"Predicted Rating: {y_pred[test_indices.index(idx)]}")
    print(f"Actual Rating: {testing_df['Rating'].iloc[idx]}")
    print()

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['Negative', 'Mostly Negative', 'Neutral', 'Mostly Positive', 'Positive']))

import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['1', '2', '3', '4', '5'], yticklabels=['1', '2', '3', '4', '5'])
plt.xlabel('Predicted Ratings')
plt.ylabel('Actual Ratings')
plt.title('Confusion Matrix')
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Compute errors
errors = np.abs(y_pred - y_test)

plt.figure(figsize=(10, 6))
plt.scatter(range(len(errors)), errors, alpha=0.5)
plt.yticks([0, 1, 2, 3, 4])
plt.xlabel('Index')
plt.ylabel('Absolute Error')
plt.title('Prediction Errors')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(errors, bins=range(1, 6), edgecolor='black')
plt.xticks([1, 2, 3, 4, 5])
plt.xlabel('Absolute Error')
plt.ylabel('Frequency')
plt.title('Histogram of Prediction Errors')
plt.grid(True)
plt.show()

# Save model and vectorizer.
joblib.dump(logreg_model, 'sentiment_model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')