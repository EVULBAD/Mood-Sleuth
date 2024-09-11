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

# Visualizations.
# 1. Confusion matrix.
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Compute confusion matrix.
cm = confusion_matrix(y_test, y_pred)

# Create confusion matrix.
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['1', '2', '3', '4', '5'], yticklabels=['1', '2', '3', '4', '5'])
plt.xlabel('Predicted Ratings')
plt.ylabel('Actual Ratings')
plt.title('Confusion Matrix')
plt.show()

# 2. Prediction error scatter plot.
import numpy as np
import matplotlib.pyplot as plt

# Compute errors.
errors = np.abs(y_pred - y_test)

# Create scatter plot.
plt.figure(figsize=(10, 6))
plt.scatter(range(len(errors)), errors, alpha=0.5)
plt.yticks([0, 1, 2, 3, 4])
plt.xlabel('Index')
plt.ylabel('Absolute Error')
plt.title('Prediction Errors')
plt.grid(True)
plt.show()

# 3. Word cloud.
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# Create dictionary to map ratings to merged sentiment labels.
sentiment_labels = {
    1: 'Negative',
    2: 'Negative',
    3: 'Neutral',
    4: 'Positive',
    5: 'Positive'
}

# Create custom stopwords to avoid words that don't provide useful sentiment information.
custom_stopwords = set(STOPWORDS)
custom_stopwords.update([
    "hotel", "room", "stay", "trip", "night", "rooms", "mcdonalds", "disney", "ride", "park", "day", "resort", "went", "time", "place", "check", "staff", "check-in", "checkin", "checkout", "facility", "booking", "reservation", "visit", "restaurant", "attraction", "experience", "city", "town", "area", "neighborhood", "people", "guests", "spot", "entry", "exit", "lobby", "bar", "kitchen", "table", "dish", "menu", "cuisine", "option", "choice", "event", "show", "performance", "exhibit", "display", "showcase", "presentation", "activity", "program", "schedule", "plan", "good", "disneyland", "great", "stayed", "really", "dont", "got", "wa", "kid", "food", "world", "make", "family", "looked", "need", "still", "shower", "two", "child", "one", "said", "say", "asked", "told", "hour", "minute", "hours", "minutes", "second", "desk", "thing", "go", "better", "u", "visited", "felt", "maybe", "arrived", "around", "pm", "am", "nice", "many", "get", "getting", "took", "way", "think", "take", "look", "looking", "return", "let", "saw", "paris", "place", "area", "location", "address", "unit", "number", "street", "avenue", "road", "lane", "drive", "square", "building", "floor", "room", "suite", "office", "block", "district", "region", "zone", "section", "part", "side", "corner", "item", "object", "thing", "aspect", "feature", "aspect", "detail", "image", "color", "style", "shape", "design", "appearance", "size", "dimension", "measure", "scale", "capacity", "volume", "quantity", "amount", "level", "stage", "phase", "context", "background", "information", "data", "input", "output", "result", "report", "summary", "overview", "analysis", "review", "comment", "feedback", "opinion", "perspective", "viewpoint", "judgment", "assessment", "evaluation", "note", "record", "entry", "detail", "description", "narrative", "account", "story", "event", "occurrence", "incident", "happening", "experience", "case", "example", "instance", "fact", "event", "situation", "lot", "small", "definitely", "quite", "bit", "use", "guest", "end", "thought", "called", "given", "first", "staying", "decided", "decide", "spent", "bathroom", "going", "checked", "week", "actually", "person", "husband", "wife", "main", "beautiful", "standard", "another", "entire", "half", "character", "hong", "kong", "know", "want", "new orleans", "overall", "little", "land", "used", "feel"
])

# Function to generate and display word cloud with stopwords.
def generate_word_cloud(category, df):
    all_text = " ".join(review for review in df['Preprocessed'])
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=custom_stopwords).generate(all_text)

    # Create word cloud.
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Hide the axis
    plt.title(f'{category} Word Cloud')
    plt.show()

# Map the ratings to the combined sentiment labels
mapped_df = training_df.copy()
mapped_df['Sentiment'] = mapped_df['Rating'].map(sentiment_labels)

# Create separate word clouds for each combined sentiment category.
for sentiment in ['Negative', 'Neutral', 'Positive']:
    filtered_df = mapped_df[mapped_df['Sentiment'] == sentiment]
    generate_word_cloud(sentiment, filtered_df)

# Calculate the percentage of predictions with absolute error 0, 1, or 2.
total_predictions = len(errors)
error_0 = np.sum(errors == 0)
error_1 = np.sum(errors == 1)
error_2 = np.sum(errors == 2)
error_3 = np.sum(errors == 3)
error_4 = np.sum(errors == 4)

percent_error_0 = (error_0 / total_predictions) * 100
percent_error_1 = (error_1 / total_predictions) * 100
percent_error_2 = (error_2 / total_predictions) * 100
percent_error_3 = (error_3 / total_predictions) * 100
percent_error_4 = (error_4 / total_predictions) * 100

print(f"Percentage of predictions with an error of 0 points: {percent_error_0:.2f}%")
print(f"Percentage of predictions with an error of 1 point: {percent_error_1:.2f}%")
print(f"Percentage of predictions with an error of 2 points: {percent_error_2:.2f}%")
print(f"Percentage of predictions with an error of 3 points: {percent_error_3:.2f}%")
print(f"Percentage of predictions with an error of 4 points: {percent_error_4:.2f}%")

# Save model and vectorizer.
# joblib.dump(logreg_model, 'sentiment_model.joblib')
# joblib.dump(vectorizer, 'vectorizer.joblib')
