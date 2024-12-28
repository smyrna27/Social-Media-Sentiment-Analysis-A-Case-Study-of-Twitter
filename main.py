# Social Media Sentiment Analysis
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset (use correct path to CSV file)
file_path = '/content/test.csv'

# Try reading the CSV file with different encodings (in case of UnicodeDecodeError)
try:
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    print("File loaded successfully with 'ISO-8859-1' encoding!")
    
except UnicodeDecodeError:
    try:
        df = pd.read_csv(file_path, encoding='utf-16')
        print("File loaded successfully with 'utf-16' encoding!")
    
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='utf-8', errors='ignore')
        print("File loaded successfully, ignoring problematic characters!")

# Show the first few rows of the dataset
print(df.head())

# Preprocess the dataset
# Drop rows where essential columns ('text' or 'sentiment') are missing
df.dropna(subset=['text', 'sentiment'], inplace=True)

# Convert the target sentiment to numerical labels if it's not numeric
# Assuming 'sentiment' is categorical with labels: 'positive', 'negative', 'neutral'
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0, 'neutral': 2})

# Feature (X) - 'text' column; Target (y) - 'sentiment' column
X = df['text']
y = df['sentiment']

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text data into numerical format using CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Initialize and train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Predict on the test set
y_pred = model.predict(X_test_vec)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Display evaluation metrics
print(f"Model Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Sample Predictions for first 5 tweets in the test set
sample_data = X_test[:5]
sample_vec = vectorizer.transform(sample_data)
sample_predictions = model.predict(sample_vec)

print(f"Sample Predictions for the first 5 tweets: {sample_predictions}")
