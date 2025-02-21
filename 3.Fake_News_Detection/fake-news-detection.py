# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load the dataset
df = pd.read_csv('news.csv')

# Inspect the first few rows
print("Dataset preview:")
print(df.head())

# Step 2: Prepare features and labels
# Assuming 'text' contains the news articles and 'label' contains REAL/FAKE
X = df['text']
y = df['label']

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Vectorize the text data
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
# Fit and transform the training data, transform the test data
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Step 4: Train the PassiveAggressiveClassifier
clf = PassiveAggressiveClassifier(max_iter=50)
clf.fit(X_train_vectorized, y_train)

# Step 5: Make predictions and evaluate the model
y_pred = clf.predict(X_test_vectorized)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

# Display confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Display detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 6: Function to predict new articles
def predict_news(text):
    # Vectorize the input text
    text_vectorized = vectorizer.transform([text])
    # Predict using the trained classifier
    prediction = clf.predict(text_vectorized)
    return prediction[0]

# Example usage
sample_text = "This is a sample news article to test the model."
result = predict_news(sample_text)
print(f"\nPrediction for sample text: {result}")