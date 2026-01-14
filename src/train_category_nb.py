import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
DATA_PATH = "data/processed/emails_labeled.csv"
df = pd.read_csv(DATA_PATH)

print("Columns:", df.columns.tolist())

# Use correct columns (from your dataset)
X = df["text"].astype(str)
y = df["target"]

print("Text samples:")
print(X.head(3))

# Vectorization
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.9,
    min_df=2
)

X_vec = vectorizer.fit_transform(X)

# Train Naive Bayes on FULL dataset (Milestone 2 – baseline)
model = MultinomialNB()
model.fit(X_vec, y)

# Predict on same data (allowed for baseline)
y_pred = model.predict(X_vec)

# Evaluation
print("\n📊 Baseline Naive Bayes Results (Full Dataset Evaluation)")
print("Accuracy:", accuracy_score(y, y_pred))
print("\nClassification Report:")
print(classification_report(y, y_pred))
