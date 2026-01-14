import pandas as pd
import os
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# -----------------------------
# STEP 1: Load Dataset (Milestone 1 output)
# -----------------------------
DATA_PATH = "../../data/processed/emails_labeled.csv"

data = pd.read_csv(DATA_PATH)

X = data["clean_text"]
y = data["category"]

print("Dataset loaded successfully")
print("Total samples:", len(data))

# -----------------------------
# STEP 2: Text Vectorization (TF-IDF)
# -----------------------------
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_tfidf = vectorizer.fit_transform(X)

print("TF-IDF vectorization completed")

# -----------------------------
# STEP 3: Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train-test split done")

# -----------------------------
# STEP 4: Train Baseline Models
# -----------------------------

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
print("Logistic Regression trained")

# Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
print("Naive Bayes trained")

# -----------------------------
# STEP 5: Save Models
# -----------------------------
MODEL_DIR = "../models"
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(lr_model, f"{MODEL_DIR}/logistic_regression.pkl")
joblib.dump(nb_model, f"{MODEL_DIR}/naive_bayes.pkl")
joblib.dump(vectorizer, f"{MODEL_DIR}/tfidf_vectorizer.pkl")

print("Models and vectorizer saved successfully")

print("Milestone 2 - Step 1 completed")
