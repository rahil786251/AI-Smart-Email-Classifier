print("TRAIN_BASELINE SCRIPT STARTED")

import os
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# -------------------------------------------------
# STEP 1: Find PROJECT ROOT correctly
# -------------------------------------------------
CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(CURRENT_FILE)
        )
    )
)

# -------------------------------------------------
# STEP 2: Dataset path (CONFIRMED)
# -------------------------------------------------
DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "email_labeled.csv"
)

print("Dataset path:", DATA_PATH)

# -------------------------------------------------
# STEP 3: Load dataset
# -------------------------------------------------
data = pd.read_csv(DATA_PATH)

X = data["clean_text"]
y = data["category"]

print("Dataset loaded")
print("Total records:", len(data))

# -------------------------------------------------
# STEP 4: TF-IDF Vectorization
# -------------------------------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# -------------------------------------------------
# STEP 5: Train-test split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------------------------
# STEP 6: Train models
# -------------------------------------------------
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

print("Baseline models trained")

# -------------------------------------------------
# STEP 7: Save models
# -------------------------------------------------
MODEL_DIR = os.path.join(
    PROJECT_ROOT, "submission", "Milestone2", "models"
)
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(lr_model, os.path.join(MODEL_DIR, "logistic_regression.pkl"))
joblib.dump(nb_model, os.path.join(MODEL_DIR, "naive_bayes.pkl"))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))

print("Models saved successfully")
print("Milestone 2 – Step 1 DONE")
