# src/train_urgency.py  (safer version)
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LABELED_CSV = os.path.join(ROOT, "data", "processed", "emails_labeled.csv")
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(LABELED_CSV):
    raise SystemExit(f"Missing labeled file: {LABELED_CSV}")

print("Reading labeled data:", LABELED_CSV)
df = pd.read_csv(LABELED_CSV, encoding="utf-8")

# Find a text column to use
text_col = None
for c in ["clean_text","__text","body","text","message","content"]:
    if c in df.columns:
        text_col = c
        break
if text_col is None:
    cols = df.columns.tolist()
    if 'urgency' in cols:
        idx = cols.index('urgency')
        if idx > 0:
            text_col = cols[idx-1]
if text_col is None:
    raise SystemExit("No text column found in labeled CSV. Add 'clean_text' or 'body' or similar.")

print("Using text column:", text_col)
df[text_col] = df[text_col].fillna("").astype(str)

# Normalize urgency labels
df['urgency'] = df['urgency'].astype(str).str.lower().str.strip()
allowed = ['high','medium','low']
df = df[df['urgency'].isin(allowed)]
n = df.shape[0]
print(f"Number of labeled rows used: {n}")
if n < 10:
    print("Warning: very small labeled set (n<10). Results will be unreliable.")

X_text = df[text_col]
y = df['urgency']

# Vectorize
vec = TfidfVectorizer(stop_words="english", max_features=8000)
X = vec.fit_transform(X_text)

# Stratify only when possible
stratify = y if len(y.value_counts()) > 1 and all(y.value_counts() >= 2) else None
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
except Exception:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = MultinomialNB()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Avoid undefined-metric warnings by setting zero_division and by only requesting classes present in y_test
present_labels = sorted(list(set(y_test.unique()) | set(y_pred)))
print("Labels present in evaluation:", present_labels)

report = classification_report(y_test, y_pred, labels=present_labels, zero_division=0)
print("\nUrgency classification results")
print("Accuracy:", acc)
print("\nClassification report:\n", report)

# Compute confusion matrix for present labels only
try:
    cm = confusion_matrix(y_test, y_pred, labels=present_labels)
    print("\nConfusion matrix (rows=true, cols=pred):\n", cm)
except Exception as e:
    print("Could not compute confusion matrix:", e)

# Save model + vectorizer
joblib.dump(clf, os.path.join(MODEL_DIR, "email_urgency_nb.pkl"))
joblib.dump(vec, os.path.join(MODEL_DIR, "email_urgency_tfidf.pkl"))
print("\nSaved urgency model to:", os.path.join(MODEL_DIR, "email_urgency_nb.pkl"))
print("Saved vectorizer to:", os.path.join(MODEL_DIR, "email_urgency_tfidf.pkl"))
