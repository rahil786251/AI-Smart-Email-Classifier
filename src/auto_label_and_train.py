# src/auto_label_and_train.py
"""
Unified auto-label + train script.
Handles two cases:
- Text CSV with columns like: id, subject, body, clean_text
- Bag-of-words CSV with many token-count columns (like Enron bag-of-words)

Outputs:
- data/processed/emails_labeled.csv  (id/subject/body/clean_text/category/urgency)
- models/email_category_nb.pkl
- prints accuracy and classification report
"""
import os, sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_CSV = os.path.join(ROOT, "data", "raw", "emails.csv")
OUT_LABELED = os.path.join(ROOT, "data", "processed", "emails_labeled.csv")
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(os.path.dirname(OUT_LABELED), exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(RAW_CSV):
    raise SystemExit(f"Input not found: {RAW_CSV}")

print("Reading:", RAW_CSV)
# try utf-8 then latin1
try:
    df = pd.read_csv(RAW_CSV, encoding="utf-8", low_memory=False)
except Exception:
    df = pd.read_csv(RAW_CSV, encoding="latin1", low_memory=False)

print("Dataset shape:", df.shape)
nrows = len(df)

# helper: join subject+body into text
def get_text_col(df):
    for candidate in ["clean_text", "body", "message", "text", "content"]:
        if candidate in df.columns:
            return candidate
    # try subject + last column heuristics
    if "subject" in df.columns and df.shape[1] >= 2:
        return None  # we'll compose from subject+last
    return None

text_col = get_text_col(df)
is_bow = df.shape[1] > 50 and text_col is None  # heuristic: many columns -> bag-of-words

if is_bow:
    print("Detected bag-of-words style dataset (many columns).")
    # use token columns directly as X
    # attempt to preserve an identifier column
    id_col = None
    for c in ["Email No.","id","email_id","index"]:
        if c in df.columns:
            id_col = c; break
    # heuristic label rules similar to previous script
    req_cols = [c for c in df.columns if c in ['request','requested','requests','requesting']]
    complaint_keywords = ['not','issue','problem','refund','error','fail','failure','delay','delayed']
    complaint_cols = [c for c in df.columns if any(k in c.lower() for k in complaint_keywords)]
    urgency_tokens = [c for c in df.columns if c.lower() in ['asap','immediately','urgent','priority']]
    has_spam = 'spam' in df.columns
    has_feedback = 'feedback' in df.columns

    df['_is_spam'] = df['spam']>0 if has_spam else False
    df['_is_request'] = df[req_cols].sum(axis=1)>0 if req_cols else False
    df['_is_feedback'] = df['feedback']>0 if has_feedback else False
    df['_is_complaint'] = df[complaint_cols].sum(axis=1)>0 if complaint_cols else False

    def assign_category_bow(r):
        if r['_is_spam']:
            return 'spam'
        if r['_is_request']:
            return 'request'
        if r['_is_complaint']:
            return 'complaint'
        if r['_is_feedback']:
            return 'feedback'
        return 'feedback'

    df['category'] = df.apply(assign_category_bow, axis=1)
    df['_urg_score'] = 0
    for t in urgency_tokens:
        if t in df.columns:
            df['_urg_score'] += df[t]
    df['urgency'] = np.where(df['_urg_score']>0,'high','low')

    # features: all numeric columns except helper ones
    exclude = ['Email No.','category','urgency','_is_spam','_is_request','_is_feedback','_is_complaint','_urg_score']
    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols].fillna(0).astype(int)
    y = df['category']

    # train/test
    if nrows < 10:
        print("Dataset too small for reliable train/test (n < 10). Saving labels only.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
        model = MultinomialNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test,y_pred)
        print("Accuracy:", acc)
        print(classification_report(y_test,y_pred))
        joblib.dump(model, os.path.join(MODEL_DIR, "email_category_nb.pkl"))
else:
    print("Detected text-style dataset (subject/body). Using TF-IDF + keyword heuristics for labels.")
    # Build text column
    if text_col:
        df['__text'] = df[text_col].fillna("").astype(str)
    else:
        # compose from subject + last column
        last_col = df.columns[-1]
        subj = df['subject'].fillna("").astype(str) if 'subject' in df.columns else ""
        last = df[last_col].fillna("").astype(str)
        df['__text'] = (subj + " " + last).astype(str)

    # Simple keyword heuristics for category and urgency
    # category keywords
    complaint_kw = ["refund","not delivered","not received","unable to","cannot","can't","error","fail","issue","problem","payment","charged","late","delay"]
    request_kw = ["request","feature","can you add","please add","would like","could you","how do i","help me","support"]
    feedback_kw = ["love","great","thanks","thank you","awesome","good","nice","well done","appreciate"]
    spam_kw = ["win","free","click","http","claim","prize","offer","buy now","subscribe","unsubscribe"]

    def detect_category(text):
        t = text.lower()
        for k in spam_kw:
            if k in t: return 'spam'
        for k in complaint_kw:
            if k in t: return 'complaint'
        for k in request_kw:
            if k in t: return 'request'
        for k in feedback_kw:
            if k in t: return 'feedback'
        return 'feedback'

    df['category'] = df['__text'].apply(detect_category)

    # urgency
    urg_kw = ["urgent","asap","immediately","right away","as soon as possible"]
    def detect_urgency(text):
        t = text.lower()
        for k in urg_kw:
            if k in t: return 'high'
        return 'low'
    df['urgency'] = df['__text'].apply(detect_urgency)

    # If dataset too small, warn but continue
    if nrows < 5:
        print(f"Warning: very small dataset (n={nrows}). Model results will be meaningless.")

    # Vectorize and train
    vec = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vec.fit_transform(df['__text'].fillna("").astype(str))
    y = df['category']

    if nrows < 3:
        print("Too few rows to train. Saving labeled CSV only.")
    else:
        # if some classes have only 1 sample, stratify will fail: fallback to non-stratified split
        try:
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
        except Exception:
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
        model = MultinomialNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print("Accuracy:", acc)
        print(classification_report(y_test, y_pred))
        joblib.dump((model, vec), os.path.join(MODEL_DIR, "email_category_nb_tfidf.pkl"))

# Save labeled CSV (include subject/body if present)
out_cols = []
if 'id' in df.columns:
    out_cols.append('id')
elif 'Email No.' in df.columns:
    out_cols.append('Email No.')
if 'subject' in df.columns:
    out_cols.append('subject')
# include body or clean_text if present
for c in ['body','clean_text','__text']:
    if c in df.columns:
        out_cols.append(c)
        break
out_cols += ['category','urgency']
out_df = df[out_cols].copy()
out_df.to_csv(OUT_LABELED, index=False)
print("Saved labeled CSV to:", OUT_LABELED)
print("Saved model (if trained) to:", MODEL_DIR)

