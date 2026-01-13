import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# Load trained datasets
data = pd.read_csv("data/processed/emails_with_urgency.csv")

X = data["text"].astype(str)
y_category = data["target"]
y_urgency = data["urgency"]

# Vectorizer
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_vec = vectorizer.fit_transform(X)

# Category Model
category_model = MultinomialNB()
category_model.fit(X_vec, y_category)

# Urgency Model
urgency_model = LogisticRegression(max_iter=1000)
urgency_model.fit(X_vec, y_urgency)

def predict_email(text):
    vec = vectorizer.transform([text])

    category = category_model.predict(vec)[0]
    urgency = urgency_model.predict(vec)[0]

    # Enterprise rule: Spam is always low urgency
    if category == 1:
        urgency = "low"

    return int(category), urgency


# Manual test
if __name__ == "__main__":
    sample = "My internet is down, please fix immediately"
    print(predict_email(sample))
