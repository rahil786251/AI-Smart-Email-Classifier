import pandas as pd
import re

# Paths
INPUT_PATH = "data/processed/emails_labeled.csv"
OUTPUT_PATH = "data/processed/emails_with_urgency.csv"

# Load dataset
df = pd.read_csv(INPUT_PATH)

# Urgency keywords
HIGH_URGENCY = [
    "urgent", "asap", "immediately", "right away",
    "important", "critical", "emergency", "now"
]

MEDIUM_URGENCY = [
    "please", "request", "help", "support", "issue",
    "problem", "unable", "delay"
]

def detect_urgency(text, target):
    text = str(text).lower()

    # 1️⃣ High urgency keywords
    for word in HIGH_URGENCY:
        if word in text:
            return "high"

    # 2️⃣ Spam → Low urgency
    if target == 1:
        return "low"

    # 3️⃣ Medium urgency keywords
    for word in MEDIUM_URGENCY:
        if word in text:
            return "medium"

    # 4️⃣ Default
    return "low"

# Apply urgency detection
df["urgency"] = df.apply(
    lambda row: detect_urgency(row["text"], row["target"]),
    axis=1
)

# Save new dataset
df.to_csv(OUTPUT_PATH, index=False)

print("✅ Urgency labels generated successfully!")
print(df["urgency"].value_counts())
