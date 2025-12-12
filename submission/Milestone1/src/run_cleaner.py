# src/run_cleaner.py
import os
import sys
import pandas as pd

# Ensure the 'src' folder is importable when running "python src\run_cleaner.py"
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
SRC_DIR = os.path.abspath(THIS_DIR)   # src folder path

# Put src dir at front of sys.path so we can import preprocess.py directly
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from preprocess import clean_email
except Exception as e:
    raise SystemExit(f"Failed to import clean_email from src/preprocess.py — error: {e}")

# Input/Output paths (change if your filenames are different)
INPUT = os.path.join(PROJECT_ROOT, "data", "raw", "emails.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUT = os.path.join(OUT_DIR, "emails_clean.csv")

# Make sure output folder exists
os.makedirs(OUT_DIR, exist_ok=True)

# Read CSV (detect separator automatically is not safe; expecting standard CSV)
if not os.path.exists(INPUT):
    raise SystemExit(f"Input file not found: {INPUT}\nMake sure your dataset is at data\\raw\\emails.csv")

print("Reading input:", INPUT)
df = pd.read_csv(INPUT)

# Try to detect the main text column
possible_text_cols = ["body", "text", "message", "content", "email_body"]
text_col = None
for c in possible_text_cols:
    if c in df.columns:
        text_col = c
        break

if text_col is None:
    # fallback: if there are only 2-3 columns, pick the last column as body
    if len(df.columns) >= 1:
        text_col = df.columns[-1]
        print(f"Warning: couldn't find standard text column; using last column: '{text_col}'")
    else:
        raise SystemExit("No usable text column found in CSV. Open the CSV and check column names.")

print("Using text column:", text_col)
df["clean_text"] = df[text_col].fillna("").astype(str).apply(clean_email)

df.to_csv(OUTPUT, index=False)
print("Cleaning completed! Saved to:", OUTPUT)
