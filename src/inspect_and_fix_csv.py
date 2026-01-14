# src/inspect_and_fix_csv.py
import csv
import os
import sys
import pandas as pd

INPUT = os.path.join("data", "raw", "emails.csv")
OUTPUT_FIXED = os.path.join("data", "raw", "emails_fixed.csv")

def show_head(path, n=50):
    print(f"\n--- First {n} lines of {path} ---")
    with open(path, "rb") as f:
        raw = f.read(8192*10)
    try:
        text = raw.decode("utf-8")
    except Exception:
        text = raw.decode("latin1")
    lines = text.splitlines()
    for i, line in enumerate(lines[:n], start=1):
        print(f"{i:03d}: {line}")
    print("--- end preview ---\n")

def try_sniffer(sample):
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",","\t",";","|"])
        return dialect.delimiter
    except Exception:
        return None

def attempt_read_with_delim(path, delim):
    try:
        print(f"Trying pandas.read_csv with delimiter = {repr(delim)}")
        df = pd.read_csv(path, sep=delim, engine="python", encoding="utf-8")
        print("Success. Shape:", df.shape)
        return df
    except Exception as e:
        print("Failed read with", repr(delim), ":", e)
        return None

def fallback_line_split(path, out_path):
    print("Falling back to conservative line-splitting (split into at most 3 columns).")
    rows = []
    with open(path, "rb") as f:
        raw = f.read()
    try:
        text = raw.decode("utf-8")
        encoding = "utf-8"
    except:
        text = raw.decode("latin1")
        encoding = "latin1"
    for i, line in enumerate(text.splitlines()):
        # skip empty lines
        if not line.strip():
            continue
        # If the line appears to be a python code line (contains 'def ' or 'import '),
        # we still try to keep it, but mark it.
        # Split into at most 3 parts: id, subject, body
        parts = line.split(",", 2)  # max 3 parts
        if len(parts) == 1:
            # maybe tab-separated or only body present
            parts = line.split("\t", 2)
        # normalize to 3 columns
        while len(parts) < 3:
            parts.append("")
        rows.append(parts[:3])
    # Create DataFrame
    df = pd.DataFrame(rows, columns=["id","subject","body"])
    # Save cleaned CSV using utf-8
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved fallback fixed CSV to: {out_path} (rows: {len(df)})")
    return df

def main():
    if not os.path.exists(INPUT):
        print("Input file not found:", INPUT)
        sys.exit(1)

    # 1) show head
    show_head(INPUT, n=50)

    # 2) sample for sniffer
    sample = ""
    with open(INPUT, "rb") as f:
        raw = f.read(8192)
    try:
        sample = raw.decode("utf-8")
    except:
        sample = raw.decode("latin1")

    delim = try_sniffer(sample)
    if delim:
        print("Sniffer thinks delimiter is:", repr(delim))
        df = attempt_read_with_delim(INPUT, delim)
        if df is not None:
            # save a normalized CSV with utf-8 and standard columns if needed
            df.to_csv(OUTPUT_FIXED, index=False, encoding="utf-8")
            print("Saved normalized CSV to:", OUTPUT_FIXED)
            return
    else:
        print("Could not detect delimiter automatically.")

    # 3) try common delimiters
    for d in [",", "\t", ";", "|"]:
        df = attempt_read_with_delim(INPUT, d)
        if df is not None:
            df.to_csv(OUTPUT_FIXED, index=False, encoding="utf-8")
            print("Saved normalized CSV to:", OUTPUT_FIXED)
            return

    # 4) fallback: line-splitting
    df = fallback_line_split(INPUT, OUTPUT_FIXED)
    print("Fallback produced a file. Inspect the first lines of", OUTPUT_FIXED)
    show_head(OUTPUT_FIXED, n=20)

if __name__ == "__main__":
    main()
