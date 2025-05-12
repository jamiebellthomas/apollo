import pandas as pd
import sqlite3
import sys

csv_path = "nasdaq_exteral_data.csv"
db_path = "news.db"
table_name = "news"
chunksize = 100_000

# Check if the CSV file exists
try:
    with open(csv_path, 'r', encoding='utf-8') as f:
        pass
except FileNotFoundError:
    print(f"[ERROR] CSV file {csv_path} not found.")
    print("[INFO] Downloading CSV file...")
    import os
    os.system(f"wget https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/nasdaq_exteral_data.csv")
    print("[INFO] CSV file downloaded.")


# Step 1: Count total number of rows (efficiently)
print("[INFO] Counting total number of rows...")
with open(csv_path, 'r', encoding='utf-8') as f:
    total_rows = sum(1 for _ in f) - 1  # subtract header

print(f"[INFO] Total rows in CSV: {total_rows:,}")

# Step 2: Load into SQLite with progress
conn = sqlite3.connect(db_path)
print(f"[INFO] Loading CSV into SQLite DB ({db_path})...")

for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize)):
    chunk.to_sql(table_name, conn, if_exists='append', index=False)
    rows_so_far = (i + 1) * chunksize
    percent = min(rows_so_far / total_rows * 100, 100)
    print(f"[INFO] Processed chunk {i + 1} — {rows_so_far:,} rows ({percent:.2f}%)")

conn.close()
print("[DONE] All chunks written to database.")
