import sqlite3
import pandas as pd
import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}

def is_technically_valid_url(url, timeout=20):
    try:
        response = requests.get(url, headers=HEADERS, allow_redirects=True, stream=True, timeout=timeout)
        if response.status_code == 404:
            print(f"[WARNING] {url} → 404 Not Found")
            return False
        return 200 <= response.status_code <= 399
    except Exception as e:
        print(f"[WARNING] {url} → {e}")
        return False


# Step 1: Open DB and load only non-null Articles
conn = sqlite3.connect("news.db")  

# Step 2: Load data into DataFrame
df = pd.read_sql_query("SELECT * FROM news WHERE Article IS NOT NULL", conn)
conn.close()
print(f"[INFO] Loaded {len(df)} rows with non-null Articles.")

# Step 3: Save as a new database
df.to_sql("news_cleaned", sqlite3.connect("news_cleaned.db"), if_exists="replace", index=False)
print("[INFO] Saved cleaned data to news_cleaned.db.")

# Step 4: Print old and new row counts
conn = sqlite3.connect("news.db")
old_count = pd.read_sql_query("SELECT COUNT(*) FROM news", conn).iloc[0, 0]
conn.close()
conn = sqlite3.connect("news_cleaned.db")
new_count = pd.read_sql_query("SELECT COUNT(*) FROM news_cleaned", conn).iloc[0, 0]
conn.close()
print(f"[INFO] Old row count: {old_count}")
print(f"[INFO] New row count: {new_count}")