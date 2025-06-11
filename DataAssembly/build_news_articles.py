import pandas as pd
import sqlite3
import os
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
import csv


def check_raw_data_exists():
    """
    Check if a file exists at the given path.
    """
    # Check if csv file exists
    print(f"[INFO] Checking if raw data file exists at {config.NEWS_CSV_PATH}...")
    if not os.path.exists(config.NEWS_CSV_PATH):
        print("[INFO] Raw data file not found.")
        print("[INFO] Downloading raw data file...")
        # Run wget command to download the CSV file
        os.system("wget https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/nasdaq_exteral_data.csv")
        # Move the downloaded file to the correct directory
        print("[INFO] Moving downloaded file to the correct directory...")
        os.rename(config.ORIGINAL_PATH, config.NEWS_CSV_PATH)

def convert_to_sqlite():
    """
    Convert the CSV file to an SQLite database. Because the CSV file is large, we will read it in chunks.
    """
    print("[INFO] Converting CSV file to SQLite database...")


    
    # Check if the CSV file exists
    if not os.path.exists(config.NEWS_CSV_PATH):
        print(f"[ERROR] CSV file {config.NEWS_CSV_PATH} not found.")
        return

    # Connect to the SQLite database
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()

    # Drop the old table if it exists
    cursor.execute(f"DROP TABLE IF EXISTS {config.NEWS_TABLE_NAME}")

    # Create the news articles table if it doesn't exist
    cursor.execute(f"""
    CREATE TABLE {config.NEWS_TABLE_NAME} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        Date TEXT,
        Article_title TEXT,
        Stock_symbol TEXT,
        Url TEXT,
        Publisher TEXT,
        Author TEXT,
        Article TEXT,
        Lsa_summary TEXT,
        Luhn_summary TEXT,
        Textrank_summary TEXT,
        Lexrank_summary TEXT
    )   
    """)
    total_processed_rows = 0
    # Read the CSV file in chunks and insert into the database
    for chunk in pd.read_csv(
        config.NEWS_CSV_PATH,
        chunksize=100_000,
        quoting=csv.QUOTE_NONE,         # <- disables quote parsing
        escapechar='\\',                # <- optional: handle embedded quotes safely
        on_bad_lines='skip',            # <- skip rows that are still broken
        encoding='utf-8',
        #low_memory=False,
        dtype=str,
        engine='python'  # <- use 'python' engine for better compatibility with large files
    ):
        # Print percentage of rows processed
        total_rows = sum(1 for _ in open(config.NEWS_CSV_PATH, 'r', encoding='utf-8')) - 1
        rows_so_far = len(chunk)
        total_processed_rows += rows_so_far
        percent = (total_processed_rows / total_rows) * 100
        print(f"[INFO] Processed chunk with {len(chunk)} rows ({percent:.2f}%)")
        if 'Unnamed: 0' in chunk.columns:
            chunk = chunk.drop(columns=['Unnamed: 0'])
        chunk.to_sql(config.NEWS_TABLE_NAME, conn, if_exists='append', index=False)

    
    # Commit changes and close connection
    conn.commit()
    conn.close()


def clean_db():
    """
    Clean the SQLite database by removing rows with null Article and filtering tickers.
    """
    print("[INFO] Cleaning the SQLite database...")
    
    # Connect to the SQLite database
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()

    # Print current row count
    cursor.execute(f"SELECT COUNT(*) FROM {config.NEWS_TABLE_NAME}")
    row_count = cursor.fetchone()[0]
    print(f"[INFO] Current row count in {config.NEWS_TABLE_NAME}: {row_count}")

    # Load tickers from metadata CSV
    df_metadata = pd.read_csv(config.METADATA_CSV_FILEPATH)
    tickers = df_metadata['Symbol'].unique()

    # Remove rows with null content
    cursor.execute(f"""
    DELETE FROM {config.NEWS_TABLE_NAME} WHERE Article IS NULL OR Article = '';
    """)
    
    # Filter the DataFrame to only include rows with tickers in the metadata
    cursor.execute(f"""
    DELETE FROM {config.NEWS_TABLE_NAME} WHERE Stock_symbol NOT IN ({','.join(['?'] * len(tickers))});
    """, tickers.tolist())

    # Print new row count
    cursor.execute(f"SELECT COUNT(*) FROM {config.NEWS_TABLE_NAME}")
    row_count = cursor.fetchone()[0]
    print(f"[INFO] New row count in {config.NEWS_TABLE_NAME}: {row_count}")

    # Commit changes and close connection
    conn.commit()
    conn.close()


if __name__ == "__main__":
    check_raw_data_exists()
    convert_to_sqlite()
    clean_db()
    print("[DONE] News articles database built successfully.")
