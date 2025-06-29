import pandas as pd
import sqlite3
import yfinance as yf
from time import sleep
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import config

def create_pricing_db():
    """
    Main function to build the pricing database.
    It reads tickers from a CSV, downloads their historical prices,
    and inserts them into an SQLite database.
    """
    
    print("[INFO] Starting to build the pricing database...")
    # === Step 1: Load Ticker List ===
    df = pd.read_csv(config.METADATA_CSV_FILEPATH)
    tickers = df['Symbol'].unique()

    # === Step 2: Connect to SQLite DB ===
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()

    # === Step 3: Create Table (if not exists) ===
    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {config.PRICING_TABLE_NAME} (
        ticker TEXT,
        date DATE,
        adjusted_close REAL,
        PRIMARY KEY (ticker, date)
    );
    """)
    conn.commit()

    # === Step 4: Download & Insert Per Ticker ===
    for ticker in tickers:
        print(f"Processing {ticker}...")

        try:
            data = yf.download(ticker, start=config.START_DATE, end=config.END_DATE, progress=False)
            if data.empty:
                print(f" - No data for {ticker}")
                continue

            # Use 'Close' because it's now auto-adjusted by default
            df_prices = data.reset_index()[['Date', 'Close']]
            df_prices.columns = ['date', 'adjusted_close']
            df_prices['ticker'] = ticker
            df_prices = df_prices[['ticker', 'date', 'adjusted_close']]
            df_prices['date'] = pd.to_datetime(df_prices['date'])

            # === NEW FILTER: Only accept if exactly 3269 rows ===
            if len(df_prices) != config.REQUIRED_ROW_COUNT:
                print(f" - Skipped: has {len(df_prices)} rows (requires {config.REQUIRED_ROW_COUNT})")
                continue

            # Remove already-inserted dates
            existing_dates = pd.read_sql_query(
                f"SELECT date FROM {config.PRICING_TABLE_NAME} WHERE ticker = ?",
                conn, params=(ticker,)
            )
            existing_dates = pd.to_datetime(existing_dates['date']).dt.date
            df_prices = df_prices[~df_prices['date'].dt.date.isin(existing_dates)]

            if df_prices.empty:
                print(" - All records already inserted.")
                continue

            # Insert new rows
            df_prices.to_sql(config.PRICING_TABLE_NAME, conn, if_exists='append', index=False)
            print(f" - Inserted {len(df_prices)} new rows.")

            sleep(1)  # polite delay to avoid throttling

        except Exception as e:
            print(f" - Error with {ticker}: {e}")

    # === Done ===
    conn.close()
    print("All tickers processed.")


