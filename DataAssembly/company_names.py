import pandas as pd
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
import sqlite3

def extract():

    # STEP 1: Get S&P 500 tickers with sector info from Wikipedia
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    sp500_df = pd.read_html(url, header=0)[0]

    # STEP 2: Filter by sectors FinBERT-style
    target_sectors = {"Information Technology", "Consumer Discretionary", "Financials"}
    filtered_df = sp500_df[sp500_df["GICS Sector"].isin(target_sectors)]

    # STEP 3: Remove tickers that aren't entirely alphanumeric
    def is_alphanumeric(ticker):
        return ticker.isalnum()
    
    filtered_df = filtered_df[filtered_df['Symbol'].apply(is_alphanumeric)]

    # Optional: Save to CSV
    filtered_df.to_csv(config.METADATA_CSV_FILEPATH, index=False)

    print(f"Selected {len(filtered_df)} companies from S&P 500 in sectors: {', '.join(target_sectors)}")


def see_which_have_existed_for_long_enough():
    """
    Check which companies have existed long enough to have at least 3269 trading days, those that have not are removed from the metadata file
    """
    downloaded = pd.read_csv(config.METADATA_CSV_FILEPATH)
    downloaded_tickers = downloaded['Symbol'].unique()

    # Now we open the SQLite database, read the config.PRICING_TABLE_NAME table, 
    # see which tickers made it (had sufficient yfinance data)

    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f"SELECT DISTINCT ticker FROM {config.PRICING_TABLE_NAME}")
    existing_tickers = [row[0] for row in cursor.fetchall()]
    conn.close()

    # Now see which are in downloaded but not in existing_tickers
    missing_tickers = set(downloaded_tickers) - set(existing_tickers)
    # Remove missing tickers from downloaded DataFrame
    downloaded = downloaded[~downloaded['Symbol'].isin(missing_tickers)]
    # And write the cleaned DataFrame back to CSV
    downloaded.to_csv(config.METADATA_CSV_FILEPATH, index=False)

    return missing_tickers

def main():
    """
    Main function to run the extraction and filtering process.
    """
    print("[INFO] Starting extraction of S&P 500 companies in target sectors from Wikipedia...")
    extract()

    