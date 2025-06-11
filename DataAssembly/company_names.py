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
    Check which companies have existed long enough to have at least 3269 trading days.
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

    # Now see which ones were removed in the filter process

    filtered_tickers = set(downloaded_tickers) & set(existing_tickers)
    print(f"Filtered tickers that have sufficient data: {len(filtered_tickers)}")

    return filtered_tickers

if __name__ == "__main__":

    filtered_tickers = see_which_have_existed_for_long_enough()
    print(f"Filtered tickers: {filtered_tickers}")
    

