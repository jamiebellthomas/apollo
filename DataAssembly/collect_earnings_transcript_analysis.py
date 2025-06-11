import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
import pandas as pd
from company_names import see_which_have_existed_for_long_enough
import sqlite3
from collect_earnings_transcript import add_to_existing

def row_counts_per_ticker():
    """
    Open config.FILING_DATES_AND_URLS_CSV, and return a dictionary with tickers as keys and row counts as values.
    """
    

    df = pd.read_csv(config.FILING_DATES_AND_URLS_CSV)
    row_counts = df['Ticker'].value_counts().to_dict()
    
    
    return row_counts

def check_missing_tickers():
    """
    Check which tickers in row_counts_per_ticker are not in see_which_have_existed_for_long_enough.
    """
    row_counts = row_counts_per_ticker()
    missing_tickers = see_which_have_existed_for_long_enough()
    missing_vals = {ticker: row_counts[ticker] for ticker in row_counts if ticker in missing_tickers}

    return missing_vals

def check_db_alignment():
    """
    Clean the missing tickers from the filing dates and URLs CSV.
    """
    missing_vals = check_missing_tickers()
    df = pd.read_csv(config.FILING_DATES_AND_URLS_CSV)
    
    if not missing_vals:
        print("[INFO] No missing tickers found.")
    
    else:

        # Filter out rows with tickers in missing_vals
        for ticker in missing_vals.keys():
            df = df[df['Ticker'] != ticker]
        
        # Save the cleaned DataFrame back to CSV
        df.to_csv(config.FILING_DATES_AND_URLS_CSV, index=False)
        print("[INFO] Cleaned missing tickers from filing dates and URLs CSV.")

    # Final sanity check, open the DB_PATH and see if the number of unique tickers in PRICING_TABLE_NAME
    # matches the number of unique tickers in the cleaned CSV 
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()

    conn.commit()
    # Get distinct tickers from the PRICING_TABLE_NAME
    cursor.execute(f"SELECT DISTINCT ticker FROM {config.PRICING_TABLE_NAME}")
    existing_tickers = [row[0] for row in cursor.fetchall()]
    conn.close()
    cleaned_tickers = df['Ticker'].unique().tolist()
    if set(existing_tickers) == set(cleaned_tickers):
        print("[INFO] All tickers in the database match the cleaned CSV.")
    else:
        print("[WARNING] There are discrepancies between the database and the cleaned CSV tickers.")
        print(f"Database tickers: {len(existing_tickers)}")
        print(f"Cleaned CSV tickers: {len(cleaned_tickers)}")

def check_url_counter():
    """
    Check the number of rows associated with each ticker, if it isnt 39-41, 
    we have, flag it and print for manual investigation
    """
    bad_tickers = []
    row_counts = row_counts_per_ticker()
    for ticker, count in row_counts.items():
        if count < 36 or count > 45:
            bad_tickers.append((ticker, count))
            print(f"[WARNING] Ticker {ticker} has {count} rows, which is outside the expected range (36-45).")

    

    return bad_tickers

def retry_bad_tickers():
    """
    Retry the bad tickers by adding them to the existing CSV.
    """
    bad_tickers = check_url_counter()
    if not bad_tickers:
        print("[INFO] No bad tickers found.")
        return

    # open the filing dates and URLs CSV, remove the bad tickers, and then add them back
    df = pd.read_csv(config.FILING_DATES_AND_URLS_CSV)
    for ticker, count in bad_tickers:
        print(f"Retrying ticker: {ticker} with count: {count}")
        # Remove the ticker from the DataFrame
        df = df[df['Ticker'] != ticker]

    # Write back the cleaned DataFrame
    df.to_csv(config.FILING_DATES_AND_URLS_CSV, index=False)

    # Now add the bad tickers back to the existing CSV
    for ticker, _ in bad_tickers:
        add_to_existing(ticker)

    # If there are still bad tickers, remove them from database, the filings_dates_and urls csv and call clean metadata function
    new_bad_tickers = check_url_counter()


    if new_bad_tickers:

        for ticker, _ in new_bad_tickers:
            print(f"Removing ticker: {ticker} from database and CSV due to persistent issues.")
            # Remove from the database
            conn = sqlite3.connect(config.DB_PATH)
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM {config.PRICING_TABLE_NAME} WHERE ticker = ?", (ticker,))
            conn.commit()
            conn.close()

            # Remove from the CSV
            df = pd.read_csv(config.FILING_DATES_AND_URLS_CSV)
            df = df[df['Ticker'] != ticker]
            df.to_csv(config.FILING_DATES_AND_URLS_CSV, index=False)

            check_db_alignment()

    
    see_which_have_existed_for_long_enough()






if __name__ == "__main__":
    check_url_counter()
    check_db_alignment()
    retry_bad_tickers()

