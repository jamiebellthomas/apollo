import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import config
import pandas as pd
from DataAssembly.MetaData.company_names import see_which_have_existed_for_long_enough
import sqlite3
from collect_earnings_transcript import add_to_existing

def row_counts_per_ticker():
    """
    Open config.FILING_DATES_AND_URLS_CSV, and return a dictionary containing the 10-Q and 10-K count for each ticker.
    """
    df = pd.read_csv(config.FILING_DATES_AND_URLS_CSV)
    
    # For each ticker work out the number of 10-K and 10-Q filings
    # There is a column called "Form Type" that either has "10-K" or "10-Q" in it for each row
    # I want specific counts for the number of 10-Q and 10-K filings per ticker in the form of a nested dictionary {ticker: {10-K: count, 10-Q: count}}
    row_counts = {}
    for ticker in df['Ticker'].unique():
        ticker_df = df[df['Ticker'] == ticker]
        count_10k = ticker_df[ticker_df['Form Type'] == '10-K (Annual report)'].shape[0]
        count_10q = ticker_df[ticker_df['Form Type'] == '10-Q (Quarterly report)'].shape[0]
        row_counts[ticker] = {'10-K': count_10k, '10-Q': count_10q}

    return row_counts

def check_missing_tickers():
    """
    Check which tickers in row_counts_per_ticker are not in see_which_have_existed_for_long_enough.
    """
    row_counts = row_counts_per_ticker()
    ticker_list = see_which_have_existed_for_long_enough()
    # see_which_have_existed_for_long_enough returns a list of tickers that have existed for long enough, and are in the database
    # (RETURNS THE CURRENT GLOBAL WORKING LIST OF TICKERS, and ENSURES THAT METADATA_CSV_FILEPATH IS UP TO DATE WITH THE DATABASE)
    # If we have any in row_counts that are in ticker_list, we have a problem and need to remove them from the CSV
    missing_tickers = list(set(row_counts.keys()) - set(ticker_list))

    return missing_tickers

def check_db_alignment():
    """
    Removes tickers that are not in the database from the filing dates and URLs CSV.
    This is a sanity check to ensure that the tickers in the CSV match those in the database.
    """
    missing_vals = check_missing_tickers()
    df = pd.read_csv(config.FILING_DATES_AND_URLS_CSV)
    
    if len(missing_vals) == 0:
        print("[INFO] No missing tickers found.")
    
    else:

        # Filter out rows with tickers in missing_vals
        for ticker in missing_vals:
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
    for ticker, counts in row_counts.items():
        # 10-K and 10-Q filings should be around 40-42 for each ticker
        if not (13 <= counts['10-K'] <= 14) or not (39 <= counts['10-Q'] <= 41):
            print(f"[WARNING] Ticker {ticker} has unusual filing counts: {counts}")
            bad_tickers.append((ticker, counts))

    

    return bad_tickers

def retry_bad_tickers():
    """
    This function will retry the bad tickers that were found in check_url_counter.
    It will remove them from the filing dates and URLs CSV, and then add them back to the existing CSV, using 
    add_to_existing function. This retry attempt is to see if the issue was temporary, and if the ticker can be added back successfully.
    If the ticker still has issues, it will be removed from the database and the filing dates and URLs CSV.
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


    add_to_existing([ticker for ticker, _ in bad_tickers])

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

def clean_ammendment_results():
    """
    There are ammendment documents (10-Q/A and 10-K/A) that are not needed for our analysis, as market reacts to original
    documents, not the ammendments.
    Remove these from the filing dates and URLs CSV.
    """

    df = pd.read_csv(config.FILING_DATES_AND_URLS_CSV)
    df = df[~df['Form Type'].str.contains("/A", na=False)]
    df.to_csv(config.FILING_DATES_AND_URLS_CSV, index=False)
    print("[INFO] Cleaned 10-Q/A and 10-K/A filings from the filing dates and URLs CSV.")



def main():
    """
    Main function to run the checks and clean up the filing dates and URLs CSV, and ensure the database is aligned with cleaned CSV
    """
    print("[INFO] Starting the filing dates and URLs CSV cleanup process...")
    # 1. Remove ammentment reports (10-Q/A and 10-K/A) from the filing dates and 
    # URLs CSV, as these are usless for our investigation
    clean_ammendment_results()
    # 2. Retry bad tickers, to see if scraping issues were temporary or not for malformed tickers
    # If they are still bad, remove them from the database and the filing dates and URLs CSV
    retry_bad_tickers()
    # retry_bad_tickers also runs check_db_alignment to ensure that the tickers in the database match those in the CSV



if __name__ == "__main__":
    main()

    

    

