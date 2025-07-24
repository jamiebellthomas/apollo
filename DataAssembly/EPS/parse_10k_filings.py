import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import config
from parse_10q_filings import extract_eps_openai, extract_eps_from_openai_result, check_eps_formatting, extract_eps, convert_sec_url_to_accession

from sec_downloader import Downloader
import sec_parser as sp
import pandas as pd
import re
import warnings
import time
from openai import OpenAI


warnings.filterwarnings("ignore", category=UserWarning)
    
def main(start:int):
    """
    Main function to run the EPS extraction process.
    """

    # 2. Create new DataFrame with only the necessary columns (Ticker, Form Type, URL) and save it to config.EPS_DATA_CSV if it doesn't exist
    if not os.path.exists(config.ANNUAL_EPS_DATA_CSV):
        original_df = pd.read_csv(config.FILING_DATES_AND_URLS_CSV)
        print("[INFO] EPS data CSV does not exist. Creating a new one.")
        df = original_df[['Ticker', 'Form Type', 'URL']].copy()
        df = df[df['Form Type'] == '10-K (Annual report)'].copy()
        df.to_csv(config.ANNUAL_EPS_DATA_CSV, index=False)
    else:
        df = pd.read_csv(config.ANNUAL_EPS_DATA_CSV)
    

    # 3. If there is no 'Accession Number' column, create it
    # This will be taken from the URLs which are in the format:
    # https://www.sec.gov/Archives/edgar/data/320193/000032019324000081/aapl-20240629.htm
    # And convert it to the format: 
    # 0000320193-24-000081

    df['Accession Number'] = df['URL'].apply(convert_sec_url_to_accession)
    # Now for each row we will build the query in the format 'Ticker/AccessionNumber'
    df['Query'] = df.apply(lambda row: f"{row['Ticker']}/{row['Accession Number']}", axis=1)
    # Now for each row, apply the extract_eps function to the 'Query' column, if the 'Form Type' column contains '10-Q', save it to the
    # quarterly_raw_eps, and quarterly_diluted_eps column, if it contains '10-K', save it to the annual_raw_eps, and annual_diluted_eps column
    # If there are no columns called quarterly_raw_eps, quarterly_diluted_eps, annual_raw_eps, and annual_diluted_eps, create them
    if 'annual_raw_eps' not in df.columns:
        df['annual_raw_eps'] = None
    if 'annual_diluted_eps' not in df.columns:
        df['annual_diluted_eps'] = None

    
    # Now we will iterate over each row and apply the extract_eps function to the 'Query' column
    # We will also keep track of the rows that failed to process, so we can retry them later
    print("[INFO] Starting EPS extraction process...")
    # Initialize a list to keep track of bad row indices
    total_rows = len(df)
    bad_row_indices = []
    for index, row in df.iterrows():

        # If the row already has any eps values, skip it
        if (
        (pd.notnull(row['annual_raw_eps']) and pd.notnull(row['annual_diluted_eps']))
    ):
            continue

        if index < start:
            continue

        print(f"[INFO] Processing row {index + 1}/{total_rows}: {row['Query']}")

        

        if index % 5 == 0:  # Print every 100 rows
            percent = (index + 1) / total_rows * 100
            print(f"[INFO] Processing row {index + 1}/{total_rows} ({percent:.2f}%), Saving progress...")
            df.to_csv(config.ANNUAL_EPS_DATA_CSV, index=False)  # Save progress

        
        try:
            query = row['Query']
            basic_eps, diluted_eps = extract_eps(query,quarterly_report=False)
            df.at[index, 'annual_raw_eps'] = basic_eps
            df.at[index, 'annual_diluted_eps'] = diluted_eps
        except Exception as e:
            print(f"[ERROR] Failed to process row {index + 1}: {e}")
            bad_row_indices.append((index,query))
            df.at[index, 'annual_raw_eps'] = None
            df.at[index, 'annual_diluted_eps'] = None
            continue
        print("---------------------")

    print(f"[INFO] Finished processing {total_rows} rows.")
    # Save the updated DataFrame to the CSV
    df.to_csv(config.ANNUAL_EPS_DATA_CSV, index=False)
    if len(bad_row_indices) > 0:
        print(f"[WARNING] Some rows failed to process: {bad_row_indices}")
    else:
        print("[INFO] All rows processed successfully.")

if __name__ == "__main__":
    main(start=200)
    # data = (extract_relevant_eps_data_html("https://www.sec.gov/Archives/edgar/data/820313/000155837024013696/aph-20240930x10q.htm"))
    # for i in data:
    #     print("-------------------")
    #     print(i)
    # print(extract_eps("https://www.sec.gov/Archives/edgar/data/820313/000155837024013696/aph-20240930x10q.htm"))





    


    

    


