import pandas as pd
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

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
