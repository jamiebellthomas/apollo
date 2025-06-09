import pandas as pd

# STEP 1: Get S&P 500 tickers with sector info from Wikipedia
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
sp500_df = pd.read_html(url, header=0)[0]

# STEP 2: Filter by sectors FinBERT-style
target_sectors = {"Information Technology", "Consumer Discretionary", "Financials"}
filtered_df = sp500_df[sp500_df["GICS Sector"].isin(target_sectors)]

# Optional: Save to CSV
filtered_df.to_csv("Data/filtered_sp500_finbert_style.csv", index=False)

print(f"Selected {len(filtered_df)} companies:")
print(filtered_df[["Symbol", "Security", "GICS Sector"]].head())