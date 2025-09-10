"""
Configuration file for the Thesis
This will store all environment variables and constants used throughout the project.
"""
# all api keys have been regenerated and redacted for security purposes
# --- Credentials ---
EMAIL = "REDACTED"
COMPANY_NAME = "REDACTED"
OPENAI_API_KEY = "REDACTED"
SEC_API_KEY = "REDACTED"
GROQ_API_KEY = "REDACTED"
GIT_TOKEN = "REDACTED"
HUGGING_FACE = "REDACTED"
NASDAQ_API_KEY = "REDACTED"
FINNHUB_API_KEY = "REDACTED"
# --- Filepaths ---
METADATA_CSV_FILEPATH = "Data/filtered_sp500_metadata.csv"
ORIGINAL_PATH = "nasdaq_exteral_data.csv"
NEWS_CSV_PATH_ORIGIN = "Data/nasdaq_exteral_data.csv"
NEWS_CSV_PATH_FORMATTED_ROWS = "Data/news_formatted_rows.csv"
NEWS_CSV_PATH_CLEAN = "Data/news_formatted_rows_clean.csv"
NEWS_CSV_PATH_ASSOCIATED_TICKERS = "Data/news_associated_tickers.csv"
DB_PATH = "Data/momentum_data.db"
SEC_DOWNLOADS = "SEC_Downloads"
DATA_FOLDER = "Data"
FILING_DATES_AND_URLS_CSV = "Data/filing_dates_and_urls.csv"
EPS_DATA_CSV = "Data/eps_data.csv"
QUARTERLY_EPS_DATA_CSV = "Data/eps_data_10q.csv"
ANNUAL_EPS_DATA_CSV = "Data/eps_data_10k.csv"
NEWS_FACTS = "Data/facts_output.jsonl"
SUBGRAPHS_JSONL = "Data/subgraphs.jsonl"
CLUSTER_CENTROIDS = "Data/cluster_centroids.jsonl"
EPS_SURPRISES = "Data/eps_surprises_quarterly_2012_2024.csv"


# --- Database Table Names ---
NEWS_TABLE_NAME = "news_articles"
PRICING_TABLE_NAME = "daily_prices"

# --- Date Range DO NOT CHANGE ---
START_DATE = "2012-01-01"
END_DATE = "2024-12-31"
REQUIRED_ROW_COUNT = 3269

# --- KG Parameters ---
WINDOW_DAYS   = 90 # 1 quarter roughly
SENTIMENT_MIN = 0.10 