"""
Configuration file for the Thesis
This will store all environment variables and constants used throughout the project.
"""
# --- Credentials ---
EMAIL = "jamiebt@live.co.uk"
COMPANY_NAME = "ApolloResearch"
OPENAI_API_KEY = "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"
SEC_API_KEY = "ec5c67df2f4fc7b712e3854cece2de54d1c1cc2523ee690ded5b02de2f92461f"
GROQ_API_KEY = "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"
GIT_TOKEN = "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"
HUGGING_FACE = "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"
NASDAQ_API_KEY = "Vpuv3bsLL6YRtN14TueF"
FINNHUB_API_KEY = "d2cals1r01qihtcqtd00d2cals1r01qihtcqtd0g"
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