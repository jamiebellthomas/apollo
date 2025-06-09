"""
Configuration file for the Thesis
This will store all environment variables and constants used throughout the project.
"""

# --- Filepaths ---
METADATA_CSV_FILEPATH = "Data/filtered_sp500_metadata.csv"
NEWS_CSV_PATH = "Data/nasdaq_external_data.csv"
DB_PATH = "Data/momentum_data.db"

# --- Database Table Names ---
NEWS_TABLE_NAME = "news_articles"
PRICING_TABLE_NAME = "daily_prices"

# --- Date Range ---
START_DATE = "2012-01-01"
END_DATE = "2024-12-31"
REQUIRED_ROW_COUNT = 3269