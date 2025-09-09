# DataAssembly/AssetPrices Directory

This directory contains tools for building and analyzing asset pricing databases, with a focus on historical stock price data collection, storage, and visualization. The primary functionality centers around creating a comprehensive SQLite database of stock prices and providing analysis tools for financial research, particularly for Post-Earnings Announcement Drift (PEAD) studies.

## Directory Structure

```
DataAssembly/AssetPrices/
├── README.md                    # This file
└── build_pricing_db.py          # Main pricing database builder and analysis tools
```

## Core Functionality

### `build_pricing_db.py`

This comprehensive script provides multiple functionalities for asset pricing data management and analysis:

## Main Functions

### 1. `create_pricing_db()`
**Purpose**: Builds a comprehensive SQLite database of historical stock prices from Yahoo Finance.

**Key Features**:
- Downloads historical price data for all tickers in the metadata CSV
- Creates SQLite database with standardized schema
- Handles data validation (requires exactly 3269 rows per ticker)
- Includes SPY (S&P 500 ETF) as benchmark data
- Implements polite delays to avoid API throttling
- Skips tickers with insufficient or missing data
- Prevents duplicate data insertion

**Database Schema**:
```sql
CREATE TABLE daily_prices (
    ticker TEXT,
    date DATE,
    adjusted_close REAL,
    PRIMARY KEY (ticker, date)
);
```

**Configuration**:
- Uses `config.METADATA_CSV_FILEPATH` for ticker list
- Uses `config.DB_PATH` for database location
- Uses `config.START_DATE` and `config.END_DATE` for date range
- Uses `config.REQUIRED_ROW_COUNT` for data validation

### 2. `look_up_pricing(ticker, start_date, end_date, announce_date, short_window_end, medium_window_end)`
**Purpose**: Visualizes historical stock prices with optional earnings announcement analysis.

**Key Features**:
- Plots historical prices for specified ticker and date range
- Includes SPY benchmark data (rescaled for comparison)
- Optional earnings announcement date with vertical marker
- Short-term and medium-term trend analysis with fitted lines
- Publication-ready plots with proper formatting

**Parameters**:
- `ticker`: Stock symbol (e.g., 'AAPL', 'MSFT')
- `start_date`: Start date in YYYY-MM-DD format
- `end_date`: End date in YYYY-MM-DD format
- `announce_date`: Optional earnings announcement date
- `short_window_end`: Days for short-term analysis (default: 5)
- `medium_window_end`: Days for medium-term analysis (default: 60)

**Output**:
- Interactive plot showing price movements
- SPY benchmark comparison (rescaled)
- Trend lines for post-announcement periods
- Announcement date marker

### 3. `look_up_pricing_abnormal(ticker, start_date, end_date, announce_date, short_window_end, medium_window_end)`
**Purpose**: Analyzes Cumulative Abnormal Returns (CAR) for financial event studies.

**Key Features**:
- Calculates daily log returns for ticker and SPY
- Computes abnormal returns (ticker return - SPY return)
- Generates cumulative abnormal returns (CAR)
- Provides trend analysis with slope calculations
- Reports daily CAR slopes in basis points

**Methodology**:
- **Daily Returns**: `r = log(P_t / P_{t-1})`
- **Abnormal Returns**: `AR = r_ticker - r_SPY`
- **Cumulative Abnormal Returns**: `CAR = Σ(AR)`

**Output**:
- CAR plot with announcement date marker
- Short-term and medium-term trend lines
- Slope calculations in basis points per day
- Statistical analysis of post-event performance

### 4. `make_dates(date)`
**Purpose**: Utility function for creating date ranges around events.

**Functionality**:
- Takes a single date string (YYYY-MM-DD format)
- Returns list with start date (15 days before) and end date (90 days after)
- Useful for creating analysis windows around earnings announcements

### 5. `plot_average_pead_from_csv(surprises_csv, short_window_end, medium_window_end, pre_event_days, save_path)`
**Purpose**: Creates comprehensive PEAD (Post-Earnings Announcement Drift) analysis plots.

**Key Features**:
- Loads earnings surprise data from CSV
- Separates positive and negative surprise cohorts
- Calculates average CAR for each cohort
- Provides statistical analysis with slope calculations
- Creates publication-ready plots with LaTeX formatting
- Saves results to specified path

**CSV Requirements**:
- `symbol`: Stock ticker symbol
- `surprise` or `surprisePercent`: Earnings surprise value
- Date column: `announce_date`, `date`, or `period`

**Analysis Windows**:
- **Pre-event**: 20 days before announcement (default)
- **Short-term**: 0 to 15 days after announcement (default)
- **Medium-term**: 15 to 60 days after announcement (default)

**Output**:
- Average CAR plots for positive and negative surprise cohorts
- Slope annotations in basis points per day
- Shaded analysis windows
- Statistical significance indicators

### 6. `plot_pricing_over_period(ticker, save_dir)`
**Purpose**: Creates long-term historical price charts for individual stocks.

**Key Features**:
- Plots complete price history for specified ticker
- Uses LaTeX formatting for publication quality
- Saves plots to specified directory
- High-resolution output (300 DPI)
- Professional styling with grid and proper labels

## Key Concepts

### 1. Data Collection
- **Source**: Yahoo Finance via yfinance library
- **Frequency**: Daily closing prices
- **Adjustment**: Automatically adjusted for splits and dividends
- **Validation**: Requires exact row count for data quality
- **Benchmark**: Includes SPY for market comparison

### 2. Financial Analysis
- **Returns Calculation**: Log returns for statistical properties
- **Abnormal Returns**: Stock return minus market return
- **Cumulative Abnormal Returns**: Sum of abnormal returns over time
- **Event Studies**: Analysis around specific dates (earnings announcements)

### 3. PEAD Research
- **Post-Earnings Announcement Drift**: Stock price continuation after earnings
- **Surprise Classification**: Positive vs. negative earnings surprises
- **Time Windows**: Short-term (0-15 days) vs. medium-term (15-60 days)
- **Statistical Analysis**: Slope calculations and significance testing

### 4. Data Quality
- **Completeness**: Requires full time series for each ticker
- **Consistency**: Standardized date ranges across all stocks
- **Validation**: Automatic filtering of incomplete datasets
- **Deduplication**: Prevents duplicate data insertion

## Usage Examples

### Building the Database
```python
from build_pricing_db import create_pricing_db

# Build complete pricing database
create_pricing_db()
```

### Individual Stock Analysis
```python
from build_pricing_db import look_up_pricing, look_up_pricing_abnormal

# Price analysis with earnings announcement
look_up_pricing(
    ticker='AAPL',
    start_date='2023-01-01',
    end_date='2023-12-31',
    announce_date='2023-03-31'
)

# Abnormal returns analysis
look_up_pricing_abnormal(
    ticker='AAPL',
    start_date='2023-01-01',
    end_date='2023-12-31',
    announce_date='2023-03-31',
    short_window_end=5,
    medium_window_end=60
)
```

### PEAD Analysis
```python
from build_pricing_db import plot_average_pead_from_csv

# Comprehensive PEAD analysis
plot_average_pead_from_csv(
    surprises_csv="Data/eps_surprises_quarterly_2012_2024.csv",
    short_window_end=15,
    medium_window_end=60,
    pre_event_days=20,
    save_path="Plots/PEAD_demo/average_pead_plot.png"
)
```

### Long-term Price Charts
```python
from build_pricing_db import plot_pricing_over_period

# Create publication-ready price charts
plot_pricing_over_period(
    ticker="AAPL",
    save_dir="Plots/asset_price_graphs"
)
```

## Configuration

The script relies on configuration parameters from `config.py`:

```python
# Database configuration
DB_PATH = "Data/momentum_data.db"
PRICING_TABLE_NAME = "daily_prices"

# Data source configuration
METADATA_CSV_FILEPATH = "Data/filtered_sp500_metadata.csv"
START_DATE = "2012-01-01"
END_DATE = "2024-12-31"
REQUIRED_ROW_COUNT = 3269  # Expected number of trading days
```

## Dependencies

- **pandas**: Data manipulation and analysis
- **sqlite3**: Database operations
- **yfinance**: Yahoo Finance data download
- **matplotlib**: Plotting and visualization
- **numpy**: Numerical computations
- **config**: Project configuration

## Data Requirements

### Input Data
- **Metadata CSV**: Contains list of stock tickers to process
- **Configuration**: Database paths and date ranges
- **EPS Surprises CSV**: For PEAD analysis (optional)

### Output Data
- **SQLite Database**: Structured price data storage
- **Plots**: Publication-ready visualizations
- **Analysis Results**: Statistical summaries and metrics

## Performance Considerations

### Data Collection
- **Rate Limiting**: 1-second delay between API calls
- **Error Handling**: Graceful handling of missing data
- **Validation**: Automatic filtering of incomplete datasets
- **Caching**: Avoids re-downloading existing data

### Analysis
- **Efficient Queries**: Optimized SQL queries for large datasets
- **Memory Management**: Streaming data processing for large files
- **Visualization**: High-quality plots with proper formatting

## Research Applications

### 1. Event Studies
- Earnings announcement analysis
- Merger and acquisition studies
- Regulatory change impacts
- Market microstructure research

### 2. PEAD Research
- Post-earnings announcement drift
- Information processing efficiency
- Market anomaly studies
- Behavioral finance research

### 3. Risk Management
- Historical volatility analysis
- Correlation studies
- Portfolio optimization
- Stress testing

### 4. Academic Research
- Publication-ready visualizations
- Reproducible analysis pipelines
- Standardized methodologies
- Statistical significance testing

## Notes

- All functions use LaTeX formatting for publication-quality output
- Database operations are atomic and handle concurrent access
- Error handling ensures robust operation with large datasets
- Plots are automatically saved with high resolution
- Statistical calculations follow standard financial research practices
- The system is designed for reproducibility and academic use
