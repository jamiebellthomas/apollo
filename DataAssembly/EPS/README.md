# DataAssembly/EPS Directory

This directory contains comprehensive tools for collecting, processing, and analyzing earnings-related data from multiple sources. The EPS (Earnings Per Share) data assembly pipeline handles SEC filings, earnings transcripts, and financial data extraction using both automated parsing and AI-powered analysis.

## Directory Structure

```
DataAssembly/EPS/
├── README.md                                    # This file
├── finnhub_extractor.py                         # Finnhub API data extraction
├── parse_10k_filings.py                         # 10-K annual report parsing
├── parse_10q_filings.py                         # 10-Q quarterly report parsing
├── collect_earnings_transcript.py               # Earnings transcript collection
├── collect_earnings_transcript_analysis.py      # Transcript analysis and validation
├── split_10q_and_10k.py                         # Data splitting utilities
├── profile_parse_sec_filings.py                 # Performance profiling tools
└── analyse_10q_results.ipynb                    # Jupyter notebook for analysis
```

## Core Data Collection Scripts

### 1. `finnhub_extractor.py`
**Purpose**: Extracts earnings estimates and surprises from Finnhub API for comprehensive financial data collection.

**Key Features**:
- Downloads quarterly earnings estimates and actual results
- Handles API rate limiting with configurable delays
- Supports batch processing with resume capability
- Validates data completeness and quality
- Exports to standardized CSV format

**Configuration**:
```python
START_YEAR = 2012
END_YEAR = 2024
SLEEP_SECONDS = 60.00  # API rate limiting
MAX_RETRIES = 6
EST_OUT = "Data/eps_estimates_quarterly_2012_2024.csv"
SURP_OUT = "Data/eps_surprises_quarterly_2012_2024.csv"
```

**Output Files**:
- `eps_estimates_quarterly_2012_2024.csv`: Analyst earnings estimates
- `eps_surprises_quarterly_2012_2024.csv`: Actual vs. estimated earnings

**Usage**:
```python
from finnhub_extractor import main
main()  # Run full extraction process
```

### 2. `parse_10k_filings.py`
**Purpose**: Extracts EPS data from SEC 10-K annual reports using AI-powered parsing.

**Key Features**:
- Downloads and parses SEC 10-K filings
- Uses OpenAI GPT models for intelligent data extraction
- Handles multiple model fallbacks for reliability
- Extracts both basic and diluted EPS values
- Validates extracted data format and quality
- Supports batch processing with progress tracking

**AI Integration**:
- Uses OpenAI API for intelligent document parsing
- Implements model rotation for API limits
- Validates extraction results for accuracy
- Handles various document formats and structures

**Data Extraction**:
- **Basic EPS**: Primary earnings per share calculation
- **Diluted EPS**: Includes potential share dilution
- **Validation**: Format checking and data quality assurance

**Usage**:
```python
from parse_10k_filings import main
main(start_index=0)  # Start from specific index
```

### 3. `parse_10q_filings.py`
**Purpose**: Extracts EPS data from SEC 10-Q quarterly reports with advanced parsing capabilities.

**Key Features**:
- Comprehensive 10-Q quarterly report processing
- AI-powered EPS extraction using OpenAI models
- Robust error handling and retry mechanisms
- Data validation and quality checks
- Progress tracking and resume capability
- Support for large-scale batch processing

**Advanced Features**:
- **Smart Parsing**: AI identifies EPS data in complex documents
- **Format Validation**: Ensures extracted data meets requirements
- **Error Recovery**: Handles parsing failures gracefully
- **Batch Processing**: Efficient processing of large datasets

**Data Quality**:
- Validates EPS format: `basic_eps: value, diluted_eps: value`
- Handles negative values and missing data
- Ensures numerical accuracy and consistency

**Usage**:
```python
from parse_10q_filings import extract_eps, main
# Extract single filing
result = extract_eps("AAPL/0000320193-24-000081")
# Run full batch processing
main(start_index=0)
```

### 4. `collect_earnings_transcript.py`
**Purpose**: Automated collection of earnings call transcripts using web scraping.

**Key Features**:
- Selenium-based web scraping of SEC EDGAR database
- Automated search and download of earnings transcripts
- Handles dynamic web content and JavaScript
- Batch processing with progress tracking
- Error handling and retry mechanisms
- Database integration for storage

**Web Scraping Capabilities**:
- **SEC EDGAR Search**: Automated search interface interaction
- **Dynamic Content**: Handles JavaScript-rendered content
- **Date Range Filtering**: Configurable time periods
- **Form Type Selection**: Targets specific document types

**Data Management**:
- Stores transcripts in SQLite database
- Tracks download progress and status
- Handles duplicate detection
- Manages file organization and storage

**Usage**:
```python
from collect_earnings_transcript import sec_search
ticker_list = ["AAPL", "MSFT", "GOOGL"]
sec_search(ticker_list)
```

### 5. `collect_earnings_transcript_analysis.py`
**Purpose**: Analysis and validation of collected earnings transcripts.

**Key Features**:
- Validates transcript data completeness
- Analyzes filing patterns and counts
- Ensures database alignment and consistency
- Provides data quality metrics
- Handles missing data identification

**Analysis Functions**:
- **Row Count Analysis**: Counts 10-K and 10-Q filings per ticker
- **Missing Ticker Detection**: Identifies incomplete data
- **Database Alignment**: Ensures CSV and database consistency
- **Data Validation**: Checks data integrity and completeness

**Quality Assurance**:
- Validates ticker existence in database
- Checks filing count consistency
- Identifies data gaps and missing information
- Provides comprehensive data quality reports

**Usage**:
```python
from collect_earnings_transcript_analysis import check_db_alignment
check_db_alignment()  # Validate data consistency
```

## Data Processing Utilities

### 6. `split_10q_and_10k.py`
**Purpose**: Separates and cleans 10-K and 10-Q data for specialized processing.

**Key Features**:
- Splits combined filing data into separate datasets
- Removes irrelevant columns for each filing type
- Handles data cleaning and standardization
- Creates specialized datasets for annual vs. quarterly analysis

**Data Separation**:
- **10-K Data**: Annual reports with annual EPS data
- **10-Q Data**: Quarterly reports with quarterly EPS data
- **Column Management**: Removes irrelevant columns per filing type
- **Data Integrity**: Maintains data consistency during splitting

**Usage**:
```python
from split_10q_and_10k import split_10k_data_and_clean
split_10k_data_and_clean(
    original_csv="combined_data.csv",
    annual_csv="annual_data.csv", 
    quarterly_csv="quarterly_data.csv"
)
```

### 7. `profile_parse_sec_filings.py`
**Purpose**: Performance profiling and optimization tools for SEC filing processing.

**Key Features**:
- Memory usage monitoring and analysis
- CPU time and wall clock time measurement
- Batch processing performance evaluation
- Resource utilization optimization
- Performance bottleneck identification

**Profiling Metrics**:
- **Memory Usage**: RSS memory and peak memory tracking
- **CPU Time**: Process CPU time vs. wall clock time
- **Batch Performance**: Large-scale processing efficiency
- **Resource Optimization**: Identifies performance improvements

**Usage**:
```python
from profile_parse_sec_filings import profile_function_batch
results = profile_function_batch(extract_eps, query_list)
print(f"CPU Time: {results['cpu_time']:.2f}s")
print(f"Memory Used: {results['rss_memory_used']:.2f} MB")
```

## Data Analysis

### 8. `analyse_10q_results.ipynb`
**Purpose**: Jupyter notebook for comprehensive analysis of quarterly EPS data.

**Analysis Capabilities**:
- Data quality assessment and validation
- Statistical analysis of EPS trends
- Visualization of earnings patterns
- Comparative analysis across companies
- Time series analysis and forecasting

## Key Data Concepts

### 1. SEC Filing Types
- **10-K**: Annual reports containing comprehensive financial information
- **10-Q**: Quarterly reports with interim financial results
- **8-K**: Current reports for significant events
- **Transcripts**: Earnings call transcripts and presentations

### 2. EPS Metrics
- **Basic EPS**: Net income divided by weighted average shares outstanding
- **Diluted EPS**: Includes potential share dilution from options, warrants, etc.
- **Quarterly EPS**: Three-month period earnings
- **Annual EPS**: Full-year earnings summary

### 3. Data Sources
- **SEC EDGAR**: Official SEC filing database
- **Finnhub API**: Financial data and estimates
- **Yahoo Finance**: Market data and historical prices
- **Company Websites**: Direct earnings announcements

### 4. Data Quality Standards
- **Completeness**: All required fields populated
- **Accuracy**: Validated against source documents
- **Consistency**: Standardized formats and units
- **Timeliness**: Current and up-to-date information

## Configuration

### API Configuration
```python
# Finnhub API
FINNHUB_API_KEY = "your_api_key_here"

# OpenAI API
OPENAI_API_KEY = "your_openai_key_here"
```

### File Paths
```python
# Data files
EPS_DATA_CSV = "Data/eps_data.csv"
ANNUAL_EPS_DATA_CSV = "Data/annual_eps_data.csv"
QUARTERLY_EPS_DATA_CSV = "Data/quarterly_eps_data.csv"
FILING_DATES_AND_URLS_CSV = "Data/filing_dates_and_urls.csv"
```

### Processing Parameters
```python
# Time ranges
START_YEAR = 2012
END_YEAR = 2024

# API limits
SLEEP_SECONDS = 60.00
MAX_RETRIES = 6
```

## Dependencies

### Core Libraries
- **pandas**: Data manipulation and analysis
- **requests**: HTTP API interactions
- **sqlite3**: Database operations
- **selenium**: Web scraping automation

### AI and Parsing
- **openai**: GPT model integration
- **sec_parser**: SEC document parsing
- **sec_downloader**: SEC filing downloads

### Analysis and Visualization
- **matplotlib**: Plotting and visualization
- **numpy**: Numerical computations
- **jupyter**: Interactive analysis notebooks

## Usage Workflow

### 1. Data Collection
```bash
# Extract earnings estimates and surprises
python finnhub_extractor.py

# Collect SEC filing data
python collect_earnings_transcript.py

# Parse 10-K annual reports
python parse_10k_filings.py

# Parse 10-Q quarterly reports  
python parse_10q_filings.py
```

### 2. Data Processing
```bash
# Split and clean data
python split_10q_and_10k.py

# Validate data quality
python collect_earnings_transcript_analysis.py
```

### 3. Performance Optimization
```bash
# Profile processing performance
python profile_parse_sec_filings.py
```

### 4. Analysis
```bash
# Run Jupyter notebook analysis
jupyter notebook analyse_10q_results.ipynb
```

## Data Quality Assurance

### Validation Checks
- **Format Validation**: Ensures EPS data follows expected format
- **Range Validation**: Checks for reasonable EPS values
- **Completeness Check**: Verifies all required fields are populated
- **Consistency Check**: Validates data across different sources

### Error Handling
- **API Failures**: Retry mechanisms and fallback strategies
- **Parsing Errors**: Graceful handling of document parsing failures
- **Data Corruption**: Detection and recovery from data issues
- **Network Issues**: Robust handling of connectivity problems

## Performance Considerations

### Optimization Strategies
- **Batch Processing**: Efficient handling of large datasets
- **Rate Limiting**: Respectful API usage with delays
- **Memory Management**: Efficient memory usage for large files
- **Parallel Processing**: Concurrent operations where possible

### Monitoring
- **Progress Tracking**: Real-time processing status
- **Performance Metrics**: CPU, memory, and time monitoring
- **Error Logging**: Comprehensive error tracking and reporting
- **Resource Usage**: Disk space and network usage monitoring

## Notes

- All scripts include comprehensive error handling and logging
- Data validation ensures high-quality, reliable datasets
- AI-powered parsing provides intelligent document analysis
- Performance profiling enables optimization of large-scale processing
- The system is designed for reproducibility and academic research
- All data sources are properly attributed and documented
