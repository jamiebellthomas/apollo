# DataAssembly/MetaData Directory

This directory contains tools for collecting, filtering, and managing company metadata, specifically focused on S&P 500 companies within target sectors relevant to financial analysis and earnings prediction research. The metadata management system ensures data quality and consistency across the entire research pipeline.

## Directory Structure

```
DataAssembly/MetaData/
├── README.md                    # This file
└── company_names.py             # Company metadata extraction and management
```

## Core Functionality

### `company_names.py`

This comprehensive script provides metadata management for company selection and validation within the research pipeline.

## Main Functions

### 1. `extract()`
**Purpose**: Extracts and filters S&P 500 company data from Wikipedia, focusing on specific sectors relevant to financial analysis.

**Key Features**:
- Downloads current S&P 500 company list from Wikipedia
- Filters companies by target sectors (Information Technology, Consumer Discretionary, Financials)
- Validates ticker symbols for alphanumeric format
- Saves filtered results to standardized CSV format
- Provides comprehensive logging and progress tracking

**Target Sectors**:
- **Information Technology**: Tech companies and software firms
- **Consumer Discretionary**: Retail, automotive, and consumer services
- **Financials**: Banks, insurance, and financial services

**Data Validation**:
- **Ticker Format**: Ensures ticker symbols are purely alphanumeric
- **Sector Filtering**: Focuses on sectors with high earnings volatility
- **Data Quality**: Validates company information completeness

**Output**:
- Saves filtered company list to `config.METADATA_CSV_FILEPATH`
- Includes company symbols, names, and sector information
- Provides count of selected companies

**Usage**:
```python
from company_names import extract
extract()  # Run extraction and filtering
```

### 2. `see_which_have_existed_for_long_enough()`
**Purpose**: Validates company data availability by checking which companies have sufficient historical trading data in the database.

**Key Features**:
- Cross-references metadata CSV with pricing database
- Identifies companies with insufficient historical data
- Removes companies that don't meet data requirements
- Updates metadata CSV to reflect available data
- Ensures data consistency across the research pipeline

**Data Requirements**:
- **Minimum Trading Days**: Requires 3269 trading days (approximately 13 years)
- **Data Completeness**: Ensures full historical price coverage
- **Database Alignment**: Maintains consistency between metadata and pricing data

**Validation Process**:
1. Reads company list from metadata CSV
2. Queries pricing database for available tickers
3. Identifies missing or incomplete data
4. Removes companies with insufficient data
5. Updates metadata CSV with validated companies

**Quality Assurance**:
- **Data Integrity**: Ensures all companies have complete price histories
- **Consistency Check**: Validates metadata against actual data availability
- **Automatic Cleanup**: Removes invalid entries automatically

**Usage**:
```python
from company_names import see_which_have_existed_for_long_enough
missing_tickers = see_which_have_existed_for_long_enough()
print(f"Removed {len(missing_tickers)} companies with insufficient data")
```

### 3. `main()`
**Purpose**: Orchestrates the complete metadata extraction and validation process.

**Workflow**:
1. **Extraction**: Downloads and filters S&P 500 companies
2. **Validation**: Checks data availability and completeness
3. **Cleanup**: Removes companies with insufficient data
4. **Reporting**: Provides comprehensive status updates

**Process Flow**:
```python
def main():
    print("[INFO] Starting extraction of S&P 500 companies...")
    extract()  # Extract and filter companies
    # Additional validation steps can be added here
```

## Key Concepts

### 1. Sector Selection Strategy
The target sectors are chosen based on research relevance:

**Information Technology**:
- High earnings volatility and growth potential
- Strong correlation with market movements
- Rich earnings announcement data

**Consumer Discretionary**:
- Economic sensitivity and cyclical patterns
- Diverse business models and earnings patterns
- Strong seasonal and quarterly variations

**Financials**:
- Regulatory reporting requirements
- Interest rate sensitivity
- Complex earnings structures

### 2. Data Quality Standards
**Ticker Validation**:
- Purely alphanumeric characters
- No special characters or symbols
- Consistent formatting across sources

**Historical Data Requirements**:
- Minimum 3269 trading days (13+ years)
- Complete price history without gaps
- Consistent data quality and availability

**Sector Consistency**:
- Aligned with GICS (Global Industry Classification Standard)
- Consistent with financial research standards
- Focused on high-volatility sectors

### 3. Database Integration
**Pricing Database Alignment**:
- Cross-references with `config.PRICING_TABLE_NAME`
- Ensures metadata matches available data
- Maintains referential integrity

**Configuration Management**:
- Uses `config.METADATA_CSV_FILEPATH` for output
- Integrates with `config.DB_PATH` for validation
- Follows project-wide configuration standards

## Configuration

### File Paths
```python
# Metadata configuration
METADATA_CSV_FILEPATH = "Data/filtered_sp500_metadata.csv"
DB_PATH = "Data/momentum_data.db"
PRICING_TABLE_NAME = "daily_prices"
```

### Sector Configuration
```python
# Target sectors for analysis
target_sectors = {
    "Information Technology",
    "Consumer Discretionary", 
    "Financials"
}
```

### Data Requirements
```python
# Minimum data requirements
MIN_TRADING_DAYS = 3269  # Approximately 13 years
```

## Dependencies

### Core Libraries
- **pandas**: Data manipulation and analysis
- **sqlite3**: Database operations and queries
- **requests**: HTTP requests for web scraping (via pandas)

### Data Sources
- **Wikipedia**: S&P 500 company list and sector information
- **SQLite Database**: Historical pricing data validation
- **Configuration**: Project-wide settings and paths

## Usage Examples

### Complete Metadata Pipeline
```python
from company_names import main

# Run complete extraction and validation
main()
```

### Individual Functions
```python
from company_names import extract, see_which_have_existed_for_long_enough

# Extract companies from Wikipedia
extract()

# Validate data availability
missing = see_which_have_existed_for_long_enough()
print(f"Companies removed: {missing}")
```

### Custom Sector Filtering
```python
# Modify target sectors in the script
target_sectors = {
    "Information Technology",
    "Health Care",  # Add additional sectors
    "Consumer Discretionary"
}
```

## Data Output

### Metadata CSV Structure
The output CSV contains the following columns:

```csv
Symbol,GICS Sector,Security,GICS Sub Industry,Headquarters Location,Date first added,CIK,Founded
AAPL,Information Technology,Apple Inc.,Technology Hardware Storage & Peripherals,Cupertino California,1976-12-31,320193,1976
MSFT,Information Technology,Microsoft Corporation,Systems Software,Redmond Washington,1994-06-01,789019,1975
```

**Column Descriptions**:
- **Symbol**: Stock ticker symbol
- **GICS Sector**: Global Industry Classification Standard sector
- **Security**: Company name
- **GICS Sub Industry**: Detailed industry classification
- **Headquarters Location**: Company headquarters
- **Date first added**: When added to S&P 500
- **CIK**: Central Index Key (SEC identifier)
- **Founded**: Company founding year

## Quality Assurance

### Validation Checks
- **Ticker Format**: Ensures alphanumeric ticker symbols
- **Sector Alignment**: Validates against GICS standards
- **Data Completeness**: Checks for missing company information
- **Database Consistency**: Verifies data availability

### Error Handling
- **Network Issues**: Graceful handling of Wikipedia access problems
- **Data Parsing**: Robust parsing of HTML tables
- **Database Errors**: Proper error handling for database operations
- **File Operations**: Safe CSV reading and writing

## Performance Considerations

### Optimization Strategies
- **Efficient Queries**: Optimized database queries for large datasets
- **Memory Management**: Efficient pandas operations
- **Batch Processing**: Handles large company lists efficiently
- **Caching**: Avoids redundant data downloads

### Scalability
- **Large Datasets**: Handles full S&P 500 company list
- **Database Integration**: Efficient cross-referencing with pricing data
- **Memory Usage**: Optimized for large metadata operations

## Research Applications

### 1. Earnings Prediction
- **Sector Focus**: Targets high-volatility sectors for earnings analysis
- **Data Quality**: Ensures sufficient historical data for modeling
- **Consistency**: Maintains standardized company universe

### 2. Financial Analysis
- **Market Research**: Provides curated company universe
- **Sector Analysis**: Enables sector-specific research
- **Historical Studies**: Supports long-term trend analysis

### 3. Academic Research
- **Reproducibility**: Standardized company selection process
- **Data Quality**: High-quality, validated datasets
- **Documentation**: Clear methodology and selection criteria

## Integration with Research Pipeline

### Data Flow
1. **Extraction**: Wikipedia → Filtered CSV
2. **Validation**: CSV → Database cross-reference
3. **Cleanup**: Remove insufficient data
4. **Integration**: Feed into pricing and earnings pipelines

### Downstream Usage
- **Pricing Data**: Used by `DataAssembly/AssetPrices/`
- **Earnings Data**: Used by `DataAssembly/EPS/`
- **Analysis**: Used by `Analysis/` and `Baselines/`

## Notes

- All functions include comprehensive error handling and logging
- Data validation ensures high-quality, reliable company universe
- Sector selection is optimized for earnings prediction research
- The system maintains consistency across all research components
- Regular updates ensure current and accurate company information
- Integration with configuration system enables flexible deployment
