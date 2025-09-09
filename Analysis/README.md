# Analysis Directory

This directory contains comprehensive analysis scripts and results for evaluating the performance of heterogeneous graph neural network (HeteroGNN) models on earnings prediction tasks. The analysis focuses on both classification performance metrics and financial impact assessment through Cumulative Abnormal Returns (CAR) analysis.

## Directory Structure

```
Analysis/
├── README.md                           # This file
├── aggregate_consensus.py              # Core consensus aggregation analysis
├── analyse_many_models.py              # Multi-model CAR comparison analysis
├── analyse_mean_car.py                 # Mean CAR analysis across models
├── analyse_one_model.py                # Single model CAR analysis
├── analyze_aggregated_coverage.py      # Coverage analysis for aggregated results
├── plot_medium_aggregate.py            # Medium-term aggregate plotting
├── plot_single_model_car.py            # Single model CAR plotting
```

## Core Analysis Scripts

### 1. `aggregate_consensus.py`
**Purpose**: Core consensus aggregation analysis that combines predictions from multiple models using majority voting.

**Key Features**:
- Aggregates predictions from all models within each architecture (heterognn, heterognn2, etc.)
- Uses majority voting to create consensus predictions
- Calculates comprehensive performance metrics (accuracy, precision, recall, F1-score)
- Generates CAR (Cumulative Abnormal Returns) analysis for predicted positive events
- Creates comparison plots against positive EPS surprises baseline
- Supports both regular and medium-term analysis periods


### 2. `analyse_many_models.py`
**Purpose**: Multi-model CAR analysis comparing average CARs across different model architectures.

**Key Features**:
- Analyzes multiple model results directories simultaneously
- Compares average CARs against positive EPS surprises baseline
- Supports both manual directory specification and automatic discovery
- Generates comprehensive comparison plots with statistical analysis
- Configurable analysis periods (pre-event, early post-event, late post-event)

**Configuration**:
- Modify `MANUAL_RESULTS_DIRECTORIES` or use automatic discovery
- Adjust `DAYS_BEFORE`, `DAYS_AFTER`, and `MID_POINT_START` parameters
- Set benchmark ticker (default: SPY)

### 3. `analyse_mean_car.py`
**Purpose**: Mean CAR analysis with standard deviation (shred) across all models.

**Key Features**:
- Calculates mean CAR and standard deviation across multiple models
- Provides statistical analysis of CAR distributions
- Generates plots showing mean CAR with confidence intervals
- Supports both individual model analysis and ensemble analysis

### 4. `analyse_one_model.py`
**Purpose**: Single model CAR analysis for detailed examination of individual model performance.

**Key Features**:
- Analyzes CAR for a single model's predictions
- Compares against positive EPS surprises baseline
- Supports multiple analysis modes:
  - Manual configuration with specific dates and tickers
  - Results folder mode (reads from test_predictions.csv)
  - EPS surprises analysis mode
- Generates detailed individual stock analysis

**Configuration**:
- Set `RESULTS_FOLDER` to point to specific model results
- Configure analysis parameters (days before/after, benchmark ticker)
- Choose analysis mode via `MANUAL_MODE` flag

### 5. `analyze_aggregated_coverage.py`
**Purpose**: Coverage analysis for aggregated results, measuring how well models identify actual positive events.

**Key Features**:
- Counts actual positive events identified by at least one model
- Calculates coverage statistics for each model architecture
- Provides insights into model ensemble effectiveness
- Generates coverage reports and statistics

### 6. `plot_medium_aggregate.py`
**Purpose**: Medium-term aggregate plotting focused on the 15-40 day post-event period.

**Key Features**:
- Specialized for medium-term PEAD (Post-Earnings Announcement Drift) analysis
- Focuses on day 15 to day 40 after earnings announcements
- Generates medium-term specific CAR analysis
- Creates comparison plots for medium-term performance

### 7. `plot_single_model_car.py`
**Purpose**: Single model CAR plotting with simplified visualization.

**Key Features**:
- Simplified CAR analysis for individual models
- Focuses on medium-term period (day 15-40)
- Generates clean, publication-ready plots
- Compares positive predictions against test set average


## Key Analysis Concepts

### 1. Consensus Aggregation
Models within each architecture are combined using majority voting to create more robust predictions. This approach:
- Reduces individual model bias
- Improves prediction stability
- Provides confidence measures through vote counts

### 2. CAR Analysis
Cumulative Abnormal Returns analysis measures the financial impact of predictions:
- **Abnormal Returns**: Stock returns minus benchmark (SPY) returns
- **CAR**: Cumulative sum of abnormal returns over time
- **Analysis Periods**:
  - Pre-event: 20 days before earnings announcement
  - Early post-event: Days 0-14 after announcement
  - Late post-event: Days 15-40 after announcement (PEAD period)

### 3. Performance Metrics
- **Accuracy**: Overall prediction correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC**: Area Under the ROC Curve

### 4. Baseline Comparisons
All analyses compare model predictions against:
- **Positive EPS Surprises**: Actual positive earnings surprises
- **Test Set Average**: Average CAR for all test set instances
- **Random Baseline**: Expected performance from random predictions

## Usage Instructions

### Running Consensus Aggregation
```bash
python aggregate_consensus.py
```

### Analyzing Multiple Models
```bash
python analyse_many_models.py
```

### Single Model Analysis
```bash
python analyse_one_model.py
```

### Coverage Analysis
```bash
python analyze_aggregated_coverage.py
```

## Configuration

Most scripts use configurable parameters at the top of the file:

```python
# Analysis parameters
DAYS_BEFORE = 20          # Days before event to analyze
DAYS_AFTER = 40           # Days after event to analyze
MID_POINT_START = 15      # Start of late post-event period
BENCHMARK_TICKER = "SPY"  # Benchmark for abnormal returns
```

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- sqlite3
- pathlib
- json

## Data Requirements

The analysis scripts expect:
- Model results in `Results/` directory with `test_predictions.csv` files
- Stock price data in `Data/momentum_data.db`
- EPS surprises data in `Data/eps_surprises_quarterly_2012_2024.csv`
- Test set data in `Data/test_set.csv`

## Notes

- All scripts use LaTeX formatting for publication-ready plots
- Results are automatically saved to appropriate subdirectories
- Scripts handle missing data gracefully with appropriate warnings
- Analysis periods are configurable but optimized for PEAD research
- All financial calculations use SPY as the market benchmark
