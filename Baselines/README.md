# Baselines Directory

This directory contains baseline classification methods for earnings prediction tasks, providing comparison points for evaluating the performance of more sophisticated heterogeneous graph neural network (HeteroGNN) models. The baselines implement various heuristic and machine learning approaches using different feature combinations.

## Directory Structure

```
Baselines/
├── README.md                           # This file
├── calculate_auc_all_baselines.py      # AUC calculation for all baselines
├── eps_only/                           # EPS-only baseline methods
├── neural_net/                         # Neural network baseline methods
├── sentiment/                          # Sentiment-based baseline methods
└── weighted_sentiment/                 # Time-weighted sentiment baseline methods
```

## Core Baseline Scripts

### 1. `calculate_auc_all_baselines.py`
**Purpose**: Comprehensive AUC analysis for all baseline methods, providing standardized performance comparison.

**Key Features**:
- Loads results from all baseline experiments
- Calculates AUC scores from confusion matrices
- Generates comparison plots and summary statistics
- Creates standardized performance reports
- Supports both individual baseline analysis and ensemble comparison

**Usage**:
```bash
python calculate_auc_all_baselines.py
```

## Baseline Method Categories

### 1. EPS-Only Baselines (`eps_only/`)

#### `base_eps_positive.py`
**Purpose**: Simple heuristic baseline using only EPS surprise information for positive EPS cases.

**Methodology**:
- Filters data to include only positive EPS surprise cases
- Always predicts positive (label = 1) for positive EPS cases
- Evaluates against true labels to measure baseline performance
- Provides insight into the predictive power of EPS surprises alone

**Key Features**:
- Simple heuristic approach
- Focuses on positive EPS cases only
- No machine learning involved
- Fast execution and easy interpretation

#### `positive_only/base_eps_all.py`
**Purpose**: EPS-only baseline for all data (both positive and negative EPS cases).

**Methodology**:
- Uses EPS surprise as the primary predictor
- Predicts positive when EPS surprise > 0, negative otherwise
- Evaluates on complete test set
- Provides comprehensive EPS-based baseline

### 2. Neural Network Baselines (`neural_net/`)

#### `all_data/base_nn_all.py`
**Purpose**: Neural network baseline using quantitative features from primary ticker data.

**Architecture**:
- Simple feedforward neural network with batch normalization
- Multiple hidden layers with dropout for regularization
- Uses quantitative features from `get_ticker_node_features` method
- Binary classification with BCEWithLogitsLoss

**Key Features**:
- Configurable network architecture (hidden size, layers, dropout)
- Batch normalization for training stability
- Early stopping to prevent overfitting
- Training history visualization
- Standardized feature preprocessing

#### `positive_eps_only/base_nn_positive.py`
**Purpose**: Neural network baseline trained only on positive EPS cases.

**Methodology**:
- Same architecture as full neural network
- Trained exclusively on positive EPS surprise cases
- Tests generalization to positive EPS scenarios
- Provides comparison for domain-specific training

### 3. Sentiment-Based Baselines (`sentiment/`)

#### `sentiment_baseline.py`
**Purpose**: Sentiment-based baseline using EPS surprise and average sentiment.

**Methodology**:
- Combines EPS surprise with average sentiment from fact list
- Prediction logic:
  - Negative EPS surprise → predict negative
  - Positive EPS + negative sentiment → predict negative
  - Positive EPS + positive sentiment → predict positive
- Simple heuristic combining financial and textual signals

**Key Features**:
- Multi-modal approach (financial + textual)
- Simple rule-based decision making
- Fast execution
- Interpretable predictions

**Variants**:
- `all_data/`: Uses all available data
- `positive_eps_only/`: Focuses on positive EPS cases
- `threshold_0_2/`: Uses sentiment threshold of 0.2

### 4. Weighted Sentiment Baselines (`weighted_sentiment/`)

#### `weighted_sentiment_baseline.py`
**Purpose**: Advanced sentiment baseline with time-weighted sentiment calculation.

**Methodology**:
- Uses time-weighted average sentiment instead of simple average
- More recent articles receive higher weights
- Linear scaling: weight = 1 - (delta_days / max_days)
- Articles older than max_days (default 90) get zero weight
- Combines with EPS surprise for final prediction

**Key Features**:
- Temporal weighting of sentiment information
- Configurable time window (max_days parameter)
- More sophisticated than simple sentiment averaging
- Accounts for information recency

**Variants**:
- `all_data/`: Uses all available data
- `positive_eps_only/`: Focuses on positive EPS cases
- `threshold_0_1_positive_eps_only/`: Uses sentiment threshold of 0.1

## Baseline Comparison Framework

### Performance Metrics
All baselines are evaluated using standardized metrics:
- **Accuracy**: Overall prediction correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **AUC**: Area Under the ROC Curve (approximated from confusion matrix)

### Data Splits
Baselines support multiple data configurations:
- **All Data**: Complete test set evaluation
- **Positive EPS Only**: Subset with positive EPS surprises
- **Threshold Variants**: Different sentiment thresholds for classification

### Feature Categories
1. **EPS-Only**: Uses only earnings surprise information
2. **Quantitative**: Uses numerical features from ticker data
3. **Sentiment**: Uses textual sentiment from news articles
4. **Hybrid**: Combines multiple feature types

## Key Baseline Concepts

### 1. Heuristic Baselines
Simple rule-based approaches that provide fundamental performance bounds:
- **EPS-Only**: Tests predictive power of earnings surprises
- **Sentiment**: Tests predictive power of news sentiment
- **Combined**: Tests synergy between financial and textual signals

### 2. Machine Learning Baselines
More sophisticated approaches using traditional ML:
- **Neural Networks**: Tests performance of standard deep learning
- **Feature Engineering**: Uses domain-specific feature extraction
- **Regularization**: Prevents overfitting with dropout and batch normalization

### 3. Temporal Weighting
Advanced approaches that account for information recency:
- **Linear Decay**: Recent information gets higher weight
- **Time Windows**: Information older than threshold is ignored
- **Adaptive Weighting**: Weights based on information relevance

### 4. Domain Adaptation
Specialized approaches for specific scenarios:
- **Positive EPS Focus**: Optimized for positive earnings surprises
- **Threshold Tuning**: Optimized sentiment thresholds
- **Feature Selection**: Different feature combinations

## Usage Instructions

### Running Individual Baselines
```bash
# EPS-only baseline
python eps_only/base_eps_positive.py

# Neural network baseline
python neural_net/all_data/base_nn_all.py

# Sentiment baseline
python sentiment/sentiment_baseline.py

# Weighted sentiment baseline
python weighted_sentiment/weighted_sentiment_baseline.py
```

### Running AUC Analysis
```bash
python calculate_auc_all_baselines.py
```

## Configuration

Most baseline scripts support configurable parameters:

### Neural Network Parameters
```python
hidden_size = 64              # Hidden layer size
num_hidden_layers = 2         # Number of hidden layers
dropout_rate = 0.3            # Dropout probability
learning_rate = 0.001         # Learning rate
batch_size = 32               # Batch size
epochs = 100                  # Maximum epochs
```

### Sentiment Parameters
```python
sentiment_threshold = 0.2     # Sentiment classification threshold
max_days = 90                 # Maximum days for temporal weighting
```

### Data Parameters
```python
test_size = 0.2               # Test set proportion
random_state = 42             # Random seed for reproducibility
```

## Dependencies

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- torch (for neural network baselines)
- json

## Data Requirements

The baseline scripts expect:
- Test data in JSONL format with required fields:
  - `eps_surprise`: EPS surprise value
  - `label`: True binary label
  - `fact_list`: List of facts with sentiment scores
  - `delta_days`: Days between fact and event date
- Quantitative features from ticker data (for neural network baselines)

## Baseline Selection Guide

### For Simple Heuristics
- **EPS-Only**: When you want to test pure financial signal strength
- **Sentiment**: When you want to test pure textual signal strength
- **Combined**: When you want to test simple multi-modal approaches

### For Machine Learning
- **Neural Network**: When you want to test traditional deep learning
- **Feature Engineering**: When you want to test domain-specific features
- **Regularization**: When you want to test overfitting prevention

### For Advanced Approaches
- **Weighted Sentiment**: When you want to test temporal information weighting
- **Threshold Tuning**: When you want to optimize classification boundaries
- **Domain Adaptation**: When you want to test specialized approaches

## Notes

- All baselines use standardized evaluation metrics for fair comparison
- Results are automatically saved with consistent naming conventions
- Scripts handle missing data gracefully with appropriate warnings
- Neural network baselines include training history visualization
- Sentiment baselines support multiple threshold configurations
- Temporal weighting uses linear decay with configurable time windows
- All baselines are designed to be easily extensible and modifiable
