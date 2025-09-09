# KG Directory

The KG (Knowledge Graph) directory contains a comprehensive suite of tools for building, training, and evaluating heterogeneous graph neural networks (GNNs) for financial prediction tasks. This system processes financial news facts and company data to create heterogeneous graphs that capture relationships between news events and companies, then uses advanced GNN architectures to predict post-earnings announcement drift (PEAD).

## Directory Structure

```
KG/
├── README.md                           # This file
├── SubGraph.py                         # Core SubGraph data structure and encoding
├── SubGraphDataLoader.py               # Data loading and preprocessing utilities
├── EdgeDecay.py                        # Temporal decay functions for edge weighting
├── HeteroGNN.py                        # Original heterogeneous GNN implementation
├── HeteroGNN2.py                       # Temporal-aware GNN with learned edge encoding
├── HeteroGNN3.py                       # GNN without temporal encoding (sentiment only)
├── HeteroGNN4.py                       # Attention-based GNN with GATv2Conv
├── HeteroGNN5.py                       # Enhanced attention GNN with advanced features
├── HeteroGNN5Explainer.py              # Model explainability and attention analysis
├── run.py                              # Main training script with comprehensive logging
├── run_many.py                         # Multi-run experiment orchestration
├── create_raw_subgraphs.py             # SubGraph construction from raw data
├── cache_dataset.py                    # Dataset caching for fast loading
├── download_model.py                   # Sentence transformer model download utility
├── model_exponential_decay.ipynb       # Temporal decay function analysis notebook
├── subgraph_analysis.ipynb             # SubGraph structure and statistics analysis
├── model_cache/                        # Cached sentence transformer models
└── dataset_cache/                      # Cached processed datasets
```

## Core Components

### 1. SubGraph Data Structure (`SubGraph.py`)

The fundamental data structure representing a financial event graph:

**Key Features:**
- **Primary Ticker**: The main company for the earnings event
- **Reported Date**: Earnings announcement date
- **EPS Surprise**: Earnings per share surprise value
- **Fact List**: Collection of news facts with sentiment, temporal, and event information
- **Label**: Binary prediction target (positive/negative PEAD)

**Core Methods:**
- `encode()`: Converts SubGraph to numerical representations using sentence transformers
- `get_fact_node_features()`: Generates fact node embeddings with text and event encoding
- `get_ticker_node_features()`: Computes company node features from historical price data
- `get_edges()`: Builds fact-to-company edge connectivity with temporal weighting
- `to_pyg_data()`: Converts to PyTorch Geometric HeteroData format
- `visualise_numpy_graph_interactive()`: Creates interactive graph visualizations

**Feature Engineering:**
- **Fact Features**: Text embeddings (768D) + event type embeddings (768D) = 1536D
- **Company Features**: 27 financial indicators (returns, volatility, technical indicators)
- **Edge Features**: Sentiment scores + temporal decay weights

### 2. Data Loading (`SubGraphDataLoader.py`)

Comprehensive data loading and preprocessing system:

**Key Features:**
- **Fixed Data Splits**: Maintains consistent train/test splits across experiments
- **Filtering**: Removes subgraphs with insufficient facts or negative EPS surprises
- **Cluster Filtering**: Optional removal of specific event clusters
- **Batch Processing**: Efficient loading with progress tracking

**Data Validation:**
- Minimum fact count requirements
- EPS surprise positivity constraints
- Temporal window validation
- Data consistency checks

### 3. Temporal Decay Functions (`EdgeDecay.py`)

Mathematical functions for temporal edge weighting:

**Available Functions:**
- `linear()`: Linear decay from 1.0 to final_weight
- `exponential()`: Exponential decay with configurable rate
- `logarithmic()`: Logarithmic decay for gradual reduction
- `sigmoid()`: S-shaped decay curve
- `quadratic()`: Quadratic decay function

**Parameters:**
- `decay_days`: Time window for decay application
- `final_weight`: Minimum weight at end of window

### 4. GNN Model Architectures

#### HeteroGNN (`HeteroGNN.py`)
**Original Implementation:**
- Type-specific input encoders for facts and companies
- Learned edge gating with sentiment and decay
- HeteroConv stack with GraphConv layers
- Flexible readout options (fact, company, concat, gated)
- Primary-ticker-aware pooling

#### HeteroGNN2 (`HeteroGNN2.py`)
**Temporal-Aware Model:**
- Advanced edge attribute encoder with Time2Vec
- Learnable temporal decay parameters
- Improved MLP architecture for edge features
- Residual connections and layer normalization
- Enhanced temporal modeling capabilities

#### HeteroGNN3 (`HeteroGNN3.py`)
**No Temporal Encoding:**
- Simplified architecture without temporal components
- Sentiment-only edge attributes
- Baseline for temporal encoding ablation
- Faster training and inference

#### HeteroGNN4 (`HeteroGNN4.py`)
**Attention-Based Model:**
- GATv2Conv with edge-feature-aware attention
- TimeEdgeBuilder for temporal edge features
- Optional funnel mode (fact→company only)
- Top-k pre-gating for computational efficiency
- Multi-head attention mechanisms

#### HeteroGNN5 (`HeteroGNN5.py`)
**Enhanced Attention Model:**
- Advanced attention temperature control
- Entropy regularization for attention sparsity
- Time bucket embeddings for coarse temporal regimes
- Enhanced edge features (absolute sentiment, polarity bits)
- Training-time jitter for robustness
- Monte Carlo dropout support
- Manual attention layer implementation for regularization

### 5. Training Infrastructure (`run.py`)

Comprehensive training system with advanced features:

**Key Features:**
- **Comprehensive Logging**: Experiment tracking with hyperparameters and results
- **Run Scraping**: Automatic removal of runs that end too early
- **Multiple Loss Functions**: BCE, weighted BCE, focal loss options
- **Learning Rate Scheduling**: Step, cosine, plateau, one-cycle schedulers
- **Early Stopping**: Configurable patience and minimum epochs
- **Data Caching**: Fast dataset loading with validation
- **Device Management**: Automatic CUDA/MPS/CPU selection
- **Gradient Clipping**: Prevents exploding gradients

**Training Configuration:**
- Batch processing with rate limiting
- Robust error handling and retry logic
- Memory-efficient data loading
- Comprehensive metrics tracking
- Model checkpointing and restoration

### 6. Multi-Run Experiments (`run_many.py`)

Orchestration system for robust performance evaluation:

**Features:**
- **Multiple Seeds**: Random seed generation for statistical robustness
- **Model Comparison**: Side-by-side evaluation of different architectures
- **Summary Statistics**: Mean, std, min, max across runs
- **Run Management**: Handles scraped and failed runs gracefully
- **Comprehensive Reporting**: Detailed results and performance analysis

### 7. Data Construction (`create_raw_subgraphs.py`)

Pipeline for building SubGraphs from raw financial data:

**Process:**
1. **EPS Data Loading**: Loads earnings surprises and dates
2. **Facts Loading**: Processes news facts with sentiment filtering
3. **Temporal Indexing**: Creates efficient time-based lookups
4. **Window Selection**: Extracts facts within temporal windows
5. **Label Calculation**: Computes PEAD labels from price data
6. **SubGraph Assembly**: Creates final SubGraph objects

**Key Functions:**
- `load_eps()`: Earnings data preprocessing
- `load_facts()`: News facts with sentiment filtering
- `build_ticker_index()`: Efficient temporal indexing
- `select_facts_for_instance()`: Window-based fact selection
- `calculate_label()`: PEAD label computation from price data

### 8. Dataset Caching (`cache_dataset.py`)

Efficient dataset caching system:

**Features:**
- **Atomic Writes**: Safe concurrent access with file locking
- **Data Validation**: Ensures cache integrity
- **Separate Caches**: Training and testing data cached separately
- **Timeout Handling**: Robust error recovery
- **Progress Tracking**: Detailed caching status reporting

## Key Concepts

### Heterogeneous Graph Structure
- **Node Types**: Facts (news events) and Companies (tickers)
- **Edge Types**: Fact→Company (mentions) and Company→Fact (mentioned_in)
- **Edge Attributes**: Sentiment scores and temporal decay weights
- **Node Features**: Text embeddings for facts, financial indicators for companies

### Temporal Modeling
- **Time Windows**: Configurable lookback periods (default: 60 days)
- **Decay Functions**: Multiple mathematical decay options
- **Time2Vec**: Learned temporal representations
- **Event Timing**: Delta days from earnings announcement

### Attention Mechanisms
- **Multi-Head Attention**: Parallel attention computation
- **Edge-Aware Attention**: Incorporates edge features in attention
- **Temperature Scaling**: Controls attention sharpness
- **Entropy Regularization**: Encourages focused attention patterns

### Financial Prediction
- **PEAD Labels**: Post-earnings announcement drift prediction
- **Cumulative Abnormal Returns**: Market-adjusted performance
- **Slope Thresholds**: Configurable performance criteria
- **Benchmark Comparison**: SPY-adjusted returns

## Usage

### Basic Training
```python
from run import run_training

model, test_metrics, history = run_training(
    model_type="heterognn5",
    n_facts=35,
    hidden_channels=128,
    num_layers=4,
    epochs=100,
    lr=1e-5
)
```

### Multi-Run Experiments
```python
from run_many import run_multiple_experiments

results, summary = run_multiple_experiments(
    num_runs=10,
    model_type="heterognn5"
)
```

### Data Construction
```python
from create_raw_subgraphs import build_subgraphs_jsonl

build_subgraphs_jsonl(
    eps_csv_path="Data/eps_surprises.csv",
    facts_jsonl_path="Data/facts.jsonl",
    out_path="Data/subgraphs.jsonl"
)
```

## Configuration

### Model Parameters
- **Hidden Channels**: Model dimension (128-1024)
- **Number of Layers**: GNN depth (2-6)
- **Attention Heads**: Multi-head attention (4-8)
- **Dropout Rates**: Feature, edge, and final dropout
- **Readout Mode**: fact, company, concat, or gated

### Training Parameters
- **Batch Size**: Training batch size (16-64)
- **Learning Rate**: Optimizer learning rate (1e-6 to 1e-3)
- **Weight Decay**: L2 regularization strength
- **Patience**: Early stopping patience
- **Gradient Clipping**: Maximum gradient norm

### Data Parameters
- **Minimum Facts**: Required facts per subgraph
- **Window Days**: Temporal lookback window
- **Sentiment Minimum**: Minimum sentiment magnitude
- **Train/Val/Test Ratios**: Data split proportions

## Dependencies

### Core Libraries
- **PyTorch**: Deep learning framework
- **PyTorch Geometric**: Graph neural network library
- **Sentence Transformers**: Text embedding models
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning utilities

### Optional Libraries
- **Matplotlib**: Plotting and visualization
- **Plotly**: Interactive visualizations
- **NetworkX**: Graph analysis
- **SQLite3**: Database operations

## Performance Considerations

### Memory Management
- **Batch Processing**: Configurable batch sizes
- **Gradient Accumulation**: For large effective batch sizes
- **Model Caching**: Reuse of sentence transformer models
- **Dataset Caching**: Preprocessed data storage

### Computational Efficiency
- **Edge Dropout**: Reduces graph connectivity during training
- **Top-k Filtering**: Limits edges per company node
- **Mixed Precision**: Optional FP16 training
- **Device Optimization**: Automatic GPU/CPU selection

### Scalability
- **Distributed Training**: Multi-GPU support
- **Data Parallelism**: Batch-level parallelization
- **Model Parallelism**: Layer-level distribution
- **Asynchronous Loading**: Non-blocking data pipeline

## Research Applications

### Financial Prediction
- **Earnings Surprise Prediction**: Anticipating market reactions
- **PEAD Modeling**: Post-earnings announcement drift
- **Sentiment Analysis**: News impact on stock prices
- **Event Study**: Financial event analysis

### Graph Learning
- **Heterogeneous GNNs**: Multi-type node and edge modeling
- **Temporal GNNs**: Time-aware graph neural networks
- **Attention Mechanisms**: Interpretable graph attention
- **Financial Knowledge Graphs**: Domain-specific graph construction

### Model Interpretability
- **Attention Visualization**: Understanding model focus
- **Edge Importance**: Identifying critical relationships
- **Temporal Patterns**: Time-based model behavior
- **Feature Attribution**: Input feature importance

## Notes

- **Data Requirements**: Requires processed EPS and news facts data
- **Model Selection**: Choose architecture based on computational constraints
- **Hyperparameter Tuning**: Extensive configuration options available
- **Reproducibility**: Fixed random seeds for consistent results
- **Monitoring**: Comprehensive logging and experiment tracking
- **Validation**: Robust data validation and error handling
