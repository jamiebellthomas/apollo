# Apollo: Financial News Analysis and Earnings Prediction System

Apollo is a comprehensive research system for analyzing financial news and predicting post-earnings announcement drift (PEAD) using heterogeneous graph neural networks. The system processes financial news articles, earnings data, and market information to build knowledge graphs that capture relationships between news events and companies, then uses advanced machine learning models to predict market reactions.

## System Overview

Apollo combines multiple data sources and advanced machine learning techniques to create a robust financial prediction system:

- **Data Assembly**: Comprehensive pipeline for collecting and processing financial data
- **Knowledge Graph Construction**: Building heterogeneous graphs from news facts and company data
- **Graph Neural Networks**: Advanced GNN architectures for financial prediction
- **Analysis and Evaluation**: Comprehensive performance analysis and model comparison
- **Explainability**: Model interpretability and attention analysis

## Directory Structure

```
apollo/
├── README.md                    # This file
├── config.py                    # Global configuration settings
├── requirements.txt             # Python dependencies
├── DataAssembly/                # Data collection and processing pipeline
│   ├── README.md               # Data assembly documentation
│   ├── AssetPrices/            # Historical stock price data collection
│   ├── EPS/                    # Earnings per share data extraction
│   ├── MetaData/               # Company metadata and filtering
│   └── NewsArticles/           # Financial news processing and analysis
├── KG/                         # Knowledge Graph and GNN implementation
│   ├── README.md               # GNN system documentation
│   ├── SubGraph.py             # Core graph data structure
│   ├── HeteroGNN*.py           # Multiple GNN architectures
│   ├── run.py                  # Training and evaluation scripts
│   └── model_cache/            # Cached ML models
├── Analysis/                   # Performance analysis and visualization
│   ├── README.md               # Analysis documentation
│   ├── analyse_*.py            # Analysis scripts
│   └── plot_*.py               # Visualization scripts
├── Baselines/                  # Baseline model implementations
│   ├── README.md               # Baseline documentation
│   ├── eps_only/               # EPS-only prediction baselines
│   ├── neural_net/             # Neural network baselines
│   ├── sentiment/              # Sentiment-based baselines
│   └── weighted_sentiment/     # Time-weighted sentiment baselines
├── Explainability/             # Model interpretability and analysis
│   ├── attention_explainability.py    # Attention mechanism analysis
│   ├── cluster_analysis.py            # Event cluster analysis
│   └── temporal_attention_analysis.py # Temporal attention patterns
└── Data/                       # Processed datasets and results
    ├── eps_data.csv           # Earnings per share data
    ├── facts_output.jsonl     # Processed news facts
    ├── subgraphs.jsonl        # Constructed knowledge graphs
    └── momentum_data.db       # Historical price database
```

## Key Components

### 1. Data Assembly Pipeline (`DataAssembly/`)

Comprehensive data collection and processing system:

- **AssetPrices/**: Historical stock price data collection and analysis
- **EPS/**: Earnings per share data extraction from SEC filings and APIs
- **MetaData/**: Company metadata management and S&P 500 filtering
- **NewsArticles/**: Financial news processing, sentiment analysis, and fact extraction

**Data Flow:**
1. Company metadata extraction and filtering
2. Historical price data collection
3. Earnings data extraction from SEC filings
4. News article processing and fact extraction
5. Data validation and quality assurance

### 2. Knowledge Graph System (`KG/`)

Advanced heterogeneous graph neural network implementation:

- **SubGraph Construction**: Building financial event graphs from news facts
- **Multiple GNN Architectures**: 5 different model variants with increasing sophistication
- **Temporal Modeling**: Advanced time-aware graph processing
- **Attention Mechanisms**: Interpretable attention-based graph learning
- **Training Infrastructure**: Comprehensive training with logging and evaluation

**Model Variants:**
- **HeteroGNN**: Original implementation with basic temporal encoding
- **HeteroGNN2**: Temporal-aware model with learned edge encoding
- **HeteroGNN3**: Simplified model without temporal components
- **HeteroGNN4**: Attention-based model with GATv2Conv
- **HeteroGNN5**: Enhanced attention model with advanced features

### 3. Analysis and Evaluation (`Analysis/`)

Comprehensive performance analysis and model comparison:

- **Model Performance**: Accuracy, precision, recall, F1-score, AUC analysis
- **Temporal Analysis**: Time-based performance patterns
- **Aggregation Methods**: Consensus and ensemble analysis
- **Visualization**: Performance comparison plots and charts

### 4. Baseline Models (`Baselines/`)

Multiple baseline implementations for performance comparison:

- **EPS-Only**: Simple earnings surprise-based predictions
- **Neural Networks**: Traditional MLP baselines
- **Sentiment-Based**: News sentiment-only predictions
- **Weighted Sentiment**: Time-weighted sentiment analysis

### 5. Explainability (`Explainability/`)

Model interpretability and attention analysis:

- **Attention Visualization**: Understanding model focus patterns
- **Cluster Analysis**: Event type clustering and analysis
- **Temporal Patterns**: Time-based attention behavior
- **Misclassification Analysis**: Understanding prediction errors

## Quick Start

### 1. Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Download required models
cd KG/
python download_model.py
```

### 2. Data Preparation
```bash
# Run data assembly pipeline (see DataAssembly/README.md for details)
cd DataAssembly/
python MetaData/company_names.py
python AssetPrices/build_pricing_db.py
python EPS/parse_10k_filings.py
python NewsArticles/restore_row_structure.py
```

### 3. Knowledge Graph Construction
```bash
# Build SubGraphs from processed data
cd KG/
python create_raw_subgraphs.py
```

### 4. Model Training
```bash
# Train a single model
python run.py

# Run multiple experiments for statistical robustness
python run_many.py
```

### 5. Analysis and Evaluation
```bash
# Analyze aggregate model performance
python Analysis/analyse_many_models.py
python Analysis/aggregate_consensus.py
```

## Research Applications

### Financial Prediction
- **Earnings Surprise Prediction**: Anticipating market reactions to earnings
- **PEAD Modeling**: Post-earnings announcement drift prediction
- **Sentiment Impact Analysis**: News sentiment effects on stock prices
- **Event Study Research**: Financial event analysis and market efficiency

### Machine Learning Research
- **Heterogeneous GNNs**: Multi-type node and edge modeling
- **Temporal Graph Learning**: Time-aware graph neural networks
- **Attention Mechanisms**: Interpretable graph attention research
- **Financial Knowledge Graphs**: Domain-specific graph construction

## Dependencies

### Core Requirements
- **Python 3.8+**: Modern Python with type hints support
- **PyTorch**: Deep learning framework
- **PyTorch Geometric**: Graph neural network library
- **Transformers**: NLP model support
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning utilities



## Contact

For questions, issues, or collaboration opportunities, please contact [your-email@domain.com] or open an issue on the project repository.
