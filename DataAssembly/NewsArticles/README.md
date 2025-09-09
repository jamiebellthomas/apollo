# DataAssembly/NewsArticles Directory

This directory contains a comprehensive suite of tools for processing, analyzing, and extracting structured information from financial news articles. The system handles everything from raw data acquisition and cleaning to advanced NLP processing, sentiment analysis, and event extraction using Large Language Models (LLMs).

## Directory Structure

```
DataAssembly/NewsArticles/
├── README.md                           # This file
├── article_fact_extractor.py           # LLM-based fact extraction pipeline
├── cluster_event_type.py               # Event type clustering and analysis
├── extract_other_relevant_companies.py # Secondary ticker extraction
├── finBERT_sentiment.py                # Financial sentiment analysis
├── news_csv_analysis.ipynb             # News data analysis notebook
├── plot_benchmarks.ipynb               # Benchmark plotting notebook
├── restore_row_structure.py            # CSV structure repair utilities
├── row_parser.py                       # Custom CSV parsing and filtering
└── ticker_keyword_lookup.py            # Company name to ticker mapping
```

## Core Functionality

### 1. Data Acquisition and Cleaning

#### `restore_row_structure.py`
**Purpose**: Downloads and repairs malformed CSV files from external sources, specifically handling the FNSPID dataset.

**Key Features**:
- **Data Download**: Automated downloading of raw news datasets using wget
- **Row Structure Repair**: Fixes broken CSV formatting by identifying row boundaries
- **Streaming Processing**: Handles large files efficiently without loading into memory
- **Pattern Recognition**: Uses regex patterns to identify proper row starts
- **Progress Tracking**: Comprehensive logging and progress reporting

**Main Functions**:
- `download_raw_data()`: Downloads datasets from external URLs
- `fix_newlines_by_index_pattern_streaming()`: Repairs CSV structure using streaming
- `repair_fns_news_csv()`: Master function for complete CSV repair
- `analyse_csv()`: Analyzes and samples CSV content

**Usage**:
```python
from restore_row_structure import repair_fns_news_csv, analyse_csv

# Repair malformed CSV
repair_fns_news_csv(original_path, final_output_path)

# Analyze results
analyse_csv(filepath, num_samples=3)
```

#### `row_parser.py`
**Purpose**: Advanced CSV parsing with custom field handling and data filtering capabilities.

**Key Features**:
- **Custom Parser**: Handles malformed CSV rows with proper quote handling
- **Data Validation**: Validates row structure and field counts
- **EPS Filtering**: Filters news data to match EPS dataset tickers
- **Progress Tracking**: Efficient processing of large datasets
- **Error Handling**: Robust handling of malformed rows

**Main Functions**:
- `parse_row_custom()`: Custom CSV row parser with quote handling
- `sample_and_export_cleaned_csv()`: Processes and exports cleaned CSV data
- `filter_news_by_eps_tickers()`: Filters news to match EPS dataset companies

**Usage**:
```python
from row_parser import sample_and_export_cleaned_csv, filter_news_by_eps_tickers

# Clean and export CSV
sample_and_export_cleaned_csv(input_path, output_path, num_samples=3)

# Filter by EPS tickers
filter_news_by_eps_tickers(news_csv_path, eps_csv_path, output_csv_path)
```

### 2. Company and Ticker Management

#### `ticker_keyword_lookup.py`
**Purpose**: Comprehensive mapping of company names to stock tickers for accurate entity recognition.

**Key Features**:
- **Complete Coverage**: Maps 150+ S&P 500 companies to their tickers
- **Multiple Aliases**: Includes various company name variations and aliases
- **Case Sensitivity**: Handles different capitalization patterns
- **Special Characters**: Properly handles special characters and punctuation

**Data Structure**:
```python
TICKER_KEYWORD_LOOKUP = {
    "AAPL": ["AAPL", "Apple"],
    "MSFT": ["MSFT", "Microsoft"],
    "AMZN": ["AMZN", "Amazon"],
    # ... 150+ more companies
}
```

**Usage**:
```python
from ticker_keyword_lookup import TICKER_KEYWORD_LOOKUP

# Access ticker mappings
apple_terms = TICKER_KEYWORD_LOOKUP["AAPL"]  # ["AAPL", "Apple"]
```

#### `extract_other_relevant_companies.py`
**Purpose**: Extracts secondary companies mentioned in news articles using both LLM-based and regex-based approaches.

**Key Features**:
- **LLM Extraction**: Uses multiple language models for intelligent company extraction
- **Regex Fallback**: Fast regex-based extraction using ticker keyword lookup
- **Model Fallback**: Automatic switching between models on rate limits
- **Async Processing**: High-performance asynchronous processing
- **Duplicate Handling**: Removes duplicate articles and cleans data

**Main Functions**:
- `generate_related_ticker_prompt()`: Creates prompts for LLM-based extraction
- `extract_relevant_tickers_openai()`: LLM-based ticker extraction with multiple providers
- `populate_associated_tickers_column_async()`: Async processing of large datasets
- `populate_associated_tickers_with_regex()`: Fast regex-based extraction
- `clear_duplicate_articles()`: Removes duplicate articles from dataset

**Usage**:
```python
from extract_other_relevant_companies import (
    populate_associated_tickers_with_regex,
    clear_duplicate_articles
)

# Fast regex-based extraction
df = populate_associated_tickers_with_regex(df)

# Remove duplicates
clear_duplicate_articles(df)
```

### 3. Sentiment Analysis

#### `finBERT_sentiment.py`
**Purpose**: Financial-specific sentiment analysis using the FinBERT model for accurate financial text understanding.

**Key Features**:
- **FinBERT Model**: Uses ProsusAI/finbert for financial domain expertise
- **Multi-Device Support**: Automatic device selection (CUDA, MPS, CPU)
- **Comprehensive Output**: Provides labels, probabilities, and confidence scores
- **Performance Optimization**: Caching, mixed precision, and efficient inference
- **Apple Silicon Support**: Optimized for M1/M2/M3/M4 chips with MPS

**Main Functions**:
- `finbert_sentiment()`: Analyzes sentiment of financial text
- `_load_finbert_sentiment()`: Cached model loading for efficiency
- `_pick_device()`: Intelligent device selection

**Output Format**:
```python
{
    "label": "positive",           # negative/neutral/positive
    "confidence": 0.85,            # confidence in prediction
    "p_negative": 0.05,            # probability of negative
    "p_neutral": 0.10,             # probability of neutral
    "p_positive": 0.85,            # probability of positive
    "score": 0.80,                 # continuous sentiment [-1,1]
    "score_conf": 0.72             # confidence-weighted score
}
```

**Usage**:
```python
from finBERT_sentiment import finbert_sentiment

# Analyze sentiment
result = finbert_sentiment("Apple reported strong quarterly earnings")
print(result["label"])  # "positive"
```

### 4. Advanced Event Extraction

#### `article_fact_extractor.py`
**Purpose**: Sophisticated LLM-based pipeline for extracting structured facts from news articles with high throughput and reliability.

**Key Features**:
- **Async Pipeline**: Memory-bounded, batched asynchronous processing
- **GPU Safety**: Controlled concurrency for LLM inference
- **Crash Recovery**: Resumable processing with progress tracking
- **Multiple Models**: Support for various LLM providers (Ollama, OpenAI, Groq)
- **Structured Output**: Extracts facts in standardized JSON format
- **Validation**: Comprehensive response validation and error handling

**Extraction Schema**:
```json
{
    "date": "2024-01-15",
    "tickers": ["AAPL", "MSFT"],
    "raw_text": "Apple and Microsoft announced a strategic partnership...",
    "event_type": "partnership",
    "sentiment": 0.7
}
```

**Main Functions**:
- `build_llm_prompt()`: Creates structured prompts for fact extraction
- `validate_llm_response()`: Validates and parses LLM responses
- `call_llm_async()`: Asynchronous LLM inference
- `run_pipeline_async()`: Complete async processing pipeline
- `iter_csv()`: Efficient CSV streaming

**Configuration**:
```python
# Pipeline settings
MAX_WORKERS = 8                     # Worker coroutines
GPU_CONCURRENCY = 4                 # Simultaneous LLM calls
FLUSH_EVERY = 100                   # Facts per disk flush
MAX_ATTEMPTS = 3                    # Retries per article
MODEL_NAME = "llama3.3:70b"        # LLM model
```

**Usage**:
```python
from article_fact_extractor import main

# Run complete extraction pipeline
main()
```

### 5. Event Clustering and Analysis

#### `cluster_event_type.py`
**Purpose**: Clusters and analyzes event types extracted from news articles using semantic embeddings and machine learning.

**Key Features**:
- **Semantic Clustering**: Uses sentence transformers for event type clustering
- **Visualization**: t-SNE visualization of event clusters
- **Artifact Generation**: Creates comprehensive cluster artifacts
- **In-Place Annotation**: Adds cluster IDs to existing fact files
- **Caching**: Efficient model loading and caching

**Main Functions**:
- `collect_event_types()`: Extracts unique event types from facts
- `get_embeddings()`: Generates semantic embeddings for event types
- `get_clusters()`: Performs K-means clustering
- `reduce_dimensionality()`: t-SNE dimensionality reduction
- `plot_two_dim_embeddings()`: Visualizes clusters
- `save_cluster_artifacts()`: Saves cluster analysis results
- `annotate_facts_with_clusters_inplace()`: Adds cluster IDs to facts

**Output Artifacts**:
- **event_cluster_map.json**: Mapping of event types to cluster IDs
- **event_clusters.csv**: Cluster information with names and sizes
- **cluster_centroids.jsonl**: Cluster centroids for downstream use
- **event_types.txt**: Complete vocabulary of event types
- **clusters.png**: t-SNE visualization of clusters

**Usage**:
```python
from cluster_event_type import main

# Run complete clustering pipeline
main()
```

### 6. Analysis and Visualization

#### `news_csv_analysis.ipynb`
**Purpose**: Comprehensive analysis of news data including distribution analysis, temporal patterns, and data quality assessment.

**Key Features**:
- **Data Exploration**: Statistical analysis of news dataset
- **Temporal Analysis**: Time-series analysis of news patterns
- **Quality Assessment**: Data quality metrics and validation
- **Visualization**: Charts and graphs for data understanding

#### `plot_benchmarks.ipynb`
**Purpose**: Benchmarking and performance analysis of news processing pipelines.

**Key Features**:
- **Performance Metrics**: Processing speed and accuracy analysis
- **Model Comparison**: Comparison of different LLM models
- **Scalability Analysis**: Performance under different loads
- **Visualization**: Benchmark charts and performance graphs

## Key Concepts

### 1. Data Pipeline Architecture
**Multi-Stage Processing**:
1. **Raw Data Acquisition**: Download from external sources
2. **Structure Repair**: Fix malformed CSV files
3. **Data Cleaning**: Parse and validate data
4. **Entity Extraction**: Identify companies and tickers
5. **Sentiment Analysis**: Analyze financial sentiment
6. **Event Extraction**: Extract structured facts using LLMs
7. **Clustering**: Group similar events semantically
8. **Analysis**: Generate insights and visualizations

### 2. LLM Integration
**Model Support**:
- **Local Models**: Ollama integration for privacy
- **Cloud APIs**: OpenAI and Groq for scalability
- **Fallback Strategy**: Automatic model switching on failures
- **Rate Limiting**: Intelligent handling of API limits

**Prompt Engineering**:
- **Structured Prompts**: Consistent, detailed prompts for extraction
- **Validation**: Comprehensive response validation
- **Error Handling**: Robust error recovery and retry logic

### 3. Performance Optimization
**Async Processing**:
- **Concurrent Workers**: Multiple parallel processing streams
- **Memory Bounded**: Controlled memory usage for large datasets
- **Batch Processing**: Efficient batch operations
- **Progress Tracking**: Real-time progress monitoring

**Caching and Efficiency**:
- **Model Caching**: Cached model loading for repeated use
- **Streaming**: Memory-efficient streaming for large files
- **Incremental Processing**: Resume from interruption points

## Configuration

### File Paths
```python
# News data paths
NEWS_CSV_PATH_ORIGIN = "Data/news_raw.csv"
NEWS_CSV_PATH_FORMATTED_ROWS = "Data/news_formatted.csv"
NEWS_CSV_PATH_CLEAN = "Data/news_clean.csv"
NEWS_CSV_PATH_ASSOCIATED_TICKERS = "Data/news_with_tickers.csv"
NEWS_FACTS = "Data/facts_output.jsonl"

# Cluster artifacts
CLUSTER_CENTROIDS = "Data/cluster_centroids.jsonl"
```

### Model Configuration
```python
# LLM settings
MODEL_NAME = "llama3.3:70b"
MAX_WORKERS = 8
GPU_CONCURRENCY = 4
FLUSH_EVERY = 100
MAX_ATTEMPTS = 3

# Sentiment analysis
FINBERT_MODEL = "ProsusAI/finbert"
MAX_LENGTH = 512
```

## Dependencies

### Core Libraries
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **torch**: PyTorch for deep learning models
- **transformers**: Hugging Face transformers library
- **sentence-transformers**: Semantic embeddings
- **scikit-learn**: Machine learning algorithms
- **matplotlib/seaborn**: Visualization

### LLM and NLP
- **openai**: OpenAI API client
- **httpx**: Async HTTP client
- **ollama**: Local LLM integration
- **asyncio**: Asynchronous programming

### Data Processing
- **csv**: CSV file handling
- **json**: JSON processing
- **re**: Regular expressions
- **pathlib**: Path manipulation

## Usage Examples

### Complete News Processing Pipeline
```python
# 1. Download and repair raw data
from restore_row_structure import repair_fns_news_csv
repair_fns_news_csv(original_path, formatted_path)

# 2. Parse and clean data
from row_parser import sample_and_export_cleaned_csv
sample_and_export_cleaned_csv(formatted_path, clean_path)

# 3. Extract associated tickers
from extract_other_relevant_companies import populate_associated_tickers_with_regex
df = pd.read_csv(clean_path)
df = populate_associated_tickers_with_regex(df)

# 4. Extract facts using LLM
from article_fact_extractor import main
main()  # Processes all articles and extracts facts

# 5. Cluster event types
from cluster_event_type import main
main()  # Clusters events and generates artifacts
```

### Sentiment Analysis
```python
from finBERT_sentiment import finbert_sentiment

# Analyze individual articles
result = finbert_sentiment("Apple reported record quarterly revenue")
print(f"Sentiment: {result['label']} (confidence: {result['confidence']:.2f})")
```

### Event Clustering
```python
from cluster_event_type import collect_event_types, get_embeddings, get_clusters

# Collect and cluster event types
event_types = collect_event_types()
embeddings = get_embeddings(event_types)
labels, kmeans = get_clusters(embeddings, n_clusters=60)
```

## Data Output

### Fact Extraction Output
The fact extraction pipeline produces structured JSONL files with the following format:
```json
{
    "date": "2024-01-15",
    "tickers": ["AAPL", "MSFT"],
    "raw_text": "Apple and Microsoft announced a strategic partnership...",
    "event_type": "partnership",
    "sentiment": 0.7,
    "source_article_index": 12345,
    "event_cluster_id": 15
}
```

### Cluster Artifacts
- **event_cluster_map.json**: Complete mapping of event types to clusters
- **event_clusters.csv**: Cluster metadata with names and sizes
- **cluster_centroids.jsonl**: Cluster centroids for downstream analysis
- **clusters.png**: t-SNE visualization of event clusters

## Quality Assurance

### Data Validation
- **Row Structure**: Validates CSV row structure and field counts
- **Ticker Format**: Ensures proper ticker symbol formatting
- **Response Validation**: Validates LLM responses against schema
- **Cluster Quality**: Validates cluster assignments and centroids

### Error Handling
- **Graceful Degradation**: Continues processing despite individual failures
- **Retry Logic**: Automatic retry with exponential backoff
- **Progress Recovery**: Resumes from interruption points
- **Comprehensive Logging**: Detailed logging for debugging

## Performance Considerations

### Scalability
- **Streaming Processing**: Handles datasets of any size
- **Memory Efficiency**: Bounded memory usage regardless of dataset size
- **Parallel Processing**: Multi-core and GPU utilization
- **Incremental Updates**: Processes only new or changed data

### Optimization Strategies
- **Model Caching**: Avoids repeated model loading
- **Batch Operations**: Efficient batch processing
- **Async I/O**: Non-blocking file and network operations
- **Smart Batching**: Optimal batch sizes for different operations

## Research Applications

### 1. Financial News Analysis
- **Event Detection**: Identifies significant corporate events
- **Sentiment Tracking**: Monitors market sentiment over time
- **Company Relationships**: Maps inter-company relationships
- **Market Impact**: Analyzes news impact on stock prices

### 2. Machine Learning Research
- **Event Classification**: Supervised learning for event categorization
- **Sentiment Analysis**: Financial domain sentiment analysis
- **Entity Recognition**: Company and ticker identification
- **Clustering**: Unsupervised event type discovery

### 3. Academic Research
- **Reproducibility**: Standardized data processing pipeline
- **Data Quality**: High-quality, validated datasets
- **Methodology**: Clear methodology and documentation
- **Extensibility**: Modular design for research extensions

## Integration with Research Pipeline

### Data Flow
1. **Raw News** → **Cleaned News** → **News with Tickers** → **Extracted Facts** → **Clustered Events**
2. **Integration Points**: Connects with EPS data, pricing data, and analysis modules
3. **Downstream Usage**: Feeds into GNN models, baselines, and explainability analysis

### Configuration Integration
- **Centralized Config**: Uses project-wide configuration system
- **Path Management**: Consistent file path management
- **Model Settings**: Unified model configuration across components

## Notes

- All scripts include comprehensive error handling and logging
- The system is designed for both research and production use
- Modular architecture allows for easy extension and modification
- Extensive documentation and examples for all major functions
- Performance optimizations for handling large-scale financial datasets
- Integration with the broader research pipeline for comprehensive analysis
