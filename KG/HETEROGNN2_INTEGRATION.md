# HeteroGNN2 Integration

## Overview

HeteroGNN2 is a temporal-aware heterogeneous graph neural network that has been integrated into the existing `run.py` infrastructure. It extends the original HeteroGNN with temporal edge encoding capabilities.

## Key Features

### 1. Temporal Edge Encoding
- **EdgeAttrEncoder**: Processes edge attributes (sentiment, decay) with temporal awareness
- **Time2Vec**: Sinusoidal encoding for temporal patterns
- **Decay Conversion**: Converts decay values to temporal deltas for better modeling

### 2. SAGEConv Architecture
- Uses GraphSAGE convolution layers instead of GCN
- Better handling of heterogeneous graphs
- Improved scalability for large graphs

### 3. Compatible Data Format
- Works with the existing data format from `run.py`
- Supports the same edge types: `('fact', 'mentions', 'company')` and `('company', 'mentioned_in', 'fact')`
- Handles edge attributes in `[sentiment, decay]` format

## Usage

### Basic Usage

```python
from run import run_training

# Run with HeteroGNN2
model, metrics, history = run_training(
    model_type="heterognn2",  # Use the new model
    time_dim=8,               # Temporal encoding dimension
    hidden_channels=64,       # Hidden layer size
    num_layers=2,             # Number of GNN layers
    readout="concat",         # Graph pooling method
    # ... other parameters
)
```

### Comparison with Original HeteroGNN

```python
# Original HeteroGNN
model1, metrics1, history1 = run_training(
    model_type="heterognn",
    # ... parameters
)

# HeteroGNN2 (temporal-aware)
model2, metrics2, history2 = run_training(
    model_type="heterognn2",
    time_dim=8,  # Additional parameter for temporal encoding
    # ... same other parameters
)
```

## Model Architecture

### EdgeAttrEncoder
```python
class EdgeAttrEncoder(nn.Module):
    def __init__(self, time_dim=8):
        # MLP: [sentiment, decay, time2vec] -> [1] (gate)
        
    def time2vec(self, delta_t):
        # Sinusoidal encoding for temporal patterns
        
    def forward(self, sentiment, delta_t, lambda_decay=0.01):
        # Convert decay to delta_t, apply temporal encoding, output gate
```

### HeteroGNN2
```python
class HeteroGNN2(nn.Module):
    def __init__(self, metadata, hidden_channels=128, num_layers=2, 
                 feature_dropout=0.2, edge_dropout=0.0, final_dropout=0.0,
                 readout="concat", time_dim=8):
        # Lazy initialization of node type encoders
        # SAGEConv layers for message passing
        # Temporal edge encoding
        # Flexible readout options
```

## Key Differences from Original HeteroGNN

| Feature | HeteroGNN | HeteroGNN2 |
|---------|-----------|------------|
| Convolution | GCNConv | SAGEConv |
| Edge Weights | Direct support | Temporal encoding |
| Temporal Awareness | None | Time2Vec + decay conversion |
| Edge Processing | Simple linear gate | MLP with temporal features |
| Scalability | Standard | Better for large graphs |

## Testing

### Unit Tests
```bash
python test_heterognn2.py
```

### Integration Example
```bash
python run_heterognn2_example.py
```

### Full Training
```bash
python run.py  # Uses HeteroGNN2 by default now
```

## Configuration Options

### Model Parameters
- `model_type`: "heterognn" or "heterognn2"
- `time_dim`: Temporal encoding dimension (default: 8)
- `hidden_channels`: Hidden layer size
- `num_layers`: Number of GNN layers
- `readout`: "fact", "company", "concat", or "gated"

### Training Parameters
- All existing parameters from `run_training()` work the same
- `time_dim` is the only new parameter specific to HeteroGNN2

## Performance Considerations

### Advantages of HeteroGNN2
1. **Temporal Awareness**: Better modeling of time-dependent patterns
2. **Scalability**: SAGEConv handles large graphs better
3. **Flexibility**: More sophisticated edge processing

### Memory Usage
- Slightly higher memory usage due to temporal encoding
- Time2Vec requires additional parameters
- Edge encoder MLP adds computational overhead

## Example Results

The model produces outputs compatible with the existing evaluation pipeline:
- Graph-level predictions (logits)
- Compatible with BCEWithLogitsLoss
- Same evaluation metrics (accuracy, AUC, precision, recall, F1)

## Integration Status

✅ **Fully Integrated**
- Works with existing `run.py` infrastructure
- Compatible data format
- Same training pipeline
- Same evaluation metrics
- Comprehensive logging and experiment tracking

## Files Modified

1. **HeteroGNN2.py**: Main model implementation
2. **run.py**: Added model_type parameter and HeteroGNN2 support
3. **test_heterognn2.py**: Unit tests
4. **run_heterognn2_example.py**: Usage examples

## Next Steps

1. **Hyperparameter Tuning**: Optimize `time_dim` and other temporal parameters
2. **Ablation Studies**: Compare with and without temporal encoding
3. **Performance Analysis**: Benchmark against original HeteroGNN
4. **Feature Engineering**: Explore additional temporal features 