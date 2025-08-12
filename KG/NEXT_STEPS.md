# Next Steps for Model Improvement

## Current Best Configuration

**Performance:**
- **AUC**: 0.645
- **F1-Score**: 0.41
- **Recall**: 35%
- **Precision**: 49%
- **Accuracy**: 65.5%

**Configuration:**
- **Text encoder**: `all-mpnet-base-v2` (768d embeddings)
- **Loss**: Standard `BCEWithLogitsLoss`
- **Threshold**: 0.5 (for evaluation)
- **Splitting**: Random (not time-aware)
- **Readout**: "fact"
- **Dropout**: 0.4 (feature), 0.2 (edge), 0.3 (final)

## 1. Threshold Optimization

### Try Even Lower Thresholds
```python
# Test thresholds: 0.2, 0.25, 0.3, 0.35, 0.4
thresholds = [0.2, 0.25, 0.3, 0.35, 0.4]
for threshold in thresholds:
    test_metrics = evaluate(model, test_loader, device, 
                          detailed_analysis=True, threshold=threshold)
    print(f"Threshold {threshold}: F1={test_metrics['f1']:.3f}, "
          f"Recall={test_metrics['recall']:.3f}")
```

### Dynamic Threshold Selection
```python
# Find optimal threshold on validation set
from sklearn.metrics import f1_score
import numpy as np

def find_optimal_threshold(model, val_loader, device):
    model.eval()
    all_probs = []
    all_y = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu())
            all_y.append(batch.y.cpu())
    
    all_probs = torch.cat(all_probs).numpy()
    all_y = torch.cat(all_y).numpy()
    
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        preds = (all_probs >= threshold).astype(int)
        f1 = f1_score(all_y, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1
```

## 2. Model Architecture Experiments

### Try Different Readout Methods
```python
# Test all readout options
readout_options = ["fact", "company", "concat", "gated"]

for readout in readout_options:
    model = HeteroGNN(
        metadata=metadata,
        hidden_channels=128,
        num_layers=2,
        readout=readout,
        # ... other params
    )
    # Train and evaluate
```

### Increase Model Capacity
```python
# Try deeper networks
configs = [
    {"hidden_channels": 256, "num_layers": 3},
    {"hidden_channels": 128, "num_layers": 4},
    {"hidden_channels": 512, "num_layers": 2},
]

for config in configs:
    model = HeteroGNN(
        metadata=metadata,
        **config,
        # ... other params
    )
```

### Try Different GNN Layers
```python
# Experiment with different convolution types
# Modify HeteroGNN.py to support:
# - GraphSAGE
# - GAT (Graph Attention Networks)
# - GIN (Graph Isomorphism Networks)
```

## 3. Data Augmentation Techniques

### Edge Dropout During Training
```python
# Increase edge dropout for regularization
edge_dropout_values = [0.1, 0.2, 0.3, 0.4]

for edge_dropout in edge_dropout_values:
    # Train with different edge dropout rates
    pass
```

### Feature Noise Injection
```python
# Add noise to node features during training
def add_feature_noise(data, noise_std=0.1):
    """Add Gaussian noise to node features."""
    for node_type in data.node_types:
        if hasattr(data[node_type], 'x'):
            noise = torch.randn_like(data[node_type].x) * noise_std
            data[node_type].x = data[node_type].x + noise
    return data
```

### Temporal Data Augmentation
```python
# Since we have temporal information, try:
# 1. Time-based data augmentation
# 2. Sliding window approaches
# 3. Temporal masking
```

## 4. Advanced Loss Functions

### Focal Loss with Optimal Parameters
```python
# Try different focal loss configurations
focal_configs = [
    {"alpha": 1.0, "gamma": 2.0},
    {"alpha": 2.0, "gamma": 2.0},
    {"alpha": 1.0, "gamma": 3.0},
]

for config in focal_configs:
    criterion = FocalLoss(**config)
    # Train and evaluate
```

### Label Smoothing
```python
class LabelSmoothingBCELoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, inputs, targets):
        # Apply label smoothing
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(inputs, targets)
```

## 5. Ensemble Methods

### Model Averaging
```python
# Train multiple models with different seeds
models = []
for seed in [42, 123, 456, 789, 999]:
    set_seed(seed)
    model = train_model(seed=seed)
    models.append(model)

# Average predictions
def ensemble_predict(models, data_loader, device):
    all_probs = []
    for model in models:
        model.eval()
        probs = []
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(device)
                logits = model(batch)
                probs.append(torch.sigmoid(logits).cpu())
        all_probs.append(torch.cat(probs))
    
    # Average probabilities
    avg_probs = torch.stack(all_probs).mean(dim=0)
    return avg_probs
```

### Cross-Validation Ensemble
```python
# Use k-fold cross-validation
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
models = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
    train_data = [dataset[i] for i in train_idx]
    val_data = [dataset[i] for i in val_idx]
    
    model = train_model(train_data, val_data)
    models.append(model)
```

## 6. Temporal Decay System Investigation

### Analyze Decay Patterns
```python
# Investigate the temporal decay system
def analyze_temporal_patterns():
    """Analyze how temporal decay affects predictions."""
    
    # 1. Check decay values distribution
    decay_values = []
    for graph in dataset:
        for edge_attr in graph['fact', 'mentions', 'company'].edge_attr:
            decay_values.append(edge_attr[1].item())  # decay weight
    
    print(f"Decay stats: mean={np.mean(decay_values):.3f}, "
          f"std={np.std(decay_values):.3f}")
    
    # 2. Check correlation between decay and predictions
    # 3. Try different decay functions
    # 4. Analyze temporal patterns in predictions
```

### Try Different Decay Functions
```python
# Test different decay types in EdgeDecay.py
decay_types = ["linear", "exponential", "logarithmic", "sigmoid", "quadratic"]

for decay_type in decay_types:
    # Train with different decay functions
    pass
```

## 7. Feature Engineering

### Temporal Features
```python
# Add more temporal features to nodes
def add_temporal_features(graph):
    """Add temporal features to fact nodes."""
    
    # 1. Days since earliest fact
    # 2. Days until latest fact
    # 3. Temporal position (0-1)
    # 4. Seasonal features
    
    return graph
```

### Sentiment Aggregation
```python
# Aggregate sentiment features
def aggregate_sentiment_features(graph):
    """Create aggregated sentiment features."""
    
    # 1. Mean sentiment per company
    # 2. Sentiment variance
    # 3. Sentiment trend (positive/negative)
    
    return graph
```

## 8. Hyperparameter Optimization

### Bayesian Optimization
```python
# Use Optuna for hyperparameter optimization
import optuna

def objective(trial):
    # Define hyperparameter search space
    hidden_channels = trial.suggest_categorical('hidden_channels', [64, 128, 256])
    num_layers = trial.suggest_int('num_layers', 1, 4)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    
    # Train model with these hyperparameters
    model = train_model(hidden_channels=hidden_channels, 
                       num_layers=num_layers, lr=lr, weight_decay=weight_decay)
    
    # Return validation F1 score
    return validation_f1_score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

## 9. Data Quality Improvements

### Label Consistency Check
```python
# Check for label inconsistencies
def check_label_consistency():
    """Check if same ticker has consistent labels."""
    
    ticker_labels = {}
    for graph, sg in zip(dataset, raw_sg):
        ticker = sg.primary_ticker
        label = graph.y.item()
        
        if ticker not in ticker_labels:
            ticker_labels[ticker] = []
        ticker_labels[ticker].append(label)
    
    # Check for inconsistencies
    inconsistencies = []
    for ticker, labels in ticker_labels.items():
        if len(set(labels)) > 1:
            inconsistencies.append((ticker, labels))
    
    print(f"Found {len(inconsistencies)} tickers with inconsistent labels")
    return inconsistencies
```

### Feature Selection
```python
# Use feature importance to select best features
from sklearn.feature_selection import SelectKBest, f_classif

def select_best_features(X, y, k=20):
    """Select k best features using ANOVA F-test."""
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    return X_selected, selector.get_support()
```

## 10. Evaluation Improvements

### Stratified Metrics
```python
# Calculate metrics per ticker/company
def stratified_evaluation(model, test_loader, device):
    """Evaluate model performance per company."""
    
    company_metrics = {}
    
    for batch in test_loader:
        # Get predictions
        # Group by company
        # Calculate metrics per company
    
    return company_metrics
```

### Confidence Calibration
```python
# Calibrate model confidence
from sklearn.calibration import CalibratedClassifierCV

def calibrate_confidence(model, val_loader, device):
    """Calibrate model confidence using validation set."""
    
    # Get predictions on validation set
    # Use Platt scaling or isotonic regression
    # Return calibrated model
```

## Implementation Priority

**High Priority:**
1. Threshold optimization (quick win)
2. Different readout methods (easy to implement)
3. Model ensemble (significant improvement potential)

**Medium Priority:**
4. Hyperparameter optimization
5. Advanced loss functions
6. Data augmentation

**Low Priority:**
7. Feature engineering
8. Data quality improvements
9. Evaluation improvements

## Expected Improvements

- **Threshold optimization**: +5-10% F1 improvement
- **Model ensemble**: +10-15% F1 improvement
- **Hyperparameter optimization**: +5-10% F1 improvement
- **Combined approaches**: +20-30% F1 improvement potential

## Quick Start Commands

```bash
# 1. Test different thresholds
python -c "
from run import run_training
model, metrics, history = run_training(threshold=0.2)
print(f'Threshold 0.2: F1={metrics[\"f1\"]:.3f}')
"

# 2. Test different readout methods
python -c "
for readout in ['fact', 'company', 'concat', 'gated']:
    model, metrics, history = run_training(readout=readout)
    print(f'Readout {readout}: F1={metrics[\"f1\"]:.3f}')
"

# 3. Train ensemble
python ensemble_training.py
``` 