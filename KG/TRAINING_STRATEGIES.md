# Training Strategies to Reduce Validation Loss

## Current Settings (Recommended)
```python
hidden_channels=64,      # More model capacity
num_layers=2,           # Deeper model
feature_dropout=0.3,    # Moderate regularization
edge_dropout=0.2,       # Moderate regularization
final_dropout=0.2,      # Moderate regularization
batch_size=32,          # Better gradient estimates
lr=5e-4,               # Stable learning rate
weight_decay=1e-4,      # Light regularization
lr_scheduler="cosine",  # Gradual LR decay
```

## Strategy 1: Increase Model Capacity
```python
hidden_channels=128,    # Double the capacity
num_layers=3,          # Even deeper
feature_dropout=0.2,   # Less dropout for more capacity
edge_dropout=0.1,
final_dropout=0.1,
```

## Strategy 2: Reduce Regularization
```python
feature_dropout=0.1,    # Much less dropout
edge_dropout=0.0,       # No edge dropout
final_dropout=0.1,      # Minimal dropout
weight_decay=1e-5,      # Very light regularization
```

## Strategy 3: Optimize Learning Rate
```python
lr=1e-3,               # Higher initial LR
lr_scheduler="plateau", # Reduce LR when validation plateaus
lr_gamma=0.3,          # More aggressive LR reduction
```

## Strategy 4: Try Different Loss Functions
```python
# Option A: Weighted BCE
loss_type="weighted_bce"

# Option B: Standard BCE with higher pos_weight
loss_type="bce"
# (pos_weight will be computed automatically)

# Option C: Focal Loss with different gamma
# Modify FocalLoss class to use gamma=1.0 instead of 2.0
```

## Strategy 5: Data Augmentation
```python
# Increase batch size for better gradient estimates
batch_size=16,         # Smaller batches, more updates

# Or use larger batches with gradient accumulation
batch_size=64,         # Larger batches
```

## Strategy 6: Architecture Changes
```python
# Try different readout strategies
readout="concat",      # Concatenate fact and company features
readout="gated",       # Use gated attention
readout="company",     # Use only company features
```

## Strategy 7: Learning Rate Scheduling
```python
# Option A: Step decay
lr_scheduler="step",
lr_step_size=15,       # Reduce LR every 15 epochs
lr_gamma=0.5,          # Halve the LR

# Option B: Cosine annealing (current)
lr_scheduler="cosine",

# Option C: Plateau-based
lr_scheduler="plateau",
lr_gamma=0.3,          # Reduce by 70% when plateauing
```

## Quick Test Scripts

### Test Model Capacity
```python
model, metrics, history = run_training(
    hidden_channels=128,
    num_layers=3,
    feature_dropout=0.1,
    edge_dropout=0.0,
    final_dropout=0.1,
    lr=3e-4,
    weight_decay=1e-5,
    epochs=30,
    early_stopping=True,
    patience=15,
)
```

### Test Lower Regularization
```python
model, metrics, history = run_training(
    feature_dropout=0.1,
    edge_dropout=0.0,
    final_dropout=0.1,
    weight_decay=1e-5,
    lr_scheduler="plateau",
    lr_gamma=0.3,
    epochs=30,
    early_stopping=True,
    patience=15,
)
```

### Test Higher Learning Rate
```python
model, metrics, history = run_training(
    lr=1e-3,
    lr_scheduler="step",
    lr_step_size=10,
    lr_gamma=0.5,
    batch_size=16,
    epochs=30,
    early_stopping=True,
    patience=15,
)
```

## Monitoring Tips

1. **Watch the learning rate**: Should decrease gradually
2. **Check train vs val loss gap**: If train loss is much lower, you're overfitting
3. **Monitor AUC**: Often more stable than accuracy
4. **Look for convergence**: Loss should stabilize after ~20-30 epochs

## Common Issues & Solutions

### High Validation Loss, Low Training Loss
- **Problem**: Overfitting
- **Solution**: Increase dropout, weight decay, or reduce model capacity

### Both Losses High
- **Problem**: Underfitting
- **Solution**: Increase model capacity, reduce regularization, or train longer

### Unstable Training
- **Problem**: Learning rate too high
- **Solution**: Reduce learning rate or use better scheduler

### Slow Convergence
- **Problem**: Learning rate too low
- **Solution**: Increase learning rate or use warmup

## Recommended Next Steps

1. **Try Strategy 1** (Increase Model Capacity) first
2. **If overfitting**: Use Strategy 2 (Reduce Regularization)
3. **If underfitting**: Use Strategy 3 (Optimize Learning Rate)
4. **Experiment with different readout strategies**
5. **Monitor and adjust based on results** 