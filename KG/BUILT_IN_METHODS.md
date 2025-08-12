# Built-in PyTorch Methods for Model Improvement

## 🎯 Available Built-in Loss Functions

### 1. **BCEWithLogitsLoss with pos_weight** ✅ (Currently Using)
```python
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```
- **Pros**: Built-in, stable, handles class imbalance
- **Cons**: May not be optimal for all cases

### 2. **BCEWithLogitsLoss with Label Smoothing**
```python
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, label_smoothing=0.1)
```
- **Pros**: Reduces overconfidence, better generalization
- **Cons**: May reduce performance on easy examples

### 3. **Focal Loss (Custom Implementation)**
```python
criterion = FocalLoss(alpha=pos_weight, gamma=2.0)
```
- **Pros**: Focuses on hard examples, handles class imbalance
- **Cons**: Custom implementation, more complex

## 🚀 Available Built-in Optimizers

### 1. **Adam** ✅ (Currently Using)
```python
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
```
- **Pros**: Adaptive learning rates, good for most cases
- **Cons**: May converge to suboptimal solutions

### 2. **AdamW**
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
```
- **Pros**: Better weight decay implementation, often better generalization
- **Cons**: May need different hyperparameters

### 3. **SGD with Momentum**
```python
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
```
- **Pros**: Often better final performance, more stable
- **Cons**: Requires careful learning rate tuning

### 4. **RMSprop**
```python
optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
```
- **Pros**: Good for non-convex optimization
- **Cons**: May be less stable than Adam

## 📈 Available Built-in Learning Rate Schedulers

### 1. **ReduceLROnPlateau** ✅ (Currently Using)
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
```
- **Pros**: Reduces LR when validation doesn't improve
- **Cons**: May reduce LR too early

### 2. **StepLR**
```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
```
- **Pros**: Simple, predictable
- **Cons**: May not adapt to actual training progress

### 3. **CosineAnnealingLR**
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
```
- **Pros**: Smooth LR reduction, often better final performance
- **Cons**: May need more epochs to converge

### 4. **OneCycleLR**
```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr*10, epochs=epochs, steps_per_epoch=len(train_loader))
```
- **Pros**: Super-convergence, often faster training
- **Cons**: Requires careful tuning, may be unstable

### 5. **ExponentialLR**
```python
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
```
- **Pros**: Smooth exponential decay
- **Cons**: May decay too quickly

## 🔧 Other Built-in Features

### 1. **Gradient Clipping**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```
- **Pros**: Prevents gradient explosion
- **Cons**: May slow down training

### 2. **Mixed Precision Training**
```python
from torch.cuda.amp import autocast, GradScaler
```
- **Pros**: Faster training, less memory usage
- **Cons**: May reduce numerical precision

### 3. **DataParallel/DistributedDataParallel**
```python
model = torch.nn.DataParallel(model)
```
- **Pros**: Multi-GPU training
- **Cons**: Communication overhead

## 🧪 Quick Test Commands

### Test Label Smoothing:
```bash
python -c "from run import run_training; run_training(loss_type='bce_label_smooth', epochs=10)"
```

### Test AdamW:
```bash
python -c "from run import run_training; run_training(optimizer_type='adamw', epochs=10)"
```

### Test SGD:
```bash
python -c "from run import run_training; run_training(optimizer_type='sgd', lr=1e-4, epochs=10)"
```

### Test OneCycleLR:
```bash
python -c "from run import run_training; run_training(lr_scheduler='one_cycle', epochs=10)"
```

## 📊 Expected Performance Ranking

Based on typical performance for imbalanced datasets:

1. **AdamW + Label Smoothing** (Best generalization)
2. **SGD + CosineAnnealingLR** (Best final performance)
3. **Adam + OneCycleLR** (Fastest convergence)
4. **Adam + ReduceLROnPlateau** (Current setup)
5. **RMSprop + StepLR** (Most stable)

## 🎯 Recommended Next Steps

1. **Try AdamW + Label Smoothing** (most likely to improve)
2. **Try SGD + CosineAnnealingLR** (best final performance)
3. **Try OneCycleLR** (fastest training)
4. **Experiment with different thresholds** (0.3, 0.4, 0.5, 0.6)
5. **Try ensemble methods** (combine multiple models) 