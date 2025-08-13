#!/usr/bin/env python3
"""
Neural Network baseline classifier for POSITIVE EPS cases only.
Uses simple features extracted from the data.
Matches training data conditions by filtering to positive EPS cases.
"""

import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

class SimpleNN(nn.Module):
    """Simple neural network for binary classification with batch normalization."""
    def __init__(self, input_size, hidden_size=64, num_hidden_layers=2, dropout_rate=0.3):
        super(SimpleNN, self).__init__()
        
        layers = []
        current_size = input_size
        
        # Build hidden layers
        for i in range(num_hidden_layers):
            layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            current_size = hidden_size
            # Reduce hidden size for subsequent layers
            if i < num_hidden_layers - 1:
                hidden_size = hidden_size // 2
        
        # Output layer (no sigmoid since we use BCEWithLogitsLoss)
        layers.append(nn.Linear(current_size, 1))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

def load_data(file_path):
    """Load data from JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def extract_simple_features(data):
    """Extract simple features from positive EPS cases only."""
    features = []
    labels = []
    eps_surprises = []
    valid_samples = 0
    
    print("Extracting simple features from positive EPS cases only...")
    
    # Filter to positive EPS cases first
    positive_eps_data = [item for item in data if item.get('eps_surprise') is not None and item['eps_surprise'] > 0]
    print(f"Filtered to {len(positive_eps_data)} positive EPS cases (out of {len(data)} total)")
    
    for i, item in enumerate(positive_eps_data):
        if i % 500 == 0:
            print(f"Processing sample {i}/{len(positive_eps_data)}")
        
        try:
            # Extract basic features that we can get from the data
            eps_surprise = item.get('eps_surprise', 0.0)
            if eps_surprise is None:
                eps_surprise = 0.0
            
            fact_count = item.get('fact_count', 0)
            fact_list = item.get('fact_list', [])
            
            # Calculate some simple features from facts
            total_sentiment = sum(f.get('sentiment', 0.0) for f in fact_list)
            avg_sentiment = total_sentiment / len(fact_list) if fact_list else 0.0
            
            # Count positive and negative facts
            positive_facts = sum(1 for f in fact_list if f.get('sentiment', 0.0) > 0)
            negative_facts = sum(1 for f in fact_list if f.get('sentiment', 0.0) < 0)
            
            # Create a simple feature vector
            feature_vector = [
                eps_surprise,  # EPS surprise
                fact_count,    # Number of facts
                total_sentiment,  # Total sentiment
                avg_sentiment,    # Average sentiment
                positive_facts,   # Number of positive facts
                negative_facts,   # Number of negative facts
                len(fact_list) - positive_facts - negative_facts,  # Neutral facts
                max(0, eps_surprise),  # Positive EPS surprise
                min(0, eps_surprise),  # Negative EPS surprise
                abs(eps_surprise),     # Absolute EPS surprise
            ]
            
            features.append(feature_vector)
            labels.append(item['label'])
            eps_surprises.append(eps_surprise)
            valid_samples += 1
                
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    print(f"Successfully extracted features for {valid_samples}/{len(positive_eps_data)} positive EPS samples")
    
    return np.array(features), np.array(labels), np.array(eps_surprises)

def train_model(X_train, y_train, X_val, y_val, input_size, epochs=100, batch_size=32, lr=0.001, num_hidden_layers=2):
    """Train the neural network model."""
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Calculate class weights for weighted BCE loss
    num_positive = np.sum(y_train == 1)
    num_negative = np.sum(y_train == 0)
    total_samples = len(y_train)
    
    # Weight positive class more heavily to address imbalance
    pos_weight = num_negative / num_positive  # This gives more weight to positive class
    print(f"Class distribution: {num_negative} negative, {num_positive} positive")
    print(f"Positive class weight: {pos_weight:.2f}")
    
    # Initialize model
    model = SimpleNN(input_size=input_size, num_hidden_layers=num_hidden_layers)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))  # Use BCEWithLogitsLoss for weighted loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    print("Training neural network...")
    print(f"Architecture: {input_size} -> {num_hidden_layers} hidden layers -> 1 output")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss)
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_losses[-1]:.4f}, Val Loss = {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model, train_losses, val_losses

def evaluate_predictions(y_true, y_pred, y_prob):
    """Evaluate predictions and print metrics."""
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Print results
    print("=" * 50)
    print("NEURAL NETWORK BASELINE RESULTS - POSITIVE EPS CASES ONLY")
    print("=" * 50)
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print()
    
    print("Confusion Matrix:")
    print("                 Predicted")
    print("                0    1")
    print(f"Actual 0    {cm[0,0]:4d} {cm[0,1]:4d}")
    print(f"       1    {cm[1,0]:4d} {cm[1,1]:4d}")
    print()
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("Detailed Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print()
    
    # Print classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))
    
    return cm, accuracy, precision, recall, f1

def plot_confusion_matrix(cm, save_path=None):
    """Plot confusion matrix as a heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix - Neural Network Baseline (Positive EPS Only)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix plot saved to {save_path}")
    
    plt.show()

def plot_training_history(train_losses, val_losses, save_path=None):
    """Plot training history."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', alpha=0.7)
    plt.plot(val_losses, label='Validation Loss', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History - Neural Network Baseline (Positive EPS Only)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()

def main():
    """Main function to run the neural network baseline on positive EPS cases only."""
    # Load data
    print("Loading data from Data/subgraphs.jsonl...")
    data = load_data('Data/subgraphs.jsonl')
    print(f"Loaded {len(data)} samples")
    print()
    
    # Extract features (already filtered to positive EPS cases)
    features, labels, eps_surprises = extract_simple_features(data)
    
    if len(features) == 0:
        print("No valid features extracted. Exiting.")
        return
    
    print(f"Feature matrix shape: {features.shape}")
    print(f"Number of features per sample: {features.shape[1]}")
    print()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print()
    
    # Train model
    input_size = features.shape[1]
    num_hidden_layers = 3  # Configurable: try different values like 2, 3, 4
    model, train_losses, val_losses = train_model(
        X_train_scaled, y_train, X_test_scaled, y_test, input_size, num_hidden_layers=num_hidden_layers
    )
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        logits = model(X_test_tensor)
        probabilities = torch.sigmoid(logits).numpy().flatten()
        predictions = (probabilities > 0.5).astype(int)
    
    # Evaluate predictions
    cm, accuracy, precision, recall, f1 = evaluate_predictions(y_test, predictions, probabilities)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, 'Baselines/neural_net/positive_eps_only/nn_baseline_positive_confusion_matrix.png')
    
    # Plot training history
    plot_training_history(train_losses, val_losses, 'Baselines/neural_net/positive_eps_only/nn_baseline_positive_training_history.png')
    
    # Save results to file
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'total_samples': int(len(data)),
        'positive_eps_samples': int(len(features)),
        'training_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'input_size': int(input_size),
        'num_hidden_layers': int(num_hidden_layers),
        'positive_predictions': int(np.sum(predictions == 1)),
        'negative_predictions': int(np.sum(predictions == 0)),
        'true_positives': int(np.sum((predictions == 1) & (y_test == 1))),
        'true_negatives': int(np.sum((predictions == 0) & (y_test == 0))),
        'false_positives': int(np.sum((predictions == 1) & (y_test == 0))),
        'false_negatives': int(np.sum((predictions == 0) & (y_test == 1)))
    }
    
    with open('Baselines/neural_net/positive_eps_only/nn_baseline_positive_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Results saved to Baselines/neural_net/positive_eps_only/nn_baseline_positive_results.json")

if __name__ == "__main__":
    main() 