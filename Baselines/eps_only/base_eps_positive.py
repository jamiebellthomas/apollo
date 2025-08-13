#!/usr/bin/env python3
"""
EPS-based baseline classifier for POSITIVE EPS cases only.
Heuristic: Always predict positive for positive EPS cases (matching training data).
"""

import json
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    """Load data from JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def eps_baseline_predict(data):
    """Make predictions based on EPS surprise heuristic for positive EPS cases only."""
    predictions = []
    true_labels = []
    eps_surprises = []
    
    # Filter to only include positive EPS cases (matching training data)
    filtered_data = [item for item in data if item['eps_surprise'] is not None and item['eps_surprise'] > 0]
    
    print(f"Filtered to {len(filtered_data)} positive EPS cases (out of {len(data)} total)")
    
    for item in filtered_data:
        eps_surprise = item['eps_surprise']
        true_label = item['label']
        
        # Since we're only looking at positive EPS cases, all predictions will be 1
        # But we can still evaluate against the true labels
        prediction = 1  # Always predict positive for positive EPS cases
        
        predictions.append(prediction)
        true_labels.append(true_label)
        eps_surprises.append(eps_surprise)
    
    return np.array(predictions), np.array(true_labels), np.array(eps_surprises)

def evaluate_predictions(y_true, y_pred):
    """Evaluate predictions and print metrics."""
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Print results
    print("=" * 50)
    print("EPS BASELINE RESULTS - POSITIVE EPS CASES ONLY")
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
    plt.title('Confusion Matrix - EPS Baseline (Positive EPS Only)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix plot saved to {save_path}")
    
    plt.show()

def analyze_eps_distribution(eps_surprises, y_true):
    """Analyze the distribution of EPS surprises by true label."""
    eps_positive = eps_surprises[y_true == 1]
    eps_negative = eps_surprises[y_true == 0]
    
    print("EPS Surprise Distribution Analysis (Positive EPS Cases Only):")
    print(f"Positive labels (1): {len(eps_positive)} samples")
    print(f"  Mean EPS surprise: {np.mean(eps_positive):.4f}")
    print(f"  Std EPS surprise:  {np.std(eps_positive):.4f}")
    print(f"  Min EPS surprise:  {np.min(eps_positive):.4f}")
    print(f"  Max EPS surprise:  {np.max(eps_positive):.4f}")
    print()
    
    print(f"Negative labels (0): {len(eps_negative)} samples")
    print(f"  Mean EPS surprise: {np.mean(eps_negative):.4f}")
    print(f"  Std EPS surprise:  {np.std(eps_negative):.4f}")
    print(f"  Min EPS surprise:  {np.min(eps_negative):.4f}")
    print(f"  Max EPS surprise:  {np.max(eps_negative):.4f}")
    print()

def main():
    """Main function to run the EPS baseline on positive EPS cases only."""
    # Load data
    print("Loading data from Data/subgraphs.jsonl...")
    data = load_data('Data/subgraphs.jsonl')
    print(f"Loaded {len(data)} samples")
    print()
    
    # Make predictions
    print("Making predictions using EPS surprise heuristic...")
    predictions, true_labels, eps_surprises = eps_baseline_predict(data)
    
    # Evaluate predictions
    cm, accuracy, precision, recall, f1 = evaluate_predictions(true_labels, predictions)
    
    # Analyze EPS distribution
    analyze_eps_distribution(eps_surprises, true_labels)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, 'Baselines/positive_eps_only/eps_baseline_positive_confusion_matrix.png')
    
    # Save results to file
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'total_samples': int(len(data)),
        'filtered_samples': int(len(predictions)),
        'positive_predictions': int(np.sum(predictions == 1)),
        'negative_predictions': int(np.sum(predictions == 0)),
        'true_positives': int(np.sum((predictions == 1) & (true_labels == 1))),
        'true_negatives': int(np.sum((predictions == 0) & (true_labels == 0))),
        'false_positives': int(np.sum((predictions == 1) & (true_labels == 0))),
        'false_negatives': int(np.sum((predictions == 0) & (true_labels == 1)))
    }
    
    with open('Baselines/positive_eps_only/eps_baseline_positive_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Results saved to Baselines/positive_eps_only/eps_baseline_positive_results.json")

if __name__ == "__main__":
    main() 