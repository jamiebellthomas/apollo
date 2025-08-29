#!/usr/bin/env python3
"""
Weighted Sentiment-based baseline classifier.
Uses EPS surprise and time-weighted average sentiment to predict stock price movements.
More recent articles are given higher weight in the sentiment calculation.
"""

import json
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import math

def load_data(file_path):
    """Load data from JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def calculate_weighted_sentiment(fact_list, max_days=90):
    """
    Calculate time-weighted average sentiment from fact list.
    
    Args:
        fact_list: List of facts with sentiment and delta_days
        max_days: Maximum days for linear scaling (default 90)
    
    Returns:
        Weighted average sentiment
    """
    if not fact_list:
        return 0.0
    
    total_weighted_sentiment = 0.0
    total_weight = 0.0
    
    for fact in fact_list:
        sentiment = fact.get('sentiment', 0.0)
        delta_days = fact.get('delta_days', 0)
        
        # Calculate weight using linear scaling: 1 - d/90
        # More recent articles (smaller delta_days) get higher weight
        # Articles older than 90 days get weight 0
        if delta_days >= max_days:
            weight = 0.0
        else:
            weight = 1.0 - (delta_days / max_days)
        
        total_weighted_sentiment += sentiment * weight
        total_weight += weight
    
    if total_weight == 0:
        return 0.0
    
    return total_weighted_sentiment / total_weight

def weighted_sentiment_baseline_predict(data, max_days=90):
    """Make predictions based on EPS surprise and weighted average sentiment."""
    predictions = []
    true_labels = []
    eps_surprises = []
    weighted_sentiments = []
    
    print(f"Making predictions using EPS surprise + weighted sentiment (max_days={max_days})...")
    
    for item in data:
        eps_surprise = item.get('eps_surprise', 0.0)
        if eps_surprise is None:
            eps_surprise = 0.0
        
        fact_list = item.get('fact_list', [])
        weighted_sentiment = calculate_weighted_sentiment(fact_list, max_days)
        true_label = item['label']
        
        # Prediction logic:
        # If eps_surprise is negative -> predict 0
        # If eps_surprise is positive and weighted_sentiment is negative -> predict 0  
        # If eps_surprise is positive and weighted_sentiment is positive -> predict 1
        if eps_surprise <= 0:
            prediction = 0
        elif weighted_sentiment <= 0:
            prediction = 0
        else:
            prediction = 1
        
        predictions.append(prediction)
        true_labels.append(true_label)
        eps_surprises.append(eps_surprise)
        weighted_sentiments.append(weighted_sentiment)
    
    return np.array(predictions), np.array(true_labels), np.array(eps_surprises), np.array(weighted_sentiments)

def weighted_sentiment_baseline_predict_positive_eps_only(data, max_days=90):
    """Make predictions for positive EPS cases only using weighted sentiment."""
    predictions = []
    true_labels = []
    eps_surprises = []
    weighted_sentiments = []
    
    # Filter to only include positive EPS cases
    filtered_data = [item for item in data if item.get('eps_surprise', 0.0) is not None and item.get('eps_surprise', 0.0) > 0]
    
    print(f"Filtered to {len(filtered_data)} positive EPS cases (out of {len(data)} total)")
    print(f"Using weighted sentiment with max_days={max_days}")
    
    for item in filtered_data:
        eps_surprise = item.get('eps_surprise', 0.0)
        fact_list = item.get('fact_list', [])
        weighted_sentiment = calculate_weighted_sentiment(fact_list, max_days)
        true_label = item['label']
        
        # For positive EPS cases only:
        # If weighted_sentiment is negative -> predict 0
        # If weighted_sentiment is positive -> predict 1
        if weighted_sentiment <= 0:
            prediction = 0
        else:
            prediction = 1
        
        predictions.append(prediction)
        true_labels.append(true_label)
        eps_surprises.append(eps_surprise)
        weighted_sentiments.append(weighted_sentiment)
    
    return np.array(predictions), np.array(true_labels), np.array(eps_surprises), np.array(weighted_sentiments)

def weighted_sentiment_baseline_predict_threshold(data, threshold=0.2, max_days=90):
    """Make predictions using weighted sentiment threshold."""
    predictions = []
    true_labels = []
    eps_surprises = []
    weighted_sentiments = []
    
    print(f"Making predictions using EPS surprise + weighted sentiment threshold {threshold} (max_days={max_days})...")
    
    for item in data:
        eps_surprise = item.get('eps_surprise', 0.0)
        if eps_surprise is None:
            eps_surprise = 0.0
        
        fact_list = item.get('fact_list', [])
        weighted_sentiment = calculate_weighted_sentiment(fact_list, max_days)
        true_label = item['label']
        
        # Prediction logic with weighted sentiment threshold:
        # If eps_surprise is negative -> predict 0
        # If eps_surprise is positive and weighted_sentiment <= threshold -> predict 0  
        # If eps_surprise is positive and weighted_sentiment > threshold -> predict 1
        if eps_surprise <= 0:
            prediction = 0
        elif weighted_sentiment <= threshold:
            prediction = 0
        else:
            prediction = 1
        
        predictions.append(prediction)
        true_labels.append(true_label)
        eps_surprises.append(eps_surprise)
        weighted_sentiments.append(weighted_sentiment)
    
    return np.array(predictions), np.array(true_labels), np.array(eps_surprises), np.array(weighted_sentiments)

def weighted_sentiment_baseline_predict_threshold_positive_eps_only(data, threshold=0.1, max_days=90):
    """Make predictions using weighted sentiment threshold for positive EPS cases only."""
    predictions = []
    true_labels = []
    eps_surprises = []
    weighted_sentiments = []
    
    # Filter to only include positive EPS cases
    filtered_data = [item for item in data if item.get('eps_surprise', 0.0) is not None and item.get('eps_surprise', 0.0) > 0]
    
    print(f"Filtered to {len(filtered_data)} positive EPS cases (out of {len(data)} total)")
    print(f"Making predictions using weighted sentiment threshold {threshold} (max_days={max_days})...")
    
    for item in filtered_data:
        eps_surprise = item.get('eps_surprise', 0.0)
        fact_list = item.get('fact_list', [])
        weighted_sentiment = calculate_weighted_sentiment(fact_list, max_days)
        true_label = item['label']
        
        # Prediction logic with weighted sentiment threshold for positive EPS only:
        # If weighted_sentiment <= threshold -> predict 0
        # If weighted_sentiment > threshold -> predict 1
        if weighted_sentiment <= threshold:
            prediction = 0
        else:
            prediction = 1
        
        predictions.append(prediction)
        true_labels.append(true_label)
        eps_surprises.append(eps_surprise)
        weighted_sentiments.append(weighted_sentiment)
    
    return np.array(predictions), np.array(true_labels), np.array(eps_surprises), np.array(weighted_sentiments)

def evaluate_predictions(y_true, y_pred, title):
    """Evaluate predictions and print metrics."""
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Print results
    print("=" * 50)
    print(f"WEIGHTED SENTIMENT BASELINE RESULTS - {title}")
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

def plot_confusion_matrix(cm, title, save_path=None):
    """Plot confusion matrix as a heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(f'Confusion Matrix - Weighted Sentiment Baseline ({title})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix plot saved to {save_path}")
    
    plt.show()

def analyze_weighted_sentiment_distribution(eps_surprises, weighted_sentiments, y_true, title):
    """Analyze the distribution of EPS surprises and weighted sentiments by true label."""
    eps_positive = eps_surprises[y_true == 1]
    eps_negative = eps_surprises[y_true == 0]
    sent_positive = weighted_sentiments[y_true == 1]
    sent_negative = weighted_sentiments[y_true == 0]
    
    print(f"Distribution Analysis - {title}:")
    print(f"Positive labels (1): {len(eps_positive)} samples")
    print(f"  Mean EPS surprise: {np.mean(eps_positive):.4f}")
    print(f"  Mean weighted sentiment: {np.mean(sent_positive):.4f}")
    print(f"  Std EPS surprise: {np.std(eps_positive):.4f}")
    print(f"  Std weighted sentiment: {np.std(sent_positive):.4f}")
    print()
    
    print(f"Negative labels (0): {len(eps_negative)} samples")
    print(f"  Mean EPS surprise: {np.mean(eps_negative):.4f}")
    print(f"  Mean weighted sentiment: {np.mean(sent_negative):.4f}")
    print(f"  Std EPS surprise: {np.std(eps_negative):.4f}")
    print(f"  Std weighted sentiment: {np.std(sent_negative):.4f}")
    print()

def plot_weight_function(max_days=90):
    """Plot the linear weight function to visualize how weights change over time."""
    days = np.arange(0, max_days + 1)
    weights = np.where(days < max_days, 1.0 - (days / max_days), 0.0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(days, weights, 'b-', linewidth=2)
    plt.xlabel('Days Before Earnings Report')
    plt.ylabel('Weight')
    plt.title(f'Linear Weight Function (max_days={max_days})')
    plt.grid(True, alpha=0.3)
    
    # Add some reference points
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Weight = 0.5')
    plt.axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='Weight = 0.1')
    
    # Find days where weight crosses these thresholds
    half_weight_days = max_days * 0.5
    tenth_weight_days = max_days * 0.9
    
    plt.axvline(x=half_weight_days, color='r', linestyle=':', alpha=0.7, label=f'Half-weight: {half_weight_days:.1f} days')
    plt.axvline(x=tenth_weight_days, color='orange', linestyle=':', alpha=0.7, label=f'Tenth-weight: {tenth_weight_days:.1f} days')
    
    plt.legend()
    plt.tight_layout()
    
    save_path = f'Baselines/weighted_sentiment/weight_function_max_days_{max_days}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Weight function plot saved to {save_path}")
    plt.show()

def main():
    """Main function to run the weighted sentiment baseline."""
    # Load data
    print("Loading data from Data/subgraphs.jsonl...")
    data = load_data('Data/subgraphs.jsonl')
    print(f"Loaded {len(data)} samples")
    print()
    
    # Set max_days for time weighting
    max_days = 90  # Linear scaling over 90 days
    
    # Plot weight function for visualization
    print("Plotting weight function...")
    plot_weight_function(max_days)
    print()
    
    # Experiment 1: All data
    print("EXPERIMENT 1: ALL DATA")
    print("=" * 50)
    predictions, true_labels, eps_surprises, weighted_sentiments = weighted_sentiment_baseline_predict(data, max_days)
    cm, accuracy, precision, recall, f1 = evaluate_predictions(true_labels, predictions, "ALL DATA")
    analyze_weighted_sentiment_distribution(eps_surprises, weighted_sentiments, true_labels, "ALL DATA")
    plot_confusion_matrix(cm, "ALL DATA", f'Baselines/weighted_sentiment/all_data/weighted_sentiment_baseline_all_confusion_matrix_max_days_{max_days}.png')
    
    # Save results for all data
    results_all = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'total_samples': int(len(data)),
        'max_days': int(max_days),
        'positive_predictions': int(np.sum(predictions == 1)),
        'negative_predictions': int(np.sum(predictions == 0)),
        'true_positives': int(np.sum((predictions == 1) & (true_labels == 1))),
        'true_negatives': int(np.sum((predictions == 0) & (true_labels == 0))),
        'false_positives': int(np.sum((predictions == 1) & (true_labels == 0))),
        'false_negatives': int(np.sum((predictions == 0) & (true_labels == 1)))
    }
    
    with open(f'Baselines/weighted_sentiment/all_data/weighted_sentiment_baseline_all_results_max_days_{max_days}.json', 'w') as f:
        json.dump(results_all, f, indent=2)
    
    print(f"Results saved to Baselines/weighted_sentiment/all_data/weighted_sentiment_baseline_all_results_max_days_{max_days}.json")
    print()
    
    # Experiment 2: Positive EPS only
    print("EXPERIMENT 2: POSITIVE EPS ONLY")
    print("=" * 50)
    predictions_pos, true_labels_pos, eps_surprises_pos, weighted_sentiments_pos = weighted_sentiment_baseline_predict_positive_eps_only(data, max_days)
    cm_pos, accuracy_pos, precision_pos, recall_pos, f1_pos = evaluate_predictions(true_labels_pos, predictions_pos, "POSITIVE EPS ONLY")
    analyze_weighted_sentiment_distribution(eps_surprises_pos, weighted_sentiments_pos, true_labels_pos, "POSITIVE EPS ONLY")
    plot_confusion_matrix(cm_pos, "POSITIVE EPS ONLY", f'Baselines/weighted_sentiment/positive_eps_only/weighted_sentiment_baseline_positive_confusion_matrix_max_days_{max_days}.png')
    
    # Save results for positive EPS only
    results_pos = {
        'accuracy': float(accuracy_pos),
        'precision': float(precision_pos),
        'recall': float(recall_pos),
        'f1_score': float(f1_pos),
        'confusion_matrix': cm_pos.tolist(),
        'total_samples': int(len(data)),
        'filtered_samples': int(len(predictions_pos)),
        'max_days': int(max_days),
        'positive_predictions': int(np.sum(predictions_pos == 1)),
        'negative_predictions': int(np.sum(predictions_pos == 0)),
        'true_positives': int(np.sum((predictions_pos == 1) & (true_labels_pos == 1))),
        'true_negatives': int(np.sum((predictions_pos == 0) & (true_labels_pos == 0))),
        'false_positives': int(np.sum((predictions_pos == 1) & (true_labels_pos == 0))),
        'false_negatives': int(np.sum((predictions_pos == 0) & (true_labels_pos == 1)))
    }
    
    with open(f'Baselines/weighted_sentiment/positive_eps_only/weighted_sentiment_baseline_positive_results_max_days_{max_days}.json', 'w') as f:
        json.dump(results_pos, f, indent=2)
    
    print(f"Results saved to Baselines/weighted_sentiment/positive_eps_only/weighted_sentiment_baseline_positive_results_max_days_{max_days}.json")
    print()
    
    # Experiment 3: Weighted sentiment threshold 0.1 for positive EPS only
    print("EXPERIMENT 3: WEIGHTED SENTIMENT THRESHOLD 0.1 (POSITIVE EPS ONLY)")
    print("=" * 50)
    predictions_thresh_pos, true_labels_thresh_pos, eps_surprises_thresh_pos, weighted_sentiments_thresh_pos = weighted_sentiment_baseline_predict_threshold_positive_eps_only(data, threshold=0.1, max_days=max_days)
    cm_thresh_pos, accuracy_thresh_pos, precision_thresh_pos, recall_thresh_pos, f1_thresh_pos = evaluate_predictions(true_labels_thresh_pos, predictions_thresh_pos, "WEIGHTED SENTIMENT THRESHOLD 0.1 (POSITIVE EPS ONLY)")
    analyze_weighted_sentiment_distribution(eps_surprises_thresh_pos, weighted_sentiments_thresh_pos, true_labels_thresh_pos, "WEIGHTED SENTIMENT THRESHOLD 0.1 (POSITIVE EPS ONLY)")
    plot_confusion_matrix(cm_thresh_pos, "WEIGHTED SENTIMENT THRESHOLD 0.1 (POSITIVE EPS ONLY)", f'Baselines/weighted_sentiment/threshold_0_1_positive_eps_only/weighted_sentiment_baseline_threshold_0_1_positive_eps_only_confusion_matrix_max_days_{max_days}.png')
    
    # Save results for weighted sentiment threshold 0.1 (positive EPS only)
    results_thresh_pos = {
        'accuracy': float(accuracy_thresh_pos),
        'precision': float(precision_thresh_pos),
        'recall': float(recall_thresh_pos),
        'f1_score': float(f1_thresh_pos),
        'confusion_matrix': cm_thresh_pos.tolist(),
        'total_samples': int(len(data)),
        'filtered_samples': int(len(predictions_thresh_pos)),
        'threshold': 0.1,
        'max_days': int(max_days),
        'positive_predictions': int(np.sum(predictions_thresh_pos == 1)),
        'negative_predictions': int(np.sum(predictions_thresh_pos == 0)),
        'true_positives': int(np.sum((predictions_thresh_pos == 1) & (true_labels_thresh_pos == 1))),
        'true_negatives': int(np.sum((predictions_thresh_pos == 0) & (true_labels_thresh_pos == 0))),
        'false_positives': int(np.sum((predictions_thresh_pos == 1) & (true_labels_thresh_pos == 0))),
        'false_negatives': int(np.sum((predictions_thresh_pos == 0) & (true_labels_thresh_pos == 1)))
    }
    
    with open(f'Baselines/weighted_sentiment/threshold_0_1_positive_eps_only/weighted_sentiment_baseline_threshold_0_1_positive_eps_only_results_max_days_{max_days}.json', 'w') as f:
        json.dump(results_thresh_pos, f, indent=2)
    
    print(f"Results saved to Baselines/weighted_sentiment/threshold_0_1_positive_eps_only/weighted_sentiment_baseline_threshold_0_1_positive_eps_only_results_max_days_{max_days}.json")

if __name__ == "__main__":
    main()
