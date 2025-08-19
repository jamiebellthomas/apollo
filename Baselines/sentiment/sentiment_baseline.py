#!/usr/bin/env python3
"""
Sentiment-based baseline classifier.
Uses EPS surprise and average sentiment to predict stock price movements.
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

def calculate_average_sentiment(fact_list):
    """Calculate average sentiment from fact list."""
    if not fact_list:
        return 0.0
    
    sentiments = [f.get('sentiment', 0.0) for f in fact_list]
    return sum(sentiments) / len(sentiments)

def sentiment_baseline_predict(data):
    """Make predictions based on EPS surprise and average sentiment."""
    predictions = []
    true_labels = []
    eps_surprises = []
    avg_sentiments = []
    
    print("Making predictions using EPS surprise + sentiment heuristic...")
    
    for item in data:
        eps_surprise = item.get('eps_surprise', 0.0)
        if eps_surprise is None:
            eps_surprise = 0.0
        
        fact_list = item.get('fact_list', [])
        avg_sentiment = calculate_average_sentiment(fact_list)
        true_label = item['label']
        
        # Prediction logic:
        # If eps_surprise is negative -> predict 0
        # If eps_surprise is positive and avg_sentiment is negative -> predict 0  
        # If eps_surprise is positive and avg_sentiment is positive -> predict 1
        if eps_surprise <= 0:
            prediction = 0
        elif avg_sentiment <= 0:
            prediction = 0
        else:
            prediction = 1
        
        predictions.append(prediction)
        true_labels.append(true_label)
        eps_surprises.append(eps_surprise)
        avg_sentiments.append(avg_sentiment)
    
    return np.array(predictions), np.array(true_labels), np.array(eps_surprises), np.array(avg_sentiments)

def sentiment_baseline_predict_positive_eps_only(data):
    """Make predictions for positive EPS cases only."""
    predictions = []
    true_labels = []
    eps_surprises = []
    avg_sentiments = []
    
    # Filter to only include positive EPS cases
    filtered_data = [item for item in data if item.get('eps_surprise', 0.0) is not None and item.get('eps_surprise', 0.0) > 0]
    
    print(f"Filtered to {len(filtered_data)} positive EPS cases (out of {len(data)} total)")
    
    for item in filtered_data:
        eps_surprise = item.get('eps_surprise', 0.0)
        fact_list = item.get('fact_list', [])
        avg_sentiment = calculate_average_sentiment(fact_list)
        true_label = item['label']
        
        # For positive EPS cases only:
        # If avg_sentiment is negative -> predict 0
        # If avg_sentiment is positive -> predict 1
        if avg_sentiment <= 0:
            prediction = 0
        else:
            prediction = 1
        
        predictions.append(prediction)
        true_labels.append(true_label)
        eps_surprises.append(eps_surprise)
        avg_sentiments.append(avg_sentiment)
    
    return np.array(predictions), np.array(true_labels), np.array(eps_surprises), np.array(avg_sentiments)

def sentiment_baseline_predict_threshold_0_2(data, threshold=0.2):
    """Make predictions using sentiment threshold of 0.2."""
    predictions = []
    true_labels = []
    eps_surprises = []
    avg_sentiments = []
    
    print("Making predictions using EPS surprise + sentiment threshold 0.2...")
    
    for item in data:
        eps_surprise = item.get('eps_surprise', 0.0)
        if eps_surprise is None:
            eps_surprise = 0.0
        
        fact_list = item.get('fact_list', [])
        avg_sentiment = calculate_average_sentiment(fact_list)
        true_label = item['label']
        
        # Prediction logic with sentiment threshold 0.2:
        # If eps_surprise is negative -> predict 0
        # If eps_surprise is positive and avg_sentiment <= 0.2 -> predict 0  
        # If eps_surprise is positive and avg_sentiment > 0.2 -> predict 1
        if eps_surprise <= 0:
            prediction = 0
        elif avg_sentiment <= threshold:
            prediction = 0
        else:
            prediction = 1
        
        predictions.append(prediction)
        true_labels.append(true_label)
        eps_surprises.append(eps_surprise)
        avg_sentiments.append(avg_sentiment)
    
    return np.array(predictions), np.array(true_labels), np.array(eps_surprises), np.array(avg_sentiments)

def evaluate_predictions(y_true, y_pred, title):
    """Evaluate predictions and print metrics."""
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Print results
    print("=" * 50)
    print(f"SENTIMENT BASELINE RESULTS - {title}")
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
    plt.title(f'Confusion Matrix - Sentiment Baseline ({title})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix plot saved to {save_path}")
    
    plt.show()

def analyze_sentiment_distribution(eps_surprises, avg_sentiments, y_true, title):
    """Analyze the distribution of EPS surprises and sentiments by true label."""
    eps_positive = eps_surprises[y_true == 1]
    eps_negative = eps_surprises[y_true == 0]
    sent_positive = avg_sentiments[y_true == 1]
    sent_negative = avg_sentiments[y_true == 0]
    
    print(f"Distribution Analysis - {title}:")
    print(f"Positive labels (1): {len(eps_positive)} samples")
    print(f"  Mean EPS surprise: {np.mean(eps_positive):.4f}")
    print(f"  Mean sentiment: {np.mean(sent_positive):.4f}")
    print(f"  Std EPS surprise: {np.std(eps_positive):.4f}")
    print(f"  Std sentiment: {np.std(sent_positive):.4f}")
    print()
    
    print(f"Negative labels (0): {len(eps_negative)} samples")
    print(f"  Mean EPS surprise: {np.mean(eps_negative):.4f}")
    print(f"  Mean sentiment: {np.mean(sent_negative):.4f}")
    print(f"  Std EPS surprise: {np.std(eps_negative):.4f}")
    print(f"  Std sentiment: {np.std(sent_negative):.4f}")
    print()

def main():
    """Main function to run the sentiment baseline."""
    # Load data
    print("Loading data from Data/subgraphs.jsonl...")
    data = load_data('Data/subgraphs.jsonl')
    print(f"Loaded {len(data)} samples")
    print()
    
    # Experiment 1: All data
    print("EXPERIMENT 1: ALL DATA")
    print("=" * 50)
    predictions, true_labels, eps_surprises, avg_sentiments = sentiment_baseline_predict(data)
    cm, accuracy, precision, recall, f1 = evaluate_predictions(true_labels, predictions, "ALL DATA")
    analyze_sentiment_distribution(eps_surprises, avg_sentiments, true_labels, "ALL DATA")
    plot_confusion_matrix(cm, "ALL DATA", 'Baselines/sentiment/all_data/sentiment_baseline_all_confusion_matrix.png')
    
    # Save results for all data
    results_all = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'total_samples': int(len(data)),
        'positive_predictions': int(np.sum(predictions == 1)),
        'negative_predictions': int(np.sum(predictions == 0)),
        'true_positives': int(np.sum((predictions == 1) & (true_labels == 1))),
        'true_negatives': int(np.sum((predictions == 0) & (true_labels == 0))),
        'false_positives': int(np.sum((predictions == 1) & (true_labels == 0))),
        'false_negatives': int(np.sum((predictions == 0) & (true_labels == 1)))
    }
    
    with open('Baselines/sentiment/all_data/sentiment_baseline_all_results.json', 'w') as f:
        json.dump(results_all, f, indent=2)
    
    print("Results saved to Baselines/sentiment/all_data/sentiment_baseline_all_results.json")
    print()
    
    # Experiment 2: Positive EPS only
    print("EXPERIMENT 2: POSITIVE EPS ONLY")
    print("=" * 50)
    predictions_pos, true_labels_pos, eps_surprises_pos, avg_sentiments_pos = sentiment_baseline_predict_positive_eps_only(data)
    cm_pos, accuracy_pos, precision_pos, recall_pos, f1_pos = evaluate_predictions(true_labels_pos, predictions_pos, "POSITIVE EPS ONLY")
    analyze_sentiment_distribution(eps_surprises_pos, avg_sentiments_pos, true_labels_pos, "POSITIVE EPS ONLY")
    plot_confusion_matrix(cm_pos, "POSITIVE EPS ONLY", 'Baselines/sentiment/positive_eps_only/sentiment_baseline_positive_confusion_matrix.png')
    
    # Save results for positive EPS only
    results_pos = {
        'accuracy': float(accuracy_pos),
        'precision': float(precision_pos),
        'recall': float(recall_pos),
        'f1_score': float(f1_pos),
        'confusion_matrix': cm_pos.tolist(),
        'total_samples': int(len(data)),
        'filtered_samples': int(len(predictions_pos)),
        'positive_predictions': int(np.sum(predictions_pos == 1)),
        'negative_predictions': int(np.sum(predictions_pos == 0)),
        'true_positives': int(np.sum((predictions_pos == 1) & (true_labels_pos == 1))),
        'true_negatives': int(np.sum((predictions_pos == 0) & (true_labels_pos == 0))),
        'false_positives': int(np.sum((predictions_pos == 1) & (true_labels_pos == 0))),
        'false_negatives': int(np.sum((predictions_pos == 0) & (true_labels_pos == 1)))
    }
    
    with open('Baselines/sentiment/positive_eps_only/sentiment_baseline_positive_results.json', 'w') as f:
        json.dump(results_pos, f, indent=2)
    
    print("Results saved to Baselines/sentiment/positive_eps_only/sentiment_baseline_positive_results.json")
    print()
    
    # Experiment 3: Sentiment threshold 0.2
    print("EXPERIMENT 3: SENTIMENT THRESHOLD 0.2")
    print("=" * 50)
    predictions_thresh, true_labels_thresh, eps_surprises_thresh, avg_sentiments_thresh = sentiment_baseline_predict_threshold_0_2(data, threshold=0.2)
    cm_thresh, accuracy_thresh, precision_thresh, recall_thresh, f1_thresh = evaluate_predictions(true_labels_thresh, predictions_thresh, "SENTIMENT THRESHOLD 0.2")
    analyze_sentiment_distribution(eps_surprises_thresh, avg_sentiments_thresh, true_labels_thresh, "SENTIMENT THRESHOLD 0.2")
    plot_confusion_matrix(cm_thresh, "SENTIMENT THRESHOLD 0.2", 'Baselines/sentiment/sentiment_threshold_0_2/sentiment_baseline_threshold_0_2_confusion_matrix.png')
    
    # Save results for sentiment threshold 0.2
    results_thresh = {
        'accuracy': float(accuracy_thresh),
        'precision': float(precision_thresh),
        'recall': float(recall_thresh),
        'f1_score': float(f1_thresh),
        'confusion_matrix': cm_thresh.tolist(),
        'total_samples': int(len(data)),
        'positive_predictions': int(np.sum(predictions_thresh == 1)),
        'negative_predictions': int(np.sum(predictions_thresh == 0)),
        'true_positives': int(np.sum((predictions_thresh == 1) & (true_labels_thresh == 1))),
        'true_negatives': int(np.sum((predictions_thresh == 0) & (true_labels_thresh == 0))),
        'false_positives': int(np.sum((predictions_thresh == 1) & (true_labels_thresh == 0))),
        'false_negatives': int(np.sum((predictions_thresh == 0) & (true_labels_thresh == 1)))
    }
    
    with open('Baselines/sentiment/sentiment_threshold_0_2/sentiment_baseline_threshold_0_2_results.json', 'w') as f:
        json.dump(results_thresh, f, indent=2)
    
    print("Results saved to Baselines/sentiment/sentiment_threshold_0_2/sentiment_baseline_threshold_0_2_results.json")

if __name__ == "__main__":
    main()
