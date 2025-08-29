#!/usr/bin/env python3
"""
Calculate AUC (Area Under the ROC Curve) for all baselines.
This script reads the results from all baseline experiments and computes AUC scores.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve
import os
import glob

def load_baseline_results(baseline_path):
    """Load results from a baseline JSON file."""
    try:
        with open(baseline_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {baseline_path} not found")
        return None
    except json.JSONDecodeError:
        print(f"Warning: {baseline_path} is not valid JSON")
        return None

def calculate_auc_from_confusion_matrix(cm):
    """
    Calculate AUC from confusion matrix.
    This is an approximation since we only have binary predictions, not probabilities.
    """
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    
    # Calculate TPR (True Positive Rate) and FPR (False Positive Rate)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # For binary predictions, AUC is approximately (TPR + (1 - FPR)) / 2
    # This is a simplified calculation since we don't have probability scores
    auc = (tpr + (1 - fpr)) / 2
    
    return auc, tpr, fpr

def analyze_baseline_results():
    """Analyze all baseline results and calculate AUC."""
    
    baseline_results = {}
    
    # 1. EPS Only Baseline
    print("Analyzing EPS Only Baseline...")
    eps_results = {}
    
    # All data
    eps_all_path = "Baselines/eps_only/positive_only/eps_baseline_all_results.json"
    eps_all = load_baseline_results(eps_all_path)
    if eps_all:
        auc, tpr, fpr = calculate_auc_from_confusion_matrix(eps_all['confusion_matrix'])
        eps_results['all_data'] = {
            'accuracy': eps_all['accuracy'],
            'precision': eps_all['precision'],
            'recall': eps_all['recall'],
            'f1_score': eps_all['f1_score'],
            'auc': auc,
            'tpr': tpr,
            'fpr': fpr
        }
    
    # Positive EPS only
    eps_pos_path = "Baselines/eps_only/eps_baseline_positive_results.json"
    eps_pos = load_baseline_results(eps_pos_path)
    if eps_pos:
        auc, tpr, fpr = calculate_auc_from_confusion_matrix(eps_pos['confusion_matrix'])
        eps_results['positive_eps_only'] = {
            'accuracy': eps_pos['accuracy'],
            'precision': eps_pos['precision'],
            'recall': eps_pos['recall'],
            'f1_score': eps_pos['f1_score'],
            'auc': auc,
            'tpr': tpr,
            'fpr': fpr
        }
    
    baseline_results['eps_only'] = eps_results
    
    # 2. Sentiment Baseline
    print("Analyzing Sentiment Baseline...")
    sentiment_results = {}
    
    # All data
    sentiment_all_path = "Baselines/sentiment/all_data/sentiment_baseline_all_results.json"
    sentiment_all = load_baseline_results(sentiment_all_path)
    if sentiment_all:
        auc, tpr, fpr = calculate_auc_from_confusion_matrix(sentiment_all['confusion_matrix'])
        sentiment_results['all_data'] = {
            'accuracy': sentiment_all['accuracy'],
            'precision': sentiment_all['precision'],
            'recall': sentiment_all['recall'],
            'f1_score': sentiment_all['f1_score'],
            'auc': auc,
            'tpr': tpr,
            'fpr': fpr
        }
    
    # Positive EPS only
    sentiment_pos_path = "Baselines/sentiment/positive_eps_only/sentiment_baseline_positive_results.json"
    sentiment_pos = load_baseline_results(sentiment_pos_path)
    if sentiment_pos:
        auc, tpr, fpr = calculate_auc_from_confusion_matrix(sentiment_pos['confusion_matrix'])
        sentiment_results['positive_eps_only'] = {
            'accuracy': sentiment_pos['accuracy'],
            'precision': sentiment_pos['precision'],
            'recall': sentiment_pos['recall'],
            'f1_score': sentiment_pos['f1_score'],
            'auc': auc,
            'tpr': tpr,
            'fpr': fpr
        }
    
    # Threshold 0.2
    sentiment_thresh_path = "Baselines/sentiment/sentiment_threshold_0_2/sentiment_baseline_threshold_0_2_results.json"
    sentiment_thresh = load_baseline_results(sentiment_thresh_path)
    if sentiment_thresh:
        auc, tpr, fpr = calculate_auc_from_confusion_matrix(sentiment_thresh['confusion_matrix'])
        sentiment_results['threshold_0_2'] = {
            'accuracy': sentiment_thresh['accuracy'],
            'precision': sentiment_thresh['precision'],
            'recall': sentiment_thresh['recall'],
            'f1_score': sentiment_thresh['f1_score'],
            'auc': auc,
            'tpr': tpr,
            'fpr': fpr
        }
    
    baseline_results['sentiment'] = sentiment_results
    
    # 3. Weighted Sentiment Baseline
    print("Analyzing Weighted Sentiment Baseline...")
    weighted_sentiment_results = {}
    
    # All data
    ws_all_path = "Baselines/weighted_sentiment/all_data/weighted_sentiment_baseline_all_results_max_days_90.json"
    ws_all = load_baseline_results(ws_all_path)
    if ws_all:
        auc, tpr, fpr = calculate_auc_from_confusion_matrix(ws_all['confusion_matrix'])
        weighted_sentiment_results['all_data'] = {
            'accuracy': ws_all['accuracy'],
            'precision': ws_all['precision'],
            'recall': ws_all['recall'],
            'f1_score': ws_all['f1_score'],
            'auc': auc,
            'tpr': tpr,
            'fpr': fpr
        }
    
    # Positive EPS only
    ws_pos_path = "Baselines/weighted_sentiment/positive_eps_only/weighted_sentiment_baseline_positive_results_max_days_90.json"
    ws_pos = load_baseline_results(ws_pos_path)
    if ws_pos:
        auc, tpr, fpr = calculate_auc_from_confusion_matrix(ws_pos['confusion_matrix'])
        weighted_sentiment_results['positive_eps_only'] = {
            'accuracy': ws_pos['accuracy'],
            'precision': ws_pos['precision'],
            'recall': ws_pos['recall'],
            'f1_score': ws_pos['f1_score'],
            'auc': auc,
            'tpr': tpr,
            'fpr': fpr
        }
    
    # Threshold 0.1 positive EPS only
    ws_thresh_path = "Baselines/weighted_sentiment/threshold_0_1_positive_eps_only/weighted_sentiment_baseline_threshold_0_1_positive_eps_only_results_max_days_90.json"
    ws_thresh = load_baseline_results(ws_thresh_path)
    if ws_thresh:
        auc, tpr, fpr = calculate_auc_from_confusion_matrix(ws_thresh['confusion_matrix'])
        weighted_sentiment_results['threshold_0_1_positive_eps_only'] = {
            'accuracy': ws_thresh['accuracy'],
            'precision': ws_thresh['precision'],
            'recall': ws_thresh['recall'],
            'f1_score': ws_thresh['f1_score'],
            'auc': auc,
            'tpr': tpr,
            'fpr': fpr
        }
    
    baseline_results['weighted_sentiment'] = weighted_sentiment_results
    
    # 4. Neural Network Baseline
    print("Analyzing Neural Network Baseline...")
    nn_results = {}
    
    # All data
    nn_all_path = "Baselines/neural_net/all_data/nn_baseline_all_results.json"
    nn_all = load_baseline_results(nn_all_path)
    if nn_all:
        auc, tpr, fpr = calculate_auc_from_confusion_matrix(nn_all['confusion_matrix'])
        nn_results['all_data'] = {
            'accuracy': nn_all['accuracy'],
            'precision': nn_all['precision'],
            'recall': nn_all['recall'],
            'f1_score': nn_all['f1_score'],
            'auc': auc,
            'tpr': tpr,
            'fpr': fpr
        }
    
    # Positive EPS only
    nn_pos_path = "Baselines/neural_net/positive_eps_only/nn_baseline_positive_results.json"
    nn_pos = load_baseline_results(nn_pos_path)
    if nn_pos:
        auc, tpr, fpr = calculate_auc_from_confusion_matrix(nn_pos['confusion_matrix'])
        nn_results['positive_eps_only'] = {
            'accuracy': nn_pos['accuracy'],
            'precision': nn_pos['precision'],
            'recall': nn_pos['recall'],
            'f1_score': nn_pos['f1_score'],
            'auc': auc,
            'tpr': tpr,
            'fpr': fpr
        }
    
    baseline_results['neural_net'] = nn_results
    
    return baseline_results

def print_auc_summary(baseline_results):
    """Print a summary of AUC scores for all baselines."""
    
    print("=" * 80)
    print("AUC SUMMARY FOR ALL BASELINES")
    print("=" * 80)
    
    for baseline_name, experiments in baseline_results.items():
        print(f"\n{baseline_name.upper().replace('_', ' ')}:")
        print("-" * 50)
        
        for exp_name, metrics in experiments.items():
            print(f"  {exp_name.replace('_', ' ').title()}:")
            print(f"    Accuracy:  {metrics['accuracy']:.4f}")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall:    {metrics['recall']:.4f}")
            print(f"    F1-Score:  {metrics['f1_score']:.4f}")
            print(f"    AUC:       {metrics['auc']:.4f}")
            print(f"    TPR:       {metrics['tpr']:.4f}")
            print(f"    FPR:       {metrics['fpr']:.4f}")
            print()

def create_auc_comparison_plot(baseline_results):
    """Create a comparison plot of AUC scores."""
    
    # Prepare data for plotting
    baseline_names = []
    experiment_names = []
    auc_scores = []
    accuracy_scores = []
    
    for baseline_name, experiments in baseline_results.items():
        for exp_name, metrics in experiments.items():
            baseline_names.append(baseline_name.replace('_', ' ').title())
            experiment_names.append(exp_name.replace('_', ' ').title())
            auc_scores.append(metrics['auc'])
            accuracy_scores.append(metrics['accuracy'])
    
    # Create DataFrame
    df = pd.DataFrame({
        'Baseline': baseline_names,
        'Experiment': experiment_names,
        'AUC': auc_scores,
        'Accuracy': accuracy_scores
    })
    
    # Create comparison plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # AUC comparison
    sns.barplot(data=df, x='Baseline', y='AUC', hue='Experiment', ax=ax1)
    ax1.set_title('AUC Scores by Baseline and Experiment')
    ax1.set_ylabel('AUC Score')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(title='Experiment', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Accuracy comparison
    sns.barplot(data=df, x='Baseline', y='Accuracy', hue='Experiment', ax=ax2)
    ax2.set_title('Accuracy Scores by Baseline and Experiment')
    ax2.set_ylabel('Accuracy Score')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(title='Experiment', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('Baselines/auc_comparison_plot.png', dpi=300, bbox_inches='tight')
    print("AUC comparison plot saved to Baselines/auc_comparison_plot.png")
    plt.show()
    
    return df

def append_auc_to_results_files(baseline_results):
    """Append AUC scores to all existing results JSON files."""
    
    print("\nAppending AUC to results files...")
    
    # Define file paths and their corresponding baseline/experiment
    file_mappings = [
        # EPS Only Baseline
        ("Baselines/eps_only/positive_only/eps_baseline_all_results.json", "eps_only", "all_data"),
        ("Baselines/eps_only/eps_baseline_positive_results.json", "eps_only", "positive_eps_only"),
        
        # Sentiment Baseline
        ("Baselines/sentiment/all_data/sentiment_baseline_all_results.json", "sentiment", "all_data"),
        ("Baselines/sentiment/positive_eps_only/sentiment_baseline_positive_results.json", "sentiment", "positive_eps_only"),
        ("Baselines/sentiment/sentiment_threshold_0_2/sentiment_baseline_threshold_0_2_results.json", "sentiment", "threshold_0_2"),
        
        # Weighted Sentiment Baseline
        ("Baselines/weighted_sentiment/all_data/weighted_sentiment_baseline_all_results_max_days_90.json", "weighted_sentiment", "all_data"),
        ("Baselines/weighted_sentiment/positive_eps_only/weighted_sentiment_baseline_positive_results_max_days_90.json", "weighted_sentiment", "positive_eps_only"),
        ("Baselines/weighted_sentiment/threshold_0_1_positive_eps_only/weighted_sentiment_baseline_threshold_0_1_positive_eps_only_results_max_days_90.json", "weighted_sentiment", "threshold_0_1_positive_eps_only"),
        
        # Neural Network Baseline
        ("Baselines/neural_net/all_data/nn_baseline_all_results.json", "neural_net", "all_data"),
        ("Baselines/neural_net/positive_eps_only/nn_baseline_positive_results.json", "neural_net", "positive_eps_only"),
    ]
    
    updated_files = []
    
    for file_path, baseline_name, experiment_name in file_mappings:
        try:
            # Check if the baseline and experiment exist in our results
            if (baseline_name in baseline_results and 
                experiment_name in baseline_results[baseline_name]):
                
                # Load existing results
                with open(file_path, 'r') as f:
                    existing_results = json.load(f)
                
                # Get AUC metrics
                metrics = baseline_results[baseline_name][experiment_name]
                
                # Add AUC-related fields
                existing_results['auc'] = metrics['auc']
                existing_results['tpr'] = metrics['tpr']
                existing_results['fpr'] = metrics['fpr']
                
                # Save updated results
                with open(file_path, 'w') as f:
                    json.dump(existing_results, f, indent=2)
                
                updated_files.append(file_path)
                print(f"  ✓ Updated {file_path}")
                
            else:
                print(f"  ⚠ Skipped {file_path} (no AUC data available)")
                
        except FileNotFoundError:
            print(f"  ✗ File not found: {file_path}")
        except json.JSONDecodeError:
            print(f"  ✗ Invalid JSON: {file_path}")
        except Exception as e:
            print(f"  ✗ Error updating {file_path}: {str(e)}")
    
    print(f"\nUpdated {len(updated_files)} results files with AUC scores.")

def save_auc_results(baseline_results, df):
    """Save AUC results to JSON and CSV files."""
    
    # Save detailed results
    with open('Baselines/auc_results_detailed.json', 'w') as f:
        json.dump(baseline_results, f, indent=2)
    
    # Save summary DataFrame
    df.to_csv('Baselines/auc_results_summary.csv', index=False)
    
    print("AUC results saved to:")
    print("  - Baselines/auc_results_detailed.json")
    print("  - Baselines/auc_results_summary.csv")
    
    # Append AUC to all existing results files
    append_auc_to_results_files(baseline_results)

def main():
    """Main function to calculate and analyze AUC for all baselines."""
    
    print("Calculating AUC for all baselines...")
    print("=" * 50)
    
    # Analyze all baseline results
    baseline_results = analyze_baseline_results()
    
    # Print summary
    print_auc_summary(baseline_results)
    
    # Create comparison plot
    df = create_auc_comparison_plot(baseline_results)
    
    # Save results
    save_auc_results(baseline_results, df)
    
    print("\nAUC analysis complete!")

if __name__ == "__main__":
    main()
