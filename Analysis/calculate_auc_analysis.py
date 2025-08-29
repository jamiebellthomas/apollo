#!/usr/bin/env python3
"""
Calculate AUC (Area Under the ROC Curve) for all aggregated results in Analysis directory.
This script reads the aggregated_results.csv files and computes AUC scores, then appends them to performance.json files.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import os
import glob

def load_aggregated_results(csv_path):
    """Load aggregated results from CSV file."""
    try:
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        print(f"Warning: {csv_path} not found")
        return None
    except Exception as e:
        print(f"Warning: Error loading {csv_path}: {str(e)}")
        return None

def calculate_auc_from_predictions(df):
    """
    Calculate AUC from aggregated results DataFrame.
    
    Args:
        df: DataFrame with Actual_Label and Aggregated_Predicted_Label columns
    
    Returns:
        auc: Area Under the ROC Curve
        tpr: True Positive Rate (Recall)
        fpr: False Positive Rate
    """
    if df is None or df.empty:
        return None, None, None
    
    # Extract true labels and predicted labels
    y_true = df['Actual_Label'].values
    y_pred = df['Aggregated_Predicted_Label'].values
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if cm.shape != (2, 2):
        print(f"Warning: Unexpected confusion matrix shape: {cm.shape}")
        return None, None, None
    
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    
    # Calculate TPR (True Positive Rate) and FPR (False Positive Rate)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # For binary predictions, AUC is approximately (TPR + (1 - FPR)) / 2
    # This is a simplified calculation since we don't have probability scores
    auc = (tpr + (1 - fpr)) / 2
    
    return auc, tpr, fpr

def load_performance_json(json_path):
    """Load existing performance.json file."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {json_path} not found")
        return None
    except json.JSONDecodeError:
        print(f"Warning: {json_path} is not valid JSON")
        return None

def save_performance_json(performance_data, json_path):
    """Save performance data to JSON file."""
    try:
        with open(json_path, 'w') as f:
            json.dump(performance_data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving {json_path}: {str(e)}")
        return False

def analyze_all_models():
    """Analyze all models in the Analysis directory."""
    
    # Get all heterognn directories
    analysis_dir = "Analysis"
    model_dirs = [d for d in os.listdir(analysis_dir) if d.startswith('heterognn') and os.path.isdir(os.path.join(analysis_dir, d))]
    model_dirs.sort()  # Sort to process in order
    
    print(f"Found {len(model_dirs)} model directories: {model_dirs}")
    print("=" * 60)
    
    all_results = {}
    
    for model_dir in model_dirs:
        print(f"\nProcessing {model_dir}...")
        model_path = os.path.join(analysis_dir, model_dir)
        
        model_results = {}
        
        # Process both regular and medium results
        for suffix in ['', '_medium']:
            csv_path = os.path.join(model_path, f'aggregated_results{suffix}.csv')
            json_path = os.path.join(model_path, f'performance{suffix}.json')
            
            print(f"  Checking {csv_path}...")
            
            # Load aggregated results
            df = load_aggregated_results(csv_path)
            if df is None:
                print(f"    Skipped (no CSV data)")
                continue
            
            # Calculate AUC
            auc, tpr, fpr = calculate_auc_from_predictions(df)
            if auc is None:
                print(f"    Skipped (could not calculate AUC)")
                continue
            
            # Load existing performance data
            performance_data = load_performance_json(json_path)
            if performance_data is None:
                print(f"    Skipped (no performance.json)")
                continue
            
            # Add AUC-related fields
            performance_data['auc'] = float(auc)
            performance_data['tpr'] = float(tpr)
            performance_data['fpr'] = float(fpr)
            
            # Save updated performance data
            if save_performance_json(performance_data, json_path):
                print(f"    ✓ Updated {json_path}")
                print(f"      AUC: {auc:.4f}, TPR: {tpr:.4f}, FPR: {fpr:.4f}")
                
                # Store results for summary
                experiment_name = 'regular' if suffix == '' else 'medium'
                model_results[experiment_name] = {
                    'auc': auc,
                    'tpr': tpr,
                    'fpr': fpr,
                    'accuracy': performance_data.get('accuracy', 0),
                    'precision': performance_data.get('precision', 0),
                    'recall': performance_data.get('recall', 0),
                    'f1_score': performance_data.get('f1_score', 0)
                }
            else:
                print(f"    ✗ Failed to update {json_path}")
        
        all_results[model_dir] = model_results
    
    return all_results

def print_auc_summary(all_results):
    """Print a summary of AUC scores for all models."""
    
    print("\n" + "=" * 80)
    print("AUC SUMMARY FOR ALL MODELS")
    print("=" * 80)
    
    for model_name, experiments in all_results.items():
        print(f"\n{model_name.upper()}:")
        print("-" * 50)
        
        for exp_name, metrics in experiments.items():
            print(f"  {exp_name.title()}:")
            print(f"    Accuracy:  {metrics['accuracy']:.4f}")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall:    {metrics['recall']:.4f}")
            print(f"    F1-Score:  {metrics['f1_score']:.4f}")
            print(f"    AUC:       {metrics['auc']:.4f}")
            print(f"    TPR:       {metrics['tpr']:.4f}")
            print(f"    FPR:       {metrics['fpr']:.4f}")
            print()

def create_auc_comparison_plot(all_results):
    """Create a comparison plot of AUC scores."""
    
    # Prepare data for plotting
    model_names = []
    experiment_names = []
    auc_scores = []
    accuracy_scores = []
    
    for model_name, experiments in all_results.items():
        for exp_name, metrics in experiments.items():
            model_names.append(model_name.replace('heterognn', 'HeteroGNN'))
            experiment_names.append(exp_name.title())
            auc_scores.append(metrics['auc'])
            accuracy_scores.append(metrics['accuracy'])
    
    # Create DataFrame
    df = pd.DataFrame({
        'Model': model_names,
        'Experiment': experiment_names,
        'AUC': auc_scores,
        'Accuracy': accuracy_scores
    })
    
    # Create comparison plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # AUC comparison
    sns.barplot(data=df, x='Model', y='AUC', hue='Experiment', ax=ax1)
    ax1.set_title('AUC Scores by Model and Experiment')
    ax1.set_ylabel('AUC Score')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(title='Experiment', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Accuracy comparison
    sns.barplot(data=df, x='Model', y='Accuracy', hue='Experiment', ax=ax2)
    ax2.set_title('Accuracy Scores by Model and Experiment')
    ax2.set_ylabel('Accuracy Score')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(title='Experiment', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('Analysis/auc_comparison_plot.png', dpi=300, bbox_inches='tight')
    print("AUC comparison plot saved to Analysis/auc_comparison_plot.png")
    plt.show()
    
    return df

def save_auc_results(all_results, df):
    """Save AUC results to JSON and CSV files."""
    
    # Save detailed results
    with open('Analysis/auc_results_detailed.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save summary DataFrame
    df.to_csv('Analysis/auc_results_summary.csv', index=False)
    
    print("AUC results saved to:")
    print("  - Analysis/auc_results_detailed.json")
    print("  - Analysis/auc_results_summary.csv")

def main():
    """Main function to calculate and analyze AUC for all models."""
    
    print("Calculating AUC for all models in Analysis directory...")
    print("=" * 60)
    
    # Analyze all models
    all_results = analyze_all_models()
    
    # Print summary
    print_auc_summary(all_results)
    
    # Create comparison plot
    df = create_auc_comparison_plot(all_results)
    
    # Save results
    save_auc_results(all_results, df)
    
    print("\nAUC analysis complete!")

if __name__ == "__main__":
    main()
