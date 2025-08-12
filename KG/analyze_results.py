#!/usr/bin/env python3
"""
Script to analyze and compare results from different experiments.
"""

import json
import os
from pathlib import Path
import pandas as pd
from datetime import datetime

def list_experiments(results_dir="results"):
    """List all experiments in the results directory."""
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"❌ Results directory '{results_dir}' not found")
        return []
    
    experiments = []
    for exp_dir in results_path.iterdir():
        if exp_dir.is_dir() and exp_dir.name.replace("_", "").replace("/", "").isdigit():
            experiments.append(exp_dir)
    
    return sorted(experiments, key=lambda x: x.name, reverse=True)

def load_experiment(exp_dir):
    """Load experiment data from directory."""
    exp_data = {
        "timestamp": exp_dir.name,
        "directory": str(exp_dir)
    }
    
    # Load hyperparameters
    hyperparams_file = exp_dir / "hyperparameters.txt"
    if hyperparams_file.exists():
        with open(hyperparams_file, 'r') as f:
            exp_data["hyperparameters"] = f.read()
    
    # Load results
    results_file = exp_dir / "results.json"
    if results_file.exists():
        with open(results_file, 'r') as f:
            results = json.load(f)
            exp_data["results"] = results
    
    # Load data split
    split_file = exp_dir / "data_split.json"
    if split_file.exists():
        with open(split_file, 'r') as f:
            exp_data["data_split"] = json.load(f)
    
    return exp_data

def compare_experiments(experiments, metric="f1"):
    """Compare experiments based on a specific metric."""
    comparison = []
    
    for exp_dir in experiments:
        exp_data = load_experiment(exp_dir)
        
        if "results" in exp_data and "test_metrics" in exp_data["results"]:
            test_metrics = exp_data["results"]["test_metrics"]
            comparison.append({
                "timestamp": exp_data["timestamp"],
                "auc": test_metrics.get("auc", float("nan")),
                "f1": test_metrics.get("f1", float("nan")),
                "accuracy": test_metrics.get("acc", float("nan")),
                "recall": test_metrics.get("recall", float("nan")),
                "precision": test_metrics.get("precision", float("nan")),
                "best_epoch": exp_data["results"].get("best_epoch", "unknown")
            })
    
    return pd.DataFrame(comparison)

def analyze_experiment(exp_dir):
    """Detailed analysis of a single experiment."""
    exp_data = load_experiment(exp_dir)
    
    print(f"=== EXPERIMENT ANALYSIS: {exp_dir.name} ===\n")
    
    # Print hyperparameters
    if "hyperparameters" in exp_data:
        print("HYPERPARAMETERS:")
        print(exp_data["hyperparameters"])
        print()
    
    # Print results
    if "results" in exp_data:
        results = exp_data["results"]
        test_metrics = results.get("test_metrics", {})
        
        print("TEST METRICS:")
        print(f"  AUC: {test_metrics.get('auc', 'N/A'):.4f}")
        print(f"  F1: {test_metrics.get('f1', 'N/A'):.4f}")
        print(f"  Accuracy: {test_metrics.get('acc', 'N/A'):.4f}")
        print(f"  Precision: {test_metrics.get('precision', 'N/A'):.4f}")
        print(f"  Recall: {test_metrics.get('recall', 'N/A'):.4f}")
        print(f"  Best Epoch: {results.get('best_epoch', 'N/A')}")
        print()
    
    # Print data split summary
    if "data_split" in exp_data:
        split = exp_data["data_split"]
        print("DATA SPLIT:")
        print(f"  Train: {len(split.get('train', []))} samples")
        print(f"  Val: {len(split.get('val', []))} samples")
        print(f"  Test: {len(split.get('test', []))} samples")
        print()

def main():
    """Main function to analyze results."""
    print("=== EXPERIMENT ANALYSIS TOOL ===\n")
    
    # List all experiments
    experiments = list_experiments()
    
    if not experiments:
        print("No experiments found!")
        return
    
    print(f"Found {len(experiments)} experiments:\n")
    
    # Show recent experiments
    for i, exp_dir in enumerate(experiments[:5]):
        print(f"{i+1}. {exp_dir.name}")
    
    if len(experiments) > 5:
        print(f"... and {len(experiments) - 5} more")
    
    print("\n" + "="*50 + "\n")
    
    # Compare all experiments
    print("EXPERIMENT COMPARISON:")
    df = compare_experiments(experiments)
    if not df.empty:
        print(df.to_string(index=False))
    else:
        print("No results found for comparison")
    
    print("\n" + "="*50 + "\n")
    
    # Analyze most recent experiment
    if experiments:
        print("MOST RECENT EXPERIMENT:")
        analyze_experiment(experiments[0])

if __name__ == "__main__":
    main() 