#!/usr/bin/env python3
"""
Script to run multiple training experiments with different random seeds.
This script imports the run_training function from run.py and executes it
multiple times with different seeds to get robust performance estimates.
"""

import random
import time
import json
import os
from datetime import datetime
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple

# Import the run_training function from run.py
from run import run_training


def run_multiple_experiments(
    num_runs: int = 5,
    seed_range: Tuple[int, int] = (0, 1000000)
) -> List[Dict]:
    """
    Run multiple training experiments with different random seeds.
    
    Args:
        num_runs: Number of experiments to run
        seed_range: Tuple of (min_seed, max_seed) for random seed generation
    
    Returns:
        List of dictionaries containing results from each run
    """
    
    all_results = []
    seeds_used = []
    
    print(f"Starting {num_runs} training runs with different random seeds...")
    print(f"Seed range: {seed_range}")
    print("=" * 60)
    
    for run_num in range(1, num_runs + 1):
        # Generate a random seed
        seed = random.randint(seed_range[0], seed_range[1])
        seeds_used.append(seed)
        
        print(f"\n{'='*20} RUN {run_num}/{num_runs} {'='*20}")
        print(f"Using seed: {seed}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        start_time = time.time()
        
        try:
            # Run the training with the current seed
            # Using the same hyperparameters as in run.py
            model, test_metrics, history = run_training(
                # Model configuration
                model_type="heterognn",  # You can change this
                time_dim=8,
                
                # Data configuration
                n_facts=35,
                limit=None,
                use_cache=True,
                
                # Model architecture
                hidden_channels=128,
                num_layers=4,
                feature_dropout=0.3,
                edge_dropout=0.1,
                final_dropout=0.2,
                readout="company",
                
                # Training configuration
                batch_size=24,
                epochs=100,
                lr=1e-5,
                weight_decay=1e-4,
                seed=seed,  # This is the key parameter that changes
                grad_clip=1.0,
                ckpt_path=f"best_model_run_{run_num}.pt",
                loss_type="weighted_bce",
                early_stopping=True,
                patience=20,
                lr_scheduler="cosine",
                lr_step_size=10,
                lr_gamma=0.5,
                time_aware_split=False,
                optimizer_type="adam",
                
                # Data splitting
                train_ratio=0.7,
                val_ratio=0.15,
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Collect results
            run_result = {
                "run_number": run_num,
                "seed": seed,
                "duration_seconds": duration,
                "start_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "test_metrics": test_metrics,
                "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
                "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
                "best_val_loss": min(history["val_loss"]) if history["val_loss"] else None,
                "epochs_trained": len(history["train_loss"]) if history["train_loss"] else 0,
            }
            
            all_results.append(run_result)
            
            print(f"✅ Run {run_num} completed successfully!")
            print(f"   Duration: {duration:.2f} seconds")
            print(f"   Test Accuracy: {test_metrics['acc']:.4f}")
            print(f"   Test AUC: {test_metrics.get('auc', float('nan')):.4f}")
            if 'precision' in test_metrics:
                print(f"   Test Precision: {test_metrics['precision']:.4f}")
                print(f"   Test Recall: {test_metrics['recall']:.4f}")
                print(f"   Test F1: {test_metrics['f1']:.4f}")
            
        except Exception as e:
            print(f"❌ Run {run_num} failed with error: {e}")
            run_result = {
                "run_number": run_num,
                "seed": seed,
                "error": str(e),
                "start_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }
            all_results.append(run_result)
    
    # Calculate summary statistics
    successful_runs = [r for r in all_results if "error" not in r]
    
    if successful_runs:
        accuracies = [r["test_metrics"]["acc"] for r in successful_runs]
        aucs = [r["test_metrics"].get("auc", float('nan')) for r in successful_runs if not np.isnan(r["test_metrics"].get("auc", float('nan')))]
        
        summary = {
            "total_runs": num_runs,
            "successful_runs": len(successful_runs),
            "failed_runs": num_runs - len(successful_runs),
            "seeds_used": seeds_used,
            "accuracy_stats": {
                "mean": np.mean(accuracies),
                "std": np.std(accuracies),
                "min": np.min(accuracies),
                "max": np.max(accuracies),
                "values": accuracies
            },
            "auc_stats": {
                "mean": np.mean(aucs) if aucs else float('nan'),
                "std": np.std(aucs) if aucs else float('nan'),
                "min": np.min(aucs) if aucs else float('nan'),
                "max": np.max(aucs) if aucs else float('nan'),
                "values": aucs
            },
            "duration_stats": {
                "mean": np.mean([r["duration_seconds"] for r in successful_runs]),
                "std": np.std([r["duration_seconds"] for r in successful_runs]),
                "total": sum([r["duration_seconds"] for r in successful_runs])
            }
        }
    else:
        summary = {
            "total_runs": num_runs,
            "successful_runs": 0,
            "failed_runs": num_runs,
            "seeds_used": seeds_used,
            "error": "All runs failed"
        }
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Total runs: {num_runs}")
    print(f"Successful runs: {len(successful_runs)}")
    print(f"Failed runs: {num_runs - len(successful_runs)}")
    
    if successful_runs:
        print(f"\nAccuracy - Mean: {summary['accuracy_stats']['mean']:.4f} ± {summary['accuracy_stats']['std']:.4f}")
        print(f"Accuracy - Range: [{summary['accuracy_stats']['min']:.4f}, {summary['accuracy_stats']['max']:.4f}]")
        
        if not np.isnan(summary['auc_stats']['mean']):
            print(f"AUC - Mean: {summary['auc_stats']['mean']:.4f} ± {summary['auc_stats']['std']:.4f}")
            print(f"AUC - Range: [{summary['auc_stats']['min']:.4f}, {summary['auc_stats']['max']:.4f}]")
        
        print(f"\nTotal training time: {summary['duration_stats']['total']:.2f} seconds")
        print(f"Average time per run: {summary['duration_stats']['mean']:.2f} seconds")
    
    return all_results


if __name__ == "__main__":
    # Configuration - you can modify these parameters
    NUM_RUNS = 20  # Number of experiments to run
    SEED_RANGE = (0, 1000000)  # Range for random seed generation
    
    print("Multiple Training Runs Script")
    print("=" * 40)
    print(f"Number of runs: {NUM_RUNS}")
    print(f"Seed range: {SEED_RANGE}")
    print("Note: Results are automatically saved to the Results directory by run.py")
     
    # Run the experiments
    results = run_multiple_experiments(
        num_runs=NUM_RUNS,
        seed_range=SEED_RANGE
    )
    
    print("\nAll experiments completed!") 