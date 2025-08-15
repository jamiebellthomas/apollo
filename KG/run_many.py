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
    seed_range: Tuple[int, int] = (0, 1000000),
    model_type: str = "heterognn2"
) -> List[Dict]:
    """
    Run multiple training experiments with different random seeds for a specific model type.
    
    Args:
        num_runs: Number of experiments to run
        seed_range: Tuple of (min_seed, max_seed) for random seed generation
        model_type: Type of model to train ("heterognn", "heterognn2", "heterognn3", "heterognn4")
    
    Returns:
        List of dictionaries containing results from each run
    """
    
    all_results = []
    seeds_used = []
    
    print(f"Starting {num_runs} training runs for {model_type} with different random seeds...")
    print(f"Seed range: {seed_range}")
    print("=" * 60)

    for run_num in range(1, num_runs + 1):
        # Generate a random seed
        seed = random.randint(seed_range[0], seed_range[1])
        seeds_used.append(seed)
        
        print(f"\n{'='*20} RUN {run_num}/{num_runs} - {model_type.upper()} {'='*20}")
        print(f"Using seed: {seed}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        start_time = time.time()
        
        try:
            # Configuration variables - all in one place
            TIME_DIM = 16  # Used for HeteroGNN2 and HeteroGNN4
            ENABLE_RUN_SCRAPING = True  # Set to True to enable run scraping
            MIN_EPOCHS_AFTER_PATIENCE = 10
            BATCH_SIZE = 32
            EPOCHS = 100
            if model_type == "heterognn2":
                LEARNING_RATE = 3e-5
            elif model_type == "heterognn4":
                LEARNING_RATE = 3e-6
            else:
                LEARNING_RATE = 1e-5
            
            if model_type == "heterognn4":
                HIDDEN_CHANNELS = 1024
            else:
                HIDDEN_CHANNELS = 128
            NUM_LAYERS = 4
            READOUT = "company"
            LOSS_TYPE = "weighted_bce"
            
            # Attention parameters (for HeteroGNN4)
            HEADS = 8  # Number of attention heads
            FUNNEL_TO_PRIMARY = False  # If True: only ('fact','mentions','company') relation is used
            TOPK_PER_PRIMARY = 15  # If set, keep top-k incoming fact edges per primary before attention
            
            # Run the training with the current seed
            # Using the same hyperparameters as in run.py
            result = run_training(
                # Model configuration
                model_type=model_type,  # Use the parameter instead of hardcoded value
                time_dim=TIME_DIM,  # Use the configuration variable
                
                # Data configuration
                n_facts=35,
                limit=100,
                use_cache=True,
                
                # Model architecture
                hidden_channels=HIDDEN_CHANNELS,
                num_layers=NUM_LAYERS,
                feature_dropout=0.3,
                edge_dropout=0.1,
                final_dropout=0.2,
                readout=READOUT,
                heads=HEADS,
                funnel_to_primary=FUNNEL_TO_PRIMARY,
                topk_per_primary=TOPK_PER_PRIMARY,
                
                # Training configuration
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                lr=LEARNING_RATE,
                weight_decay=1e-4,
                seed=seed,  # This is the key parameter that changes
                grad_clip=1.0,
                ckpt_path=f"best_model_run_{run_num}.pt",
                loss_type=LOSS_TYPE,
                early_stopping=True,
                patience=20,
                lr_scheduler="cosine",
                lr_step_size=10,
                lr_gamma=0.5,
                time_aware_split=False,
                optimizer_type="adam",
                
                # Run scraping configuration
                enable_run_scraping=ENABLE_RUN_SCRAPING,  # Use the configuration variable
                min_epochs_after_patience=MIN_EPOCHS_AFTER_PATIENCE,  # Use the configuration variable
                
                # Data splitting
                train_ratio=0.8,
                val_ratio=0.2,
            )
            
            # Check if run was scraped (returns None values)
            if result[0] is None or result[1] is None or result[2] is None:
                print(f"⚠️  Run {run_num} was scraped due to early termination")
                run_result = {
                    "run_number": run_num,
                    "seed": seed,
                    "model_type": model_type,
                    "status": "scraped",
                    "reason": "Training ended too early after patience threshold",
                    "start_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                }
                all_results.append(run_result)
                continue
            
            model, test_metrics, history = result
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Collect results
            run_result = {
                "run_number": run_num,
                "seed": seed,
                "model_type": model_type,
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
                "model_type": model_type,
                "error": str(e),
                "start_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            }
            all_results.append(run_result)
    
    # Calculate summary statistics
    successful_runs = [r for r in all_results if "error" not in r and "status" not in r]
    scraped_runs = [r for r in all_results if "status" in r and r["status"] == "scraped"]
    failed_runs = [r for r in all_results if "error" in r]
    
    if successful_runs:
        accuracies = [r["test_metrics"]["acc"] for r in successful_runs]
        aucs = [r["test_metrics"].get("auc", float('nan')) for r in successful_runs if not np.isnan(r["test_metrics"].get("auc", float('nan')))]
        
        summary = {
            "model_type": model_type,
            "total_runs": num_runs,
            "successful_runs": len(successful_runs),
            "scraped_runs": len(scraped_runs),
            "failed_runs": len(failed_runs),
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
            "model_type": model_type,
            "total_runs": num_runs,
            "successful_runs": 0,
            "scraped_runs": len(scraped_runs),
            "failed_runs": len(failed_runs),
            "seeds_used": seeds_used,
            "error": "No successful runs"
        }
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"EXPERIMENT SUMMARY - {model_type.upper()}")
    print("=" * 60)
    print(f"Total runs: {num_runs}")
    print(f"Successful runs: {len(successful_runs)}")
    print(f"Scraped runs: {len(scraped_runs)}")
    print(f"Failed runs: {len(failed_runs)}")
    
    if successful_runs:
        print(f"\nAccuracy - Mean: {summary['accuracy_stats']['mean']:.4f} ± {summary['accuracy_stats']['std']:.4f}")
        print(f"Accuracy - Range: [{summary['accuracy_stats']['min']:.4f}, {summary['accuracy_stats']['max']:.4f}]")
        
        if not np.isnan(summary['auc_stats']['mean']):
            print(f"AUC - Mean: {summary['auc_stats']['mean']:.4f} ± {summary['auc_stats']['std']:.4f}")
            print(f"AUC - Range: [{summary['auc_stats']['min']:.4f}, {summary['auc_stats']['max']:.4f}]")
        
        print(f"\nTotal training time: {summary['duration_stats']['total']:.2f} seconds")
        print(f"Average time per run: {summary['duration_stats']['mean']:.2f} seconds")
    
    if scraped_runs:
        print(f"\n⚠️  {len(scraped_runs)} runs were scraped due to early termination")
    
    return all_results, summary


def run_all_model_types(
    num_runs: int = 5,
    seed_range: Tuple[int, int] = (0, 1000000),
    model_types: List[str] = None
) -> Dict[str, List[Dict]]:
    """
    Run multiple training experiments for all model types.
    
    Args:
        num_runs: Number of experiments to run per model type
        seed_range: Tuple of (min_seed, max_seed) for random seed generation
        model_types: List of model types to run. If None, runs all available types.
    
    Returns:
        Dictionary mapping model types to their results
    """
    
    if model_types is None:
        model_types = ["heterognn", "heterognn2", "heterognn3", "heterognn4"]
    
    all_model_results = {}
    all_summaries = {}
    
    print("=" * 80)
    print("MULTI-MODEL TYPE TRAINING EXPERIMENTS")
    print("=" * 80)
    print(f"Model types to run: {model_types}")
    print(f"Runs per model type: {num_runs}")
    print(f"Total expected runs: {len(model_types) * num_runs}")
    
    for i, model_type in enumerate(model_types, 1):
        print(f"\n{'='*20} MODEL TYPE {i}/{len(model_types)}: {model_type.upper()} {'='*20}")
        
        # Run experiments for this model type
        results, summary = run_multiple_experiments(
            num_runs=num_runs,
            seed_range=seed_range,
            model_type=model_type
        )
        
        all_model_results[model_type] = results
        all_summaries[model_type] = summary
        
        print(f"\n✅ Completed all runs for {model_type}")
    
    # Print overall summary
    print("\n" + "=" * 80)
    print("OVERALL EXPERIMENT SUMMARY")
    print("=" * 80)
    
    for model_type, summary in all_summaries.items():
        print(f"\n{model_type.upper()}:")
        print(f"  Successful runs: {summary['successful_runs']}/{summary['total_runs']}")
        if summary['successful_runs'] > 0:
            print(f"  Accuracy: {summary['accuracy_stats']['mean']:.4f} ± {summary['accuracy_stats']['std']:.4f}")
            if not np.isnan(summary['auc_stats']['mean']):
                print(f"  AUC: {summary['auc_stats']['mean']:.4f} ± {summary['auc_stats']['std']:.4f}")
        print(f"  Scraped runs: {summary['scraped_runs']}")
        print(f"  Failed runs: {summary['failed_runs']}")
    
    return all_model_results, all_summaries


if __name__ == "__main__":
    # Configuration - you can modify these parameters
    NUM_RUNS = 40  # Number of experiments to run per model type
    SEED_RANGE = (0, 1000000)  # Range for random seed generation
    MODEL_TYPES = ["heterognn4"]  # Model types to run
    
    print("Multiple Training Runs Script - All Model Types")
    print("=" * 50)
    print(f"Number of runs per model type: {NUM_RUNS}")
    print(f"Total runs: {NUM_RUNS * len(MODEL_TYPES)}")
    print(f"Seed range: {SEED_RANGE}")
    print(f"Model types: {MODEL_TYPES}")
    print("\nModel descriptions:")
    print("  - heterognn: Original model with temporal encoding (sentiment + decay)")
    print("  - heterognn2: Temporal-aware model with learned temporal encoding")
    print("  - heterognn3: No temporal encoding (sentiment only)")
    print("  - heterognn4: Attention model with GATv2Conv and temporal encoding")
    print("\nNote: Results are automatically saved to the Results directory by run.py")
     
    # Run the experiments for all model types
    all_results, all_summaries = run_all_model_types(
        num_runs=NUM_RUNS,
        seed_range=SEED_RANGE,
        model_types=MODEL_TYPES
    )
    
    print("\nAll experiments completed!")
    print(f"Results saved for {len(all_results)} model types:")
    for model_type in all_results.keys():
        print(f"  - {model_type}: {len(all_results[model_type])} runs") 