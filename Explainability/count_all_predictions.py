#!/usr/bin/env python3
"""
Script to count predicted labels (1s) in all samples files.
"""

import json
from pathlib import Path

def count_all_predictions():
    """Count predicted labels for all directories."""
    results_dir = Path("../Results/heterognn5")
    if not results_dir.exists():
        print("Results directory not found")
        return
    
    directories = sorted([d for d in results_dir.iterdir() if d.is_dir()])
    print(f"Found {len(directories)} directories to check")
    print("\nPredicted Label Counts:")
    print("=" * 50)
    print(f"{'Directory':<20} {'Label 1 Count':<12} {'File Type':<15}")
    print("-" * 50)
    
    total_label_1 = 0
    total_samples = 0
    
    for directory in directories:
        # Check for both possible filenames
        samples_file_top25 = directory / "samples_top_25_facts.json"
        samples_file_all = directory / "samples_all_facts.json"
        
        if samples_file_top25.exists():
            samples_file = samples_file_top25
            file_type = "top_25_facts"
        elif samples_file_all.exists():
            samples_file = samples_file_all
            file_type = "all_facts"
        else:
            print(f"{directory.name:<20} No samples file found")
            continue
        
        try:
            with open(samples_file, 'r') as f:
                data = json.load(f)
            
            # Count predicted_label: 1
            label_1_count = sum(1 for key in data.keys() 
                              if data[key]['sample_metadata']['predicted_label'] == 1)
            
            print(f"{directory.name:<20} {label_1_count:<12} {file_type:<15}")
            total_label_1 += label_1_count
            total_samples += len(data)
            
        except Exception as e:
            print(f"{directory.name:<20} ERROR: {e}")
    
    print("-" * 50)
    print(f"{'TOTAL':<20} {total_label_1:<12}")
    print(f"Total samples: {total_samples}")
    print(f"Average positive predictions per directory: {total_label_1/len(directories):.1f}")

if __name__ == "__main__":
    count_all_predictions()
