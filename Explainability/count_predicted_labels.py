#!/usr/bin/env python3
"""
Script to count predicted labels in all samples_all_facts.json files.
"""

import json
from pathlib import Path

def count_predicted_labels():
    """Count predicted labels for all directories."""
    results_dir = Path("../Results/heterognn5")
    if not results_dir.exists():
        print("Results directory not found")
        return
    
    directories = sorted([d for d in results_dir.iterdir() if d.is_dir()])
    print(f"Found {len(directories)} directories to check")
    print("\nPredicted Label Counts:")
    print("=" * 60)
    print(f"{'Directory':<20} {'Label 0':<8} {'Label 1':<8} {'Total':<8}")
    print("-" * 60)
    
    total_label_0 = 0
    total_label_1 = 0
    
    for directory in directories:
        samples_file = directory / "samples_all_facts.json"
        if not samples_file.exists():
            print(f"{directory.name:<20} {'MISSING':<8} {'MISSING':<8} {'MISSING':<8}")
            continue
        
        try:
            with open(samples_file, 'r') as f:
                data = json.load(f)
            
            label_0_count = 0
            label_1_count = 0
            
            for sample_key, sample_data in data.items():
                if sample_key.isdigit():  # Only process numeric sample keys
                    predicted_label = sample_data.get('sample_metadata', {}).get('predicted_label')
                    if predicted_label == 0:
                        label_0_count += 1
                    elif predicted_label == 1:
                        label_1_count += 1
            
            total_samples = label_0_count + label_1_count
            print(f"{directory.name:<20} {label_0_count:<8} {label_1_count:<8} {total_samples:<8}")
            
            total_label_0 += label_0_count
            total_label_1 += label_1_count
            
        except Exception as e:
            print(f"{directory.name:<20} {'ERROR':<8} {'ERROR':<8} {'ERROR':<8} ({e})")
    
    print("-" * 60)
    print(f"{'TOTAL':<20} {total_label_0:<8} {total_label_1:<8} {total_label_0 + total_label_1:<8}")
    print("=" * 60)

if __name__ == "__main__":
    count_predicted_labels()
