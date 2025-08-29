#!/usr/bin/env python3
"""
Script to fix predicted labels in all samples_all_facts.json files.
Gets the correct y_pred from each directory's replicate_eval and applies it.
"""

import json
import sys
from pathlib import Path
from replicate_eval import replicate_evaluation

def fix_predicted_labels_for_directory(directory_path):
    """Fix predicted labels for a single directory."""
    print(f"Processing {directory_path.name}...")
    
    # Get the correct y_pred from replicate_eval
    try:
        results = replicate_evaluation(str(directory_path))
        y_pred = results['y_pred']
        if hasattr(y_pred, 'tolist'):
            y_pred = y_pred.tolist()
        print(f"  Got y_pred with {sum(y_pred)} positive predictions")
    except Exception as e:
        print(f"  ERROR getting y_pred: {e}")
        return False
    
    # Load the samples file
    samples_file = directory_path / "samples_all_facts.json"
    if not samples_file.exists():
        print(f"  No samples_all_facts.json found")
        return False
    
    try:
        with open(samples_file, 'r') as f:
            data = json.load(f)
        
        print(f"  Before fix: {sum(1 for key in data.keys() if data[key]['sample_metadata']['predicted_label'] == 1)} positive predictions")
        
        # Update each sample's predicted_label with the correct y_pred value
        for i in range(len(y_pred)):
            if str(i) in data:
                data[str(i)]['sample_metadata']['predicted_label'] = int(y_pred[i])
        
        print(f"  After fix: {sum(1 for key in data.keys() if data[key]['sample_metadata']['predicted_label'] == 1)} positive predictions")
        
        # Save the corrected file
        with open(samples_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"  ✅ Fixed {directory_path.name}!")
        return True
        
    except Exception as e:
        print(f"  ERROR processing file: {e}")
        return False

def main():
    """Fix predicted labels for all directories."""
    results_dir = Path("../Results/heterognn5")
    if not results_dir.exists():
        print("Results directory not found")
        return
    
    directories = sorted([d for d in results_dir.iterdir() if d.is_dir()])
    print(f"Found {len(directories)} directories to process")
    
    successful_fixes = 0
    total_processed = 0
    
    for directory in directories:
        if fix_predicted_labels_for_directory(directory):
            successful_fixes += 1
        total_processed += 1
        print()  # Add spacing between directories
    
    print(f"Summary: Fixed {successful_fixes}/{total_processed} directories")

if __name__ == "__main__":
    main()
