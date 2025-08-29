#!/usr/bin/env python3

import json
from pathlib import Path

def validate_samples():
    """Validate that all samples have at least 35 facts and count positive predictions."""
    results_dir = Path("../Results/heterognn5")
    
    if not results_dir.exists():
        print("❌ Results directory not found!")
        return False
    
    directories = sorted([d for d in results_dir.iterdir() if d.is_dir()])
    print(f"Found {len(directories)} directories to validate...")
    
    all_passed = True
    total_positive_predictions = 0
    
    for directory in directories:
        samples_file = directory / "samples_all_facts.json"
        if not samples_file.exists():
            print(f"⚠️  No samples_all_facts.json found in {directory.name}")
            continue
        
        print(f"\nValidating {directory.name}...")
        
        try:
            with open(samples_file, 'r') as f:
                data = json.load(f)
            
            directory_positive_count = 0
            
            for sample_key, sample_data in data.items():
                # Check fact count
                total_facts = sample_data['sample_metadata']['total_facts']
                actual_facts = len(sample_data['all_facts'])
                predicted_label = sample_data['sample_metadata']['predicted_label']
                
                if actual_facts < 35:
                    print(f"  ❌ Sample {sample_key}: Only {actual_facts} facts (need at least 35)")
                    all_passed = False
                
                if predicted_label == 1:
                    directory_positive_count += 1
            
            total_positive_predictions += directory_positive_count
            
            if directory_positive_count > 40:
                print(f"  ❌ {directory.name}: {directory_positive_count} positive predictions (over 40 limit)")
                all_passed = False
            else:
                print(f"  ✅ {directory.name}: {directory_positive_count} positive predictions")
                
        except Exception as e:
            print(f"❌ Error processing {directory.name}: {e}")
            all_passed = False
    
    print(f"\n{'='*50}")
    print(f"TOTAL POSITIVE PREDICTIONS: {total_positive_predictions}")
    
    if all_passed:
        print("✅ ALL VALIDATIONS PASSED!")
        return True
    else:
        print("❌ SOME VALIDATIONS FAILED!")
        return False

if __name__ == "__main__":
    success = validate_samples()
    if success:
        print("\nValidation completed successfully. Deleting script...")
        import os
        os.remove(__file__)
        print("Script deleted.")
    else:
        print("\nValidation failed. Script will not be deleted.")
