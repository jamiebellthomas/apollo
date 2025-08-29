#!/usr/bin/env python3

print("Test script starting...")

from pathlib import Path
import json

# Check if results directory exists
results_dir = Path("../Results/heterognn5")
print(f"Results dir exists: {results_dir.exists()}")

# Check if we can find directories
directories = [d for d in results_dir.iterdir() if d.is_dir()]
print(f"Found {len(directories)} directories")

# Test with first directory
if directories:
    test_dir = directories[0]
    print(f"Testing with directory: {test_dir.name}")
    
    samples_file = test_dir / "samples_all_facts.json"
    print(f"Samples file exists: {samples_file.exists()}")
    
    if samples_file.exists():
        with open(samples_file, 'r') as f:
            data = json.load(f)
        print(f"Loaded data with {len(data)} samples")
        
        # Check first sample
        first_sample = list(data.values())[0]
        print(f"First sample has {len(first_sample['all_facts'])} facts")
        
        # Check if facts have cluster_id
        facts_with_cluster = sum(1 for fact in first_sample['all_facts'] if fact.get('cluster_id') is not None)
        print(f"Facts with cluster_id: {facts_with_cluster}")

print("Test script completed.")
