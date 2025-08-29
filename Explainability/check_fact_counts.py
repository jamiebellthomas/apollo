#!/usr/bin/env python3

import json
from pathlib import Path
from collections import defaultdict

def check_fact_counts():
    """Check the distribution of fact counts per sample."""
    results_dir = Path("../Results/heterognn5")
    
    if not results_dir.exists():
        print("❌ Results directory not found!")
        return
    
    directories = sorted([d for d in results_dir.iterdir() if d.is_dir()])
    print(f"Checking fact counts across {len(directories)} directories...")
    
    fact_counts = []
    total_samples = 0
    
    for directory in directories:
        samples_file = directory / "samples_all_facts.json"
        if not samples_file.exists():
            continue
        
        try:
            with open(samples_file, 'r') as f:
                data = json.load(f)
            
            for sample_key, sample_data in data.items():
                actual_facts = len(sample_data['all_facts'])
                fact_counts.append(actual_facts)
                total_samples += 1
                
        except Exception as e:
            print(f"❌ Error processing {directory.name}: {e}")
    
    if not fact_counts:
        print("❌ No data found!")
        return
    
    fact_counts.sort()
    
    print(f"\n{'='*60}")
    print(f"FACT COUNT SUMMARY (Total samples: {total_samples})")
    print(f"{'='*60}")
    print(f"Minimum facts per sample: {min(fact_counts)}")
    print(f"Maximum facts per sample: {max(fact_counts)}")
    print(f"Average facts per sample: {sum(fact_counts) / len(fact_counts):.1f}")
    print(f"Median facts per sample: {fact_counts[len(fact_counts)//2]}")
    
    # Show distribution
    print(f"\n{'='*60}")
    print("FACT COUNT DISTRIBUTION")
    print(f"{'='*60}")
    
    ranges = [
        (35, 50, "35-50"),
        (51, 100, "51-100"), 
        (101, 200, "101-200"),
        (201, 500, "201-500"),
        (501, 1000, "501-1000"),
        (1001, float('inf'), "1000+")
    ]
    
    for min_count, max_count, label in ranges:
        if max_count == float('inf'):
            count = sum(1 for x in fact_counts if x >= min_count)
        else:
            count = sum(1 for x in fact_counts if min_count <= x < max_count)
        percentage = (count / len(fact_counts)) * 100
        print(f"{label:8}: {count:4} samples ({percentage:5.1f}%)")
    
    # Show some examples
    print(f"\n{'='*60}")
    print("SAMPLE EXAMPLES")
    print(f"{'='*60}")
    print(f"Lowest fact counts: {fact_counts[:5]}")
    print(f"Highest fact counts: {fact_counts[-5:]}")

if __name__ == "__main__":
    check_fact_counts()
