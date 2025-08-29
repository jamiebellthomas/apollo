#!/usr/bin/env python3
print("Starting aggregation test...")
import pandas as pd
from pathlib import Path

results_dir = Path("../Results/heterognn5")
print(f"Results dir exists: {results_dir.exists()}")

if results_dir.exists():
    directories = [d for d in results_dir.iterdir() if d.is_dir()]
    print(f"Found {len(directories)} directories")
    
    # Test loading one file
    test_dir = directories[0]
    test_file = test_dir / "cluster_analysis.csv"
    print(f"Test file exists: {test_file.exists()}")
    
    if test_file.exists():
        df = pd.read_csv(test_file)
        print(f"Loaded {len(df)} rows")
        df['model'] = test_dir.name
        print(f"Added model column")
        
        # Test aggregation
        print(f"Unique clusters: {df['cluster_id'].nunique()}")
        print(f"Average attention score: {df['avg_attention_score'].mean():.6f}")
        
        # Save test output
        df.to_csv("test_output.csv", index=False)
        print("Saved test output")
