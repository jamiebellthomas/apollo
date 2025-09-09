#!/usr/bin/env python3
"""
Script to consolidate cluster analysis data.
Creates a CSV showing fact counts across all prediction outcomes (TP, TN, FP, FN)
and attention rankings from positive/negative cluster analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def consolidate_cluster_analysis():
    """Consolidate cluster analysis data for all directories."""
    print("🔍 Consolidating cluster analysis data...")
    
    # Find all HeteroGNN5 results directories
    results_dir = Path("../Results/heterognn5")
    if not results_dir.exists():
        print("❌ Results directory not found!")
        return
    
    directories = sorted([d for d in results_dir.iterdir() if d.is_dir()])
    print(f"Found {len(directories)} directories to process...")
    
    # Process each directory
    for directory in directories:
        print(f"\nProcessing {directory.name}...")
        consolidate_directory(directory)

def consolidate_directory(directory_path):
    """Consolidate cluster analysis for a single directory."""
    # Check if all required files exist
    required_files = [
        "prediction_outcome_TP.csv",
        "prediction_outcome_TN.csv", 
        "prediction_outcome_FP.csv",
        "prediction_outcome_FN.csv",
        "cluster_analysis_positive.csv",
        "cluster_analysis_negative.csv"
    ]
    
    missing_files = []
    for file in required_files:
        if not (directory_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"  ❌ Missing files: {missing_files}")
        return
    
    # Load all the CSV files
    print("  Loading prediction outcome files...")
    tp_df = pd.read_csv(directory_path / "prediction_outcome_TP.csv")
    tn_df = pd.read_csv(directory_path / "prediction_outcome_TN.csv")
    fp_df = pd.read_csv(directory_path / "prediction_outcome_FP.csv")
    fn_df = pd.read_csv(directory_path / "prediction_outcome_FN.csv")
    
    print("  Loading cluster analysis files...")
    pos_df = pd.read_csv(directory_path / "cluster_analysis_positive.csv")
    neg_df = pd.read_csv(directory_path / "cluster_analysis_negative.csv")
    
    # Create attention ranking dictionaries
    print("  Creating attention rankings...")
    pos_rankings = {}
    for idx, row in pos_df.iterrows():
        pos_rankings[row['cluster_id']] = idx + 1  # 1-based ranking
    
    neg_rankings = {}
    for idx, row in neg_df.iterrows():
        neg_rankings[row['cluster_id']] = idx + 1  # 1-based ranking
    
    # Create fact count dictionaries
    print("  Creating fact count dictionaries...")
    tp_counts = dict(zip(tp_df['cluster_id'], tp_df['fact_count']))
    tn_counts = dict(zip(tn_df['cluster_id'], tn_df['fact_count']))
    fp_counts = dict(zip(fp_df['cluster_id'], fp_df['fact_count']))
    fn_counts = dict(zip(fn_df['cluster_id'], fn_df['fact_count']))
    
    # Get all unique cluster IDs
    all_clusters = set()
    all_clusters.update(tp_df['cluster_id'].tolist())
    all_clusters.update(tn_df['cluster_id'].tolist())
    all_clusters.update(fp_df['cluster_id'].tolist())
    all_clusters.update(fn_df['cluster_id'].tolist())
    all_clusters.update(pos_df['cluster_id'].tolist())
    all_clusters.update(neg_df['cluster_id'].tolist())
    
    print(f"  Found {len(all_clusters)} unique clusters")
    
    # Create consolidated data
    consolidated_data = []
    for cluster_id in sorted(all_clusters):
        consolidated_data.append({
            'cluster_id': cluster_id,
            'fact_count_TP': tp_counts.get(cluster_id, 0),
            'fact_count_TN': tn_counts.get(cluster_id, 0),
            'fact_count_FP': fp_counts.get(cluster_id, 0),
            'fact_count_FN': fn_counts.get(cluster_id, 0),
            'average_attention_rank_positive': pos_rankings.get(cluster_id, np.nan),
            'average_attention_rank_negative': neg_rankings.get(cluster_id, np.nan)
        })
    
    # Create DataFrame and save
    consolidated_df = pd.DataFrame(consolidated_data)
    output_file = directory_path / "consolidated_cluster_analysis.csv"
    consolidated_df.to_csv(output_file, index=False)
    
    print(f"  ✅ Saved {output_file}")
    print(f"  📊 Summary:")
    print(f"    Total clusters: {len(consolidated_df)}")
    print(f"    Clusters with TP facts: {len(consolidated_df[consolidated_df['fact_count_TP'] > 0])}")
    print(f"    Clusters with TN facts: {len(consolidated_df[consolidated_df['fact_count_TN'] > 0])}")
    print(f"    Clusters with FP facts: {len(consolidated_df[consolidated_df['fact_count_FP'] > 0])}")
    print(f"    Clusters with FN facts: {len(consolidated_df[consolidated_df['fact_count_FN'] > 0])}")

def main():
    """Main function."""
    consolidate_cluster_analysis()
    print(f"\n✅ Consolidation completed!")

if __name__ == "__main__":
    main()
