#!/usr/bin/env python3
"""
Script to aggregate cluster analysis data from all test directories.
Creates summary CSV files showing cluster performance across all models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

def main():
    """Main function to aggregate cluster analysis data."""
    print("🔍 Aggregating cluster analysis data...")
    
    # Find all HeteroGNN5 results directories
    results_dir = Path("../Results/heterognn5")
    if not results_dir.exists():
        print("❌ Results directory not found!")
        return
    
    directories = sorted([d for d in results_dir.iterdir() if d.is_dir()])
    print(f"Found {len(directories)} directories to process...")
    
    # Initialize aggregated data structures
    overall_data = []
    positive_data = []
    negative_data = []
    
    # Process each directory
    for i, directory in enumerate(directories):
        print(f"Processing {directory.name} ({i+1}/{len(directories)})...")
        
        # Load overall cluster analysis
        overall_file = directory / "cluster_analysis.csv"
        if overall_file.exists():
            df = pd.read_csv(overall_file)
            df['model'] = directory.name
            overall_data.append(df)
            print(f"  ✅ Loaded overall data: {len(df)} rows")
        else:
            print(f"  ❌ No overall file found")
        
        # Load positive cluster analysis
        positive_file = directory / "cluster_analysis_positive.csv"
        if positive_file.exists():
            df = pd.read_csv(positive_file)
            df['model'] = directory.name
            positive_data.append(df)
            print(f"  ✅ Loaded positive data: {len(df)} rows")
        else:
            print(f"  ❌ No positive file found")
        
        # Load negative cluster analysis
        negative_file = directory / "cluster_analysis_negative.csv"
        if negative_file.exists():
            df = pd.read_csv(negative_file)
            df['model'] = directory.name
            negative_data.append(df)
            print(f"  ✅ Loaded negative data: {len(df)} rows")
        else:
            print(f"  ❌ No negative file found")
    
    print(f"\n📊 Combining data...")
    
    # Combine all data
    if overall_data:
        overall_df = pd.concat(overall_data, ignore_index=True)
        print(f"  Overall: {len(overall_df)} cluster entries from {len(overall_data)} models")
    else:
        overall_df = pd.DataFrame()
        print("  ❌ No overall cluster analysis data found")
    
    if positive_data:
        positive_df = pd.concat(positive_data, ignore_index=True)
        print(f"  Positive: {len(positive_df)} cluster entries from {len(positive_data)} models")
    else:
        positive_df = pd.DataFrame()
        print("  ❌ No positive cluster analysis data found")
    
    if negative_data:
        negative_df = pd.concat(negative_data, ignore_index=True)
        print(f"  Negative: {len(negative_df)} cluster entries from {len(negative_data)} models")
    else:
        negative_df = pd.DataFrame()
        print("  ❌ No negative cluster analysis data found")
    
    # Calculate aggregated statistics
    print("\n📊 Calculating aggregated statistics...")
    
    # Overall aggregation
    if not overall_df.empty:
        print("  Processing overall data...")
        overall_summary = aggregate_cluster_stats(overall_df, "overall")
        overall_summary.to_csv("aggregated_cluster_analysis_overall.csv", index=False)
        print(f"  ✅ Saved aggregated_cluster_analysis_overall.csv")
    
    # Positive aggregation
    if not positive_df.empty:
        print("  Processing positive data...")
        positive_summary = aggregate_cluster_stats(positive_df, "positive")
        positive_summary.to_csv("aggregated_cluster_analysis_positive.csv", index=False)
        print(f"  ✅ Saved aggregated_cluster_analysis_positive.csv")
    
    # Negative aggregation
    if not negative_df.empty:
        print("  Processing negative data...")
        negative_summary = aggregate_cluster_stats(negative_df, "negative")
        negative_summary.to_csv("aggregated_cluster_analysis_negative.csv", index=False)
        print(f"  ✅ Saved aggregated_cluster_analysis_negative.csv")
    
    print(f"\n✅ Aggregation completed!")

def aggregate_cluster_stats(df, analysis_type):
    """Aggregate statistics for each cluster across all models."""
    print(f"    Aggregating {analysis_type} cluster statistics...")
    
    # Group by cluster_id
    grouped = df.groupby('cluster_id')
    print(f"    Found {len(grouped)} unique clusters")
    
    aggregated_stats = []
    
    for cluster_id, group in grouped:
        # Basic statistics
        num_models = len(group['model'].unique())
        total_fact_count = group['fact_count'].sum()
        avg_fact_count = group['fact_count'].mean()
        std_fact_count = group['fact_count'].std()
        
        # Sample statistics
        total_sample_count = group['sample_count'].sum()
        avg_sample_count = group['sample_count'].mean()
        std_sample_count = group['sample_count'].std()
        avg_percentage_of_samples = group['percentage_of_samples'].mean()
        std_percentage_of_samples = group['percentage_of_samples'].std()
        
        # Attention statistics
        avg_attention_score = group['avg_attention_score'].mean()
        std_attention_score = group['avg_attention_score'].std()
        min_attention_score = group['avg_attention_score'].min()
        max_attention_score = group['avg_attention_score'].max()
        
        # Sentiment statistics
        avg_sentiment = group['avg_sentiment'].mean()
        std_sentiment = group['avg_sentiment'].std()
        min_sentiment = group['avg_sentiment'].min()
        max_sentiment = group['avg_sentiment'].max()
        
        # Consistency metrics
        attention_cv = std_attention_score / avg_attention_score if avg_attention_score > 0 else 0
        sentiment_cv = std_sentiment / abs(avg_sentiment) if avg_sentiment != 0 else 0
        
        # Get sample event types (most common across models)
        all_event_types = []
        for event_types in group['sample_event_types']:
            if pd.notna(event_types):
                all_event_types.extend([et.strip() for et in event_types.split(';')])
        
        # Get most common event types
        event_type_counts = Counter(all_event_types)
        most_common_event_types = '; '.join([et for et, count in event_type_counts.most_common(5)])
        
        aggregated_stats.append({
            'cluster_id': cluster_id,
            'num_models': num_models,
            'total_fact_count': total_fact_count,
            'avg_fact_count': round(avg_fact_count, 2),
            'std_fact_count': round(std_fact_count, 2),
            'total_sample_count': total_sample_count,
            'avg_sample_count': round(avg_sample_count, 2),
            'std_sample_count': round(std_sample_count, 2),
            'avg_percentage_of_samples': round(avg_percentage_of_samples, 4),
            'std_percentage_of_samples': round(std_percentage_of_samples, 4),
            'avg_attention_score': round(avg_attention_score, 6),
            'std_attention_score': round(std_attention_score, 6),
            'min_attention_score': round(min_attention_score, 6),
            'max_attention_score': round(max_attention_score, 6),
            'attention_cv': round(attention_cv, 4),
            'avg_sentiment': round(avg_sentiment, 6),
            'std_sentiment': round(std_sentiment, 6),
            'min_sentiment': round(min_sentiment, 6),
            'max_sentiment': round(max_sentiment, 6),
            'sentiment_cv': round(sentiment_cv, 4),
            'most_common_event_types': most_common_event_types
        })
    
    # Convert to DataFrame and sort by average attention score
    summary_df = pd.DataFrame(aggregated_stats)
    if not summary_df.empty:
        summary_df = summary_df.sort_values('avg_attention_score', ascending=False)
    
    print(f"    Aggregated {len(summary_df)} clusters")
    return summary_df

if __name__ == "__main__":
    main()
