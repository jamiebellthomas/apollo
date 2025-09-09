#!/usr/bin/env python3
"""
Script to analyze unanimous FN samples across all models.
Find the top 20 clusters by average attention score in unanimous FN samples.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

def analyze_unanimous_fn_clusters():
    """Analyze cluster patterns in unanimous FN samples."""
    print("🔍 Analyzing unanimous FN sample clusters...")
    
    # Find all HeteroGNN5 results directories
    results_dir = Path("../Results/heterognn5")
    if not results_dir.exists():
        print("❌ Results directory not found!")
        return
    
    directories = sorted([d for d in results_dir.iterdir() if d.is_dir()])
    print(f"Found {len(directories)} directories to process...")
    
    # Get unanimous FN samples from the misclassification analysis
    misclass_file = results_dir / "sample_misclassification_analysis.csv"
    if not misclass_file.exists():
        print("❌ Misclassification analysis file not found!")
        return
    
    df = pd.read_csv(misclass_file)
    unanimous_fns = df[(df['Real_Label'] == 1) & (df['Times_Misclassified'] == 32)]
    print(f"Found {len(unanimous_fns)} unanimous FN samples")
    
    # Dictionary to store cluster statistics
    cluster_stats = defaultdict(lambda: {
        'attention_scores': [],
        'sentiments': [],
        'fact_count': 0,
        'sample_count': 0,
        'event_types': set()
    })
    
    # Process each directory
    for i, directory in enumerate(directories):
        print(f"Processing {directory.name} ({i+1}/{len(directories)})...")
        
        # Load the samples file
        samples_file = directory / "samples_all_facts.json"
        if not samples_file.exists():
            print(f"  ❌ No samples_all_facts.json found")
            continue
        
        with open(samples_file, 'r') as f:
            samples_data = json.load(f)
        
        # Process each unanimous FN sample
        for _, row in unanimous_fns.iterrows():
            ticker = row['Ticker']
            date = row['Date']
            
            # Find the sample in this directory
            sample_data = None
            for key, data in samples_data.items():
                metadata = data.get('sample_metadata', {})
                if (metadata.get('primary_ticker') == ticker and 
                    metadata.get('sample_date') == date):
                    sample_data = data
                    break
            
            if sample_data is None:
                continue
            
            # Process facts in this sample
            facts = sample_data['all_facts']
            for fact in facts:
                cluster_id = fact.get('cluster_id')
                if cluster_id is not None:
                    cluster_stats[cluster_id]['attention_scores'].append(fact['attention_score'])
                    if fact['sentiment'] is not None:
                        cluster_stats[cluster_id]['sentiments'].append(fact['sentiment'])
                    cluster_stats[cluster_id]['fact_count'] += 1
                    cluster_stats[cluster_id]['event_types'].add(fact['event_type'])
            
            # Count this sample
            for cluster_id in cluster_stats:
                cluster_stats[cluster_id]['sample_count'] += 1
    
    # Calculate average attention scores and create results
    print("\n📊 Calculating cluster statistics...")
    results = []
    
    for cluster_id, stats in cluster_stats.items():
        if stats['attention_scores']:
            avg_attention = np.mean(stats['attention_scores'])
            std_attention = np.std(stats['attention_scores'])
            avg_sentiment = np.mean(stats['sentiments']) if stats['sentiments'] else 0.0
            std_sentiment = np.std(stats['sentiments']) if len(stats['sentiments']) > 1 else 0.0
            total_facts = stats['fact_count']
            sample_count = stats['sample_count']
            
            # Get sample event types
            sample_event_types = '; '.join(sorted(list(stats['event_types']))[:5])
            
            results.append({
                'cluster_id': cluster_id,
                'avg_attention_score': round(avg_attention, 6),
                'std_attention_score': round(std_attention, 6),
                'avg_sentiment': round(avg_sentiment, 6),
                'std_sentiment': round(std_sentiment, 6),
                'total_facts': total_facts,
                'sample_count': sample_count,
                'avg_facts_per_sample': round(total_facts / sample_count, 2) if sample_count > 0 else 0,
                'negative_attention_rank': None,  # Will be filled later
                'positive_attention_rank': None,  # Will be filled later
                'sample_event_types': sample_event_types
            })
    
    # Load the aggregated cluster analysis to get attention score ranks
    print("📊 Loading attention score ranks from aggregated analysis...")
    
    # Load negative ranks
    negative_file = results_dir / "aggregated_cluster_analysis_negative.csv"
    if negative_file.exists():
        negative_df = pd.read_csv(negative_file)
        negative_rank_dict = dict(zip(negative_df['cluster_id'], negative_df['attention_score_rank']))
        print(f"  Loaded negative ranks for {len(negative_rank_dict)} clusters")
    else:
        print("  ⚠️  Negative aggregated file not found, skipping negative ranks")
        negative_rank_dict = {}
    
    # Load positive ranks
    positive_file = results_dir / "aggregated_cluster_analysis_positive.csv"
    if positive_file.exists():
        positive_df = pd.read_csv(positive_file)
        positive_rank_dict = dict(zip(positive_df['cluster_id'], positive_df['attention_score_rank']))
        print(f"  Loaded positive ranks for {len(positive_rank_dict)} clusters")
    else:
        print("  ⚠️  Positive aggregated file not found, skipping positive ranks")
        positive_rank_dict = {}
    
    # Add attention score ranks to results
    for result in results:
        cluster_id = result['cluster_id']
        result['negative_attention_rank'] = negative_rank_dict.get(cluster_id, 'N/A')
        result['positive_attention_rank'] = positive_rank_dict.get(cluster_id, 'N/A')
    
    # Sort by average attention score and get top 20
    results.sort(key=lambda x: x['avg_attention_score'], reverse=True)
    top_20 = results[:20]
    
    # Create DataFrame and reorder columns to put sample_event_types at the end
    df_results = pd.DataFrame(top_20)
    
    # Reorder columns to put sample_event_types at the end
    column_order = [
        'cluster_id', 'avg_attention_score', 'std_attention_score', 
        'avg_sentiment', 'std_sentiment', 'total_facts', 'sample_count', 
        'avg_facts_per_sample', 'negative_attention_rank', 'positive_attention_rank',
        'sample_event_types'
    ]
    df_results = df_results[column_order]
    
    output_file = results_dir / "unanimous_fn_top_clusters.csv"
    df_results.to_csv(output_file, index=False)
    
    print(f"✅ Saved {output_file}")
    print(f"📊 Top 20 clusters by average attention score in unanimous FN samples:")
    print(df_results[['cluster_id', 'avg_attention_score', 'avg_sentiment', 'total_facts', 'sample_count']].head(10))

def main():
    """Main function."""
    analyze_unanimous_fn_clusters()
    print(f"\n✅ Unanimous FN cluster analysis completed!")

if __name__ == "__main__":
    main()
