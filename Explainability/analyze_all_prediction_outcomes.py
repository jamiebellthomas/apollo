#!/usr/bin/env python3
"""
Script to analyze cluster patterns for all prediction outcomes:
- Unanimous FPs (False Positives)
- Unanimous TPs (True Positives) 
- Unanimous TNs (True Negatives)
- Mixed cases (partially correct/incorrect)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

def analyze_all_prediction_outcomes():
    """Analyze cluster patterns for all prediction outcome types."""
    print("🔍 Analyzing cluster patterns for all prediction outcomes...")
    
    # Find all HeteroGNN5 results directories
    results_dir = Path("../Results/heterognn5")
    if not results_dir.exists():
        print("❌ Results directory not found!")
        return
    
    directories = sorted([d for d in results_dir.iterdir() if d.is_dir()])
    print(f"Found {len(directories)} directories to process...")
    
    # Get sample classifications from the misclassification analysis
    misclass_file = results_dir / "sample_misclassification_analysis.csv"
    if not misclass_file.exists():
        print("❌ Misclassification analysis file not found!")
        return
    
    df = pd.read_csv(misclass_file)
    
    # Classify samples into categories
    unanimous_fps = df[(df['Real_Label'] == 0) & (df['Times_Misclassified'] == 32)]
    unanimous_tps = df[(df['Real_Label'] == 1) & (df['Times_Correctly_Identified'] == 32)]
    unanimous_tns = df[(df['Real_Label'] == 0) & (df['Times_Correctly_Identified'] == 32)]
    mixed_cases = df[(df['Times_Correctly_Identified'] > 0) & (df['Times_Misclassified'] > 0)]
    
    print(f"Found {len(unanimous_fps)} unanimous FPs")
    print(f"Found {len(unanimous_tps)} unanimous TPs")
    print(f"Found {len(unanimous_tns)} unanimous TNs")
    print(f"Found {len(mixed_cases)} mixed cases")
    
    # Analyze each category
    categories = {
        'unanimous_fps': unanimous_fps,
        'unanimous_tps': unanimous_tps,
        'unanimous_tns': unanimous_tns,
        'mixed_cases': mixed_cases
    }
    
    for category_name, category_samples in categories.items():
        if len(category_samples) > 0:
            print(f"\n📊 Analyzing {category_name}...")
            analyze_category(directories, category_samples, category_name, results_dir)
        else:
            print(f"\n⚠️  No {category_name} found, skipping...")

def analyze_category(directories, samples, category_name, results_dir):
    """Analyze cluster patterns for a specific category."""
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
        print(f"  Processing {directory.name} ({i+1}/{len(directories)})...")
        
        # Load the samples file
        samples_file = directory / "samples_all_facts.json"
        if not samples_file.exists():
            print(f"    ❌ No samples_all_facts.json found")
            continue
        
        with open(samples_file, 'r') as f:
            samples_data = json.load(f)
        
        # Process each sample in this category
        for _, row in samples.iterrows():
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
    
    # Calculate statistics and create results
    print(f"  📊 Calculating cluster statistics...")
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
    
    # Load attention score ranks
    print(f"  📊 Loading attention score ranks...")
    
    # Load negative ranks
    negative_file = results_dir / "aggregated_cluster_analysis_negative.csv"
    if negative_file.exists():
        negative_df = pd.read_csv(negative_file)
        negative_rank_dict = dict(zip(negative_df['cluster_id'], negative_df['attention_score_rank']))
    else:
        negative_rank_dict = {}
    
    # Load positive ranks
    positive_file = results_dir / "aggregated_cluster_analysis_positive.csv"
    if positive_file.exists():
        positive_df = pd.read_csv(positive_file)
        positive_rank_dict = dict(zip(positive_df['cluster_id'], positive_df['attention_score_rank']))
    else:
        positive_rank_dict = {}
    
    # Add attention score ranks to results
    for result in results:
        cluster_id = result['cluster_id']
        result['negative_attention_rank'] = negative_rank_dict.get(cluster_id, 'N/A')
        result['positive_attention_rank'] = positive_rank_dict.get(cluster_id, 'N/A')
    
    # Sort by average attention score and get top 20
    results.sort(key=lambda x: x['avg_attention_score'], reverse=True)
    top_20 = results[:20]
    
    # Create DataFrame and reorder columns
    df_results = pd.DataFrame(top_20)
    
    # Reorder columns to put sample_event_types at the end
    column_order = [
        'cluster_id', 'avg_attention_score', 'std_attention_score', 
        'avg_sentiment', 'std_sentiment', 'total_facts', 'sample_count', 
        'avg_facts_per_sample', 'negative_attention_rank', 'positive_attention_rank',
        'sample_event_types'
    ]
    df_results = df_results[column_order]
    
    # Save results
    output_file = results_dir / f"{category_name}_top_clusters.csv"
    df_results.to_csv(output_file, index=False)
    
    print(f"  ✅ Saved {output_file}")
    print(f"  📊 Top 10 clusters by average attention score in {category_name}:")
    print(df_results[['cluster_id', 'avg_attention_score', 'avg_sentiment', 'total_facts', 'sample_count']].head(10))

def main():
    """Main function."""
    analyze_all_prediction_outcomes()
    print(f"\n✅ All prediction outcome analysis completed!")

if __name__ == "__main__":
    main()
