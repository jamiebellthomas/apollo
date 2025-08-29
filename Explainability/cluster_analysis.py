#!/usr/bin/env python3
"""
Script to generate cluster analysis CSV files.
Creates cluster_analysis.csv, cluster_analysis_positive.csv, and cluster_analysis_negative.csv
for each results directory, plus weighted versions sorted by weighted attention score.
"""

import json
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np

def analyze_clusters_for_directory(directory_path):
    """Analyze clusters for a single directory."""
    print(f"Processing {directory_path.name}...")
    
    # Load the samples file
    samples_file = directory_path / "samples_all_facts.json"
    if not samples_file.exists():
        print(f"  ❌ No samples_all_facts.json found")
        return
    
    with open(samples_file, 'r') as f:
        data = json.load(f)
    
    print(f"  Loaded {len(data)} samples")
    
    # Initialize cluster data structures
    cluster_stats = defaultdict(lambda: {
        'attention_scores': [],
        'sentiments': [],
        'event_types': set(),
        'fact_count': 0,
        'samples': set()
    })
    
    cluster_stats_positive = defaultdict(lambda: {
        'attention_scores': [],
        'sentiments': [],
        'event_types': set(),
        'fact_count': 0,
        'samples': set()
    })
    
    cluster_stats_negative = defaultdict(lambda: {
        'attention_scores': [],
        'sentiments': [],
        'event_types': set(),
        'fact_count': 0,
        'samples': set()
    })
    
    # Process each sample
    for sample_key, sample_data in data.items():
        predicted_label = sample_data['sample_metadata']['predicted_label']
        
        # Choose which stats dict to use based on prediction
        if predicted_label == 1:
            current_stats = cluster_stats_positive
        else:
            current_stats = cluster_stats_negative
        
        # Process all facts in this sample
        for fact in sample_data['all_facts']:
            cluster_id = fact.get('cluster_id')
            if cluster_id is not None:
                # Add to overall stats
                cluster_stats[cluster_id]['attention_scores'].append(fact['attention_score'])
                if fact['sentiment'] is not None:
                    cluster_stats[cluster_id]['sentiments'].append(fact['sentiment'])
                cluster_stats[cluster_id]['event_types'].add(fact['event_type'])
                cluster_stats[cluster_id]['fact_count'] += 1
                cluster_stats[cluster_id]['samples'].add(sample_key)
                
                # Add to prediction-specific stats
                current_stats[cluster_id]['attention_scores'].append(fact['attention_score'])
                if fact['sentiment'] is not None:
                    current_stats[cluster_id]['sentiments'].append(fact['sentiment'])
                current_stats[cluster_id]['event_types'].add(fact['event_type'])
                current_stats[cluster_id]['fact_count'] += 1
                current_stats[cluster_id]['samples'].add(sample_key)
    
    print(f"  Found {len(cluster_stats)} clusters overall")
    print(f"  Found {len(cluster_stats_positive)} clusters in positive predictions")
    print(f"  Found {len(cluster_stats_negative)} clusters in negative predictions")
    
    # Generate CSV files sorted by average attention score
    generate_cluster_csv(directory_path, cluster_stats, "cluster_analysis.csv", "overall")
    generate_cluster_csv(directory_path, cluster_stats_positive, "cluster_analysis_positive.csv", "positive")
    generate_cluster_csv(directory_path, cluster_stats_negative, "cluster_analysis_negative.csv", "negative")
    
    # Generate CSV files sorted by weighted attention score
    generate_weighted_cluster_csv(directory_path, cluster_stats, "cluster_analysis_weighted.csv", "overall")
    generate_weighted_cluster_csv(directory_path, cluster_stats_positive, "cluster_analysis_positive_weighted.csv", "positive")
    generate_weighted_cluster_csv(directory_path, cluster_stats_negative, "cluster_analysis_negative_weighted.csv", "negative")
    
    print(f"  ✅ Generated cluster analysis files")

def generate_cluster_csv(directory_path, cluster_stats, filename, analysis_type):
    """Generate a cluster analysis CSV file sorted by average attention score."""
    output_file = directory_path / filename
    
    # Load the samples file to get total sample count
    samples_file = directory_path / "samples_all_facts.json"
    with open(samples_file, 'r') as f:
        data = json.load(f)
    total_samples = len(data)
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            'cluster_id', 'fact_count', 'sample_count', 'percentage_of_samples', 'avg_attention_score', 
            'std_attention_score', 'avg_sentiment', 
            'std_sentiment', 'sample_event_types'
        ])
        
        # Sort clusters by average attention score (descending)
        sorted_clusters = sorted(
            cluster_stats.items(),
            key=lambda x: np.mean(x[1]['attention_scores']) if x[1]['attention_scores'] else 0,
            reverse=True
        )
        
        # Write data rows
        for cluster_id, stats in sorted_clusters:
            attention_scores = stats['attention_scores']
            sentiments = stats['sentiments']
            
            # Calculate statistics
            avg_attention = np.mean(attention_scores) if attention_scores else 0.0
            std_attention = np.std(attention_scores) if len(attention_scores) > 1 else 0.0
            avg_sentiment = np.mean(sentiments) if sentiments else 0.0
            std_sentiment = np.std(sentiments) if len(sentiments) > 1 else 0.0
            
            # Calculate sample count and percentage
            sample_count = len(stats['samples'])
            percentage_of_samples = (sample_count / total_samples) * 100 if total_samples > 0 else 0.0
            
            # Get sample event types (up to 5)
            sample_event_types = '; '.join(sorted(list(stats['event_types']))[:5])
            
            writer.writerow([
                cluster_id,
                stats['fact_count'],
                sample_count,
                round(percentage_of_samples, 6),
                round(avg_attention, 6),
                round(std_attention, 6),
                round(avg_sentiment, 6),
                round(std_sentiment, 6),
                sample_event_types
            ])

def generate_weighted_cluster_csv(directory_path, cluster_stats, filename, analysis_type):
    """Generate a cluster analysis CSV file sorted by weighted attention score."""
    output_file = directory_path / filename
    
    # Calculate total facts for proportion calculation
    total_facts = sum(stats['fact_count'] for stats in cluster_stats.values())
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            'cluster_id', 'fact_count', 'cluster_proportion', 'avg_attention_score', 
            'std_attention_score', 'weighted_attention_score', 'avg_sentiment', 
            'std_sentiment', 'sample_event_types'
        ])
        
        # Sort clusters by weighted attention score (descending)
        sorted_clusters = sorted(
            cluster_stats.items(),
            key=lambda x: (np.mean(x[1]['attention_scores']) if x[1]['attention_scores'] else 0) * 
                         (x[1]['fact_count'] / total_facts if total_facts > 0 else 0),
            reverse=True
        )
        
        # Write data rows
        for cluster_id, stats in sorted_clusters:
            attention_scores = stats['attention_scores']
            sentiments = stats['sentiments']
            
            # Calculate statistics
            avg_attention = np.mean(attention_scores) if attention_scores else 0.0
            std_attention = np.std(attention_scores) if len(attention_scores) > 1 else 0.0
            avg_sentiment = np.mean(sentiments) if sentiments else 0.0
            std_sentiment = np.std(sentiments) if len(sentiments) > 1 else 0.0
            
            # Calculate cluster proportion and weighted attention score
            cluster_proportion = stats['fact_count'] / total_facts if total_facts > 0 else 0.0
            weighted_attention_score = avg_attention * cluster_proportion
            
            # Get sample event types (up to 5)
            sample_event_types = '; '.join(sorted(list(stats['event_types']))[:5])
            
            writer.writerow([
                cluster_id,
                stats['fact_count'],
                round(cluster_proportion, 6),
                round(avg_attention, 6),
                round(std_attention, 6),
                round(weighted_attention_score, 6),
                round(avg_sentiment, 6),
                round(std_sentiment, 6),
                sample_event_types
            ])

def main():
    """Main function to process all directories."""
    print("🔍 Generating cluster analysis files...")
    
    # Find all HeteroGNN5 results directories
    results_dir = Path("../Results/heterognn5")
    if not results_dir.exists():
        print("❌ Results directory not found!")
        return
    
    directories = sorted([d for d in results_dir.iterdir() if d.is_dir()])
    print(f"Found {len(directories)} directories to process...")
    
    # Process each directory
    for directory in directories:
        try:
            analyze_clusters_for_directory(directory)
        except Exception as e:
            print(f"❌ Error processing {directory.name}: {e}")
    
    print(f"\n✅ Cluster analysis completed for {len(directories)} directories!")

if __name__ == "__main__":
    main()
