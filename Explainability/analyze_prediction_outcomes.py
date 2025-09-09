#!/usr/bin/env python3
"""
Script to analyze fact composition of TP, TN, FP, and FN samples.
For each results directory, analyzes which clusters are most important for each prediction outcome.
"""

import json
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np

def analyze_prediction_outcomes():
    """Analyze fact composition for each prediction outcome type."""
    print("🔍 Analyzing prediction outcomes...")
    
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
        analyze_directory_outcomes(directory)

def analyze_directory_outcomes(directory_path):
    """Analyze prediction outcomes for a single directory."""
    # Load the samples file
    samples_file = directory_path / "samples_all_facts.json"
    if not samples_file.exists():
        print(f"  ❌ No samples_all_facts.json found")
        return
    
    # Load the test predictions file to get actual labels
    predictions_file = directory_path / "test_predictions.csv"
    if not predictions_file.exists():
        print(f"  ❌ No test_predictions.csv found")
        return
    
    with open(samples_file, 'r') as f:
        samples_data = json.load(f)
    
    # Load actual labels from test_predictions.csv
    import pandas as pd
    predictions_df = pd.read_csv(predictions_file)
    actual_labels = predictions_df['Actual_Label'].tolist()
    
    print(f"  Loaded {len(samples_data)} samples")
    print(f"  Loaded {len(actual_labels)} actual labels")
    
    # Initialize outcome data structures
    outcome_stats = {
        'TP': defaultdict(lambda: {
            'fact_count': 0,
            'attention_scores': [],
            'sentiments': [],
            'event_types': set(),
            'sample_count': 0
        }),
        'TN': defaultdict(lambda: {
            'fact_count': 0,
            'attention_scores': [],
            'sentiments': [],
            'event_types': set(),
            'sample_count': 0
        }),
        'FP': defaultdict(lambda: {
            'fact_count': 0,
            'attention_scores': [],
            'sentiments': [],
            'event_types': set(),
            'sample_count': 0
        }),
        'FN': defaultdict(lambda: {
            'fact_count': 0,
            'attention_scores': [],
            'sentiments': [],
            'event_types': set(),
            'sample_count': 0
        })
    }
    
    # Process each sample
    for i, (sample_key, sample_data) in enumerate(samples_data.items()):
        predicted_label = sample_data['sample_metadata']['predicted_label']
        actual_label = actual_labels[i]
        
        # Determine outcome type
        if predicted_label == 1 and actual_label == 1:
            outcome = 'TP'
        elif predicted_label == 0 and actual_label == 0:
            outcome = 'TN'
        elif predicted_label == 1 and actual_label == 0:
            outcome = 'FP'
        elif predicted_label == 0 and actual_label == 1:
            outcome = 'FN'
        else:
            print(f"  ⚠️  Unexpected label combination: pred={predicted_label}, actual={actual_label}")
            continue
        
        # Process facts for this sample
        for fact in sample_data['all_facts']:
            cluster_id = fact.get('cluster_id')
            if cluster_id is not None:
                # Add to outcome statistics
                outcome_stats[outcome][cluster_id]['fact_count'] += 1
                outcome_stats[outcome][cluster_id]['attention_scores'].append(fact['attention_score'])
                if fact['sentiment'] is not None:
                    outcome_stats[outcome][cluster_id]['sentiments'].append(fact['sentiment'])
                outcome_stats[outcome][cluster_id]['event_types'].add(fact['event_type'])
        
        # Increment sample count for this outcome
        for cluster_id in outcome_stats[outcome]:
            outcome_stats[outcome][cluster_id]['sample_count'] += 1
    
    # Generate CSV files for each outcome
    for outcome in ['TP', 'TN', 'FP', 'FN']:
        generate_outcome_csv(directory_path, outcome, outcome_stats[outcome])
    
    # Print summary statistics
    print(f"  📊 Outcome Summary:")
    for outcome in ['TP', 'TN', 'FP', 'FN']:
        total_samples = sum(stats['sample_count'] for stats in outcome_stats[outcome].values())
        total_facts = sum(stats['fact_count'] for stats in outcome_stats[outcome].values())
        print(f"    {outcome}: {total_samples} samples, {total_facts} facts, {len(outcome_stats[outcome])} clusters")

def generate_outcome_csv(directory_path, outcome, cluster_stats):
    """Generate CSV file for a specific outcome type."""
    output_file = directory_path / f"prediction_outcome_{outcome}.csv"
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            'cluster_id', 'fact_count', 'sample_count', 'avg_attention_score', 
            'std_attention_score', 'avg_sentiment', 'std_sentiment', 
            'sample_event_types'
        ])
        
        # Sort clusters by fact count (descending)
        sorted_clusters = sorted(
            cluster_stats.items(),
            key=lambda x: x[1]['fact_count'],
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
            
            # Get sample event types (up to 5)
            sample_event_types = '; '.join(sorted(list(stats['event_types']))[:5])
            
            writer.writerow([
                cluster_id,
                stats['fact_count'],
                stats['sample_count'],
                round(avg_attention, 6),
                round(std_attention, 6),
                round(avg_sentiment, 6),
                round(std_sentiment, 6),
                sample_event_types
            ])
    
    print(f"  ✅ Generated prediction_outcome_{outcome}.csv")

def main():
    """Main function."""
    analyze_prediction_outcomes()
    print(f"\n✅ Prediction outcome analysis completed!")

if __name__ == "__main__":
    main()
