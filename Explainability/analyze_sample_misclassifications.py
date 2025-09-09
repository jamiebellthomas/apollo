#!/usr/bin/env python3
"""
Script to analyze misclassification patterns across all models.
For each sample in the test set, shows how many times it was misclassified across all 32 models.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import defaultdict

def analyze_sample_misclassifications():
    """Analyze misclassification patterns across all models."""
    print("🔍 Analyzing sample misclassifications across all models...")
    
    # Find all HeteroGNN5 results directories
    results_dir = Path("../Results/heterognn5")
    if not results_dir.exists():
        print("❌ Results directory not found!")
        return
    
    directories = sorted([d for d in results_dir.iterdir() if d.is_dir()])
    print(f"Found {len(directories)} directories to process...")
    
    # Dictionary to store results for each sample
    sample_results = defaultdict(lambda: {
        'correct_count': 0,
        'misclassified_count': 0,
        'fp_count': 0,
        'fn_count': 0,
        'attention_entropies': [],
        'fact_count': None,
        'unique_clusters': None
    })
    
    # First, get sample stats from one directory (they should be consistent)
    print("📊 Getting sample statistics...")
    first_directory = directories[0]
    sample_stats = get_all_sample_stats(first_directory)
    
    # Process each directory
    for i, directory in enumerate(directories):
        print(f"Processing {directory.name} ({i+1}/{len(directories)})...")
        
        # Load the test predictions file
        predictions_file = directory / "test_predictions.csv"
        if not predictions_file.exists():
            print(f"  ❌ No test_predictions.csv found")
            continue
        
        df = pd.read_csv(predictions_file)
        print(f"  Loaded {len(df)} samples")
        
        # Process each sample
        for _, row in df.iterrows():
            # Create a unique key for each sample
            sample_key = f"{row['Ticker']}_{row['Reported_Date']}"
            
            actual_label = row['Actual_Label']
            predicted_label = row['Predicted_Label']
            
            # Determine if sample was correctly classified
            if actual_label == predicted_label:
                sample_results[sample_key]['correct_count'] += 1
            else:
                sample_results[sample_key]['misclassified_count'] += 1
                
                # Determine misclassification type
                if actual_label == 0 and predicted_label == 1:
                    sample_results[sample_key]['fp_count'] += 1  # False Positive
                elif actual_label == 1 and predicted_label == 0:
                    sample_results[sample_key]['fn_count'] += 1  # False Negative
            
            # Calculate attention entropy for this sample (only once per directory)
            attention_entropy = calculate_attention_entropy(directory, row['Ticker'], row['Reported_Date'])
            if attention_entropy is not None:
                sample_results[sample_key]['attention_entropies'].append(attention_entropy)
            
            # Get fact count and unique clusters (only once)
            if sample_results[sample_key]['fact_count'] is None and sample_key in sample_stats:
                sample_results[sample_key]['fact_count'] = sample_stats[sample_key]['fact_count']
                sample_results[sample_key]['unique_clusters'] = sample_stats[sample_key]['unique_clusters']
    
    # Create the final analysis DataFrame
    print("\n📊 Creating final analysis...")
    analysis_data = []
    
    for sample_key, results in sample_results.items():
        ticker, date = sample_key.split('_', 1)
        
        # Determine the most common misclassification type
        if results['fp_count'] > results['fn_count']:
            misclassification_type = 'FP'
        elif results['fn_count'] > results['fp_count']:
            misclassification_type = 'FN'
        elif results['fp_count'] == results['fn_count'] and results['fp_count'] > 0:
            misclassification_type = 'Mixed'
        else:
            misclassification_type = 'None'
        
        # Get actual label from the first occurrence (should be consistent)
        first_df = pd.read_csv(directories[0] / "test_predictions.csv")
        sample_row = first_df[(first_df['Ticker'] == ticker) & (first_df['Reported_Date'] == date)]
        if len(sample_row) > 0:
            real_label = sample_row.iloc[0]['Actual_Label']
        else:
            real_label = 'Unknown'
        
        # Calculate mean attention entropy
        mean_attention_entropy = np.mean(results['attention_entropies']) if results['attention_entropies'] else np.nan
        
        # Calculate normalized entropy
        normalized_entropy = None
        if not np.isnan(mean_attention_entropy) and results['fact_count'] and results['fact_count'] > 1:
            normalized_entropy = mean_attention_entropy / np.log(results['fact_count'])
        
        analysis_data.append({
            'Date': date,
            'Ticker': ticker,
            'Real_Label': real_label,
            'Misclassification_Type': misclassification_type,
            'Times_Correctly_Identified': results['correct_count'],
            'Times_Misclassified': results['misclassified_count'],
            'Mean_Attention_Entropy': round(mean_attention_entropy, 6) if not np.isnan(mean_attention_entropy) else np.nan,
            'Normalized_Entropy': round(normalized_entropy, 6) if normalized_entropy is not None else np.nan,
            'Fact_Count': results['fact_count'],
            'Unique_Clusters': results['unique_clusters']
        })
    
    # Create DataFrame and sort by misclassification count
    analysis_df = pd.DataFrame(analysis_data)
    analysis_df = analysis_df.sort_values('Times_Misclassified', ascending=False)
    
    # Save to CSV
    output_file = results_dir / "sample_misclassification_analysis.csv"
    analysis_df.to_csv(output_file, index=False)
    
    print(f"✅ Saved {output_file}")
    
    # Print summary statistics
    print(f"\n📊 Summary Statistics:")
    print(f"Total samples analyzed: {len(analysis_df)}")
    print(f"Samples never misclassified: {len(analysis_df[analysis_df['Times_Misclassified'] == 0])}")
    print(f"Samples misclassified at least once: {len(analysis_df[analysis_df['Times_Misclassified'] > 0])}")
    print(f"Samples misclassified 10+ times: {len(analysis_df[analysis_df['Times_Misclassified'] >= 10])}")
    print(f"Samples misclassified 20+ times: {len(analysis_df[analysis_df['Times_Misclassified'] >= 20])}")
    
    # Show top problematic samples
    print(f"\n🔍 Top 10 Most Problematic Samples:")
    print(analysis_df.head(10)[['Date', 'Ticker', 'Real_Label', 'Misclassification_Type', 'Times_Correctly_Identified', 'Times_Misclassified', 'Mean_Attention_Entropy', 'Normalized_Entropy', 'Fact_Count', 'Unique_Clusters']])
    
    # Show misclassification type breakdown
    print(f"\n📈 Misclassification Type Breakdown:")
    type_counts = analysis_df['Misclassification_Type'].value_counts()
    for misclass_type, count in type_counts.items():
        print(f"  {misclass_type}: {count} samples")

def get_all_sample_stats(directory_path):
    """Get fact count and unique clusters for all samples in a directory."""
    sample_stats = {}
    
    try:
        # Load the samples file
        samples_file = directory_path / "samples_all_facts.json"
        if not samples_file.exists():
            return sample_stats
        
        with open(samples_file, 'r') as f:
            samples_data = json.load(f)
        
        # Process each sample
        for key, sample_data in samples_data.items():
            metadata = sample_data.get('sample_metadata', {})
            ticker = metadata.get('primary_ticker')
            date = metadata.get('sample_date')
            
            if ticker and date:
                sample_key = f"{ticker}_{date}"
                facts = sample_data['all_facts']
                
                fact_count = len(facts)
                unique_clusters = len(set(fact.get('cluster_id') for fact in facts if fact.get('cluster_id') is not None))
                
                sample_stats[sample_key] = {
                    'fact_count': fact_count,
                    'unique_clusters': unique_clusters
                }
        
        print(f"  Processed {len(sample_stats)} samples")
        return sample_stats
    
    except Exception as e:
        print(f"  ⚠️  Error getting sample stats: {e}")
        return sample_stats

def calculate_attention_entropy(directory_path, ticker, date):
    """Calculate attention entropy for a specific sample."""
    try:
        # Load the samples file
        samples_file = directory_path / "samples_all_facts.json"
        if not samples_file.exists():
            return None
        
        with open(samples_file, 'r') as f:
            samples_data = json.load(f)
        
        # Find the sample by matching ticker and date
        sample_data = None
        for key, data in samples_data.items():
            metadata = data.get('sample_metadata', {})
            if (metadata.get('primary_ticker') == ticker and 
                metadata.get('sample_date') == date):
                sample_data = data
                break
        
        if sample_data is None:
            return None
        
        facts = sample_data['all_facts']
        
        if not facts:
            return 0.0
        
        # Extract attention scores
        attention_scores = [fact['attention_score'] for fact in facts]
        
        # Normalize attention scores to create a probability distribution
        total_attention = sum(attention_scores)
        if total_attention == 0:
            return 0.0
        
        probabilities = [score / total_attention for score in attention_scores]
        
        # Calculate entropy: -sum(p * log(p))
        entropy = -sum(p * np.log(p) for p in probabilities if p > 0)
        
        return entropy
    
    except Exception as e:
        return None

def main():
    """Main function."""
    analyze_sample_misclassifications()
    print(f"\n✅ Sample misclassification analysis completed!")

if __name__ == "__main__":
    main()
