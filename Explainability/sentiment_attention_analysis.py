#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

def create_sentiment_attention_plots(directory_path):
    """Create sentiment vs attention weighting plots for a single directory."""
    print(f"Processing {directory_path.name}...")
    
    # Load the samples file
    samples_file = directory_path / "samples_all_facts.json"
    if not samples_file.exists():
        print(f"  ❌ No samples_all_facts.json found")
        return
    
    with open(samples_file, 'r') as f:
        data = json.load(f)
    
    print(f"  Loaded {len(data)} samples")
    
    # Separate facts by prediction type
    positive_facts = []
    negative_facts = []
    
    for sample_key, sample_data in data.items():
        predicted_label = sample_data['sample_metadata']['predicted_label']
        
        for fact in sample_data['all_facts']:
            # Only include facts with valid sentiment scores
            if fact['sentiment'] is not None:
                fact_data = {
                    'sentiment': fact['sentiment'],
                    'attention_score': fact['attention_score'],
                    'event_type': fact['event_type']
                }
                
                if predicted_label == 1:
                    positive_facts.append(fact_data)
                else:
                    negative_facts.append(fact_data)
    
    print(f"  Positive facts: {len(positive_facts)}")
    print(f"  Negative facts: {len(negative_facts)}")
    
    # Create plots
    if positive_facts:
        create_plot(positive_facts, directory_path, "positive")
    
    if negative_facts:
        create_plot(negative_facts, directory_path, "negative")
    
    print(f"  ✅ Generated sentiment-attention plots")

def create_plot(facts_data, directory_path, prediction_type):
    """Create a sentiment vs attention weighting plot."""
    # Extract data
    sentiments = [f['sentiment'] for f in facts_data]
    attention_scores = [f['attention_score'] for f in facts_data]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with transparency
    plt.scatter(sentiments, attention_scores, alpha=0.6, s=20, color='blue')
    
    # Add trend line
    if len(sentiments) > 1:
        z = np.polyfit(sentiments, attention_scores, 1)
        p = np.poly1d(z)
        plt.plot(sentiments, p(sentiments), "r--", alpha=0.8, linewidth=2)
    
    # Calculate correlation
    correlation = np.corrcoef(sentiments, attention_scores)[0, 1]
    
    # Customize the plot
    plt.xlabel('Sentiment Score', fontsize=12)
    plt.ylabel('Attention Weighting', fontsize=12)
    plt.title(f'Sentiment vs Attention Weighting - {prediction_type.title()} Predictions\n'
              f'Correlation: {correlation:.3f} | Facts: {len(facts_data)}', fontsize=14)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add statistics text box
    stats_text = f'Mean Sentiment: {np.mean(sentiments):.3f}\n'
    stats_text += f'Mean Attention: {np.mean(attention_scores):.3f}\n'
    stats_text += f'Correlation: {correlation:.3f}\n'
    stats_text += f'Total Facts: {len(facts_data)}'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Save the plot
    output_file = directory_path / f"sentiment_attention_{prediction_type}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    ✅ Saved {prediction_type} plot: {output_file.name}")

def main():
    """Main function to process all directories."""
    print("🔍 Generating sentiment-attention plots...")
    
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
            create_sentiment_attention_plots(directory)
        except Exception as e:
            print(f"❌ Error processing {directory.name}: {e}")
    
    print(f"\n✅ Sentiment-attention analysis completed for {len(directories)} directories!")

if __name__ == "__main__":
    main()
