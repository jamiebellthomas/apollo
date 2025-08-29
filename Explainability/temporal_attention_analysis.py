#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
import seaborn as sns

# Enable LaTeX rendering
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}"
})

def parse_date(date_str):
    """Parse date string to datetime object."""
    try:
        # Handle different date formats
        if isinstance(date_str, str):
            if len(date_str) == 8:  # YYYYMMDD format
                return datetime.strptime(date_str, '%Y%m%d')
            elif len(date_str) == 10:  # YYYY-MM-DD format
                return datetime.strptime(date_str, '%Y-%m-%d')
            elif len(date_str) == 19:  # YYYY-MM-DD HH:MM:SS format
                return datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
        return None
    except:
        return None

def calculate_time_difference(fact_date, announcement_date):
    """Calculate time difference in days between fact date and announcement date."""
    if fact_date is None or announcement_date is None:
        return None
    
    try:
        fact_dt = parse_date(fact_date)
        announcement_dt = parse_date(announcement_date)
        
        if fact_dt and announcement_dt:
            time_diff = (announcement_dt - fact_dt).days
            return time_diff
    except:
        pass
    
    return None

def create_temporal_attention_plots(directory_path):
    """Create attention weighting vs time difference plots for a single directory."""
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
    all_facts = []
    
    for sample_key, sample_data in data.items():
        predicted_label = sample_data['sample_metadata']['predicted_label']
        sample_date = sample_data['sample_metadata'].get('sample_date')
        
        if not sample_date:
            continue
        
        for fact in sample_data['all_facts']:
            # Calculate time difference for this specific sample
            time_diff = calculate_time_difference(fact['date'], sample_date)
            
            if time_diff is not None:
                fact_data = {
                    'time_diff': time_diff,
                    'attention_score': fact['attention_score'],
                    'event_type': fact['event_type'],
                    'date': fact['date'],
                    'sample_date': sample_date
                }
                
                all_facts.append(fact_data)
                
                if predicted_label == 1:
                    positive_facts.append(fact_data)
                else:
                    negative_facts.append(fact_data)
    
    print(f"  Positive facts: {len(positive_facts)}")
    print(f"  Negative facts: {len(negative_facts)}")
    print(f"  Total facts: {len(all_facts)}")
    
    # Create plots
    if positive_facts:
        create_temporal_plot(positive_facts, directory_path, "positive")
    
    if negative_facts:
        create_temporal_plot(negative_facts, directory_path, "negative")
    
    if all_facts:
        create_temporal_plot(all_facts, directory_path, "overall")
    
    print(f"  ✅ Generated temporal attention plots")

def create_temporal_plot(facts_data, directory_path, prediction_type):
    """Create an attention weighting vs time difference plot."""
    # Extract data
    time_diffs = [f['time_diff'] for f in facts_data]
    attention_scores = [f['attention_score'] for f in facts_data]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with transparency
    plt.scatter(time_diffs, attention_scores, alpha=0.6, s=20, color='blue', label='Facts')
    
    # Add trend line
    if len(time_diffs) > 1:
        z = np.polyfit(time_diffs, attention_scores, 1)
        p = np.poly1d(z)
        plt.plot(time_diffs, p(time_diffs), "r--", alpha=0.8, linewidth=3, label='Linear Trend Line')
    
    # Calculate correlation
    correlation = np.corrcoef(time_diffs, attention_scores)[0, 1]
    
    # Customize the plot with LaTeX formatting
    plt.xlabel(r'Days from Fact Date to Announcement Date', fontsize=12)
    plt.ylabel(r'Attention Weighting', fontsize=12)
    plt.title(r'Attention Weighting vs Time Difference - ' + prediction_type.title() + r' Predictions' + '\n' +
              r'Correlation: ' + f'{correlation:.3f}' + r' | Facts: ' + f'{len(facts_data)}', fontsize=14)
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add legend
    plt.legend(fontsize=10)
    
    # Save the plot
    output_file = directory_path / f"temporal_attention_{prediction_type}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    ✅ Saved {prediction_type} plot: {output_file.name}")

def main():
    """Main function to process all directories."""
    print("🔍 Generating temporal attention plots...")
    
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
            create_temporal_attention_plots(directory)
        except Exception as e:
            print(f"❌ Error processing {directory.name}: {e}")
    
    print(f"\n✅ Temporal attention analysis completed for {len(directories)} directories!")

if __name__ == "__main__":
    main()
