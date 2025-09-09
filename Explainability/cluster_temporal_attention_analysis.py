#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
import seaborn as sns
from scipy import stats

# Enable LaTeX rendering for better text formatting
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

def create_cluster_temporal_attention_plots():
    """Create temporal attention plots for each cluster, aggregated across all models."""
    print("🔍 Creating cluster temporal attention analysis...")
    
    # Find all HeteroGNN5 results directories
    results_dir = Path("../Results/heterognn5")
    if not results_dir.exists():
        print("❌ Results directory not found!")
        return
    
    directories = sorted([d for d in results_dir.iterdir() if d.is_dir()])
    print(f"Found {len(directories)} directories to analyze...")
    
    # Collect cluster data from all directories
    cluster_data = {}
    
    for directory in directories:
        print(f"  Processing {directory.name}...")
        samples_file = directory / "samples_all_facts.json"
        if not samples_file.exists():
            print(f"    ❌ No samples_all_facts.json found")
            continue
        
        try:
            with open(samples_file, 'r') as f:
                data = json.load(f)
            
            # Process each sample
            for sample_key, sample_data in data.items():
                sample_date = sample_data['sample_metadata'].get('sample_date')
                
                if not sample_date:
                    continue
                
                # Process all facts in this sample
                for fact in sample_data['all_facts']:
                    cluster_id = fact.get('cluster_id')
                    if cluster_id is not None:
                        time_diff = calculate_time_difference(fact['date'], sample_date)
                        
                        if time_diff is not None:
                            if cluster_id not in cluster_data:
                                cluster_data[cluster_id] = {
                                    'time_diffs': [],
                                    'attention_scores': [],
                                    'source_dirs': set()
                                }
                            
                            cluster_data[cluster_id]['time_diffs'].append(time_diff)
                            cluster_data[cluster_id]['attention_scores'].append(fact['attention_score'])
                            cluster_data[cluster_id]['source_dirs'].add(directory.name)
            
            print(f"    ✅ Loaded {len(data)} samples")
            
        except Exception as e:
            print(f"    ❌ Error loading {directory.name}: {e}")
    
    print(f"\n📊 Cluster data summary:")
    print(f"  Total clusters found: {len(cluster_data)}")
    
    # Create plots for positive, negative, and overall predictions
    create_cluster_temporal_plots(cluster_data, "overall")
    
    # Create gradient-highlighted plots for overall predictions
    create_gradient_highlighted_plots(cluster_data, "overall")
    
    # Also create separate plots for positive and negative predictions
    create_prediction_specific_cluster_plots(directories, "positive")
    create_prediction_specific_cluster_plots(directories, "negative")
    
    print(f"\n✅ Cluster temporal attention analysis completed!")

def create_cluster_temporal_plots(cluster_data, prediction_type):
    """Create temporal attention plots for each cluster with lines of best fit."""
    print(f"    📈 Creating {prediction_type} cluster temporal plots...")
    
    # Filter clusters with sufficient data
    valid_clusters = {k: v for k, v in cluster_data.items() 
                     if len(v['time_diffs']) >= 10}  # At least 10 data points
    
    print(f"      Found {len(valid_clusters)} clusters with sufficient data")
    
    # Sort clusters by average attention score for better visualization
    sorted_clusters = sorted(valid_clusters.items(), 
                           key=lambda x: np.mean(x[1]['attention_scores']), 
                           reverse=True)
    
    # Create single plot with all clusters
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Color map for different clusters
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_clusters)))
    
    # Plot each cluster's line of best fit
    for i, (cluster_id, data) in enumerate(sorted_clusters):
        time_diffs = np.array(data['time_diffs'])
        attention_scores = np.array(data['attention_scores'])
        
        # Add line of best fit
        if len(time_diffs) > 1:
            z = np.polyfit(time_diffs, attention_scores, 1)
            p = np.poly1d(z)
            x_range = np.linspace(min(time_diffs), max(time_diffs), 100)
            ax.plot(x_range, p(x_range), color=colors[i], alpha=0.7, linewidth=2, 
                   label=f'Cluster {cluster_id} (r={np.corrcoef(time_diffs, attention_scores)[0, 1]:.3f})')
    
    # Customize plot
    ax.set_xlabel('Days from Fact to Announcement Date', fontsize=14)
    ax.set_ylabel('Attention Weighting', fontsize=14)
    ax.set_title(f'Cluster Temporal Attention Analysis - {prediction_type.title()} Predictions\n' +
                 f'All Lines of Best Fit: Attention Weighting vs Days from Announcement Date\n' +
                 f'(Aggregated from Multiple HeteroGNN5 Runs)', fontsize=16)
    
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = Path("../Results/heterognn5") / f"cluster_temporal_attention_{prediction_type}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"      ✅ Saved {prediction_type} cluster temporal plot: {output_file.name}")

def create_gradient_highlighted_plots(cluster_data, prediction_type):
    """Create plots highlighting positive and negative gradients separately."""
    print(f"    📈 Creating gradient-highlighted plots for {prediction_type} predictions...")
    
    # Filter clusters with sufficient data
    valid_clusters = {k: v for k, v in cluster_data.items() 
                     if len(v['time_diffs']) >= 10}  # At least 10 data points
    
    # Sort clusters by average attention score
    sorted_clusters = sorted(valid_clusters.items(), 
                           key=lambda x: np.mean(x[1]['attention_scores']), 
                           reverse=True)
    
    # Separate clusters by gradient direction
    positive_gradient_clusters = []
    negative_gradient_clusters = []
    
    for cluster_id, data in sorted_clusters:
        time_diffs = np.array(data['time_diffs'])
        attention_scores = np.array(data['attention_scores'])
        
        if len(time_diffs) > 1:
            # Calculate gradient (slope of line of best fit)
            z = np.polyfit(time_diffs, attention_scores, 1)
            slope = z[0]  # First coefficient is the slope
            
            if slope > 0:
                positive_gradient_clusters.append((cluster_id, data, slope))
            else:
                negative_gradient_clusters.append((cluster_id, data, slope))
    
    print(f"      Found {len(positive_gradient_clusters)} positive gradient clusters")
    print(f"      Found {len(negative_gradient_clusters)} negative gradient clusters")
    
    # Create plot highlighting positive gradients (negative gradients in grey)
    create_gradient_plot(positive_gradient_clusters, negative_gradient_clusters, 
                        "positive", prediction_type)
    
    # Create plot highlighting negative gradients (positive gradients in grey)
    create_gradient_plot(negative_gradient_clusters, positive_gradient_clusters, 
                        "negative", prediction_type)

def create_gradient_plot(highlighted_clusters, grey_clusters, gradient_type, prediction_type):
    """Create a plot highlighting specific gradient types."""
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Plot grey clusters first (background)
    for cluster_id, data, slope in grey_clusters:
        time_diffs = np.array(data['time_diffs'])
        attention_scores = np.array(data['attention_scores'])
        
        if len(time_diffs) > 1:
            z = np.polyfit(time_diffs, attention_scores, 1)
            p = np.poly1d(z)
            x_range = np.linspace(min(time_diffs), max(time_diffs), 100)
            ax.plot(x_range, p(x_range), color='lightgrey', alpha=0.3, linewidth=1)
    
    # Plot highlighted clusters with colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(highlighted_clusters)))
    
    for i, (cluster_id, data, slope) in enumerate(highlighted_clusters):
        time_diffs = np.array(data['time_diffs'])
        attention_scores = np.array(data['attention_scores'])
        
        if len(time_diffs) > 1:
            z = np.polyfit(time_diffs, attention_scores, 1)
            p = np.poly1d(z)
            x_range = np.linspace(min(time_diffs), max(time_diffs), 100)
            ax.plot(x_range, p(x_range), color=colors[i], alpha=0.8, linewidth=2.5,
                   label=f'Cluster {cluster_id} (slope={slope:.6f})')
    
    # Customize plot
    ax.set_xlabel('Days from Fact to Announcement Date', fontsize=14)
    ax.set_ylabel('Attention Weighting', fontsize=14)
    
    if gradient_type == "positive":
        title = f'Cluster Temporal Attention Analysis - {prediction_type.title()} Predictions\n' + \
                f'Positive Gradients Highlighted (Negative Gradients in Grey)\n' + \
                f'(Aggregated from Multiple HeteroGNN5 Runs)'
    else:
        title = f'Cluster Temporal Attention Analysis - {prediction_type.title()} Predictions\n' + \
                f'Negative Gradients Highlighted (Positive Gradients in Grey)\n' + \
                f'(Aggregated from Multiple HeteroGNN5 Runs)'
    
    ax.set_title(title, fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = Path("../Results/heterognn5") / f"cluster_temporal_attention_{prediction_type}_{gradient_type}_gradients.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"        ✅ Saved {gradient_type} gradient plot: {output_file.name}")

def create_prediction_specific_cluster_plots(directories, prediction_type):
    """Create temporal attention plots for specific prediction types (positive/negative)."""
    print(f"    📈 Creating {prediction_type}-specific cluster temporal plots...")
    
    # Collect cluster data for specific prediction type
    cluster_data = {}
    
    for directory in directories:
        samples_file = directory / "samples_all_facts.json"
        if not samples_file.exists():
            continue
        
        try:
            with open(samples_file, 'r') as f:
                data = json.load(f)
            
            # Process each sample
            for sample_key, sample_data in data.items():
                predicted_label = sample_data['sample_metadata']['predicted_label']
                sample_date = sample_data['sample_metadata'].get('sample_date')
                
                # Only process samples of the specified prediction type
                if (prediction_type == "positive" and predicted_label != 1) or \
                   (prediction_type == "negative" and predicted_label != 0):
                    continue
                
                if not sample_date:
                    continue
                
                # Process all facts in this sample
                for fact in sample_data['all_facts']:
                    cluster_id = fact.get('cluster_id')
                    if cluster_id is not None:
                        time_diff = calculate_time_difference(fact['date'], sample_date)
                        
                        if time_diff is not None:
                            if cluster_id not in cluster_data:
                                cluster_data[cluster_id] = {
                                    'time_diffs': [],
                                    'attention_scores': [],
                                    'source_dirs': set()
                                }
                            
                            cluster_data[cluster_id]['time_diffs'].append(time_diff)
                            cluster_data[cluster_id]['attention_scores'].append(fact['attention_score'])
                            cluster_data[cluster_id]['source_dirs'].add(directory.name)
            
        except Exception as e:
            print(f"    ❌ Error loading {directory.name}: {e}")
    
    # Create plots for this prediction type
    if cluster_data:
        create_cluster_temporal_plots(cluster_data, prediction_type)
    else:
        print(f"      ❌ No data found for {prediction_type} predictions")

def main():
    """Main function to create cluster temporal attention analysis."""
    print("🔍 Starting cluster temporal attention analysis...")
    create_cluster_temporal_attention_plots()

if __name__ == "__main__":
    main()
