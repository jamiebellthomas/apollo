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

def create_temporal_density_plots(directory_path):
    """Create density plots showing attention weighting distribution across time differences."""
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
        create_density_plot(positive_facts, directory_path, "positive")
    
    if negative_facts:
        create_density_plot(negative_facts, directory_path, "negative")
    
    if all_facts:
        create_density_plot(all_facts, directory_path, "overall")
    
    print(f"  ✅ Generated temporal density plots")

def create_density_plot(facts_data, directory_path, prediction_type):
    """Create a density plot showing attention weighting distribution across time differences."""
    # Extract data
    time_diffs = [f['time_diff'] for f in facts_data]
    attention_scores = [f['attention_score'] for f in facts_data]
    
    # Print data statistics to understand the distribution
    print(f"      Data ranges:")
    print(f"        Time differences: {min(time_diffs)} to {max(time_diffs)} days")
    print(f"        Attention scores: {min(attention_scores):.6f} to {max(attention_scores):.6f}")
    print(f"        Attention percentiles: 25th={np.percentile(attention_scores, 25):.6f}, 50th={np.percentile(attention_scores, 50):.6f}, 75th={np.percentile(attention_scores, 75):.6f}")
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: 2D Density Plot with better binning
    # Use more bins for attention scores since they're likely clustered
    time_bins = np.linspace(min(time_diffs), max(time_diffs), 20)
    
    # For attention scores, use log-spaced bins if they're heavily skewed
    if np.percentile(attention_scores, 25) > 0:  # Check if 25th percentile is not zero
        skewness_ratio = np.percentile(attention_scores, 75) / np.percentile(attention_scores, 25)
        if skewness_ratio > 10:
            # Use log-spaced bins for attention scores
            attention_bins = np.logspace(np.log10(max(0.0001, min(attention_scores))), 
                                       np.log10(max(attention_scores)), 30)
            print(f"        Using log-spaced attention bins (skewed distribution, ratio: {skewness_ratio:.1f})")
        else:
            # Use regular bins
            attention_bins = np.linspace(min(attention_scores), max(attention_scores), 30)
            print(f"        Using linear attention bins (ratio: {skewness_ratio:.1f})")
    else:
        # If 25th percentile is 0, use linear bins
        attention_bins = np.linspace(min(attention_scores), max(attention_scores), 30)
        print(f"        Using linear attention bins (25th percentile is 0)")
    
    # Create 2D histogram with custom bins
    H, xedges, yedges = np.histogram2d(time_diffs, attention_scores, bins=[time_bins, attention_bins])
    
    # Plot 2D density with better color scaling
    im = ax1.imshow(H.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                     aspect='auto', cmap='viridis', norm='log')  # Use log normalization
    
    ax1.set_xlabel(r'\textbf{Days from Fact Date to Announcement Date}', fontsize=12)
    ax1.set_ylabel(r'\textbf{Attention Weighting}', fontsize=12)
    ax1.set_title(r'\textbf{2D Density: Attention Weighting vs Time Difference (Log Scale)}', fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label(r'\textbf{Fact Count (Log Scale)}', rotation=270, labelpad=15)
    
    # Plot 2: Box Plot by Time Bins
    # Create time bins
    time_bins = np.linspace(min(time_diffs), max(time_diffs), 6)
    bin_labels = []
    binned_attention = []
    
    for i in range(len(time_bins) - 1):
        mask = (np.array(time_diffs) >= time_bins[i]) & (np.array(time_diffs) < time_bins[i+1])
        if np.sum(mask) > 0:
            bin_attention = np.array(attention_scores)[mask]
            binned_attention.append(bin_attention)
            bin_labels.append(f'{int(time_bins[i])}-{int(time_bins[i+1])}')
    
    if binned_attention:
        bp = ax2.boxplot(binned_attention, tick_labels=bin_labels, patch_artist=True, showfliers=False)
        # Color boxes
        colors = plt.cm.viridis(np.linspace(0, 1, len(bp['boxes'])))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_xlabel(r'\textbf{Time Difference Bins (Days)}', fontsize=12)
        ax2.set_ylabel(r'\textbf{Attention Weighting}', fontsize=12)
        ax2.set_title(r'\textbf{Attention Weighting Distribution by Time Bins}', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        ax2.tick_params(axis='x', rotation=45)
    
    # Add overall statistics
    correlation = np.corrcoef(time_diffs, attention_scores)[0, 1]
    mean_attention = np.mean(attention_scores)
    
    # Calculate attention by time proximity
    recent_facts = [f for f in facts_data if f['time_diff'] <= 30]  # Within 30 days
    older_facts = [f for f in facts_data if f['time_diff'] > 30]    # More than 30 days
    
    recent_avg = np.mean([f['attention_score'] for f in recent_facts]) if recent_facts else 0
    older_avg = np.mean([f['attention_score'] for f in older_facts]) if older_facts else 0
    
    stats_text = r'$\textbf{Correlation: }$' + f'{correlation:.3f}\n'
    stats_text += r'$\textbf{Mean Attention: }$' + f'{mean_attention:.6f}\n'
    stats_text += r'$\textbf{Recent Facts ($\leq$30d): }$' + f'{len(recent_facts)}\n'
    stats_text += r'$\textbf{Recent Avg Attention: }$' + f'{recent_avg:.6f}\n'
    stats_text += r'$\textbf{Older Facts (>30d): }$' + f'{len(older_facts)}\n'
    stats_text += r'$\textbf{Older Avg Attention: }$' + f'{older_avg:.6f}\n'
    stats_text += r'$\textbf{Total Facts: }$' + f'{len(facts_data)}'
    
    # Statistics info box removed as requested
    
    # Main title - positioned higher to avoid overlap
    fig.suptitle(f'Temporal Attention Density Analysis - {prediction_type.title()} Predictions', 
                 fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = directory_path / f"temporal_attention_density_{prediction_type}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    ✅ Saved {prediction_type} density plot: {output_file.name}")
    
    # Print summary statistics
    print(f"      Recent facts (<=30 days): {len(recent_facts)}, Avg attention: {recent_avg:.6f}")
    print(f"      Older facts (>30 days): {len(older_facts)}, Avg attention: {older_avg:.6f}")
    if len(recent_facts) > 0 and len(older_facts) > 0:
        # Perform t-test to see if difference is statistically significant
        t_stat, p_value = stats.ttest_ind([f['attention_score'] for f in recent_facts], 
                                        [f['attention_score'] for f in older_facts])
        print(f"      T-test p-value: {p_value:.4f} ({'significant' if p_value < 0.05 else 'not significant'})")

def create_aggregated_temporal_density_plot():
    """Create a single aggregated plot combining data from all heterognn5 directories."""
    print("🔍 Creating aggregated temporal attention density plot...")
    
    # Find all HeteroGNN5 results directories
    results_dir = Path("../Results/heterognn5")
    if not results_dir.exists():
        print("❌ Results directory not found!")
        return
    
    directories = sorted([d for d in results_dir.iterdir() if d.is_dir()])
    print(f"Found {len(directories)} directories to aggregate...")
    
    # Collect data from all directories
    all_positive_facts = []
    all_negative_facts = []
    all_overall_facts = []
    
    for directory in directories:
        print(f"  Processing {directory.name}...")
        samples_file = directory / "samples_all_facts.json"
        if not samples_file.exists():
            print(f"    ❌ No samples_all_facts.json found")
            continue
        
        try:
            with open(samples_file, 'r') as f:
                data = json.load(f)
            
            # Separate facts by prediction type
            for sample_key, sample_data in data.items():
                predicted_label = sample_data['sample_metadata']['predicted_label']
                sample_date = sample_data['sample_metadata'].get('sample_date')
                
                if not sample_date:
                    continue
                
                for fact in sample_data['all_facts']:
                    time_diff = calculate_time_difference(fact['date'], sample_date)
                    
                    if time_diff is not None:
                        fact_data = {
                            'time_diff': time_diff,
                            'attention_score': fact['attention_score'],
                            'event_type': fact['event_type'],
                            'date': fact['date'],
                            'sample_date': sample_date,
                            'source_dir': directory.name
                        }
                        
                        all_overall_facts.append(fact_data)
                        
                        if predicted_label == 1:
                            all_positive_facts.append(fact_data)
                        else:
                            all_negative_facts.append(fact_data)
            
            print(f"    ✅ Loaded {len(data)} samples")
            
        except Exception as e:
            print(f"    ❌ Error loading {directory.name}: {e}")
    
    print(f"\n📊 Aggregated data summary:")
    print(f"  Total positive facts: {len(all_positive_facts)}")
    print(f"  Total negative facts: {len(all_negative_facts)}")
    print(f"  Total overall facts: {len(all_overall_facts)}")
    
    # Create aggregated plots
    if all_positive_facts:
        create_aggregated_density_plot(all_positive_facts, "positive")
    
    if all_negative_facts:
        create_aggregated_density_plot(all_negative_facts, "negative")
    
    if all_overall_facts:
        create_aggregated_density_plot(all_overall_facts, "overall")
    
    print(f"\n✅ Aggregated temporal attention density analysis completed!")

def create_aggregated_density_plot(facts_data, prediction_type):
    """Create an aggregated density plot combining data from all directories."""
    # Extract data
    time_diffs = [f['time_diff'] for f in facts_data]
    attention_scores = [f['attention_score'] for f in facts_data]
    
    print(f"    📈 Creating {prediction_type} aggregated plot...")
    print(f"      Data ranges:")
    print(f"        Time differences: {min(time_diffs)} to {max(time_diffs)} days")
    print(f"        Attention scores: {min(attention_scores):.6f} to {max(attention_scores):.6f}")
    print(f"        Attention percentiles: 25th={np.percentile(attention_scores, 25):.6f}, 50th={np.percentile(attention_scores, 50):.6f}, 75th={np.percentile(attention_scores, 75):.6f}")
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: 2D Density Plot
    time_bins = np.linspace(min(time_diffs), max(time_diffs), 25)
    
    # For attention scores, use log-spaced bins if they're heavily skewed
    if np.percentile(attention_scores, 25) > 0:
        skewness_ratio = np.percentile(attention_scores, 75) / np.percentile(attention_scores, 25)
        if skewness_ratio > 10:
            attention_bins = np.logspace(np.log10(max(0.0001, min(attention_scores))), 
                                        np.log10(max(attention_scores)), 35)
            print(f"        Using log-spaced attention bins (skewed distribution, ratio: {skewness_ratio:.1f})")
        else:
            attention_bins = np.linspace(min(attention_scores), max(attention_scores), 35)
            print(f"        Using linear attention bins (ratio: {skewness_ratio:.1f})")
    else:
        attention_bins = np.linspace(min(attention_scores), max(attention_scores), 35)
        print(f"        Using linear attention bins (25th percentile is 0)")
    
    # Create 2D histogram with custom bins
    H, xedges, yedges = np.histogram2d(time_diffs, attention_scores, bins=[time_bins, attention_bins])
    
    # Plot 2D density with better color scaling
    im = ax1.imshow(H.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                     aspect='auto', cmap='viridis', norm='log')
    
    ax1.set_xlabel(r'\textbf{Days from Fact Date to Announcement Date}', fontsize=12)
    ax1.set_ylabel(r'\textbf{Attention Weighting}', fontsize=12)
    ax1.set_title(r'\textbf{Aggregated 2D Density: Attention Weighting vs Time Difference (Log Scale)}', fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label(r'\textbf{Fact Count (Log Scale)}', rotation=270, labelpad=15)
    
    # Plot 2: Box Plot by Time Bins
    # Create time bins
    time_bins = np.linspace(min(time_diffs), max(time_diffs), 8)
    bin_labels = []
    binned_attention = []
    
    for i in range(len(time_bins) - 1):
        mask = (np.array(time_diffs) >= time_bins[i]) & (np.array(time_diffs) < time_bins[i+1])
        if np.sum(mask) > 0:
            bin_attention = np.array(attention_scores)[mask]
            binned_attention.append(bin_attention)
            bin_labels.append(f'{int(time_bins[i])}-{int(time_bins[i+1])}')
    
    if binned_attention:
        bp = ax2.boxplot(binned_attention, tick_labels=bin_labels, patch_artist=True, showfliers=False)
        # Color boxes
        colors = plt.cm.viridis(np.linspace(0, 1, len(bp['boxes'])))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_xlabel(r'\textbf{Time Difference Bins (Days)}', fontsize=12)
        ax2.set_ylabel(r'\textbf{Attention Weighting}', fontsize=12)
        ax2.set_title(r'\textbf{Aggregated Attention Weighting Distribution by Time Bins}', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        ax2.tick_params(axis='x', rotation=45)
    
    # Add overall statistics
    correlation = np.corrcoef(time_diffs, attention_scores)[0, 1]
    mean_attention = np.mean(attention_scores)
    
    # Calculate attention by time proximity
    recent_facts = [f for f in facts_data if f['time_diff'] <= 30]
    older_facts = [f for f in facts_data if f['time_diff'] > 30]
    
    recent_avg = np.mean([f['attention_score'] for f in recent_facts]) if recent_facts else 0
    older_avg = np.mean([f['attention_score'] for f in older_facts]) if older_facts else 0
    
    # Main title - positioned higher to avoid overlap
    fig.suptitle(f'Aggregated Temporal Attention Density Analysis - {prediction_type.title()} Predictions\n(Combined from Multiple HeteroGNN5 Runs)', 
                 fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = Path("../Results/heterognn5") / f"aggregated_temporal_attention_density_{prediction_type}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"      ✅ Saved aggregated {prediction_type} density plot: {output_file.name}")
    
    # Print summary statistics
    print(f"        Recent facts (≤30 days): {len(recent_facts)}, Avg attention: {recent_avg:.6f}")
    print(f"        Older facts (>30 days): {len(older_facts)}, Avg attention: {older_avg:.6f}")
    if len(recent_facts) > 0 and len(older_facts) > 0:
        t_stat, p_value = stats.ttest_ind([f['attention_score'] for f in recent_facts], 
                                        [f['attention_score'] for f in older_facts])
        print(f"        T-test p-value: {p_value:.4f} ({'significant' if p_value < 0.05 else 'not significant'})")

def main():
    """Main function to process all directories."""
    print("🔍 Generating temporal attention density plots...")
    
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
            create_temporal_density_plots(directory)
        except Exception as e:
            print(f"❌ Error processing {directory.name}: {e}")
    
    print(f"\n✅ Temporal attention density analysis completed for {len(directories)} directories!")
    
    # Create aggregated plot
    print(f"\n🔗 Creating aggregated analysis...")
    create_aggregated_temporal_density_plot()

if __name__ == "__main__":
    main()
