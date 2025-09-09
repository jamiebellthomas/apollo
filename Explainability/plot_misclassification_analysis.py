#!/usr/bin/env python3
"""
Script to create plots for misclassification analysis.
- Boxplots comparing TPs vs unanimous FNs
- Scatter plots of entropy vs fact count/cluster count colored by error type
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def create_misclassification_plots():
    """Create plots for misclassification analysis."""
    print("📊 Creating misclassification analysis plots...")
    
    # Load the data
    data_file = Path("../Results/heterognn5/sample_misclassification_analysis.csv")
    if not data_file.exists():
        print("❌ Data file not found!")
        return
    
    df = pd.read_csv(data_file)
    print(f"Loaded {len(df)} samples")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Misclassification Analysis: Attention Patterns and Sample Complexity', fontsize=16, fontweight='bold')
    
    # 1. Boxplot: TPs vs Unanimous FNs (Mean Attention Entropy)
    print("Creating boxplot 1: Mean Attention Entropy...")
    create_entropy_boxplot(df, axes[0, 0], 'Mean_Attention_Entropy', 'Mean Attention Entropy')
    
    # 2. Boxplot: TPs vs Unanimous FNs (Normalized Entropy)
    print("Creating boxplot 2: Normalized Entropy...")
    create_entropy_boxplot(df, axes[0, 1], 'Normalized_Entropy', 'Normalized Entropy')
    
    # 3. Scatter plot: Entropy vs Fact Count
    print("Creating scatter plot 1: Entropy vs Fact Count...")
    create_entropy_scatter(df, axes[1, 0], 'Fact_Count', 'Fact Count')
    
    # 4. Scatter plot: Entropy vs Cluster Count
    print("Creating scatter plot 2: Entropy vs Cluster Count...")
    create_entropy_scatter(df, axes[1, 1], 'Unique_Clusters', 'Unique Clusters')
    
    # Adjust layout and save
    plt.tight_layout()
    output_file = Path("../Results/heterognn5/misclassification_analysis_plots.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved {output_file}")
    
    # Create additional detailed plots
    create_detailed_plots(df)

def create_entropy_boxplot(df, ax, entropy_col, title):
    """Create boxplot comparing TPs vs unanimous FNs."""
    # Define groups
    tps = df[(df['Real_Label'] == 1) & (df['Times_Misclassified'] == 0)]  # True Positives
    unanimous_fns = df[(df['Real_Label'] == 1) & (df['Times_Misclassified'] == 32)]  # Unanimous False Negatives
    
    # Prepare data for boxplot
    data_to_plot = []
    labels = []
    
    if len(tps) > 0:
        data_to_plot.append(tps[entropy_col].dropna())
        labels.append(f'True Positives\n(n={len(tps)})')
    
    if len(unanimous_fns) > 0:
        data_to_plot.append(unanimous_fns[entropy_col].dropna())
        labels.append(f'Unanimous False Negatives\n(n={len(unanimous_fns)})')
    
    if data_to_plot:
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        
        # Color the boxes
        colors = ['lightgreen', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title(f'{title}: TPs vs Unanimous FNs', fontweight='bold')
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
        
        # Add statistical comparison
        if len(data_to_plot) == 2:
            t_stat, p_value = perform_ttest(data_to_plot[0], data_to_plot[1])
            ax.text(0.02, 0.98, f't-test p-value: {p_value:.4f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)

def create_entropy_scatter(df, ax, x_col, x_label):
    """Create scatter plot of entropy vs specified variable."""
    # Filter out samples with missing data
    plot_df = df.dropna(subset=['Mean_Attention_Entropy', x_col])
    
    # Create color mapping for error types
    color_map = {
        'None': 'green',      # Never misclassified
        'FN': 'red',          # False Negatives
        'FP': 'orange',       # False Positives
        'Mixed': 'purple'     # Mixed errors
    }
    
    # Plot each error type
    for error_type, color in color_map.items():
        subset = plot_df[plot_df['Misclassification_Type'] == error_type]
        if len(subset) > 0:
            ax.scatter(subset[x_col], subset['Mean_Attention_Entropy'], 
                      c=color, alpha=0.6, s=50, label=f'{error_type} (n={len(subset)})')
    
    ax.set_xlabel(x_label)
    ax.set_ylabel('Mean Attention Entropy')
    ax.set_title(f'Attention Entropy vs {x_label}', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

def perform_ttest(group1, group2):
    """Perform t-test between two groups."""
    from scipy import stats
    try:
        t_stat, p_value = stats.ttest_ind(group1, group2)
        return t_stat, p_value
    except:
        return None, None

def create_detailed_plots(df):
    """Create additional detailed plots."""
    print("Creating detailed plots...")
    
    # Create a new figure for detailed analysis
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Detailed Misclassification Analysis', fontsize=16, fontweight='bold')
    
    # 1. Violin plot: Entropy distribution by misclassification type
    print("Creating violin plot...")
    plot_df = df.dropna(subset=['Mean_Attention_Entropy'])
    
    sns.violinplot(data=plot_df, x='Misclassification_Type', y='Mean_Attention_Entropy', ax=axes[0])
    axes[0].set_title('Attention Entropy Distribution by Error Type', fontweight='bold')
    axes[0].set_xlabel('Misclassification Type')
    axes[0].set_ylabel('Mean Attention Entropy')
    axes[0].grid(True, alpha=0.3)
    
    # 2. Scatter plot: Normalized Entropy vs Misclassification Count
    print("Creating normalized entropy scatter...")
    plot_df = df.dropna(subset=['Normalized_Entropy'])
    
    # Color by misclassification type
    for error_type, color in {'None': 'green', 'FN': 'red', 'FP': 'orange', 'Mixed': 'purple'}.items():
        subset = plot_df[plot_df['Misclassification_Type'] == error_type]
        if len(subset) > 0:
            axes[1].scatter(subset['Times_Misclassified'], subset['Normalized_Entropy'], 
                           c=color, alpha=0.6, s=50, label=f'{error_type} (n={len(subset)})')
    
    axes[1].set_xlabel('Times Misclassified')
    axes[1].set_ylabel('Normalized Entropy')
    axes[1].set_title('Normalized Entropy vs Misclassification Frequency', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Save detailed plots
    plt.tight_layout()
    output_file = Path("../Results/heterognn5/misclassification_detailed_plots.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved {output_file}")
    
    # Print summary statistics
    print_summary_statistics(df)

def print_summary_statistics(df):
    """Print summary statistics for the analysis."""
    print("\n📊 Summary Statistics:")
    
    # Group by misclassification type
    for error_type in ['None', 'FN', 'FP', 'Mixed']:
        subset = df[df['Misclassification_Type'] == error_type]
        if len(subset) > 0:
            print(f"\n{error_type} samples (n={len(subset)}):")
            print(f"  Mean Attention Entropy: {subset['Mean_Attention_Entropy'].mean():.3f}")
            print(f"  Mean Normalized Entropy: {subset['Normalized_Entropy'].mean():.3f}")
            print(f"  Mean Fact Count: {subset['Fact_Count'].mean():.1f}")
            print(f"  Mean Unique Clusters: {subset['Unique_Clusters'].mean():.1f}")
    
    # Compare TPs vs unanimous FNs
    tps = df[(df['Real_Label'] == 1) & (df['Times_Misclassified'] == 0)]
    unanimous_fns = df[(df['Real_Label'] == 1) & (df['Times_Misclassified'] == 32)]
    
    print(f"\n🔍 Key Comparison (TPs vs Unanimous FNs):")
    print(f"True Positives (n={len(tps)}):")
    print(f"  Mean Attention Entropy: {tps['Mean_Attention_Entropy'].mean():.3f}")
    print(f"  Mean Normalized Entropy: {tps['Normalized_Entropy'].mean():.3f}")
    
    print(f"Unanimous False Negatives (n={len(unanimous_fns)}):")
    print(f"  Mean Attention Entropy: {unanimous_fns['Mean_Attention_Entropy'].mean():.3f}")
    print(f"  Mean Normalized Entropy: {unanimous_fns['Normalized_Entropy'].mean():.3f}")

def main():
    """Main function."""
    create_misclassification_plots()
    print(f"\n✅ Misclassification analysis plots completed!")

if __name__ == "__main__":
    main()
