#!/usr/bin/env python3
"""
Summarize Comprehensive Diagnostics Results
Aggregates results from all model analyses
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

def load_diagnostic_results(results_dir="../Results/heterognn5"):
    """Load diagnostic results from all models"""
    results_dir = Path(results_dir)
    all_results = []
    
    for model_dir in results_dir.iterdir():
        if not model_dir.is_dir():
            continue
            
        diagnostic_file = model_dir / "comprehensive_diagnostics.txt"
        if not diagnostic_file.exists():
            continue
            
        try:
            # Parse the diagnostic file
            with open(diagnostic_file, 'r') as f:
                content = f.read()
            
            # Extract basic metrics
            model_results = {
                'model_name': model_dir.name,
                'content': content
            }
            
            # Parse summary statistics
            if 'SUMMARY STATISTICS:' in content:
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if 'TP:' in line and 'FP:' in line:
                        parts = line.split(',')
                        model_results['tp'] = int(parts[0].split(':')[1].strip())
                        model_results['fp'] = int(parts[1].split(':')[1].strip())
                    elif 'TN:' in line and 'FN:' in line:
                        parts = line.split(',')
                        model_results['tn'] = int(parts[0].split(':')[1].strip())
                        model_results['fn'] = int(parts[1].split(':')[1].strip())
                    elif 'Accuracy:' in line:
                        model_results['accuracy'] = float(line.split(':')[1].strip())
                    elif 'Precision:' in line:
                        model_results['precision'] = float(line.split(':')[1].strip())
                    elif 'Recall:' in line:
                        model_results['recall'] = float(line.split(':')[1].strip())
            
            all_results.append(model_results)
            
        except Exception as e:
            print(f"Error loading {model_dir.name}: {e}")
            continue
    
    return all_results

def analyze_cluster_patterns(results):
    """Analyze cluster patterns across models"""
    print("\n=== CLUSTER PATTERN ANALYSIS ===")
    
    # Extract cluster information from each model
    cluster_data = []
    
    for result in results:
        content = result['content']
        model_name = result['model_name']
        
        # Look for cluster analysis sections
        if 'Clusters with highest FN rates' in content:
            lines = content.split('\n')
            for line in lines:
                if 'Cluster ' in line and 'TP:' in line and 'FN:' in line:
                    try:
                        # Parse cluster line
                        parts = line.split(':')
                        cluster_id = parts[0].split('Cluster ')[1].strip()
                        ratio_part = parts[1].split('(')[0].strip()
                        ratio = float(ratio_part)
                        
                        tp_part = line.split('TP: ')[1].split(',')[0]
                        fn_part = line.split('FN: ')[1].split(')')[0]
                        tp_count = int(tp_part)
                        fn_count = int(fn_part)
                        
                        cluster_data.append({
                            'model': model_name,
                            'cluster_id': cluster_id,
                            'ratio': ratio,
                            'tp_count': tp_count,
                            'fn_count': fn_count,
                            'total': tp_count + fn_count
                        })
                    except:
                        continue
    
    if cluster_data:
        df = pd.DataFrame(cluster_data)
        
        # Find most problematic clusters across models
        cluster_summary = df.groupby('cluster_id').agg({
            'ratio': 'mean',
            'total': 'sum',
            'model': 'count'
        }).reset_index()
        
        cluster_summary.columns = ['cluster_id', 'avg_ratio', 'total_occurrences', 'model_count']
        cluster_summary = cluster_summary.sort_values('avg_ratio')
        
        print("\nMost problematic clusters (lowest TP/FN ratios):")
        print(cluster_summary.head(10))
        
        return df, cluster_summary
    else:
        print("No cluster data found")
        return None, None

def analyze_company_patterns(results):
    """Analyze company error patterns across models"""
    print("\n=== COMPANY PATTERN ANALYSIS ===")
    
    company_data = []
    
    for result in results:
        content = result['content']
        model_name = result['model_name']
        
        # Look for company analysis sections
        if 'Companies with Highest Error Rates' in content:
            lines = content.split('\n')
            for line in lines:
                if ':' in line and '[' in line and 'samples' in line:
                    try:
                        # Parse company line
                        parts = line.split(':')
                        ticker = parts[0].strip()
                        
                        error_rate_part = parts[1].split('(')[0].strip()
                        error_rate = float(error_rate_part)
                        
                        sample_part = line.split('[')[1].split(']')[0]
                        sample_count = int(sample_part.split()[0])
                        
                        company_data.append({
                            'model': model_name,
                            'ticker': ticker,
                            'error_rate': error_rate,
                            'sample_count': sample_count
                        })
                    except:
                        continue
    
    if company_data:
        df = pd.DataFrame(company_data)
        
        # Find companies with consistently high error rates
        company_summary = df.groupby('ticker').agg({
            'error_rate': ['mean', 'std', 'count'],
            'sample_count': 'sum'
        }).reset_index()
        
        company_summary.columns = ['ticker', 'avg_error_rate', 'error_std', 'model_count', 'total_samples']
        company_summary = company_summary.sort_values('avg_error_rate', ascending=False)
        
        print("\nCompanies with highest average error rates:")
        print(company_summary.head(15))
        
        return df, company_summary
    else:
        print("No company data found")
        return None, None

def analyze_calibration_patterns(results):
    """Analyze calibration patterns across models"""
    print("\n=== CALIBRATION PATTERN ANALYSIS ===")
    
    calibration_data = []
    
    for result in results:
        content = result['content']
        model_name = result['model_name']
        
        # Look for calibration analysis
        if 'CALIBRATION ANALYSIS' in content:
            lines = content.split('\n')
            for line in lines:
                if 'TP probabilities - Mean:' in line:
                    try:
                        mean_part = line.split('Mean: ')[1].split(',')[0]
                        tp_mean = float(mean_part)
                        
                        std_part = line.split('Std: ')[1]
                        tp_std = float(std_part)
                        
                        calibration_data.append({
                            'model': model_name,
                            'tp_mean': tp_mean,
                            'tp_std': tp_std
                        })
                    except:
                        continue
                elif 'FN probabilities - Mean:' in line:
                    try:
                        mean_part = line.split('Mean: ')[1].split(',')[0]
                        fn_mean = float(mean_part)
                        
                        std_part = line.split('Std: ')[1]
                        fn_std = float(std_part)
                        
                        # Find the corresponding TP entry
                        for entry in calibration_data:
                            if entry['model'] == model_name:
                                entry['fn_mean'] = fn_mean
                                entry['fn_std'] = fn_std
                                break
                    except:
                        continue
                elif 'Optimal threshold for F1:' in line:
                    try:
                        threshold_part = line.split('Optimal threshold for F1: ')[1].split(' ')[0]
                        optimal_threshold = float(threshold_part)
                        
                        # Find the corresponding entry
                        for entry in calibration_data:
                            if entry['model'] == model_name:
                                entry['optimal_threshold'] = optimal_threshold
                                break
                    except:
                        continue
    
    if calibration_data:
        df = pd.DataFrame(calibration_data)
        
        print("\nCalibration Summary:")
        print(f"Average TP probability: {df['tp_mean'].mean():.3f} ± {df['tp_mean'].std():.3f}")
        print(f"Average FN probability: {df['fn_mean'].mean():.3f} ± {df['fn_mean'].std():.3f}")
        print(f"Average optimal threshold: {df['optimal_threshold'].mean():.3f} ± {df['optimal_threshold'].std():.3f}")
        
        # Calculate calibration gap
        df['tp_fn_gap'] = df['tp_mean'] - df['fn_mean']
        print(f"Average TP-FN probability gap: {df['tp_fn_gap'].mean():.3f} ± {df['tp_fn_gap'].std():.3f}")
        
        return df
    else:
        print("No calibration data found")
        return None

def create_summary_report(results):
    """Create a comprehensive summary report"""
    print("=" * 80)
    print("COMPREHENSIVE DIAGNOSTICS SUMMARY REPORT")
    print("=" * 80)
    
    # Basic statistics
    print(f"\nTotal models analyzed: {len(results)}")
    
    if results:
        # Performance metrics
        accuracies = [r.get('accuracy', 0) for r in results if 'accuracy' in r]
        precisions = [r.get('precision', 0) for r in results if 'precision' in r]
        recalls = [r.get('recall', 0) for r in results if 'recall' in r]
        
        print(f"\nPerformance Summary:")
        print(f"Average Accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
        print(f"Average Precision: {np.mean(precisions):.3f} ± {np.std(precisions):.3f}")
        print(f"Average Recall: {np.mean(recalls):.3f} ± {np.std(recalls):.3f}")
        
        # Find best and worst models
        if accuracies:
            best_model = max(results, key=lambda x: x.get('accuracy', 0))
            worst_model = min(results, key=lambda x: x.get('accuracy', 0))
            
            print(f"\nBest performing model: {best_model['model_name']} (Accuracy: {best_model.get('accuracy', 0):.3f})")
            print(f"Worst performing model: {worst_model['model_name']} (Accuracy: {worst_model.get('accuracy', 0):.3f})")
    
    # Analyze patterns
    cluster_df, cluster_summary = analyze_cluster_patterns(results)
    company_df, company_summary = analyze_company_patterns(results)
    calibration_df = analyze_calibration_patterns(results)
    
    # Save summary to file
    summary_file = Path("../Results/diagnostics_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE DIAGNOSTICS SUMMARY REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Total models analyzed: {len(results)}\n\n")
        
        if results:
            f.write(f"Performance Summary:\n")
            f.write(f"Average Accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}\n")
            f.write(f"Average Precision: {np.mean(precisions):.3f} ± {np.std(precisions):.3f}\n")
            f.write(f"Average Recall: {np.mean(recalls):.3f} ± {np.std(recalls):.3f}\n\n")
            
            if accuracies:
                f.write(f"Best performing model: {best_model['model_name']} (Accuracy: {best_model.get('accuracy', 0):.3f})\n")
                f.write(f"Worst performing model: {worst_model['model_name']} (Accuracy: {worst_model.get('accuracy', 0):.3f})\n\n")
        
        if cluster_summary is not None:
            f.write("Most problematic clusters:\n")
            f.write(cluster_summary.head(10).to_string())
            f.write("\n\n")
        
        if company_summary is not None:
            f.write("Companies with highest error rates:\n")
            f.write(company_summary.head(15).to_string())
            f.write("\n\n")
        
        if calibration_df is not None:
            f.write("Calibration Summary:\n")
            f.write(f"Average TP probability: {calibration_df['tp_mean'].mean():.3f} ± {calibration_df['tp_mean'].std():.3f}\n")
            f.write(f"Average FN probability: {calibration_df['fn_mean'].mean():.3f} ± {calibration_df['fn_mean'].std():.3f}\n")
            f.write(f"Average optimal threshold: {calibration_df['optimal_threshold'].mean():.3f} ± {calibration_df['optimal_threshold'].std():.3f}\n")
            f.write(f"Average TP-FN probability gap: {calibration_df['tp_fn_gap'].mean():.3f} ± {calibration_df['tp_fn_gap'].std():.3f}\n")
    
    print(f"\n✅ Summary report saved to: {summary_file}")

def main():
    """Main function"""
    print("Loading diagnostic results...")
    results = load_diagnostic_results()
    
    if not results:
        print("No diagnostic results found!")
        return
    
    create_summary_report(results)

if __name__ == "__main__":
    main()
