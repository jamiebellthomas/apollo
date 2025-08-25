#!/usr/bin/env python3
"""
Aggregate Cluster Analysis Results Across All Models

This script analyzes patterns across all 32 models' cluster analysis results to find:
1. Most consistently influential clusters
2. Cluster performance patterns (TP vs FN ratios)
3. Sentiment patterns across clusters
4. Model-to-model consistency
5. Anomalous models or clusters
"""

import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

class ClusterAnalysisAggregator:
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.model_dirs = [d for d in results_dir.iterdir() if d.is_dir() and (d / "model.pt").exists()]
        self.aggregated_data = {}
        
    def parse_cluster_file(self, file_path: Path) -> Dict:
        """Parse a single cluster_ranking_comprehensive.txt file"""
        data = {
            'tp_clusters': {},
            'fp_clusters': {},
            'tn_clusters': {},
            'fn_clusters': {},
            'model_name': file_path.parent.name
        }
        
        current_section = None
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Parse each section
        sections = {
            'TRUE POSITIVES': 'tp_clusters',
            'FALSE POSITIVES': 'fp_clusters', 
            'TRUE NEGATIVES': 'tn_clusters',
            'FALSE NEGATIVES': 'fn_clusters'
        }
        
        for section_name, data_key in sections.items():
            # Extract section content
            section_match = re.search(f"{section_name}.*?={80}", content, re.DOTALL)
            if section_match:
                section_content = section_match.group(0)
                
                # Parse cluster rankings
                lines = section_content.split('\n')
                for line in lines:
                    # Match pattern: "1    53           24       market_rally, market_rally, market_rally, market_reaction, market_rally (sentiment: 0.550 [positive])"
                    cluster_match = re.search(r'^(\d+)\s+(\d+)\s+(\d+)\s+.*?\(sentiment: ([-\d.]+) \[(\w+)\]\)', line)
                    if cluster_match:
                        rank = int(cluster_match.group(1))
                        cluster_id = int(cluster_match.group(2))
                        count = int(cluster_match.group(3))
                        sentiment = float(cluster_match.group(4))
                        sentiment_label = cluster_match.group(5)
                        
                        data[data_key][cluster_id] = {
                            'rank': rank,
                            'count': count,
                            'sentiment': sentiment,
                            'sentiment_label': sentiment_label
                        }
                        pass
        
        return data
    
    def aggregate_all_models(self):
        """Aggregate data from all models"""
        print(f"Aggregating cluster analysis from {len(self.model_dirs)} models...")
        
        all_cluster_data = []
        successful_parses = 0
        
        for model_dir in self.model_dirs:
            cluster_file = model_dir / "cluster_ranking_comprehensive.txt"
            if cluster_file.exists():
                try:
                    model_data = self.parse_cluster_file(cluster_file)
                    all_cluster_data.append(model_data)
                    successful_parses += 1
                except Exception as e:
                    print(f"Error parsing {model_dir.name}: {e}")
            else:
                print(f"Missing cluster file for {model_dir.name}")
        
        print(f"Successfully parsed {successful_parses}/{len(self.model_dirs)} models")
        return all_cluster_data
    
    def analyze_cluster_consistency(self, all_data: List[Dict]) -> Dict:
        """Analyze which clusters appear most consistently across models"""
        print("\n=== CLUSTER CONSISTENCY ANALYSIS ===")
        
        # Count how often each cluster appears in top 10 across all models
        cluster_appearances = {
            'tp': defaultdict(int),
            'fp': defaultdict(int), 
            'tn': defaultdict(int),
            'fn': defaultdict(int)
        }
        
        cluster_avg_ranks = {
            'tp': defaultdict(list),
            'fp': defaultdict(list),
            'tn': defaultdict(list), 
            'fn': defaultdict(list)
        }
        
        cluster_avg_counts = {
            'tp': defaultdict(list),
            'fp': defaultdict(list),
            'tn': defaultdict(list),
            'fn': defaultdict(list)
        }
        
        cluster_avg_sentiments = {
            'tp': defaultdict(list),
            'fp': defaultdict(list),
            'tn': defaultdict(list),
            'fn': defaultdict(list)
        }
        
        for model_data in all_data:
            for pred_type in ['tp', 'fp', 'tn', 'fn']:
                clusters_key = f'{pred_type}_clusters'
                for cluster_id, cluster_info in model_data[clusters_key].items():
                    cluster_appearances[pred_type][cluster_id] += 1
                    cluster_avg_ranks[pred_type][cluster_id].append(cluster_info['rank'])
                    cluster_avg_counts[pred_type][cluster_id].append(cluster_info['count'])
                    cluster_avg_sentiments[pred_type][cluster_id].append(cluster_info['sentiment'])
        
        # Calculate averages
        consistency_analysis = {}
        for pred_type in ['tp', 'fp', 'tn', 'fn']:
            consistency_analysis[pred_type] = {}
            for cluster_id in cluster_appearances[pred_type]:
                consistency_analysis[pred_type][cluster_id] = {
                    'appearance_rate': cluster_appearances[pred_type][cluster_id] / len(all_data),
                    'avg_rank': np.mean(cluster_avg_ranks[pred_type][cluster_id]),
                    'avg_count': np.mean(cluster_avg_counts[pred_type][cluster_id]),
                    'avg_sentiment': np.mean(cluster_avg_sentiments[pred_type][cluster_id]),
                    'std_sentiment': np.std(cluster_avg_sentiments[pred_type][cluster_id]),
                    'total_models': cluster_appearances[pred_type][cluster_id]
                }
        
        return consistency_analysis
    
    def analyze_tp_vs_fn_patterns(self, all_data: List[Dict]) -> Dict:
        """Analyze patterns between True Positives and False Negatives"""
        print("\n=== TP vs FN PATTERN ANALYSIS ===")
        
        # For each cluster, compare TP vs FN performance
        cluster_tp_fn_analysis = defaultdict(lambda: {'tp_count': 0, 'fn_count': 0, 'tp_models': 0, 'fn_models': 0})
        
        for model_data in all_data:
            tp_clusters = set(model_data['tp_clusters'].keys())
            fn_clusters = set(model_data['fn_clusters'].keys())
            
            for cluster_id in tp_clusters:
                cluster_tp_fn_analysis[cluster_id]['tp_count'] += model_data['tp_clusters'][cluster_id]['count']
                cluster_tp_fn_analysis[cluster_id]['tp_models'] += 1
            
            for cluster_id in fn_clusters:
                cluster_tp_fn_analysis[cluster_id]['fn_count'] += model_data['fn_clusters'][cluster_id]['count']
                cluster_tp_fn_analysis[cluster_id]['fn_models'] += 1
        
        # Calculate ratios and identify problematic clusters
        problematic_clusters = []
        for cluster_id, data in cluster_tp_fn_analysis.items():
            if data['tp_models'] > 0 and data['fn_models'] > 0:
                tp_avg = data['tp_count'] / data['tp_models']
                fn_avg = data['fn_count'] / data['fn_models']
                fn_ratio = fn_avg / (tp_avg + fn_avg) if (tp_avg + fn_avg) > 0 else 0
                
                data['tp_avg'] = tp_avg
                data['fn_avg'] = fn_avg
                data['fn_ratio'] = fn_ratio
                
                if fn_ratio > 0.7:  # High FN ratio
                    problematic_clusters.append((cluster_id, fn_ratio, data))
        
        return {
            'cluster_analysis': dict(cluster_tp_fn_analysis),
            'problematic_clusters': sorted(problematic_clusters, key=lambda x: x[1], reverse=True)
        }
    
    def analyze_sentiment_patterns(self, all_data: List[Dict]) -> Dict:
        """Analyze sentiment patterns across clusters and prediction types"""
        print("\n=== SENTIMENT PATTERN ANALYSIS ===")
        
        sentiment_analysis = {
            'by_prediction_type': defaultdict(list),
            'by_cluster': defaultdict(lambda: defaultdict(list)),
            'sentiment_distribution': defaultdict(list)
        }
        
        for model_data in all_data:
            for pred_type in ['tp', 'fp', 'tn', 'fn']:
                clusters_key = f'{pred_type}_clusters'
                for cluster_id, cluster_info in model_data[clusters_key].items():
                    sentiment = cluster_info['sentiment']
                    sentiment_analysis['by_prediction_type'][pred_type].append(sentiment)
                    sentiment_analysis['by_cluster'][cluster_id][pred_type].append(sentiment)
                    sentiment_analysis['sentiment_distribution'][pred_type].append(sentiment)
        
        # Calculate statistics
        sentiment_stats = {}
        for pred_type in ['tp', 'fp', 'tn', 'fn']:
            sentiments = sentiment_analysis['by_prediction_type'][pred_type]
            if len(sentiments) > 0:
                sentiment_stats[pred_type] = {
                    'mean': np.mean(sentiments),
                    'std': np.std(sentiments),
                    'min': np.min(sentiments),
                    'max': np.max(sentiments),
                    'positive_ratio': sum(1 for s in sentiments if s > 0.1) / len(sentiments),
                    'negative_ratio': sum(1 for s in sentiments if s < -0.1) / len(sentiments)
                }
            else:
                sentiment_stats[pred_type] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'positive_ratio': 0.0,
                    'negative_ratio': 0.0
                }
        
        return {
            'raw_data': sentiment_analysis,
            'statistics': sentiment_stats
        }
    
    def identify_anomalous_models(self, all_data: List[Dict]) -> Dict:
        """Identify models that behave differently from the majority"""
        print("\n=== ANOMALOUS MODEL DETECTION ===")
        
        # Calculate average cluster rankings across all models
        avg_rankings = defaultdict(lambda: defaultdict(list))
        for model_data in all_data:
            for pred_type in ['tp', 'fp', 'tn', 'fn']:
                clusters_key = f'{pred_type}_clusters'
                for cluster_id, cluster_info in model_data[clusters_key].items():
                    avg_rankings[pred_type][cluster_id].append(cluster_info['rank'])
        
        # Calculate mean rankings
        mean_rankings = {}
        for pred_type in ['tp', 'fp', 'tn', 'fn']:
            mean_rankings[pred_type] = {}
            for cluster_id, ranks in avg_rankings[pred_type].items():
                mean_rankings[pred_type][cluster_id] = np.mean(ranks)
        
        # Find models that deviate significantly
        anomalous_models = []
        for model_data in all_data:
            model_name = model_data['model_name']
            total_deviation = 0
            deviation_count = 0
            
            for pred_type in ['tp', 'fp', 'tn', 'fn']:
                clusters_key = f'{pred_type}_clusters'
                for cluster_id, cluster_info in model_data[clusters_key].items():
                    if cluster_id in mean_rankings[pred_type]:
                        expected_rank = mean_rankings[pred_type][cluster_id]
                        actual_rank = cluster_info['rank']
                        deviation = abs(actual_rank - expected_rank)
                        total_deviation += deviation
                        deviation_count += 1
            
            if deviation_count > 0:
                avg_deviation = total_deviation / deviation_count
                if avg_deviation > 3.0:  # Threshold for anomaly
                    anomalous_models.append((model_name, avg_deviation))
        
        return {
            'mean_rankings': mean_rankings,
            'anomalous_models': sorted(anomalous_models, key=lambda x: x[1], reverse=True)
        }
    
    def generate_summary_report(self, all_data: List[Dict]) -> str:
        """Generate a comprehensive summary report"""
        print("\n=== GENERATING SUMMARY REPORT ===")
        
        # Run all analyses
        consistency_analysis = self.analyze_cluster_consistency(all_data)
        tp_fn_analysis = self.analyze_tp_vs_fn_patterns(all_data)
        sentiment_analysis = self.analyze_sentiment_patterns(all_data)
        anomalous_analysis = self.identify_anomalous_models(all_data)
        
        # Generate report
        report = []
        report.append("=" * 80)
        report.append("CLUSTER ANALYSIS AGGREGATION REPORT")
        report.append("=" * 80)
        report.append(f"Analysis of {len(all_data)} models")
        report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Most consistent clusters
        report.append("MOST CONSISTENTLY INFLUENTIAL CLUSTERS")
        report.append("-" * 50)
        for pred_type in ['tp', 'fp', 'tn', 'fn']:
            report.append(f"\n{pred_type.upper()} - Top 5 Most Consistent Clusters:")
            sorted_clusters = sorted(consistency_analysis[pred_type].items(), 
                                   key=lambda x: x[1]['appearance_rate'], reverse=True)
            for i, (cluster_id, data) in enumerate(sorted_clusters[:5]):
                report.append(f"  {i+1}. Cluster {cluster_id}: {data['appearance_rate']:.1%} of models "
                            f"(avg rank: {data['avg_rank']:.1f}, avg sentiment: {data['avg_sentiment']:.3f})")
        
        # Problematic clusters (high FN ratio)
        report.append("\n\nPROBLEMATIC CLUSTERS (High False Negative Ratio)")
        report.append("-" * 50)
        for cluster_id, fn_ratio, data in tp_fn_analysis['problematic_clusters'][:10]:
            report.append(f"Cluster {cluster_id}: FN ratio = {fn_ratio:.1%} "
                        f"(TP avg: {data['tp_avg']:.1f}, FN avg: {data['fn_avg']:.1f})")
        
        # Sentiment patterns
        report.append("\n\nSENTIMENT PATTERNS BY PREDICTION TYPE")
        report.append("-" * 50)
        for pred_type in ['tp', 'fp', 'tn', 'fn']:
            stats = sentiment_analysis['statistics'][pred_type]
            report.append(f"{pred_type.upper()}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, "
                        f"positive_ratio={stats['positive_ratio']:.1%}")
        
        # Anomalous models
        report.append("\n\nANOMALOUS MODELS (High Deviation from Average)")
        report.append("-" * 50)
        for model_name, deviation in anomalous_analysis['anomalous_models'][:5]:
            report.append(f"{model_name}: avg deviation = {deviation:.2f} ranks")
        
        # Key insights
        report.append("\n\nKEY INSIGHTS")
        report.append("-" * 50)
        
        # Find most problematic cluster
        if tp_fn_analysis['problematic_clusters']:
            worst_cluster, worst_ratio, _ = tp_fn_analysis['problematic_clusters'][0]
            report.append(f"• Most problematic cluster: {worst_cluster} (FN ratio: {worst_ratio:.1%})")
        else:
            report.append("• No problematic clusters identified")
        
        # Find most consistent cluster
        if consistency_analysis['tp']:
            most_consistent = max(consistency_analysis['tp'].items(), 
                                key=lambda x: x[1]['appearance_rate'])
            report.append(f"• Most consistent cluster: {most_consistent[0]} "
                        f"(appears in {most_consistent[1]['appearance_rate']:.1%} of models)")
        else:
            report.append("• No consistent clusters found in TP analysis")
        
        # Sentiment insight
        tp_sentiment = sentiment_analysis['statistics']['tp']['mean']
        fn_sentiment = sentiment_analysis['statistics']['fn']['mean']
        report.append(f"• TP sentiment (avg: {tp_sentiment:.3f}) vs FN sentiment (avg: {fn_sentiment:.3f})")
        
        return "\n".join(report)
    
    def save_aggregated_data(self, all_data: List[Dict], output_file: Path):
        """Save aggregated data for further analysis"""
        aggregated_data = {
            'model_data': all_data,
            'consistency_analysis': self.analyze_cluster_consistency(all_data),
            'tp_fn_analysis': self.analyze_tp_vs_fn_patterns(all_data),
            'sentiment_analysis': self.analyze_sentiment_patterns(all_data),
            'anomalous_analysis': self.identify_anomalous_models(all_data)
        }
        
        with open(output_file, 'w') as f:
            json.dump(aggregated_data, f, indent=2, default=str)
        
        print(f"Saved aggregated data to: {output_file}")

def main():
    """Main function to run the aggregation"""
    results_dir = Path("../Results/heterognn5")
    aggregator = ClusterAnalysisAggregator(results_dir)
    
    # Aggregate all model data
    all_data = aggregator.aggregate_all_models()
    
    if not all_data:
        print("No data to aggregate!")
        return
    
    # Generate summary report
    report = aggregator.generate_summary_report(all_data)
    
    # Save report
    output_file = Path("../Results/cluster_analysis_aggregation.txt")
    with open(output_file, 'w') as f:
        f.write(report)
    
    # Save aggregated data
    data_file = Path("../Results/cluster_analysis_aggregated_data.json")
    aggregator.save_aggregated_data(all_data, data_file)
    
    print(f"\n✅ Aggregation complete!")
    print(f"📄 Summary report: {output_file}")
    print(f"📊 Aggregated data: {data_file}")
    
    # Print key findings
    print("\n" + "="*50)
    print("KEY FINDINGS")
    print("="*50)
    print(report.split("KEY INSIGHTS")[-1])

if __name__ == "__main__":
    main()
