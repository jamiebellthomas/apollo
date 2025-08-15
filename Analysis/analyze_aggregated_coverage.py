#!/usr/bin/env python3
"""
Analyze aggregated results coverage for each model type.
Counts the number of actual positive events (Actual_Label == 1) that were identified 
by at least one model (Vote_Count_1 > 0) for each architecture.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

def get_model_types_from_results():
    """
    Get all model types from the Results directory.
    
    Returns:
        list: List of model type names
    """
    results_dir = Path("Results")
    model_types = []
    
    if results_dir.exists():
        for item in results_dir.iterdir():
            if item.is_dir():
                model_types.append(item.name)
    
    return sorted(model_types)

def analyze_aggregated_coverage(model_type):
    """
    Analyze aggregated results coverage for a specific model type.
    
    Args:
        model_type (str): The model type name
    
    Returns:
        dict: Dictionary with coverage statistics
    """
    aggregated_file = Path("Analysis") / model_type / "aggregated_results.csv"
    
    if not aggregated_file.exists():
        return None
    
    try:
        # Read the aggregated results
        df = pd.read_csv(aggregated_file)
        
        # Count total actual positive events
        total_actual_positives = len(df[df['Actual_Label'] == 1])
        
        # Count actual positive events identified by at least one model
        identified_positives = len(df[(df['Actual_Label'] == 1) & (df['Vote_Count_1'] > 0)])
        
        # Count total events where at least one model predicted positive
        total_predicted_by_any = len(df[df['Vote_Count_1'] > 0])
        
        # Count how many of those were actually positive
        correct_predictions_by_any = len(df[(df['Actual_Label'] == 1) & (df['Vote_Count_1'] > 0)])
        
        # Calculate precision of "any model predicts positive"
        precision_any_model = (correct_predictions_by_any / total_predicted_by_any * 100) if total_predicted_by_any > 0 else 0
        
        # Count how many of the identified actual positives were assigned majority label 1
        identified_with_majority_1 = len(df[(df['Actual_Label'] == 1) & (df['Vote_Count_1'] > 0) & (df['Vote_Count_1'] > df['Vote_Count_0'])])
        
        # Calculate percentage of identified actual positives that got majority label 1
        identified_majority_1_percent = (identified_with_majority_1 / identified_positives * 100) if identified_positives > 0 else 0
        
        # Calculate coverage percentage
        coverage_percentage = (identified_positives / total_actual_positives * 100) if total_actual_positives > 0 else 0
        
        # Count total events and predicted positives
        total_events = len(df)
        predicted_positives = len(df[df['Aggregated_Predicted_Label'] == 1])
        
        # Calculate score based on ratio of models that picked the correct label
        total_score = 0
        for _, row in df.iterrows():
            actual_label = row['Actual_Label']
            total_models = row['Vote_Count_0'] + row['Vote_Count_1']
            
            if total_models > 0:
                if actual_label == 1:
                    # For actual positive, count models that voted 1
                    correct_votes = row['Vote_Count_1']
                else:
                    # For actual negative, count models that voted 0
                    correct_votes = row['Vote_Count_0']
                
                # Calculate ratio of correct votes
                correct_ratio = correct_votes / total_models
                total_score += correct_ratio
        
        return {
            'model_type': model_type,
            'total_events': total_events,
            'total_actual_positives': total_actual_positives,
            'identified_positives': identified_positives,
            'coverage_percentage': coverage_percentage,
            'predicted_positives': predicted_positives,
            'total_predicted_by_any': total_predicted_by_any,
            'precision_any_model': precision_any_model,
            'identified_with_majority_1': identified_with_majority_1,
            'identified_majority_1_percent': identified_majority_1_percent,
            'total_score': total_score,
            'average_score': total_score / total_events if total_events > 0 else 0
        }
        
    except Exception as e:
        print(f"Error processing {model_type}: {e}")
        return None

def print_coverage_table(coverage_data):
    """
    Print coverage statistics as a formatted table.
    
    Args:
        coverage_data (list): List of coverage statistics dictionaries
    """
    if not coverage_data:
        print("No coverage data available.")
        return
    
    # Print header
    print("\n" + "="*160)
    print("AGGREGATED RESULTS COVERAGE ANALYSIS")
    print("="*160)
    
    # Print column guide
    print("\nCOLUMN GUIDE:")
    print("- Model Type: The type of model (heterognn, heterognn2, etc.)")
    print("- Total Events: Total number of events in the test set")
    print("- Actual Positives: Number of events that were actually positive (Actual_Label == 1)")
    print("- Identified: Number of actual positives that were identified by at least one model")
    print("- Coverage %: Percentage of actual positives identified (Identified / Actual Positives)")
    print("- Predicted Positives: Number of events where majority vote was positive")
    print("- Any Model Pred: Number of events where at least one model predicted positive")
    print("- Precision %: Percentage of 'any model predicted' events that were actually positive")
    print("- Identified Maj 1: Number of identified actual positives that got majority label 1")
    print("- Identified Maj 1 %: Percentage of identified actual positives that got majority label 1")
    print("- Total Score: Sum of correct vote ratios across all events")
    print("- Avg Score: Average correct vote ratio per event")
    
    print(f"\n{'Model Type':<15} {'Total Events':<12} {'Actual Positives':<16} {'Identified':<12} {'Coverage %':<12} {'Predicted Positives':<20} {'Any Model Pred':<15} {'Precision %':<12} {'Identified Maj 1':<15} {'Identified Maj 1 %':<15} {'Total Score':<12} {'Avg Score':<10}")
    print("-"*180)
    
    # Print data rows
    for data in coverage_data:
        if data:
            print(f"{data['model_type']:<15} {data['total_events']:<12} {data['total_actual_positives']:<16} "
                  f"{data['identified_positives']:<12} {data['coverage_percentage']:<12.1f} {data['predicted_positives']:<20} "
                  f"{data['total_predicted_by_any']:<15} {data['precision_any_model']:<12.1f} "
                  f"{data['identified_with_majority_1']:<15} {data['identified_majority_1_percent']:<15.1f} "
                  f"{data['total_score']:<12.1f} {data['average_score']:<10.3f}")
    
    print("-"*180)
    
    # Print summary statistics
    print("\nSUMMARY:")
    print(f"Total model types analyzed: {len(coverage_data)}")
    
    if coverage_data:
        avg_coverage = np.mean([d['coverage_percentage'] for d in coverage_data if d])
        max_coverage = max([d['coverage_percentage'] for d in coverage_data if d])
        min_coverage = min([d['coverage_percentage'] for d in coverage_data if d])
        
        avg_score = np.mean([d['average_score'] for d in coverage_data if d])
        max_score = max([d['average_score'] for d in coverage_data if d])
        min_score = min([d['average_score'] for d in coverage_data if d])
        
        avg_precision = np.mean([d['precision_any_model'] for d in coverage_data if d])
        max_precision = max([d['precision_any_model'] for d in coverage_data if d])
        min_precision = min([d['precision_any_model'] for d in coverage_data if d])
        
        print(f"Average coverage: {avg_coverage:.1f}%")
        print(f"Maximum coverage: {max_coverage:.1f}%")
        print(f"Minimum coverage: {min_coverage:.1f}%")
        print(f"Average precision: {avg_precision:.1f}%")
        print(f"Maximum precision: {max_precision:.1f}%")
        print(f"Minimum precision: {min_precision:.1f}%")
        print(f"Average score: {avg_score:.3f}")
        print(f"Maximum score: {max_score:.3f}")
        print(f"Minimum score: {min_score:.3f}")
        
        # Find best and worst performing models
        best_coverage_model = max(coverage_data, key=lambda x: x['coverage_percentage'] if x else 0)
        worst_coverage_model = min(coverage_data, key=lambda x: x['coverage_percentage'] if x else 0)
        best_precision_model = max(coverage_data, key=lambda x: x['precision_any_model'] if x else 0)
        worst_precision_model = min(coverage_data, key=lambda x: x['precision_any_model'] if x else 0)
        best_score_model = max(coverage_data, key=lambda x: x['average_score'] if x else 0)
        worst_score_model = min(coverage_data, key=lambda x: x['average_score'] if x else 0)
        
        print(f"Best coverage: {best_coverage_model['model_type']} ({best_coverage_model['coverage_percentage']:.1f}%)")
        print(f"Worst coverage: {worst_coverage_model['model_type']} ({worst_coverage_model['coverage_percentage']:.1f}%)")
        print(f"Best precision: {best_precision_model['model_type']} ({best_precision_model['precision_any_model']:.1f}%)")
        print(f"Worst precision: {worst_precision_model['model_type']} ({worst_precision_model['precision_any_model']:.1f}%)")
        print(f"Best score: {best_score_model['model_type']} ({best_score_model['average_score']:.3f})")
        print(f"Worst score: {worst_score_model['model_type']} ({worst_score_model['average_score']:.3f})")

def main():
    """
    Main function to analyze aggregated coverage for all model types.
    """
    print("Analyzing aggregated results coverage...")
    
    # Get model types from Results directory
    model_types = get_model_types_from_results()
    
    if not model_types:
        print("No model types found in Results directory.")
        return
    
    print(f"Found {len(model_types)} model types: {model_types}")
    
    # Analyze coverage for each model type
    coverage_data = []
    for model_type in model_types:
        print(f"\nProcessing {model_type}...")
        coverage = analyze_aggregated_coverage(model_type)
        coverage_data.append(coverage)
        
        if coverage:
            print(f"  Total events: {coverage['total_events']}")
            print(f"  Actual positives: {coverage['total_actual_positives']}")
            print(f"  Identified positives: {coverage['identified_positives']}")
            print(f"  Coverage: {coverage['coverage_percentage']:.1f}%")
            print(f"  Any model predicted: {coverage['total_predicted_by_any']}")
            print(f"  Precision (any model): {coverage['precision_any_model']:.1f}%")
            print(f"  Identified with majority 1: {coverage['identified_with_majority_1']}")
            print(f"  Identified majority 1 %: {coverage['identified_majority_1_percent']:.1f}%")
            print(f"  Total score: {coverage['total_score']:.1f}")
            print(f"  Average score: {coverage['average_score']:.3f}")
        else:
            print(f"  No aggregated results found for {model_type}")
    
    # Print coverage table
    print_coverage_table(coverage_data)

if __name__ == "__main__":
    main()
