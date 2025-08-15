#!/usr/bin/env python3
"""
Medium-Term Aggregate Consensus Analysis Script

This script aggregates consensus predictions from multiple models in Results directory
and analyzes CAR for the medium-term period (day 15 to day 40).
For each architecture (e.g., heterognn, heterognn2), finds all test_predictions.csv files,
aggregates predictions using majority voting, and calculates performance metrics.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import json
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Configure matplotlib for LaTeX formatting
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}",
    "axes.formatter.use_mathtext": True,
    "mathtext.fontset": "cm"
})

# Configuration constants - MEDIUM TERM PERIOD
DAYS_BEFORE = 0  # Start from event date (day 0)
DAYS_AFTER = 40   # End at day 40
START_DAY = 15    # Start calculating CAR from day 15
END_DAY = 40      # End calculating CAR at day 40
BENCHMARK_TICKER = "SPY"
DATABASE_PATH = "Data/momentum_data.db"
EPS_SURPRISES_FILE = "Data/eps_surprises_quarterly_2012_2024.csv"
EPS_CAR_CACHE_FILE = "Data/eps_car_cache_medium.csv"  # New cache file for medium-term data

def find_test_predictions_files(results_dir):
    """
    Find all test_predictions.csv files in the Results directory.
    Returns a dictionary mapping architecture names to lists of file paths.
    """
    architectures = {}
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Results directory {results_dir} does not exist!")
        return architectures
    
    # Look for architecture directories (e.g., heterognn, heterognn2)
    for arch_dir in results_path.iterdir():
        if arch_dir.is_dir() and not arch_dir.name.startswith('_'):
            arch_name = arch_dir.name
            architectures[arch_name] = []
            
            # Find all test_predictions.csv files in this architecture
            for subdir in arch_dir.iterdir():
                if subdir.is_dir():
                    test_pred_file = subdir / "test_predictions.csv"
                    if test_pred_file.exists():
                        architectures[arch_name].append(str(test_pred_file))
    
    return architectures

def load_predictions(file_path):
    """
    Load predictions from a test_predictions.csv file.
    Returns a DataFrame with Ticker, Reported_Date, Actual_Label, Predicted_Label columns.
    """
    try:
        df = pd.read_csv(file_path)
        # Create a unique identifier for each test case
        df['test_case'] = df['Ticker'] + '_' + df['Reported_Date']
        return df[['test_case', 'Ticker', 'Reported_Date', 'Actual_Label', 'Predicted_Label']]
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def aggregate_predictions(prediction_files):
    """
    Aggregate predictions from multiple files using majority voting.
    Returns a DataFrame with aggregated predictions for the 191 test cases only.
    """
    # Load the test set to get the 191 test cases
    test_set_df = pd.read_csv("Data/test_set.csv")
    test_set_df['test_case'] = test_set_df['Ticker'] + '_' + test_set_df['Reported_Date']
    test_cases = set(test_set_df['test_case'].tolist())
    
    print(f"Filtering for {len(test_cases)} test cases from test_set.csv")
    
    all_predictions = []
    
    # Load all prediction files
    for file_path in prediction_files:
        pred_df = load_predictions(file_path)
        if pred_df is not None:
            # Filter for only test cases in the test set
            pred_df = pred_df[pred_df['test_case'].isin(test_cases)]
            if len(pred_df) > 0:
                all_predictions.append(pred_df)
    
    if not all_predictions:
        print("No valid prediction files found!")
        return None
    
    # Combine all predictions
    combined_df = pd.concat(all_predictions, ignore_index=True)
    
    # Group by test case and calculate majority vote
    aggregated = []
    
    for test_case, group in combined_df.groupby('test_case'):
        # Get the first row for metadata (should be same for all rows with same test_case)
        first_row = group.iloc[0]
        
        # Calculate majority vote for predicted label
        majority_pred = group['Predicted_Label'].mode().iloc[0]
        
        # If there's a tie, use the prediction with highest frequency
        # If still tied, use the first one
        pred_counts = group['Predicted_Label'].value_counts()
        if len(pred_counts) > 1 and pred_counts.iloc[0] == pred_counts.iloc[1]:
            # Tie exists, use the first prediction
            majority_pred = group['Predicted_Label'].iloc[0]
        
        aggregated.append({
            'test_case': test_case,
            'Ticker': first_row['Ticker'],
            'Reported_Date': first_row['Reported_Date'],
            'Actual_Label': first_row['Actual_Label'],
            'Aggregated_Predicted_Label': majority_pred,
            'Num_Models': len(group),
            'Vote_Count_0': len(group[group['Predicted_Label'] == 0]),
            'Vote_Count_1': len(group[group['Predicted_Label'] == 1])
        })
    
    result_df = pd.DataFrame(aggregated)
    print(f"Aggregated predictions for {len(result_df)} test cases")
    return result_df 

def calculate_performance_metrics(aggregated_df):
    """
    Calculate performance metrics for aggregated predictions.
    """
    y_true = aggregated_df['Actual_Label']
    y_pred = aggregated_df['Aggregated_Predicted_Label']
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Get detailed classification report
    class_report = classification_report(y_true, y_pred, output_dict=True)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'classification_report': class_report
    }

def save_results(aggregated_df, metrics, output_dir, arch_name):
    """
    Save aggregated results and performance metrics.
    """
    # Create output directory
    output_path = Path(output_dir) / arch_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save aggregated results
    results_file = output_path / "aggregated_results_medium.csv"
    aggregated_df.to_csv(results_file, index=False)
    print(f"Saved aggregated results to {results_file}")
    
    # Save performance metrics
    performance_file = output_path / "performance_medium.txt"
    with open(performance_file, 'w') as f:
        f.write(f"Medium-Term Aggregated Model Performance for {arch_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Number of test cases: {len(aggregated_df)}\n")
        f.write(f"Number of models used: {aggregated_df['Num_Models'].iloc[0]}\n\n")
        f.write("Performance Metrics:\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['recall']:.4f}\n")
        f.write(f"F1-Score:  {metrics['f1_score']:.4f}\n\n")
        
        # Add confusion matrix
        f.write("Confusion Matrix:\n")
        f.write("-" * 20 + "\n")
        cm = metrics['confusion_matrix']
        f.write("                Predicted\n")
        f.write("                0    1\n")
        f.write("Actual 0    {:4d} {:4d}\n".format(cm[0, 0], cm[0, 1]))
        f.write("       1    {:4d} {:4d}\n".format(cm[1, 0], cm[1, 1]))
        f.write("\n")
        f.write("Confusion Matrix Details:\n")
        f.write(f"True Negatives (TN): {cm[0, 0]} - Correctly predicted negative\n")
        f.write(f"False Positives (FP): {cm[0, 1]} - Incorrectly predicted positive\n")
        f.write(f"False Negatives (FN): {cm[1, 0]} - Incorrectly predicted negative\n")
        f.write(f"True Positives (TP): {cm[1, 1]} - Correctly predicted positive\n\n")
        
        f.write("Detailed Classification Report:\n")
        f.write("-" * 30 + "\n")
        for label, scores in metrics['classification_report'].items():
            if isinstance(scores, dict):
                f.write(f"Class {label}:\n")
                for metric, value in scores.items():
                    if isinstance(value, float):
                        f.write(f"  {metric}: {value:.4f}\n")
                    else:
                        f.write(f"  {metric}: {value}\n")
                f.write("\n")
    
    print(f"Saved performance metrics to {performance_file}")
    
    # Also save metrics as JSON for programmatic access
    json_file = output_path / "performance_medium.json"
    with open(json_file, 'w') as f:
        # Convert numpy array to list for JSON serialization
        metrics_for_json = metrics.copy()
        metrics_for_json['confusion_matrix'] = metrics['confusion_matrix'].tolist()
        json.dump(metrics_for_json, f, indent=2)
    print(f"Saved performance metrics (JSON) to {json_file}")

def calculate_car_for_ticker_medium(ticker, event_date, start_day=15, end_day=40):
    """
    Calculate Cumulative Abnormal Returns (CAR) for a specific ticker around an event date.
    Focuses on medium-term period (day 15 to day 40 AFTER the event).
    Uses local database instead of yfinance.
    
    Args:
        ticker (str): Stock ticker symbol
        event_date (str): Event date in YYYY-MM-DD format
        start_day (int): Day after event to start CAR calculation (15 for medium-term)
        end_day (int): Day after event to end CAR calculation (40 for medium-term)
    
    Returns:
        dict: CAR data including relative days and CAR values, or None if error
    """
    try:
        # Convert event date to datetime
        event_dt = datetime.strptime(event_date, '%Y-%m-%d')
        
        # Calculate date range with buffer to account for weekends/holidays
        # We need data from event date to end_day after event
        start_date = event_dt - timedelta(days=10)  # Buffer before event
        end_date = event_dt + timedelta(days=end_day + 15)  # Buffer after end_day
        
        # Connect to database
        conn = sqlite3.connect(DATABASE_PATH)
        
        # Get ticker data
        ticker_query = """
        SELECT date, adjusted_close 
        FROM daily_prices 
        WHERE ticker = ? AND date BETWEEN ? AND ?
        ORDER BY date
        """
        ticker_df = pd.read_sql_query(ticker_query, conn, params=[ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')])
        
        # Get benchmark data
        benchmark_query = """
        SELECT date, adjusted_close 
        FROM daily_prices 
        WHERE ticker = ? AND date BETWEEN ? AND ?
        ORDER BY date
        """
        benchmark_df = pd.read_sql_query(benchmark_query, conn, params=[BENCHMARK_TICKER, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')])
        
        conn.close()
        
        if ticker_df.empty or benchmark_df.empty:
            return None
        
        # Convert date column to datetime and extract just the date part
        ticker_df['date'] = pd.to_datetime(ticker_df['date']).dt.date
        benchmark_df['date'] = pd.to_datetime(benchmark_df['date']).dt.date
        
        # Set date as index
        ticker_df.set_index('date', inplace=True)
        benchmark_df.set_index('date', inplace=True)
        
        # Calculate returns
        ticker_returns = ticker_df['adjusted_close'].pct_change().dropna()
        benchmark_returns = benchmark_df['adjusted_close'].pct_change().dropna()
        
        # Align data
        common_dates = ticker_returns.index.intersection(benchmark_returns.index)
        ticker_returns = ticker_returns[common_dates]
        benchmark_returns = benchmark_returns[common_dates]
        
        # Find event date index - look for exact match first, then closest date
        event_idx = None
        event_dt_date = event_dt.date()
        
        # First try exact match
        for i, date in enumerate(common_dates):
            if date == event_dt_date:
                event_idx = i
                break
        
        # If no exact match, find the closest date within 5 days
        if event_idx is None:
            min_diff = float('inf')
            for i, date in enumerate(common_dates):
                diff = abs((date - event_dt_date).days)
                if diff < min_diff and diff <= 5:
                    min_diff = diff
                    event_idx = i
        
        if event_idx is None:
            return None
        
        # Calculate abnormal returns
        abnormal_returns = ticker_returns - benchmark_returns
        
        # Extract window from start_day to end_day after event
        start_idx = event_idx + start_day
        end_idx = event_idx + end_day + 1
        
        # Check if we have enough data for the window
        if start_idx >= len(abnormal_returns):
            return None
        
        # Adjust end_idx if we don't have enough data
        if end_idx > len(abnormal_returns):
            end_idx = len(abnormal_returns)
        
        window_ar = abnormal_returns.iloc[start_idx:end_idx]
        
        # Check if we have at least 10 days of data (minimum for meaningful analysis)
        if len(window_ar) < 10:
            return None
        
        # Calculate cumulative abnormal returns
        car = np.cumsum(window_ar.values)
        actual_end_day = start_day + len(window_ar) - 1
        relative_days = np.arange(start_day, actual_end_day + 1)
        
        return {
            'ticker': ticker,
            'event_date': event_date,
            'relative_days': relative_days,
            'car': car,
            'abnormal_returns': window_ar.values
        }
        
    except Exception as e:
        return None

def calculate_test_set_average_car_medium(eps_car_cache_file="Data/eps_car_cache_medium.csv", test_set_file="Data/test_set.csv"):
    """
    Calculate the average CAR for all instances in the test set for medium-term period.
    
    Args:
        eps_car_cache_file (str): Path to EPS CAR cache file
        test_set_file (str): Path to test set file
    
    Returns:
        tuple: (average_car_values, relative_days) or (None, None) if error
    """
    try:
        # Load test set
        test_df = pd.read_csv(test_set_file)
        print(f"Loaded test set with {len(test_df)} instances")
        
        # Load CAR cache
        car_df = pd.read_csv(eps_car_cache_file)
        print(f"Loaded CAR cache with {len(car_df)} records")
        
        # Create a key for matching (ticker + event_date)
        test_df['key'] = test_df['Ticker'] + '_' + test_df['Reported_Date']
        car_df['key'] = car_df['ticker'] + '_' + car_df['event_date']
        
        # Find matching records
        matched_records = []
        matched_count = 0
        max_length = 0
        
        for _, test_row in test_df.iterrows():
            key = test_row['key']
            matching_car = car_df[car_df['key'] == key]
            if len(matching_car) > 0:
                car_values = np.array(eval(matching_car.iloc[0]['car_values']))
                relative_days = np.array(eval(matching_car.iloc[0]['relative_days']))
                matched_records.append(car_values)
                max_length = max(max_length, len(car_values))
                matched_count += 1
        
        print(f"Found {matched_count} matching records out of {len(test_df)} test set instances")
        
        if len(matched_records) == 0:
            print("No matching CAR records found for test set")
            return None, None
        
        # Interpolate all CAR arrays to the same length
        interpolated_cars = []
        for car_values in matched_records:
            if len(car_values) < max_length:
                # Pad with the last value
                padded_car = np.pad(car_values, (0, max_length - len(car_values)), mode='edge')
            else:
                padded_car = car_values[:max_length]
            interpolated_cars.append(padded_car)
        
        # Calculate average CAR
        avg_car = np.mean(interpolated_cars, axis=0)
        print(f"Calculated average CAR for {len(interpolated_cars)} test set instances")
        print(f"Average CAR shape: {avg_car.shape}")
        
        # Create relative days array
        relative_days = np.arange(15, 15 + max_length)
        
        return avg_car, relative_days
        
    except Exception as e:
        print(f"Error calculating test set average CAR: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def load_car_data_from_cache_medium(cache_file):
    """
    Load CAR data from cache file for medium-term period.
    
    Args:
        cache_file (str): Path to cache file
    
    Returns:
        list: List of CAR data dictionaries or None if cache doesn't exist
    """
    if not os.path.exists(cache_file):
        return None
    
    try:
        df = pd.read_csv(cache_file)
        car_data_list = []
        
        for _, row in df.iterrows():
            car_data = {
                'ticker': row['ticker'],
                'event_date': row['event_date'],
                'alpha': row['alpha'],
                'beta': row['beta'],
                'car': np.array(eval(row['car_values'])),
                'abnormal_returns': np.array(eval(row['abnormal_returns'])),
                'relative_days': np.array(eval(row['relative_days']))
            }
            car_data_list.append(car_data)
        
        print(f"Loaded {len(car_data_list)} CAR records from cache: {cache_file}")
        return car_data_list
        
    except Exception as e:
        print(f"Error loading cache: {e}")

def save_car_data_to_cache_medium(car_data_list, cache_file):
    """
    Save CAR data to a cache file for medium-term period.
    
    Args:
        car_data_list (list): List of CAR data dictionaries
        cache_file (str): Path to cache file
    """
    if not car_data_list:
        return
    
    try:
        # Prepare data for saving
        cache_data = []
        for car_data in car_data_list:
            if car_data is None:
                continue
                
            # Convert numpy arrays to lists for JSON serialization
            row_data = {
                'ticker': car_data['ticker'],
                'event_date': car_data['event_date'],
                'alpha': car_data['alpha'],
                'beta': car_data['beta'],
                'car_values': car_data['car'].tolist(),
                'abnormal_returns': car_data['abnormal_returns'].tolist(),
                'relative_days': car_data['relative_days'].tolist()
            }
            cache_data.append(row_data)
        
        # Save to CSV
        df = pd.DataFrame(cache_data)
        df.to_csv(cache_file, index=False)
        print(f"Saved {len(cache_data)} CAR records to cache: {cache_file}")
        
    except Exception as e:
        print(f"Error saving cache: {e}") 

def plot_aggregation_medium(aggregated_df, start_day=15, end_day=40, save_path=None, arch_name="Aggregated Model"):
    """
    Plot CAR analysis for aggregated model predictions (medium-term period).
    
    Args:
        aggregated_df (DataFrame): Aggregated results DataFrame
        start_day (int): Day after event to start CAR calculation (15 for medium-term)
        end_day (int): Day after event to end CAR calculation (40 for medium-term)
        save_path (str): Optional path to save the plot
        arch_name (str): Architecture name for labeling
    Returns:
        list: List of CAR data dictionaries for this architecture
    """
    # Filter for predicted positive events
    positive_events = aggregated_df[aggregated_df['Aggregated_Predicted_Label'] == 1]
    
    if len(positive_events) == 0:
        print(f"No predicted positive events found for {arch_name}")
        return []
    
    print(f"Calculating CAR for {len(positive_events)} predicted positive events...")
    
    # Load cached CAR data
    eps_car_data = load_car_data_from_cache_medium(EPS_CAR_CACHE_FILE)
    if eps_car_data is None:
        print("Error: Could not load cached CAR data")
        return []
    
    # Calculate test set average CAR
    test_avg_car, test_relative_days = calculate_test_set_average_car_medium(EPS_CAR_CACHE_FILE, "Data/test_set.csv")
    
    # Calculate CAR for each predicted positive event using cached data
    car_data_list = []
    for _, row in positive_events.iterrows():
        # Find matching event in cached data
        found = False
        for cached_event in eps_car_data:
            if (cached_event['ticker'] == row['Ticker'] and 
                cached_event['event_date'] == row['Reported_Date']):
                car_data_list.append(cached_event)
                found = True
                break
        
        if not found:
            print(f"Warning: Could not find cached CAR data for {row['Ticker']} on {row['Reported_Date']}")
    
    if not car_data_list:
        print(f"No valid CAR data found for {arch_name}")
        return []
    
    print(f"Found CAR data for {len(car_data_list)}/{len(positive_events)} events")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Store line objects and their labels for hover functionality
    line_objects = []
    line_labels = []
    
    # Plot CAR for each ticker (thin lines, semi-transparent)
    colors = plt.cm.Set3(np.linspace(0, 1, len(car_data_list)))
    
    # Store all CAR data for averaging - need to handle variable lengths
    all_car_data = []
    max_length = 0
    
    # First pass: find maximum length and collect data
    for car_data in car_data_list:
        car_length = len(car_data['car'])
        max_length = max(max_length, car_length)
        all_car_data.append(car_data['car'])
    
    # Second pass: interpolate all CAR arrays to the same length
    interpolated_cars = []
    for i, car_data in enumerate(car_data_list):
        car_values = car_data['car']
        relative_days = car_data['relative_days']
        
        # Interpolate to common length
        if len(car_values) < max_length:
            # Pad with the last value
            padded_car = np.pad(car_values, (0, max_length - len(car_values)), mode='edge')
        else:
            padded_car = car_values[:max_length]
        
        interpolated_cars.append(padded_car)
        
        # Plot individual stock line
        padded_days = np.arange(start_day, start_day + max_length)
        line, = ax.plot(padded_days, padded_car, 
                color=colors[i], linewidth=0.5, alpha=0.3, marker='o', markersize=1)
        line_objects.append(line)
        line_labels.append(f"{car_data['ticker']} on {car_data['event_date']}")
    
    # Calculate and plot average CAR for aggregated predictions
    if interpolated_cars:
        avg_car = np.mean(interpolated_cars, axis=0)
        relative_days = np.arange(start_day, start_day + max_length)
        line, = ax.plot(relative_days, avg_car, 
                label=r'$\text{' + arch_name + r' Average CAR} \quad (N = ' + str(len(interpolated_cars)) + r')$', 
                color='blue', linewidth=4, marker='s', markersize=8)
        line_objects.append(line)
        line_labels.append(f"{arch_name} Average CAR (N = {len(interpolated_cars)})")
    
    # Plot test set average CAR if available
    if test_avg_car is not None and test_relative_days is not None:
        # Interpolate test set CAR to match the plot range
        test_max_length = len(test_avg_car)
        if test_max_length < max_length:
            # Pad test set CAR to match
            padded_test_car = np.pad(test_avg_car, (0, max_length - test_max_length), mode='edge')
        else:
            padded_test_car = test_avg_car[:max_length]
        
        line, = ax.plot(relative_days, padded_test_car, 
                label=r'$\text{Test Set Average CAR} \quad (N = 191)$', 
                color='red', linewidth=4, linestyle='--', marker='o', markersize=8)
        line_objects.append(line)
        line_labels.append("Test Set Average CAR (N = 191)")
    
    # Add vertical line at start of medium-term period
    ax.axvline(x=start_day, color='red', linestyle='--', alpha=0.7, label=r'$\text{Medium-Term Start} \quad (t = 15)$')
    
    # Add horizontal line at zero
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Customize the plot
    ax.set_xlabel(r'Days After Event ($t$)', fontsize=12)
    ax.set_ylabel(r'Cumulative Abnormal Returns (CAR)', fontsize=12)
    
    # Create informative title
    title = r"$\text{Medium-Term CAR Analysis} - \text{" + arch_name + r"} \quad N = " + str(len(interpolated_cars)) + r"$"
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        # Create save directory if it doesn't exist
        save_dir = Path(save_path) / arch_name
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / "car_analysis_individual_stocks_medium.png", dpi=300, bbox_inches='tight')
        print(f"CAR plot saved to {save_dir / 'car_analysis_individual_stocks_medium.png'}")
    
    plt.show()
    
    # Print summary statistics
    print_car_summary_medium(car_data_list, start_day, arch_name)
    return car_data_list

def print_car_summary_medium(car_data_list, start_day, arch_name="Model"):
    """Print a summary of CAR results for all tickers and average statistics (medium-term)."""
    print(f"\n{'='*80}")
    print(f"MEDIUM-TERM CAR ANALYSIS SUMMARY - {arch_name}")
    print(f"{'='*80}")
    
    # Store data for average calculations
    all_start_cars = []
    all_end_cars = []
    all_car_changes = []
    all_avg_ars = []
    
    for car_data in car_data_list:
        if car_data is None:
            continue
            
        print(f"\nTicker: {car_data['ticker']}")
        print(f"Event Date: {car_data['event_date']}")
        print(f"Method: Simple Abnormal Returns (r_ticker - r_spy)")
        
        # Calculate key metrics
        car_values = car_data['car']
        start_car = car_values[0]  # CAR at day 15 (start of medium-term period)
        end_car = car_values[-1]   # CAR at day 40 (end of medium-term period)
        car_change = end_car - start_car
        avg_ar = np.mean(car_data['abnormal_returns'])
        
        print(f"CAR at day {start_day}: {start_car:.4f}")
        print(f"CAR at day 40: {end_car:.4f}")
        print(f"CAR Change (day {start_day} to 40): {car_change:.4f}")
        print(f"Average AR: {avg_ar:.4f}")
        
        # Store for averaging
        all_start_cars.append(start_car)
        all_end_cars.append(end_car)
        all_car_changes.append(car_change)
        all_avg_ars.append(avg_ar)
    
    # Print average statistics
    if all_start_cars:
        print(f"\n{'='*50}")
        print(f"AVERAGE STATISTICS ({len(all_start_cars)} events)")
        print(f"{'='*50}")
        print(f"Average CAR at day {start_day}: {np.mean(all_start_cars):.4f}")
        print(f"Average CAR at day 40: {np.mean(all_end_cars):.4f}")
        print(f"Average CAR Change (day {start_day} to 40): {np.mean(all_car_changes):.4f}")
        print(f"Average AR: {np.mean(all_avg_ars):.4f}")
        print(f"Standard Deviation of CAR Change: {np.std(all_car_changes):.4f}")

def generate_medium_term_car_cache():
    """
    Generate CAR cache for medium-term period (day 15 to day 40) from the database.
    This function calculates CAR data for all events and saves it to cache.
    """
    print("Generating medium-term CAR cache from database...")
    
    # Load EPS surprises data
    eps_df = pd.read_csv(EPS_SURPRISES_FILE)
    positive_surprises = eps_df[eps_df['surprise'] > 0]
    
    print(f"Found {len(positive_surprises)} positive EPS surprises")
    
    # Filter for events that have enough future data
    # Database goes up to 2024-12-30, so we need events that are at least 40 days before that
    max_event_date = datetime(2024, 12, 30) - timedelta(days=40)
    filtered_surprises = positive_surprises[pd.to_datetime(positive_surprises['period']) <= max_event_date]
    
    print(f"Filtered to {len(filtered_surprises)} events with sufficient future data")
    
    car_data_list = []
    successful_calculations = 0
    
    for i, (_, row) in enumerate(filtered_surprises.iterrows()):
        if i % 100 == 0:
            print(f"Processing event {i+1}/{len(filtered_surprises)}: {row['symbol']} on {row['period']}")
        
        ticker = str(row['symbol']).strip()
        event_date = str(row['period']).strip()
        
        # Calculate CAR for this event
        car_data = calculate_car_for_ticker_medium(ticker, event_date, START_DAY, END_DAY)
        
        if car_data is not None:
            # Add alpha and beta (not used in simple abnormal return method)
            car_data['alpha'] = 0.0
            car_data['beta'] = 1.0
            car_data_list.append(car_data)
            successful_calculations += 1
        else:
            # Debug: print first few failures to understand the issue
            if successful_calculations < 5:
                print(f"Failed to calculate CAR for {ticker} on {event_date}")
    
    print(f"Successfully calculated CAR for {successful_calculations}/{len(filtered_surprises)} events")
    
    # Save to cache
    if car_data_list:
        save_car_data_to_cache_medium(car_data_list, EPS_CAR_CACHE_FILE)
        print(f"Medium-term CAR cache saved to {EPS_CAR_CACHE_FILE}")
    else:
        print("No CAR data calculated - cache not created")
    
    return car_data_list

def plot_all_aggregations_comparison_medium(all_car_data, start_day=15, end_day=40, save_path=None):
    """
    Plot comparison of all aggregated CAR results from each model on the same plot.
    
    Args:
        all_car_data (dict): Dictionary mapping architecture names to their CAR data
        start_day (int): Day after event to start CAR calculation (15 for medium-term)
        end_day (int): Day after event to end CAR calculation (40 for medium-term)
        save_path (str): Optional path to save the plot
    """
    if not all_car_data:
        print("No CAR data available for comparison plot")
        return
    
    # Calculate test set average CAR
    test_avg_car, test_relative_days = calculate_test_set_average_car_medium(EPS_CAR_CACHE_FILE, "Data/test_set.csv")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Define colors for different architectures
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Plot each architecture's aggregated CAR
    for i, (arch_name, car_data_list) in enumerate(all_car_data.items()):
        if not car_data_list:
            continue
            
        # Handle variable-length CAR arrays
        all_car_arrays = []
        max_length = 0
        
        # First pass: find maximum length
        for car_data in car_data_list:
            car_length = len(car_data['car'])
            max_length = max(max_length, car_length)
            all_car_arrays.append(car_data['car'])
        
        # Second pass: interpolate all CAR arrays to the same length
        interpolated_cars = []
        for car_values in all_car_arrays:
            if len(car_values) < max_length:
                # Pad with the last value
                padded_car = np.pad(car_values, (0, max_length - len(car_values)), mode='edge')
            else:
                padded_car = car_values[:max_length]
            interpolated_cars.append(padded_car)
        
        # Calculate average CAR for this architecture
        if interpolated_cars:
            avg_car = np.mean(interpolated_cars, axis=0)
            relative_days = np.arange(start_day, start_day + max_length)
            
            color = colors[i % len(colors)]
            line, = ax.plot(relative_days, avg_car, 
                    label=r'$\text{' + arch_name + r' Average CAR} \quad (N = ' + str(len(interpolated_cars)) + r')$', 
                    color=color, linewidth=3, marker='o', markersize=6)
    
    # Plot test set average CAR if available
    if test_avg_car is not None and test_relative_days is not None:
        # Interpolate test set CAR to match the plot range
        test_max_length = len(test_avg_car)
        if test_max_length < max_length:
            # Pad test set CAR to match
            padded_test_car = np.pad(test_avg_car, (0, max_length - test_max_length), mode='edge')
        else:
            padded_test_car = test_avg_car[:max_length]
        
        line, = ax.plot(relative_days, padded_test_car, 
                label=r'$\text{Test Set Average CAR} \quad (N = 191)$', 
                color='black', linewidth=4, linestyle='--', marker='s', markersize=8)
    
    # Add vertical line at start of medium-term period
    ax.axvline(x=start_day, color='red', linestyle='--', alpha=0.7, label=r'$\text{Medium-Term Start} \quad (t = 15)$')
    
    # Add horizontal line at zero
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Customize the plot
    ax.set_xlabel(r'Days After Event ($t$)', fontsize=14)
    ax.set_ylabel(r'Cumulative Abnormal Returns (CAR)', fontsize=14)
    
    # Create informative title
    title = r"$\text{Medium-Term CAR Comparison - All Aggregated Models}$"
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        # Create save directory if it doesn't exist
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / "all_aggregations_comparison_medium.png", dpi=300, bbox_inches='tight')
        print(f"All aggregations comparison plot saved to {save_dir / 'all_aggregations_comparison_medium.png'}")
    
    plt.show()
    
    # Print summary statistics for all architectures
    print("\n" + "="*80)
    print("MEDIUM-TERM CAR COMPARISON SUMMARY - ALL ARCHITECTURES")
    print("="*80)
    
    for arch_name, car_data_list in all_car_data.items():
        if not car_data_list:
            continue
            
        print(f"\n{arch_name}:")
        print(f"  Number of events: {len(car_data_list)}")
        
        # Calculate key metrics
        all_start_cars = []
        all_end_cars = []
        all_car_changes = []
        
        for car_data in car_data_list:
            car_values = car_data['car']
            start_car = car_values[0]  # CAR at day 15
            end_car = car_values[-1]   # CAR at day 40 (or last available day)
            car_change = end_car - start_car
            
            all_start_cars.append(start_car)
            all_end_cars.append(end_car)
            all_car_changes.append(car_change)
        
        if all_start_cars:
            print(f"  Average CAR at day 15: {np.mean(all_start_cars):.4f}")
            print(f"  Average CAR at day 40: {np.mean(all_end_cars):.4f}")
            print(f"  Average CAR Change (day 15 to 40): {np.mean(all_car_changes):.4f}")
            print(f"  Standard Deviation of CAR Change: {np.std(all_car_changes):.4f}")
    
    # Print test set summary
    if test_avg_car is not None:
        print(f"\nTest Set Average (N=191):")
        print(f"  Average CAR at day 15: {test_avg_car[0]:.4f}")
        print(f"  Average CAR at day 40: {test_avg_car[-1]:.4f}")
        print(f"  Average CAR Change (day 15 to 40): {test_avg_car[-1] - test_avg_car[0]:.4f}")

def main():
    """Main function to run the medium-term aggregate consensus analysis."""
    
    # Default settings - run everything
    results_dir = 'Results'
    output_dir = 'Analysis'
    architectures = None  # Process all architectures
    plot = True  # Always generate plots
    generate_aggregation_comparison = True  # Always generate aggregation comparison
    compare_with_best = True  # Always compare with best models
    
    # Check if medium-term CAR cache exists, if not generate it
    if not os.path.exists(EPS_CAR_CACHE_FILE):
        print("Medium-term CAR cache not found. Generating from database...")
        generate_medium_term_car_cache()
    else:
        print(f"Using existing medium-term CAR cache: {EPS_CAR_CACHE_FILE}")
    
    # Find all test_predictions.csv files
    architectures_dict = find_test_predictions_files(results_dir)
    
    if not architectures_dict:
        print("No architecture directories found!")
        return
    
    print(f"Found architectures: {list(architectures_dict.keys())}")
    
    # Filter architectures if specified
    if architectures:
        architectures_dict = {k: v for k, v in architectures_dict.items() if k in architectures}
        print(f"Processing specified architectures: {list(architectures_dict.keys())}")
    
    # Store CAR data for aggregation comparison
    all_car_data = {}
    eps_car_data = None
    
    # Process each architecture
    for arch_name, prediction_files in architectures_dict.items():
        print(f"\nProcessing {arch_name}...")
        print(f"Found {len(prediction_files)} prediction files")
        
        if len(prediction_files) == 0:
            print(f"No prediction files found for {arch_name}")
            continue
        
        # Aggregate predictions
        aggregated_df = aggregate_predictions(prediction_files)
        
        if aggregated_df is None:
            print(f"Failed to aggregate predictions for {arch_name}")
            continue
        
        # Calculate performance metrics
        metrics = calculate_performance_metrics(aggregated_df)
        
        # Save results
        save_results(aggregated_df, metrics, output_dir, arch_name)
        
        # Generate CAR plots if requested
        if plot:
            print(f"\nGenerating medium-term CAR plot for {arch_name}...")
            # Get CAR data for this architecture
            car_data_list = plot_aggregation_medium(aggregated_df, START_DAY, END_DAY, output_dir, arch_name)
            # Store CAR data for final comparison
            if car_data_list:
                all_car_data[arch_name] = car_data_list
    
    # Generate final comparison plot with all architectures
    if all_car_data and generate_aggregation_comparison:
        print(f"\nGenerating final comparison plot with all architectures...")
        plot_all_aggregations_comparison_medium(all_car_data, START_DAY, END_DAY, output_dir)
    
    print("\nMedium-term aggregate consensus analysis completed!")

if __name__ == "__main__":
    main() 