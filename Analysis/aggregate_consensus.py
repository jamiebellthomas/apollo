#!/usr/bin/env python3
"""
Aggregate consensus predictions from multiple models in Results directory.
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

# Configuration constants
DAYS_BEFORE = 20
DAYS_AFTER = 40
MID_POINT_START = 15
BENCHMARK_TICKER = "SPY"
DATABASE_PATH = "Data/momentum_data.db"
EPS_SURPRISES_FILE = "Data/eps_surprises_quarterly_2012_2024.csv"
EPS_CAR_CACHE_FILE = "Data/eps_car_cache.csv"

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
    results_file = output_path / "aggregated_results.csv"
    aggregated_df.to_csv(results_file, index=False)
    print(f"Saved aggregated results to {results_file}")
    
    # Save performance metrics
    performance_file = output_path / "performance.txt"
    with open(performance_file, 'w') as f:
        f.write(f"Aggregated Model Performance for {arch_name}\n")
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
    json_file = output_path / "performance.json"
    with open(json_file, 'w') as f:
        # Convert numpy array to list for JSON serialization
        metrics_for_json = metrics.copy()
        metrics_for_json['confusion_matrix'] = metrics['confusion_matrix'].tolist()
        json.dump(metrics_for_json, f, indent=2)
    print(f"Saved performance metrics (JSON) to {json_file}")

def calculate_car_for_ticker(ticker, event_date, days_before=20, days_after=40):
    """
    Calculate Cumulative Abnormal Returns (CAR) for a specific ticker around an event date.
    Uses local database instead of yfinance.
    
    Args:
        ticker (str): Stock ticker symbol
        event_date (str): Event date in YYYY-MM-DD format
        days_before (int): Number of days before event to analyze
        days_after (int): Number of days after event to analyze
    
    Returns:
        dict: CAR data including relative days and CAR values, or None if error
    """
    try:
        # Convert event date to datetime
        event_dt = datetime.strptime(event_date, '%Y-%m-%d')
        
        # Calculate date range with more buffer to account for weekends/holidays
        start_date = event_dt - timedelta(days=days_before + 30)  # Extra buffer
        end_date = event_dt + timedelta(days=days_after + 30)
        
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
        
        # Find event date index
        event_idx = None
        for i, date in enumerate(common_dates):
            if date == event_dt.date():
                event_idx = i
                break
        
        if event_idx is None:
            return None
        
        # Check if we have enough data around the event
        if event_idx < days_before:
            return None
        
        if event_idx + days_after >= len(common_dates):
            return None
        
        # Calculate abnormal returns
        abnormal_returns = ticker_returns - benchmark_returns
        
        # Extract window around event
        start_idx = event_idx - days_before
        end_idx = event_idx + days_after + 1
        
        window_ar = abnormal_returns.iloc[start_idx:end_idx]
        
        # Calculate cumulative abnormal returns
        car = np.cumsum(window_ar.values)
        relative_days = np.arange(-days_before, days_after + 1)
        
        return {
            'ticker': ticker,
            'event_date': event_date,
            'relative_days': relative_days,
            'car': car,
            'abnormal_returns': window_ar.values
        }
        
    except Exception as e:
        return None

def read_eps_surprises_from_file(eps_file_path):
    """
    Read EPS surprises file and extract tickers and dates for positive EPS surprises.
    
    Args:
        eps_file_path (str): Path to EPS surprises CSV file
    
    Returns:
        list: List of tuples (ticker, reported_date) for positive EPS surprises
    """
    if not os.path.exists(eps_file_path):
        print(f"Error: EPS surprises file not found: {eps_file_path}")
        return []
    
    try:
        # Read the CSV file
        df = pd.read_csv(eps_file_path)
        
        # Check if required columns exist
        required_columns = ['symbol', 'period', 'surprise']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Error: Missing required columns in EPS file: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
            return []
        
        # Filter for positive EPS surprises (surprise > 0)
        positive_surprises = df[df['surprise'] > 0]
        
        if len(positive_surprises) == 0:
            print("Warning: No positive EPS surprises found")
            return []
        
        # Extract ticker and period date
        events = []
        for _, row in positive_surprises.iterrows():
            ticker = str(row['symbol']).strip()
            period_date = str(row['period']).strip()
            
            # Validate date format
            try:
                # Try to parse the date
                pd.to_datetime(period_date)
                events.append((ticker, period_date))
            except:
                print(f"Warning: Invalid date format for {ticker}: {period_date}")
                continue
        
        print(f"Found {len(events)} positive EPS surprise events")
        return events
        
    except Exception as e:
        print(f"Error reading EPS surprises file: {e}")
        return []

def load_car_data_from_cache(cache_file):
    """
    Load CAR data from cache file.
    
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
        return None

def plot_aggregation(aggregated_df, days_before=20, days_after=40, save_path=None, arch_name="Aggregated Model"):
    """
    Plot CAR analysis for aggregated model predictions.
    
    Args:
        aggregated_df (DataFrame): Aggregated results DataFrame
        days_before (int): Number of days before event
        days_after (int): Number of days after event
        save_path (str): Optional path to save the plot
        arch_name (str): Architecture name for labeling
    """
    # Filter for predicted positive events
    positive_events = aggregated_df[aggregated_df['Aggregated_Predicted_Label'] == 1]
    
    if len(positive_events) == 0:
        print(f"No predicted positive events found for {arch_name}")
        return
    
    print(f"Calculating CAR for {len(positive_events)} predicted positive events...")
    
    # Load cached CAR data
    eps_car_data = load_car_data_from_cache(EPS_CAR_CACHE_FILE)
    if eps_car_data is None:
        print("Error: Could not load cached CAR data")
        return
    
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
        return
    
    print(f"Found CAR data for {len(car_data_list)}/{len(positive_events)} events")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Set up the plot
    relative_days = np.arange(-days_before, days_after + 1)
    
    # Store line objects and their labels for hover functionality
    line_objects = []
    line_labels = []
    
    # Plot CAR for each ticker (thin lines, semi-transparent)
    colors = plt.cm.Set3(np.linspace(0, 1, len(car_data_list)))
    
    # Store all CAR data for averaging
    all_car_data = []
    
    for i, car_data in enumerate(car_data_list):
        line, = ax.plot(car_data['relative_days'], car_data['car'], 
                color=colors[i], linewidth=0.5, alpha=0.3, marker='o', markersize=1)
        all_car_data.append(car_data['car'])
        line_objects.append(line)
        line_labels.append(f"{car_data['ticker']} on {car_data['event_date']}")
    
    # Calculate and plot average CAR
    if all_car_data:
        avg_car = np.mean(all_car_data, axis=0)
        
        # Calculate precision for this architecture
        precision = calculate_average_precision_for_model_type_analysis(arch_name)
        
        # Create legend label with precision
        if precision is not None:
            legend_label = r'$\text{' + arch_name + r' Average CAR} \quad (N = ' + str(len(all_car_data)) + r', P = ' + f'{precision:.3f}' + r')$'
        else:
            legend_label = r'$\text{' + arch_name + r' Average CAR} \quad (N = ' + str(len(all_car_data)) + r')$'
        
        line, = ax.plot(relative_days, avg_car, 
                label=legend_label, 
                color='blue', linewidth=4, marker='s', markersize=8)
        line_objects.append(line)
        line_labels.append(f"{arch_name} Average CAR (N = {len(all_car_data)})")
    
    # Add vertical line at event date
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label=r'$\text{Event Date} \quad (t = 0)$')
    
    # Add horizontal line at zero
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Customize the plot
    ax.set_xlabel(r'Days Relative to Event ($t$)', fontsize=12)
    ax.set_ylabel(r'Cumulative Abnormal Returns (CAR)', fontsize=12)
    
    # Create informative title
    title = r"$\text{CAR Analysis} - \text{" + arch_name + r"} \quad N = " + str(len(all_car_data)) + r"$"
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add shaded areas for different periods
    ax.axvspan(0, MID_POINT_START, alpha=0.1, color='blue', label=r'$\text{Early Post-Event Period} \quad (0 \leq t < 15)$')
    ax.axvspan(MID_POINT_START, days_after, alpha=0.1, color='green', label=r'$\text{Late Post-Event Period} \quad (15 \leq t \leq 40)$')
    
    # Add hover functionality
    def on_hover(event):
        if event.inaxes == ax:
            # Check if mouse is over any line
            for i, line in enumerate(line_objects):
                if line.contains(event)[0]:
                    # Show tooltip with label
                    ax.set_title(f"{title}\n\nHovering: {line_labels[i]}", fontsize=14)
                    fig.canvas.draw_idle()
                    return
            # If not hovering over any line, restore original title
            ax.set_title(title, fontsize=14)
            fig.canvas.draw_idle()
    
    # Connect the hover event
    fig.canvas.mpl_connect('motion_notify_event', on_hover)
    
    plt.tight_layout()
    
    if save_path:
        # Create save directory if it doesn't exist
        save_dir = Path(save_path) / arch_name
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / "car_analysis_individual_stocks.png", dpi=300, bbox_inches='tight')
        print(f"CAR plot saved to {save_dir / 'car_analysis_individual_stocks.png'}")
    
    plt.show()
    
    # Print summary statistics
    print_car_summary(car_data_list, days_before, arch_name)

def plot_combined_car_comparison(predictions_car_data, eps_car_data, days_before, days_after, save_path=None, arch_name="Aggregated Model"):
    """
    Plot combined CAR analysis showing average CARs for aggregated predictions vs EPS surprises.
    
    Args:
        predictions_car_data (list): List of CAR data for aggregated predictions
        eps_car_data (list): List of CAR data for EPS surprises
        days_before (int): Number of days before event
        days_after (int): Number of days after event
        save_path (str): Optional path to save the plot
        arch_name (str): Architecture name for labeling
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Set up the plot
    relative_days = np.arange(-days_before, days_after + 1)
    
    # Store line objects and their labels for hover functionality
    line_objects = []
    line_labels = []
    
    # Plot predictions average CAR
    if predictions_car_data:
        predictions_cars = [data['car'] for data in predictions_car_data if data is not None]
        if predictions_cars:
            avg_predictions_car = np.mean(predictions_cars, axis=0)
            
            # Calculate precision for this architecture
            precision = calculate_average_precision_for_model_type_analysis(arch_name)
            
            # Create legend label with precision
            if precision is not None:
                legend_label = r'$\text{' + arch_name + r' Average CAR} \quad (N = ' + str(len(predictions_cars)) + r', P = ' + f'{precision:.3f}' + r')$'
            else:
                legend_label = r'$\text{' + arch_name + r' Average CAR} \quad (N = ' + str(len(predictions_cars)) + r')$'
            
            line, = ax.plot(relative_days, avg_predictions_car, 
                    label=legend_label, 
                    color='blue', linewidth=4, marker='s', markersize=8)
            line_objects.append(line)
            line_labels.append(f"{arch_name} Average CAR (N = {len(predictions_cars)})")
    
    # Plot EPS surprises average CAR
    if eps_car_data:
        eps_cars = [data['car'] for data in eps_car_data if data is not None]
        if eps_cars:
            avg_eps_car = np.mean(eps_cars, axis=0)
            line, = ax.plot(relative_days, avg_eps_car, 
                    label=r'$\text{Positive EPS Surprises Average CAR} \quad (N = ' + str(len(eps_cars)) + r')$', 
                    color='green', linewidth=4, marker='o', markersize=8)
            line_objects.append(line)
            line_labels.append(f"Positive EPS Surprises Average CAR (N = {len(eps_cars)})")
    
    # Add test set average CAR
    test_avg_car, test_relative_days = calculate_test_set_average_car()
    if test_avg_car is not None and test_relative_days is not None:
        line, = ax.plot(test_relative_days, test_avg_car, 
                label=r'$\text{Test Set Average CAR} \quad (N = 191)$', 
                color='red', linewidth=4, marker='^', markersize=8, linestyle='--')
        line_objects.append(line)
        line_labels.append(f"Test Set Average CAR (N = 191)")
    
    # Add vertical line at event date
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label=r'$\text{Event Date} \quad (t = 0)$')
    
    # Add horizontal line at zero
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Customize the plot
    ax.set_xlabel(r'Days Relative to Event ($t$)', fontsize=12)
    ax.set_ylabel(r'Cumulative Abnormal Returns (CAR)', fontsize=12)
    
    # Create informative title
    title = r"$\text{Average CAR Comparison: " + arch_name + r" vs Positive EPS Surprises}$"
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add shaded areas for different periods
    ax.axvspan(0, MID_POINT_START, alpha=0.1, color='blue', label=r'$\text{Early Post-Event Period} \quad (0 \leq t < 15)$')
    ax.axvspan(MID_POINT_START, days_after, alpha=0.1, color='green', label=r'$\text{Late Post-Event Period} \quad (15 \leq t \leq 40)$')
    
    # Add hover functionality
    def on_hover(event):
        if event.inaxes == ax:
            # Check if mouse is over any line
            for i, line in enumerate(line_objects):
                if line.contains(event)[0]:
                    # Show tooltip with label
                    ax.set_title(f"{title}\n\nHovering: {line_labels[i]}", fontsize=14)
                    fig.canvas.draw_idle()
                    return
            # If not hovering over any line, restore original title
            ax.set_title(title, fontsize=14)
            fig.canvas.draw_idle()
    
    # Connect the hover event
    fig.canvas.mpl_connect('motion_notify_event', on_hover)
    
    plt.tight_layout()
    
    if save_path:
        # Create save directory if it doesn't exist
        save_dir = Path(save_path) / arch_name
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / "average_car_comparison.png", dpi=300, bbox_inches='tight')
        print(f"Combined CAR comparison plot saved to {save_dir / 'average_car_comparison.png'}")
    
    plt.show()

def plot_aggregation_car_comparison(architectures_car_data, eps_car_data, days_before, days_after, save_path="Plots/results"):
    """
    Plot combined CAR analysis showing average CARs for all architectures vs EPS surprises.
    
    Args:
        architectures_car_data (dict): Dictionary of architecture names to list of CAR data
        eps_car_data (list): List of CAR data for EPS surprises
        days_before (int): Number of days before event
        days_after (int): Number of days after event
        save_path (str): Path to save the plot
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Set up the plot
    relative_days = np.arange(-days_before, days_after + 1)
    
    # Store line objects and their labels for hover functionality
    line_objects = []
    line_labels = []
    
    # Plot average CAR for each architecture
    colors = plt.cm.Set3(np.linspace(0, 1, len(architectures_car_data)))
    for i, (arch_name, car_data_list) in enumerate(architectures_car_data.items()):
        if car_data_list:
            cars = [data['car'] for data in car_data_list if data is not None]
            if cars:
                avg_car = np.mean(cars, axis=0)
                
                # Calculate precision for this architecture
                precision = calculate_average_precision_for_model_type_analysis(arch_name)
                
                # Create legend label with precision
                if precision is not None:
                    legend_label = r'$\text{' + arch_name + r' Average CAR} \quad (N = ' + str(len(cars)) + r', P = ' + f'{precision:.3f}' + r')$'
                else:
                    legend_label = r'$\text{' + arch_name + r' Average CAR} \quad (N = ' + str(len(cars)) + r')$'
                
                line, = ax.plot(relative_days, avg_car, 
                        label=legend_label, 
                        color=colors[i], linewidth=4, marker='s', markersize=8)
                line_objects.append(line)
                line_labels.append(f"{arch_name} Average CAR (N = {len(cars)})")
    
    # Plot EPS surprises average CAR
    if eps_car_data:
        eps_cars = [data['car'] for data in eps_car_data if data is not None]
        if eps_cars:
            avg_eps_car = np.mean(eps_cars, axis=0)
            line, = ax.plot(relative_days, avg_eps_car, 
                    label=r'$\text{Positive EPS Surprises Average CAR} \quad (N = ' + str(len(eps_cars)) + r')$', 
                    color='green', linewidth=4, marker='o', markersize=8)
            line_objects.append(line)
            line_labels.append(f"Positive EPS Surprises Average CAR (N = {len(eps_cars)})")
    
    # Add test set average CAR
    test_avg_car, test_relative_days = calculate_test_set_average_car()
    if test_avg_car is not None and test_relative_days is not None:
        line, = ax.plot(test_relative_days, test_avg_car, 
                label=r'$\text{Test Set Average CAR} \quad (N = 191)$', 
                color='red', linewidth=4, marker='^', markersize=8, linestyle='--')
        line_objects.append(line)
        line_labels.append(f"Test Set Average CAR (N = 191)")
    
    # Add vertical line at event date
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label=r'$\text{Event Date} \quad (t = 0)$')
    
    # Add horizontal line at zero
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Customize the plot
    ax.set_xlabel(r'Days Relative to Event ($t$)', fontsize=12)
    ax.set_ylabel(r'Cumulative Abnormal Returns (CAR)', fontsize=12)
    
    # Create informative title
    title = r"$\text{Average CAR Comparison: All Architectures vs Positive EPS Surprises}$"
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add shaded areas for different periods
    ax.axvspan(0, MID_POINT_START, alpha=0.1, color='blue', label=r'$\text{Early Post-Event Period} \quad (0 \leq t < 15)$')
    ax.axvspan(MID_POINT_START, days_after, alpha=0.1, color='green', label=r'$\text{Late Post-Event Period} \quad (15 \leq t \leq 40)$')
    
    # Add hover functionality
    def on_hover(event):
        if event.inaxes == ax:
            # Check if mouse is over any line
            for i, line in enumerate(line_objects):
                if line.contains(event)[0]:
                    # Show tooltip with label
                    ax.set_title(f"{title}\n\nHovering: {line_labels[i]}", fontsize=14)
                    fig.canvas.draw_idle()
                    return
            # If not hovering over any line, restore original title
            ax.set_title(title, fontsize=14)
            fig.canvas.draw_idle()
    
    # Connect the hover event
    fig.canvas.mpl_connect('motion_notify_event', on_hover)
    
    plt.tight_layout()
    
    # Create save directory if it doesn't exist
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / "aggregation_car_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Aggregation CAR comparison plot saved to {save_dir / 'aggregation_car_comparison.png'}")
    
    plt.show()

def plot_aggregation_vs_best_comparison(aggregated_car_data, best_car_data, days_before, days_after, save_path, arch_name):
    """
    Plot comparison between aggregated model and best individual model for a specific architecture.
    
    Args:
        aggregated_car_data (list): List of CAR data for aggregated predictions
        best_car_data (list): List of CAR data for best individual model predictions
        days_before (int): Number of days before event
        days_after (int): Number of days after event
        save_path (str): Path to save the plot
        arch_name (str): Architecture name for labeling
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Set up the plot
    relative_days = np.arange(-days_before, days_after + 1)
    
    # Store line objects and their labels for hover functionality
    line_objects = []
    line_labels = []
    
    # Plot aggregated model average CAR
    if aggregated_car_data:
        aggregated_cars = [data['car'] for data in aggregated_car_data if data is not None]
        if aggregated_cars:
            avg_aggregated_car = np.mean(aggregated_cars, axis=0)
            
            # Calculate precision for this architecture
            precision = calculate_average_precision_for_model_type_analysis(arch_name)
            
            # Create legend label with precision
            if precision is not None:
                legend_label = r'$\text{' + arch_name + r' Aggregated Average CAR} \quad (N = ' + str(len(aggregated_cars)) + r', P = ' + f'{precision:.3f}' + r')$'
            else:
                legend_label = r'$\text{' + arch_name + r' Aggregated Average CAR} \quad (N = ' + str(len(aggregated_cars)) + r')$'
            
            line, = ax.plot(relative_days, avg_aggregated_car, 
                    label=legend_label, 
                    color='blue', linewidth=4, marker='s', markersize=8)
            line_objects.append(line)
            line_labels.append(f"{arch_name} Aggregated Average CAR (N = {len(aggregated_cars)})")
    
    # Plot best individual model average CAR
    if best_car_data:
        best_cars = [data['car'] for data in best_car_data if data is not None]
        if best_cars:
            avg_best_car = np.mean(best_cars, axis=0)
            line, = ax.plot(relative_days, avg_best_car, 
                    label=r'$\text{' + arch_name + r' Best Individual Average CAR} \quad (N = ' + str(len(best_cars)) + r')$', 
                    color='red', linewidth=4, marker='^', markersize=8)
            line_objects.append(line)
            line_labels.append(f"{arch_name} Best Individual Average CAR (N = {len(best_cars)})")
    
    # Add vertical line at event date
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, label=r'$\text{Event Date} \quad (t = 0)$')
    
    # Add horizontal line at zero
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Customize the plot
    ax.set_xlabel(r'Days Relative to Event ($t$)', fontsize=12)
    ax.set_ylabel(r'Cumulative Abnormal Returns (CAR)', fontsize=12)
    
    # Create informative title
    title = r"$\text{Average CAR Comparison: " + arch_name + r" Aggregated vs Best Individual Model}$"
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add shaded areas for different periods
    ax.axvspan(0, MID_POINT_START, alpha=0.1, color='blue', label=r'$\text{Early Post-Event Period} \quad (0 \leq t < 15)$')
    ax.axvspan(MID_POINT_START, days_after, alpha=0.1, color='green', label=r'$\text{Late Post-Event Period} \quad (15 \leq t \leq 40)$')
    
    # Add hover functionality
    def on_hover(event):
        if event.inaxes == ax:
            # Check if mouse is over any line
            for i, line in enumerate(line_objects):
                if line.contains(event)[0]:
                    # Show tooltip with label
                    ax.set_title(f"{title}\n\nHovering: {line_labels[i]}", fontsize=14)
                    fig.canvas.draw_idle()
                    return
            # If not hovering over any line, restore original title
            ax.set_title(title, fontsize=14)
            fig.canvas.draw_idle()
    
    # Connect the hover event
    fig.canvas.mpl_connect('motion_notify_event', on_hover)
    
    plt.tight_layout()
    
    # Create save directory if it doesn't exist
    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f"{arch_name}_aggregated_vs_best_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Aggregation vs Best comparison plot saved to {save_dir / f'{arch_name}_aggregated_vs_best_comparison.png'}")
    
    plt.show()

def find_best_model_predictions(arch_name):
    """
    Find the best model predictions file for a given architecture.
    
    Args:
        arch_name (str): Architecture name (e.g., 'heterognn', 'heterognn2')
    
    Returns:
        str: Path to the best model's test_predictions.csv file, or None if not found
    """
    best_dir_name = f"_{arch_name}_best"
    best_dir_path = Path("Results") / arch_name / best_dir_name
    
    if not best_dir_path.exists():
        print(f"Best model directory not found: {best_dir_path}")
        return None
    
    test_pred_file = best_dir_path / "test_predictions.csv"
    if not test_pred_file.exists():
        print(f"Test predictions file not found in best model directory: {test_pred_file}")
        return None
    
    return str(test_pred_file)

def print_car_summary(car_data_list, days_before, arch_name="Model"):
    """Print a summary of CAR results for all tickers and average statistics."""
    print(f"\n{'='*80}")
    print(f"CAR ANALYSIS SUMMARY - {arch_name}")
    print(f"{'='*80}")
    
    # Store data for average calculations
    all_pre_event_cars = []
    all_post_event_cars = []
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
        pre_event_car = car_values[days_before]  # CAR at day 0
        post_event_car = car_values[-1]  # Final CAR
        car_change = post_event_car - pre_event_car
        avg_ar = np.mean(car_data['abnormal_returns'])
        
        print(f"Pre-event CAR (t=0): {pre_event_car:.4f}")
        print(f"Post-event CAR (t={len(car_values)-1}): {post_event_car:.4f}")
        print(f"CAR Change: {car_change:.4f}")
        print(f"Average AR: {avg_ar:.4f}")
        
        # Store for averaging
        all_pre_event_cars.append(pre_event_car)
        all_post_event_cars.append(post_event_car)
        all_car_changes.append(car_change)
        all_avg_ars.append(avg_ar)
    
    # Print average statistics
    if all_pre_event_cars:
        print(f"\n{'='*50}")
        print(f"AVERAGE STATISTICS ({len(all_pre_event_cars)} events)")
        print(f"{'='*50}")
        print(f"Average Pre-event CAR: {np.mean(all_pre_event_cars):.4f}")
        print(f"Average Post-event CAR: {np.mean(all_post_event_cars):.4f}")
        print(f"Average CAR Change: {np.mean(all_car_changes):.4f}")
        print(f"Average AR: {np.mean(all_avg_ars):.4f}")
        print(f"Standard Deviation of CAR Change: {np.std(all_car_changes):.4f}")

# =============================================================================

def calculate_test_set_average_car(eps_car_cache_file="Data/eps_car_cache.csv", test_set_file="Data/test_set.csv"):
    """
    Calculate the average CAR for all instances in the test set.
    
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
        for _, test_row in test_df.iterrows():
            key = test_row['key']
            matching_car = car_df[car_df['key'] == key]
            if len(matching_car) > 0:
                car_values = np.array(eval(matching_car.iloc[0]['car_values']))
                relative_days = np.array(eval(matching_car.iloc[0]['relative_days']))
                matched_records.append(car_values)
        
        if len(matched_records) == 0:
            print("No matching CAR records found for test set")
            return None, None
        
        # Calculate average CAR
        avg_car = np.mean(matched_records, axis=0)
        print(f"Calculated average CAR for {len(matched_records)} test set instances")
        
        return avg_car, relative_days
        
    except Exception as e:
        print(f"Error calculating test set average CAR: {e}")
        return None, None

# =============================================================================

def calculate_average_precision_for_model_type_analysis(model_type):
    """
    Calculate average precision for a model type from Analysis directory performance.json files.
    
    Args:
        model_type (str): The model type (e.g., 'heterognn', 'heterognn2')
    
    Returns:
        float: Average precision across all models of this type
    """
    import json
    
    model_type_path = os.path.join("Analysis", model_type)
    performance_file = os.path.join(model_type_path, "performance.json")
    
    if os.path.exists(performance_file):
        try:
            with open(performance_file, 'r') as f:
                performance = json.load(f)
                if 'precision' in performance:
                    return performance['precision']
        except (json.JSONDecodeError, KeyError):
            pass
    
    return None

# =============================================================================

def main():
    # Default settings - run everything
    results_dir = 'Results'
    output_dir = 'Analysis'
    architectures = None  # Process all architectures
    plot = True  # Always generate plots
    generate_aggregation_comparison = True  # Always generate aggregation comparison
    compare_with_best = True  # Always compare with best models
    
    # Find all test_predictions.csv files
    architectures_dict = find_test_predictions_files(results_dir)
    
    if not architectures_dict:
        print("No architecture directories found!")
        return
    
    print(f"Found architectures: {list(architectures_dict.keys())}")
    
    # Filter architectures if specified (but we're not using command line args anymore)
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
            print(f"\nGenerating CAR plot for {arch_name}...")
            plot_aggregation(aggregated_df, DAYS_BEFORE, DAYS_AFTER, output_dir, arch_name)
            
            # Also generate EPS comparison plot
            print(f"\nGenerating EPS comparison plot for {arch_name}...")
            
            # Get predictions CAR data using cached data
            positive_events = aggregated_df[aggregated_df['Aggregated_Predicted_Label'] == 1]
            predictions_car_data = []
            
            # Load cached CAR data if not already loaded
            if eps_car_data is None:
                eps_car_data = load_car_data_from_cache(EPS_CAR_CACHE_FILE)
                if eps_car_data is None:
                    print("Error: Could not load cached CAR data")
                    continue
            
            # Find matching events in cached data
            for _, row in positive_events.iterrows():
                found = False
                for cached_event in eps_car_data:
                    if (cached_event['ticker'] == row['Ticker'] and 
                        cached_event['event_date'] == row['Reported_Date']):
                        predictions_car_data.append(cached_event)
                        found = True
                        break
                
                if not found:
                    print(f"Warning: Could not find cached CAR data for {row['Ticker']} on {row['Reported_Date']}")
            
            # Store CAR data for aggregation comparison
            all_car_data[arch_name] = predictions_car_data
            
            print(f"Using cached EPS CAR data with {len(eps_car_data)} events")
            
            # Create comparison plot
            if predictions_car_data and eps_car_data:
                plot_combined_car_comparison(predictions_car_data, eps_car_data, DAYS_BEFORE, DAYS_AFTER, output_dir, arch_name)
            else:
                print("Skipping EPS comparison plot - insufficient data")
            
            # Compare with best individual model if requested
            if compare_with_best:
                print(f"\nComparing {arch_name} aggregated model with best individual model...")
                
                # Find best model predictions
                best_pred_file = find_best_model_predictions(arch_name)
                if best_pred_file:
                    # Load best model predictions
                    best_df = load_predictions(best_pred_file)
                    if best_df is not None:
                        # Filter for positive predictions
                        best_positive_events = best_df[best_df['Predicted_Label'] == 1]
                        best_car_data = []
                        
                        for _, row in best_positive_events.iterrows():
                            car_data = calculate_car_for_ticker(
                                row['Ticker'], 
                                row['Reported_Date'], 
                                DAYS_BEFORE, 
                                DAYS_AFTER
                            )
                            if car_data is not None:
                                best_car_data.append(car_data)
                        
                        # Create comparison plot
                        if predictions_car_data and best_car_data:
                            output_path = Path(output_dir) / arch_name
                            plot_aggregation_vs_best_comparison(predictions_car_data, best_car_data, DAYS_BEFORE, DAYS_AFTER, output_path, arch_name)
                        else:
                            print("Skipping best model comparison plot - insufficient data")
                    else:
                        print("Failed to load best model predictions")
                else:
                    print("Best model predictions file not found")
        
        # Print summary
        print(f"\nSummary for {arch_name}:")
        print(f"  Test cases: {len(aggregated_df)}")
        print(f"  Models used: {aggregated_df['Num_Models'].iloc[0]}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
    
    # Generate aggregation comparison plot if requested
    if generate_aggregation_comparison and len(all_car_data) >= 2 and eps_car_data:
        print(f"\nGenerating aggregation comparison plot...")
        print(f"Comparing {len(all_car_data)} architectures: {list(all_car_data.keys())}")
        
        # Get all architectures that have valid CAR data
        valid_architectures = {arch: car_data for arch, car_data in all_car_data.items() if car_data}
        
        if len(valid_architectures) >= 2:
            plot_aggregation_car_comparison(valid_architectures, eps_car_data, DAYS_BEFORE, DAYS_AFTER)
        else:
            print(f"Skipping aggregation comparison plot - need at least 2 architectures with valid data (found {len(valid_architectures)})")
    elif generate_aggregation_comparison:
        print("Skipping aggregation comparison plot - need at least 2 architectures with valid data")

if __name__ == "__main__":
    main() 