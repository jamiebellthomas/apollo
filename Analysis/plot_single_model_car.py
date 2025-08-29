#!/usr/bin/env python3
"""
Single Model CAR Analysis Script

This script analyzes CAR (Cumulative Abnormal Returns) in the PEAD period for a single model.
It loads predictions from a specific model path and calculates CAR for the medium-term period (day 15 to day 40).
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import json
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - SET YOUR MODEL PATH HERE
# =============================================================================
MODEL_PATH = "Results/heterognn5/20250820_124350"  # Change this to your model path

# =============================================================================
# CONSTANTS
# =============================================================================
DAYS_BEFORE = 0  # Start from event date (day 0)
DAYS_AFTER = 40   # End at day 40
START_DAY = 15    # Start calculating CAR from day 15
END_DAY = 40      # End calculating CAR at day 40
BENCHMARK_TICKER = "SPY"
DATABASE_PATH = "Data/momentum_data.db"
EPS_SURPRISES_FILE = "Data/eps_surprises_quarterly_2012_2024.csv"
EPS_CAR_CACHE_FILE = "Data/eps_car_cache_medium.csv"

# Configure matplotlib for LaTeX formatting
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath} \usepackage{amssymb}",
    "axes.formatter.use_mathtext": True,
    "mathtext.fontset": "cm"
})

def load_model_predictions(model_path):
    """
    Load predictions from a specific model's test_predictions.csv file.
    
    Args:
        model_path (str): Path to the model directory
    
    Returns:
        DataFrame: Predictions with Ticker, Reported_Date, Actual_Label, Predicted_Label columns
    """
    predictions_file = Path(model_path) / "test_predictions.csv"
    
    if not predictions_file.exists():
        print(f"Error: test_predictions.csv not found at {predictions_file}")
        return None
    
    try:
        df = pd.read_csv(predictions_file)
        # Create a unique identifier for each test case
        df['test_case'] = df['Ticker'] + '_' + df['Reported_Date']
        return df[['test_case', 'Ticker', 'Reported_Date', 'Actual_Label', 'Predicted_Label']]
    except Exception as e:
        print(f"Error loading {predictions_file}: {e}")
        return None

def calculate_performance_metrics(df):
    """
    Calculate performance metrics for the model predictions.
    
    Args:
        df (DataFrame): DataFrame with Actual_Label and Predicted_Label columns
    
    Returns:
        dict: Performance metrics
    """
    y_true = df['Actual_Label'].values
    y_pred = df['Predicted_Label'].values
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    # Generate classification report
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'classification_report': report
    }

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
        
        # Find event date index
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
        
        # Check if we have at least 10 days of data
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
                matched_records.append(car_values)
                matched_count += 1
                max_length = max(max_length, len(car_values))
        
        print(f"Matched {matched_count} records with CAR data")
        
        if matched_records:
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
            relative_days = np.arange(START_DAY, START_DAY + max_length)
            
            return avg_car, relative_days
        
        return None, None
        
    except Exception as e:
        print(f"Error calculating test set average CAR: {e}")
        return None, None

def plot_single_model_car(df, start_day=15, end_day=40, output_dir="Analysis", model_name="single_model"):
    """
    Plot CAR for a single model's predictions - simplified version.
    Shows only positive predictions vs test set average.
    
    Args:
        df (DataFrame): DataFrame with predictions
        start_day (int): Start day for CAR calculation
        end_day (int): End day for CAR calculation
        output_dir (str): Output directory
        model_name (str): Name for the model
    
    Returns:
        list: List of CAR data dictionaries
    """
    print(f"\nCalculating CAR for {model_name}...")
    
    # Calculate CAR for each prediction
    car_data_list = []
    successful_calculations = 0
    
    for _, row in df.iterrows():
        ticker = row['Ticker']
        event_date = row['Reported_Date']
        predicted_label = row['Predicted_Label']
        actual_label = row['Actual_Label']
        
        # Calculate CAR
        car_data = calculate_car_for_ticker_medium(ticker, event_date, start_day, end_day)
        
        if car_data is not None:
            car_data['predicted_label'] = predicted_label
            car_data['actual_label'] = actual_label
            car_data_list.append(car_data)
            successful_calculations += 1
    
    print(f"Successfully calculated CAR for {successful_calculations} out of {len(df)} predictions")
    
    if not car_data_list:
        print("No CAR data available for plotting")
        return []
    
    # Separate positive predictions
    predicted_positive = [data for data in car_data_list if data['predicted_label'] == 1]
    
    # Create single plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot positive predictions
    if predicted_positive:
        avg_car_pred_pos = calculate_average_car(predicted_positive)
        ax.plot(avg_car_pred_pos['days'], avg_car_pred_pos['car'], 
                label=f'Positive Predictions (N={len(predicted_positive)})', 
                color='blue', linewidth=3, marker='o', markersize=6)
    
    # Add test set average if available
    test_avg_car, test_days = calculate_test_set_average_car_medium()
    if test_avg_car is not None:
        ax.plot(test_days, test_avg_car, 
                label='Test Set Average (N=191)', 
                color='gray', linewidth=2, linestyle='--', marker='s', markersize=6)
    
    # Customize the plot
    ax.set_xlabel('Days After Event', fontsize=12)
    ax.set_ylabel('Cumulative Abnormal Returns (CAR)', fontsize=12)
    ax.set_title(f'CAR Comparison: {model_name} Positive Predictions vs Test Set', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.axvline(x=start_day, color='red', linestyle='--', alpha=0.7, label='Medium-Term Start')
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plot_file = output_path / f"{model_name}_car_analysis.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"CAR analysis plot saved to {plot_file}")
    
    plt.show()
    
    return car_data_list

def calculate_average_car(car_data_list):
    """
    Calculate average CAR from a list of CAR data dictionaries.
    
    Args:
        car_data_list (list): List of CAR data dictionaries
    
    Returns:
        dict: Average CAR data with 'days' and 'car' arrays
    """
    if not car_data_list:
        return {'days': [], 'car': []}
    
    # Find maximum length
    max_length = max(len(data['car']) for data in car_data_list)
    
    # Interpolate all CAR arrays to the same length
    interpolated_cars = []
    for data in car_data_list:
        car_values = data['car']
        if len(car_values) < max_length:
            # Pad with the last value
            padded_car = np.pad(car_values, (0, max_length - len(car_values)), mode='edge')
        else:
            padded_car = car_values[:max_length]
        interpolated_cars.append(padded_car)
    
    # Calculate average
    avg_car = np.mean(interpolated_cars, axis=0)
    days = np.arange(START_DAY, START_DAY + max_length)
    
    return {'days': days, 'car': avg_car}

def save_performance_metrics(metrics, output_dir, model_name):
    """
    Save performance metrics to files.
    
    Args:
        metrics (dict): Performance metrics
        output_dir (str): Output directory
        model_name (str): Model name
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as text file
    txt_file = output_path / f"{model_name}_performance.txt"
    with open(txt_file, 'w') as f:
        f.write(f"Performance Metrics for {model_name}\n")
        f.write("=" * 50 + "\n\n")
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
    
    # Save as JSON file
    json_file = output_path / f"{model_name}_performance.json"
    with open(json_file, 'w') as f:
        # Convert numpy array to list for JSON serialization
        metrics_for_json = metrics.copy()
        metrics_for_json['confusion_matrix'] = metrics['confusion_matrix'].tolist()
        json.dump(metrics_for_json, f, indent=2)
    
    print(f"Performance metrics saved to {txt_file} and {json_file}")

def main():
    """Main function to run the single model CAR analysis."""
    
    print("Single Model CAR Analysis")
    print("=" * 50)
    print(f"Model Path: {MODEL_PATH}")
    print(f"PEAD Period: Day {START_DAY} to Day {END_DAY}")
    print("=" * 50)
    
    # Check if model path exists
    if not Path(MODEL_PATH).exists():
        print(f"Error: Model path {MODEL_PATH} does not exist!")
        return
    
    # Extract model name from path
    model_name = Path(MODEL_PATH).name
    
    # Load model predictions
    print(f"\nLoading predictions from {MODEL_PATH}...")
    df = load_model_predictions(MODEL_PATH)
    
    if df is None:
        print("Failed to load model predictions!")
        return
    
    print(f"Loaded {len(df)} predictions")
    
    # Calculate performance metrics
    print("\nCalculating performance metrics...")
    metrics = calculate_performance_metrics(df)
    
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    
    # Save performance metrics
    save_performance_metrics(metrics, "Analysis", model_name)
    
    # Generate CAR plots
    print(f"\nGenerating CAR analysis plots...")
    car_data_list = plot_single_model_car(df, START_DAY, END_DAY, "Analysis", model_name)
    
    # Print summary statistics
    if car_data_list:
        print(f"\n" + "="*60)
        print(f"CAR ANALYSIS SUMMARY - {model_name.upper()}")
        print("="*60)
        
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
        
        print(f"Number of events with CAR data: {len(car_data_list)}")
        print(f"Average CAR at day 15: {np.mean(all_start_cars):.4f}")
        print(f"Average CAR at day 40: {np.mean(all_end_cars):.4f}")
        print(f"Average CAR Change (day 15 to 40): {np.mean(all_car_changes):.4f}")
        print(f"Standard Deviation of CAR Change: {np.std(all_car_changes):.4f}")
        
        # Compare with test set average
        test_avg_car, test_days = calculate_test_set_average_car_medium()
        if test_avg_car is not None:
            print(f"\nTest Set Average (N=191):")
            print(f"  Average CAR at day 15: {test_avg_car[0]:.4f}")
            print(f"  Average CAR at day 40: {test_avg_car[-1]:.4f}")
            print(f"  Average CAR Change (day 15 to 40): {test_avg_car[-1] - test_avg_car[0]:.4f}")
    
    print(f"\nSingle model CAR analysis completed for {model_name}!")

if __name__ == "__main__":
    main()
