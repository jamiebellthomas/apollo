#!/usr/bin/env python3
"""
Single Model CAR Analysis Script

This script analyzes CAR (Cumulative Abnormal Returns) for a single model's predictions
against the positive EPS surprises baseline.

Configure the parameters below and run the script.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
import glob
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

# =============================================================================
# CONFIGURATION - MODIFY THESE PARAMETERS
# =============================================================================

# Option 1: Manual configuration
MANUAL_MODE = False  # Set to True to use manual configuration below
MAIN_DATE = "2024-01-15"
DAYS_BEFORE = 20
MID_POINT_START = 15
DAYS_AFTER = 40
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
BENCHMARK_TICKER = "SPY"
SAVE_PLOT_PATH = "Plots/results/"

# Option 2: Results folder mode (reads from test_predictions.csv)
RESULTS_FOLDER = "Results/heterognn/20250814_182224_best"  # Path to results folder
PREDICTIONS_FILE = "test_predictions.csv"   # Name of predictions file

# Option 3: EPS Surprises analysis
EPS_SURPRISES_FILE = "Data/eps_surprises_quarterly_2012_2024.csv"  # Path to EPS surprises file
INCLUDE_EPS_ANALYSIS = True  # Set to True to include EPS surprises analysis
EPS_SAMPLE_SIZE = None  # Number of EPS events to analyze (set to None for all events)
EPS_CAR_CACHE_FILE = "Data/eps_car_cache.csv"  # Cache file for EPS CAR data
FORCE_RECALCULATE_EPS = False  # Set to True to force recalculation and update cache

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

def read_predictions_from_folder(results_folder, predictions_file="test_predictions.csv"):
    """
    Read predictions file and extract tickers and dates for predicted positive events.
    
    Args:
        results_folder (str): Path to results folder
        predictions_file (str): Name of predictions file
    
    Returns:
        list: List of tuples (ticker, reported_date) for Predicted_Label = 1
    """
    file_path = os.path.join(results_folder, predictions_file)
    
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return []
    
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Check if required columns exist
        required_columns = ['Ticker', 'Reported_Date', 'Predicted_Label']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            print(f"Available columns: {list(df.columns)}")
            return []
        
        # Filter for predicted positive events (Predicted_Label = 1)
        positive_events = df[df['Predicted_Label'] == 1]
        
        if len(positive_events) == 0:
            print("Warning: No predicted positive events found (Predicted_Label = 1)")
            return []
        
        # Extract ticker and reported date
        events = []
        for _, row in positive_events.iterrows():
            ticker = str(row['Ticker']).strip()
            reported_date = str(row['Reported_Date']).strip()
            
            # Validate date format
            try:
                # Try to parse the date
                pd.to_datetime(reported_date)
                events.append((ticker, reported_date))
            except:
                print(f"Warning: Invalid date format for {ticker}: {reported_date}")
                continue
        
        print(f"Found {len(events)} predicted positive events")
        return events
        
    except Exception as e:
        print(f"Error reading predictions file: {e}")
        return []

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

def save_car_data_to_cache(car_data_list, cache_file):
    """
    Save CAR data to a cache file for faster loading in future runs.
    
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

def plot_multi_ticker_car(car_data_list, days_before, days_after, save_path=None, dataset_name="Events"):
    """
    Plot CAR for multiple tickers on the same graph, including average CAR.
    
    Args:
        car_data_list (list): List of CAR data dictionaries
        days_before (int): Number of days before event
        days_after (int): Number of days after event
        save_path (str): Optional path to save the plot
        dataset_name (str): Name of the dataset for labeling
    """
    if not car_data_list:
        print(f"No valid CAR data to plot for {dataset_name}")
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set up the plot
    relative_days = np.arange(-days_before, days_after + 1)
    
    # Plot CAR for each ticker (thin lines, semi-transparent)
    colors = plt.cm.Set3(np.linspace(0, 1, len(car_data_list)))
    
    # Store all CAR data for averaging
    all_car_data = []
    
    # Store line objects and their labels for hover functionality
    line_objects = []
    line_labels = []
    
    for i, car_data in enumerate(car_data_list):
        if car_data is None:
            continue
            
        line, = ax.plot(car_data['relative_days'], car_data['car'], 
                color=colors[i], linewidth=0.5, alpha=0.3, marker='o', markersize=1)
        all_car_data.append(car_data['car'])
        line_objects.append(line)
        line_labels.append(f"{car_data['ticker']} on {car_data['event_date']}")
    
    # Calculate and plot average CAR
    if all_car_data:
        avg_car = np.mean(all_car_data, axis=0)
        line, = ax.plot(relative_days, avg_car, 
                label=r'$\text{' + dataset_name + r' Average CAR} \quad (N = ' + str(len(all_car_data)) + r')$', 
                linewidth=3, marker='s', markersize=6)
        line_objects.append(line)
        line_labels.append(f"{dataset_name} Average CAR (N = {len(all_car_data)})")
    
    # Add test set average CAR
    test_avg_car, test_relative_days = calculate_test_set_average_car()
    if test_avg_car is not None and test_relative_days is not None:
        line, = ax.plot(test_relative_days, test_avg_car, 
                label=r'$\text{Test Set Average CAR} \quad (N = 191)$', 
                color='red', linewidth=3, linestyle='--', marker='^', markersize=6)
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
    title = r"$\text{CAR Analysis} - \text{" + dataset_name + r"} \quad N = " + str(len(all_car_data)) + r"$"
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10, loc='best')
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
        plt.savefig(save_path+"individual_stocks.png", dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()

def plot_combined_car_analysis(predictions_car_data, eps_car_data, days_before, days_after, save_path=None):
    """
    Plot combined CAR analysis showing average CARs for predictions vs EPS surprises.
    
    Args:
        predictions_car_data (list): List of CAR data for predictions
        eps_car_data (list): List of CAR data for EPS surprises
        days_before (int): Number of days before event
        days_after (int): Number of days after event
        save_path (str): Optional path to save the plot
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
            line, = ax.plot(relative_days, avg_predictions_car, 
                    label=r'$\text{Predicted Events Average CAR} \quad (N = ' + str(len(predictions_cars)) + r')$', 
                    color='blue', linewidth=4, marker='s', markersize=8)
            line_objects.append(line)
            line_labels.append(f"Predicted Events Average CAR (N = {len(predictions_cars)})")
    
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
                color='red', linewidth=4, linestyle='--', marker='^', markersize=8)
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
    title = r"$\text{Average CAR Comparison: Model Predictions vs Positive EPS Surprises}$"
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
        plt.savefig(save_path+"average_car_comparison.png", dpi=300, bbox_inches='tight')
        print(f"Combined plot saved to {save_path}")
    
    plt.show()

def print_car_summary(car_data_list, days_before):
    """Print a summary of CAR results for all tickers and average statistics."""
    print("\n" + "="*80)
    print("CAR ANALYSIS SUMMARY")
    print("="*80)
    
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
        
        print(f"Pre-Event CAR (Day 0): {pre_event_car:.6f}")
        print(f"Post-Event CAR (Final): {post_event_car:.6f}")
        print(f"Total CAR Change: {car_change:.6f}")
        
        # Calculate average abnormal return
        avg_ar = np.mean(car_data['abnormal_returns'])
        print(f"Average Abnormal Return: {avg_ar:.6f}")
        
        # Store for averaging
        all_pre_event_cars.append(pre_event_car)
        all_post_event_cars.append(post_event_car)
        all_car_changes.append(car_change)
        all_avg_ars.append(avg_ar)
    
    # Print average statistics
    if all_pre_event_cars:
        print("\n" + "="*80)
        print("AVERAGE STATISTICS ACROSS ALL TICKERS")
        print("="*80)
        print(f"Average Pre-Event CAR: {np.mean(all_pre_event_cars):.6f}")
        print(f"Average Post-Event CAR: {np.mean(all_post_event_cars):.6f}")
        print(f"Average Total CAR Change: {np.mean(all_car_changes):.6f}")
        print(f"Average Abnormal Return: {np.mean(all_avg_ars):.6f}")
        print(f"Number of Tickers Analyzed: {len(all_pre_event_cars)}")

def main():
    """Main function to run the CAR analysis."""
    
    if MANUAL_MODE:
        # Manual mode - use specific tickers and date
        print(f"Running in MANUAL MODE")
        print(f"Looking up CAR for {len(TICKERS)} tickers...")
        print(f"Event date: {MAIN_DATE}")
        print(f"Analysis window: {DAYS_BEFORE} days before to {DAYS_AFTER} days after")
        print(f"Benchmark: {BENCHMARK_TICKER}")
        print(f"Tickers: {', '.join(TICKERS)}")
        
        # Load cached data
        eps_car_data = load_car_data_from_cache(EPS_CAR_CACHE_FILE)
        if eps_car_data is None:
            print("Error: CAR cache not available. Please ensure eps_car_cache.csv exists.")
            return
        
        # Look up CAR for each ticker in cached data
        car_data_list = []
        for ticker in TICKERS:
            print(f"\nLooking up {ticker}...")
            
            # Find matching event in cached data
            found = False
            for cached_event in eps_car_data:
                if (cached_event['ticker'] == ticker and 
                    cached_event['event_date'] == MAIN_DATE):
                    car_data_list.append(cached_event)
                    found = True
                    print(f"✓ Found cached CAR data for {ticker}")
                    break
            
            if not found:
                print(f"✗ Could not find cached data for {ticker} on {MAIN_DATE}")
        
        # Filter out None values
        valid_car_data = [data for data in car_data_list if data is not None]
        
        if not valid_car_data:
            print("\nNo valid CAR data could be found for any ticker.")
            return
        
        # Print summary
        print_car_summary(valid_car_data, DAYS_BEFORE)
        
        # Plot results
        plot_multi_ticker_car(valid_car_data, DAYS_BEFORE, DAYS_AFTER, SAVE_PLOT_PATH, "Manual Events")
        
    else:
        # Results folder mode - read from predictions file
        print(f"Running in RESULTS FOLDER MODE")
        print(f"Reading from: {RESULTS_FOLDER}")
        
        # Read predictions file
        events = read_predictions_from_folder(RESULTS_FOLDER, PREDICTIONS_FILE)
        
        if not events:
            print("No events found. Exiting.")
            return
        
        print(f"Analysis window: {DAYS_BEFORE} days before to {DAYS_AFTER} days after")
        print(f"Benchmark: {BENCHMARK_TICKER}")
        
        # Initialize variables for results
        all_car_data = []
        successful_events = 0
        
        # Try to load from cache first
        eps_car_data = None
        if not FORCE_RECALCULATE_EPS:
            eps_car_data = load_car_data_from_cache(EPS_CAR_CACHE_FILE)
        
        if eps_car_data is None:
            print("Error: CAR cache not available. Please ensure eps_car_cache.csv exists.")
            return
        
        # Use cached data for model predictions
        print("Using cached EPS data for model predictions...")
        
        for ticker, reported_date in events:
            # Find matching event in cached data
            found = False
            for cached_event in eps_car_data:
                if (cached_event['ticker'] == ticker and 
                    cached_event['event_date'] == reported_date):
                    all_car_data.append(cached_event)
                    successful_events += 1
                    found = True
                    break
            
            if not found:
                print(f"Warning: Could not find cached data for {ticker} on {reported_date}")
        
        if not all_car_data:
            print("\nNo valid CAR data could be found for any events.")
            return
        
        print(f"\nSuccessfully processed {successful_events}/{len(events)} events")
        
        # Print summary
        print_car_summary(all_car_data, DAYS_BEFORE)
        
        # Plot results - First plot: Individual predicted events + average
        print(f"\nDEBUG: Plotting {len(all_car_data)} predicted events (first plot)")
        plot_multi_ticker_car(all_car_data, DAYS_BEFORE, DAYS_AFTER, SAVE_PLOT_PATH, "Predicted Events")
        
        # Include EPS surprises analysis if enabled
        if INCLUDE_EPS_ANALYSIS:
            print(f"\n" + "="*80)
            print("EPS SURPRISES ANALYSIS")
            print("="*80)
            
            # Use the same cached data for EPS analysis
            if eps_car_data:
                print(f"Using cached EPS data with {len(eps_car_data)} events")
                
                # Print EPS summary
                print_car_summary(eps_car_data, DAYS_BEFORE)
                
                # Second plot: Comparison of averages only
                print(f"\nDEBUG: Plotting comparison of averages (second plot)")
                plot_combined_car_analysis(all_car_data, eps_car_data, DAYS_BEFORE, DAYS_AFTER, SAVE_PLOT_PATH)
            else:
                print("No cached EPS data available for comparison.")

if __name__ == "__main__":
    main() 