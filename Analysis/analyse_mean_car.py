#!/usr/bin/env python3
"""
Mean CAR Analysis Script

This script analyzes multiple model results directories and plots the mean CAR
and standard deviation (shred) across all models.

Configure the parameters below and run the script.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
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

# Analysis parameters
DAYS_BEFORE = 20
MID_POINT_START = 15
DAYS_AFTER = 40
BENCHMARK_TICKER = "SPY"

# Results directories to analyze (add your model results folders here)
# Option 1: Manual specification
MANUAL_RESULTS_DIRECTORIES = [
    "Results/20250813_122741",
    "Results/20250813_123456",  # Add more directories as needed
    "Results/20250813_124567",
    "Results/20250813_125678",
]

# Option 2: Automatic discovery (recommended)
USE_AUTO_DISCOVERY = True  # Set to True to automatically find all Results subdirectories

MODEL_TYPE = "heterognn"
RESULTS_BASE_DIR = f"Results/{MODEL_TYPE}"  # Base directory to search for model results

# EPS Surprises analysis
EPS_SURPRISES_FILE = "Data/eps_surprises_quarterly_2012_2024.csv"
EPS_CAR_CACHE_FILE = "Data/eps_car_cache.csv"
FORCE_RECALCULATE_EPS = False

# Plot settings
SAVE_PLOT_PATH = None  # Set to path if you want to save the plot

# =============================================================================

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

def process_model_results_from_cache(results_dir, eps_car_data):
    """
    Process a single model results directory using cached EPS data.
    
    Args:
        results_dir (str): Path to results directory
        eps_car_data (list): List of cached CAR data for all EPS events
    
    Returns:
        tuple: (model_name, car_data_list, total_predictions) or (None, None, None) if failed
    """
    print(f"\nProcessing model: {results_dir}")
    
    # Extract model name from directory
    model_name = os.path.basename(results_dir)
    
    # Read predictions file
    events = read_predictions_from_folder(results_dir)
    
    if not events:
        print(f"No events found for {model_name}")
        return None, None, None
    
    total_predictions = len(events)
    print(f"Found {total_predictions} predicted events")
    
    # Filter cached data for this model's predictions
    model_car_data = []
    
    for ticker, reported_date in events:
        # Find matching event in cached data
        found = False
        for cached_event in eps_car_data:
            if (cached_event['ticker'] == ticker and 
                cached_event['event_date'] == reported_date):
                model_car_data.append(cached_event)
                found = True
                break
        
        if not found:
            print(f"Warning: Could not find cached data for {ticker} on {reported_date}")
    
    if model_car_data:
        print(f"Successfully matched {len(model_car_data)}/{total_predictions} events for {model_name}")
        return model_name, model_car_data, total_predictions
    else:
        print(f"No valid CAR data could be found for {model_name}")
        return None, None, None

def plot_mean_car_with_shred(model_results, eps_car_data, days_before, days_after, save_path=None):
    """
    Plot mean CAR with standard deviation (shred) as shaded area around the line.
    
    Args:
        model_results (list): List of tuples (model_name, car_data_list, total_predictions)
        eps_car_data (list): List of CAR data for EPS surprises
        days_before (int): Number of days before event
        days_after (int): Number of days after event
        save_path (str): Optional path to save the plot
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Set up the plot
    relative_days = np.arange(-days_before, days_after + 1)
    
    # Collect mean CAR lines from each model
    model_mean_cars = []
    model_names = []
    
    for model_name, car_data_list, total_predictions in model_results:
        if car_data_list:
            model_cars = [data['car'] for data in car_data_list if data is not None]
            if model_cars:
                # Calculate mean CAR for this model
                model_mean_car = np.mean(model_cars, axis=0)
                model_mean_cars.append(model_mean_car)
                model_names.append(model_name)
    
    if not model_mean_cars:
        print("No valid CAR data found for any models")
        return
    
    # Convert to numpy array for easier computation
    model_mean_cars = np.array(model_mean_cars)
    
    # Calculate mean and standard deviation of the model-level mean CARs
    overall_mean_car = np.mean(model_mean_cars, axis=0)
    overall_std_car = np.std(model_mean_cars, axis=0)
    
    # Plot overall mean CAR with shaded area showing standard deviation
    ax.plot(relative_days, overall_mean_car, 
            label=r'$\text{Mean CAR Across Models} \quad (N = ' + str(len(model_mean_cars)) + r' \text{ models})$', 
            color='blue', linewidth=3, marker='o', markersize=6)
    
    # Add shaded area for standard deviation (spread)
    ax.fill_between(relative_days, 
                    overall_mean_car - overall_std_car, 
                    overall_mean_car + overall_std_car, 
                    alpha=0.15, 
                    color='blue', 
                    label=r'$\text{±1 Standard Deviation (Spread)}$')
    
    # Add EPS surprises baseline if available
    if eps_car_data:
        eps_cars = [data['car'] for data in eps_car_data if data is not None]
        if eps_cars:
            avg_eps_car = np.mean(eps_cars, axis=0)
            ax.plot(relative_days, avg_eps_car, 
                    label=r'$\text{Positive EPS Surprises Average CAR} \quad (N = ' + str(len(eps_cars)) + r')$', 
                    color='red', linewidth=3, marker='s', markersize=6, linestyle='--')
    
    # Add vertical line at event date
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.7, label=r'$\text{Event Date} \quad (t = 0)$')
    
    # Add horizontal line at zero
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Customize the plot
    ax.set_xlabel(r'Days Relative to Event ($t$)', fontsize=12)
    ax.set_ylabel(r'Cumulative Abnormal Returns (CAR)', fontsize=12)
    ax.set_title(r"$\text{Mean CAR with Standard Deviation Spread vs Positive EPS Surprises Baseline}$", fontsize=14)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add shaded areas for different periods
    ax.axvspan(0, MID_POINT_START, alpha=0.1, color='blue', label=r'$\text{Early Post-Event Period} \quad (0 \leq t < 15)$')
    ax.axvspan(MID_POINT_START, days_after, alpha=0.1, color='green', label=r'$\text{Late Post-Event Period} \quad (15 \leq t \leq 40)$')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Mean CAR plot saved to {save_path}")
    else:
        plt.savefig("Plots/results/mean_car_analysis.png", dpi=300, bbox_inches='tight')
        print("Mean CAR plot saved to Plots/results/mean_car_analysis.png")
    
    plt.show()

def get_results_directories(results_base_dir="Results"):
    """
    Automatically discover all subdirectories in the Results directory.
    
    Args:
        results_base_dir (str): Base directory containing model results
    
    Returns:
        list: List of full paths to all subdirectories in the Results directory
    """
    if not os.path.exists(results_base_dir):
        print(f"Warning: Results directory '{results_base_dir}' not found")
        return []
    
    try:
        # Get all subdirectories in the Results directory
        subdirs = []
        for item in os.listdir(results_base_dir):
            item_path = os.path.join(results_base_dir, item)
            if os.path.isdir(item_path):
                subdirs.append(item_path)
        
        # Sort directories for consistent ordering
        subdirs.sort()
        
        print(f"Found {len(subdirs)} results directories:")
        for subdir in subdirs:
            print(f"  - {subdir}")
        
        return subdirs
        
    except Exception as e:
        print(f"Error discovering results directories: {e}")
        return []

def main():
    """Main function to run the mean CAR analysis."""
    
    print("="*80)
    print("MEAN CAR ANALYSIS")
    print("="*80)
    
    # Process each model's results
    model_results = []
    
    # Load EPS surprises baseline
    print(f"\n" + "="*80)
    print("LOADING EPS SURPRISES BASELINE")
    print("="*80)
    
    eps_car_data = None
    if not FORCE_RECALCULATE_EPS:
        eps_car_data = load_car_data_from_cache(EPS_CAR_CACHE_FILE)
    
    if eps_car_data is None:
        print("EPS cache not available. Please run the single model analysis first to generate the cache.")
        return
    
    if USE_AUTO_DISCOVERY:
        results_dirs = get_results_directories(RESULTS_BASE_DIR)
        results_dirs = results_dirs[1:]
    else:
        results_dirs = MANUAL_RESULTS_DIRECTORIES

    for results_dir in results_dirs:
        if os.path.exists(results_dir):
            model_name, car_data_list, total_predictions = process_model_results_from_cache(results_dir, eps_car_data)
            if model_name and car_data_list:
                model_results.append((model_name, car_data_list, total_predictions))
        else:
            print(f"Warning: Results directory not found: {results_dir}")
    
    if not model_results:
        print("No valid model results found. Exiting.")
        return
    
    # Plot mean CAR with shred
    print(f"\n" + "="*80)
    print("GENERATING MEAN CAR ANALYSIS PLOT")
    print("="*80)
    
    plot_mean_car_with_shred(model_results, eps_car_data, DAYS_BEFORE, DAYS_AFTER, SAVE_PLOT_PATH)
    
    # Print summary statistics
    print(f"\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # Collect mean CAR lines from each model
    model_mean_cars = []
    for model_name, car_data_list, total_predictions in model_results:
        if car_data_list:
            model_cars = [data['car'] for data in car_data_list if data is not None]
            if model_cars:
                # Calculate mean CAR for this model
                model_mean_car = np.mean(model_cars, axis=0)
                model_mean_cars.append(model_mean_car)
    
    if model_mean_cars:
        model_mean_cars = np.array(model_mean_cars)
        overall_mean_car = np.mean(model_mean_cars, axis=0)
        overall_std_car = np.std(model_mean_cars, axis=0)
        
        pre_event_car = overall_mean_car[DAYS_BEFORE]  # CAR at day 0
        post_event_car = overall_mean_car[-1]  # Final CAR
        car_change = post_event_car - pre_event_car
        
        pre_event_std = overall_std_car[DAYS_BEFORE]  # Std at day 0
        post_event_std = overall_std_car[-1]  # Final std
        
        print(f"\nAggregated Statistics Across All Models:")
        print(f"  Number of Models: {len(model_mean_cars)}")
        print(f"  Mean Pre-Event CAR (Day 0): {pre_event_car:.6f}")
        print(f"  Mean Post-Event CAR (Final): {post_event_car:.6f}")
        print(f"  Mean Total CAR Change: {car_change:.6f}")
        print(f"  Mean Pre-Event Std (Day 0): {pre_event_std:.6f}")
        print(f"  Mean Post-Event Std (Final): {post_event_std:.6f}")

if __name__ == "__main__":
    main() 