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
import json
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

MODEL_TYPE = "heterognn5"
RESULTS_BASE_DIR = f"Results/{MODEL_TYPE}"  # Base directory to search for model results

# EPS Surprises analysis
EPS_SURPRISES_FILE = "Data/eps_surprises_quarterly_2012_2024.csv"
EPS_CAR_CACHE_FILE = "Data/eps_car_cache.csv"
FORCE_RECALCULATE_EPS = False

# Plot settings
SAVE_PLOT_PATH = None  # Set to path if you want to save the plot

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
    Shows only the specific model type defined in RESULTS_BASE_DIR.
    
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
    
    # Collect mean CAR lines from each model (for the specific model type)
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
    
    # Plot specific model type mean CAR with shaded area showing standard deviation
    if model_mean_cars:
        model_mean_cars = np.array(model_mean_cars)
        overall_mean_car = np.mean(model_mean_cars, axis=0)
        overall_std_car = np.std(model_mean_cars, axis=0)
        
        # Calculate average precision for this model type
        avg_precision = calculate_average_precision_for_model_type(MODEL_TYPE)
        
        # Create legend label with precision
        if avg_precision is not None:
            legend_label = r'$\text{' + MODEL_TYPE + r' Average CAR} \quad (N = ' + str(len(model_mean_cars)) + r' \text{ models}, P = ' + f'{avg_precision:.3f}' + r')$'
        else:
            legend_label = r'$\text{' + MODEL_TYPE + r' Average CAR} \quad (N = ' + str(len(model_mean_cars)) + r' \text{ models})$'
        
        ax.plot(relative_days, overall_mean_car, 
                label=legend_label, 
                color='blue', linewidth=3, marker='o', markersize=6)
        
        # Add shaded area for standard deviation (spread)
        ax.fill_between(relative_days, 
                        overall_mean_car - overall_std_car, 
                        overall_mean_car + overall_std_car, 
                        alpha=0.15, 
                        color='blue', 
                        label=r'$\text{' + MODEL_TYPE + r' ±1 Standard Deviation}$')
    
    # Add EPS surprises baseline if available
    if eps_car_data:
        eps_cars = [data['car'] for data in eps_car_data if data is not None]
        if eps_cars:
            avg_eps_car = np.mean(eps_cars, axis=0)
            ax.plot(relative_days, avg_eps_car, 
                    label=r'$\text{Positive EPS Surprises Average CAR} \quad (N = ' + str(len(eps_cars)) + r')$', 
                    color='red', linewidth=3, marker='^', markersize=6, linestyle='--')
    
    # Add test set average CAR
    test_avg_car, test_relative_days = calculate_test_set_average_car()
    if test_avg_car is not None and test_relative_days is not None:
        ax.plot(test_relative_days, test_avg_car, 
                label=r'$\text{Test Set Average CAR} \quad (N = 191)$', 
                color='orange', linewidth=3, marker='d', markersize=6, linestyle=':')
    
    # Add vertical line at event date
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.7, label=r'$\text{Event Date} \quad (t = 0)$')
    
    # Add horizontal line at zero
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Customize the plot
    ax.set_xlabel(r'Days Relative to Event ($t$)', fontsize=12)
    ax.set_ylabel(r'Cumulative Abnormal Returns (CAR)', fontsize=12)
    ax.set_title(r"$\text{Mean CAR Analysis: " + MODEL_TYPE + r" vs Baselines}$", fontsize=14)
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
    
    # Print summary statistics for the specific model type
    print(f"\n" + "="*80)
    print(f"{MODEL_TYPE.upper()} SUMMARY STATISTICS")
    print("="*80)
    
    if len(model_mean_cars) > 0:
        # Specific model type statistics
        overall_mean_car = np.mean(model_mean_cars, axis=0)
        overall_std_car = np.std(model_mean_cars, axis=0)
        
        pre_event_car = overall_mean_car[DAYS_BEFORE]
        post_event_car = overall_mean_car[-1]
        car_change = post_event_car - pre_event_car
        
        # Calculate average precision for this model type
        avg_precision = calculate_average_precision_for_model_type(MODEL_TYPE)
        
        print(f"\n{MODEL_TYPE} Statistics ({len(model_mean_cars)} models):")
        print(f"  Mean Pre-Event CAR (Day 0): {pre_event_car:.6f}")
        print(f"  Mean Post-Event CAR (Final): {post_event_car:.6f}")
        print(f"  Mean Total CAR Change: {car_change:.6f}")
        if avg_precision is not None:
            print(f"  Average Precision: {avg_precision:.3f}")
        else:
            print(f"  Average Precision: Not available")

def plot_all_model_types_separate(eps_car_data, days_before, days_after, save_path=None):
    """
    Plot AVERAGE CAR for each model TYPE (heterognn, heterognn2, etc.) as separate lines.
    
    Args:
        eps_car_data (list): List of CAR data for EPS surprises
        days_before (int): Number of days before event
        days_after (int): Number of days after event
        save_path (str): Optional path to save the plot
    """
    print(f"\n" + "="*80)
    print("GENERATING MODEL TYPE AVERAGES SEPARATE LINES PLOT")
    print("="*80)
    
    # Get all model types from Results directory
    results_base = "Results"
    model_types = []
    
    if os.path.exists(results_base):
        for item in os.listdir(results_base):
            item_path = os.path.join(results_base, item)
            if os.path.isdir(item_path):
                model_types.append(item)
    
    print(f"Found model types: {model_types}")
    
    # Process each model type
    model_type_results = {}
    
    for model_type in model_types:
        print(f"\nProcessing {model_type}...")
        model_type_path = os.path.join(results_base, model_type)
        
        # Get all subdirectories for this model type
        model_dirs = []
        for subitem in os.listdir(model_type_path):
            subitem_path = os.path.join(model_type_path, subitem)
            if os.path.isdir(subitem_path):
                model_dirs.append(subitem_path)
        
        print(f"Found {len(model_dirs)} directories for {model_type}")
        
        # Process all directories for this model type
        model_car_data_list = []
        
        for results_dir in model_dirs:
            model_name, car_data_list, total_predictions = process_model_results_from_cache(results_dir, eps_car_data)
            if model_name and car_data_list:
                model_car_data_list.extend(car_data_list)
        
        if model_car_data_list:
            model_type_results[model_type] = model_car_data_list
            print(f"Successfully processed {len(model_car_data_list)} events for {model_type}")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Set up the plot
    relative_days = np.arange(-days_before, days_after + 1)
    
    # Define colors for different model types
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray']
    
    # Plot each model type AVERAGE as a separate line
    for i, (model_type, car_data_list) in enumerate(model_type_results.items()):
        if car_data_list:
            # Get all individual model directories for this type
            model_type_path = os.path.join(results_base, model_type)
            model_dirs = []
            for subitem in os.listdir(model_type_path):
                subitem_path = os.path.join(model_type_path, subitem)
                if os.path.isdir(subitem_path):
                    model_dirs.append(subitem_path)
            
            # Calculate mean CAR for each individual model
            individual_model_means = []
            for results_dir in model_dirs:
                model_name, car_data_list, total_predictions = process_model_results_from_cache(results_dir, eps_car_data)
                if model_name and car_data_list:
                    model_cars = [data['car'] for data in car_data_list if data is not None]
                    if model_cars:
                        # Calculate mean CAR for this individual model
                        model_mean_car = np.mean(model_cars, axis=0)
                        individual_model_means.append(model_mean_car)
            
            if individual_model_means:
                # Calculate mean and std of the individual model means
                individual_model_means = np.array(individual_model_means)
                model_type_mean_car = np.mean(individual_model_means, axis=0)
                model_type_std_car = np.std(individual_model_means, axis=0)
                
                # Calculate average precision for this model type
                avg_precision = calculate_average_precision_for_model_type(model_type)
                
                color = colors[i % len(colors)]
                
                # Create legend label with precision
                if avg_precision is not None:
                    legend_label = r'$\text{' + model_type + r' Average CAR} \quad (N = ' + str(len(individual_model_means)) + r' \text{ models}, P = ' + f'{avg_precision:.3f}' + r')$'
                else:
                    legend_label = r'$\text{' + model_type + r' Average CAR} \quad (N = ' + str(len(individual_model_means)) + r' \text{ models})$'
                
                ax.plot(relative_days, model_type_mean_car, 
                        label=legend_label, 
                        color=color, linewidth=3, marker='o', markersize=6)
                
                # Add dotted lines for standard deviation
                ax.plot(relative_days, model_type_mean_car + model_type_std_car, 
                        color=color, linewidth=1, linestyle=':', alpha=0.7)
                ax.plot(relative_days, model_type_mean_car - model_type_std_car, 
                        color=color, linewidth=1, linestyle=':', alpha=0.7)
    
    # Add EPS surprises baseline if available
    if eps_car_data:
        eps_cars = [data['car'] for data in eps_car_data if data is not None]
        if eps_cars:
            avg_eps_car = np.mean(eps_cars, axis=0)
            ax.plot(relative_days, avg_eps_car, 
                    label=r'$\text{Positive EPS Surprises Average CAR} \quad (N = ' + str(len(eps_cars)) + r')$', 
                    color='black', linewidth=3, marker='^', markersize=6, linestyle='--')
    
    # Add test set average CAR
    test_avg_car, test_relative_days = calculate_test_set_average_car()
    if test_avg_car is not None and test_relative_days is not None:
        ax.plot(test_relative_days, test_avg_car, 
                label=r'$\text{Test Set Average CAR} \quad (N = 191)$', 
                color='cyan', linewidth=3, marker='d', markersize=6, linestyle=':')
    
    # Add vertical line at event date
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.7, label=r'$\text{Event Date} \quad (t = 0)$')
    
    # Add horizontal line at zero
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Customize the plot
    ax.set_xlabel(r'Days Relative to Event ($t$)', fontsize=12)
    ax.set_ylabel(r'Cumulative Abnormal Returns (CAR)', fontsize=12)
    ax.set_title(r"$\text{Model Type Averages - Separate Lines}$", fontsize=14)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Add shaded areas for different periods
    ax.axvspan(0, MID_POINT_START, alpha=0.1, color='blue', label=r'$\text{Early Post-Event Period} \quad (0 \leq t < 15)$')
    ax.axvspan(MID_POINT_START, days_after, alpha=0.1, color='green', label=r'$\text{Late Post-Event Period} \quad (15 \leq t \leq 40)$')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Model type averages plot saved to {save_path}")
    else:
        plt.savefig("Plots/results/model_type_averages.png", dpi=300, bbox_inches='tight')
        print("Model type averages plot saved to Plots/results/model_type_averages.png")
    
    plt.show()
    
    # Print summary statistics for each model type
    print(f"\n" + "="*80)
    print("MODEL TYPE AVERAGES SUMMARY STATISTICS")
    print("="*80)
    
    for model_type, car_data_list in model_type_results.items():
        if car_data_list:
            # Get all individual model directories for this type
            model_type_path = os.path.join(results_base, model_type)
            model_dirs = []
            for subitem in os.listdir(model_type_path):
                subitem_path = os.path.join(model_type_path, subitem)
                if os.path.isdir(subitem_path):
                    model_dirs.append(subitem_path)
            
            # Calculate mean CAR for each individual model
            individual_model_means = []
            for results_dir in model_dirs:
                model_name, car_data_list, total_predictions = process_model_results_from_cache(results_dir, eps_car_data)
                if model_name and car_data_list:
                    model_cars = [data['car'] for data in car_data_list if data is not None]
                    if model_cars:
                        # Calculate mean CAR for this individual model
                        model_mean_car = np.mean(model_cars, axis=0)
                        individual_model_means.append(model_mean_car)
            
            if individual_model_means:
                # Calculate mean and std of the individual model means
                individual_model_means = np.array(individual_model_means)
                model_type_mean_car = np.mean(individual_model_means, axis=0)
                
                pre_event_car = model_type_mean_car[DAYS_BEFORE]
                post_event_car = model_type_mean_car[-1]
                car_change = post_event_car - pre_event_car
                
                # Calculate average precision for this model type
                avg_precision = calculate_average_precision_for_model_type(model_type)
                
                print(f"\n{model_type} Statistics ({len(individual_model_means)} models):")
                print(f"  Mean Pre-Event CAR (Day 0): {pre_event_car:.6f}")
                print(f"  Mean Post-Event CAR (Final): {post_event_car:.6f}")
                print(f"  Mean Total CAR Change: {car_change:.6f}")
                if avg_precision is not None:
                    print(f"  Average Precision: {avg_precision:.3f}")
                else:
                    print(f"  Average Precision: Not available")

def calculate_average_precision_for_model_type(model_type):
    """
    Calculate average precision for all models of a given type.
    
    Args:
        model_type (str): The model type (e.g., 'heterognn', 'heterognn2')
    
    Returns:
        float: Average precision across all models of this type
    """
    
    model_type_path = os.path.join("Results", model_type)
    precisions = []
    
    if os.path.exists(model_type_path):
        for subitem in os.listdir(model_type_path):
            subitem_path = os.path.join(model_type_path, subitem)
            if os.path.isdir(subitem_path):
                results_file = os.path.join(subitem_path, "results.json")
                if os.path.exists(results_file):
                    try:
                        with open(results_file, 'r') as f:
                            results = json.load(f)
                            if 'test_metrics' in results and 'precision' in results['test_metrics']:
                                precisions.append(results['test_metrics']['precision'])
                    except (json.JSONDecodeError, KeyError):
                        continue
    
    if precisions:
        return np.mean(precisions)
    else:
        return None

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

def plot_all_models_subplots(eps_car_data, days_before, days_after, save_path=None):
    """
    Plot mean and spread of all 5 models as 5 subplots on the same graph.
    
    Args:
        eps_car_data (list): List of CAR data for EPS surprises
        days_before (int): Number of days before event
        days_after (int): Number of days after event
        save_path (str): Optional path to save the plot
    """
    print(f"\n" + "="*80)
    print("GENERATING ALL MODELS SUBPLOTS")
    print("="*80)
    
    # Get all model types from Results directory
    results_base = "Results"
    model_types = []
    
    if os.path.exists(results_base):
        for item in os.listdir(results_base):
            item_path = os.path.join(results_base, item)
            if os.path.isdir(item_path):
                model_types.append(item)
    
    # Sort to ensure consistent ordering
    model_types.sort()
    
    print(f"Found model types: {model_types}")
    
    # Process each model type
    model_type_results = {}
    
    for model_type in model_types:
        print(f"\nProcessing {model_type}...")
        model_type_path = os.path.join(results_base, model_type)
        
        # Get all subdirectories for this model type
        model_dirs = []
        for subitem in os.listdir(model_type_path):
            subitem_path = os.path.join(model_type_path, subitem)
            if os.path.isdir(subitem_path):
                model_dirs.append(subitem_path)
        
        print(f"Found {len(model_dirs)} directories for {model_type}")
        
        # Process all directories for this model type
        model_car_data_list = []
        
        for results_dir in model_dirs:
            model_name, car_data_list, total_predictions = process_model_results_from_cache(results_dir, eps_car_data)
            if model_name and car_data_list:
                model_car_data_list.extend(car_data_list)
        
        if model_car_data_list:
            model_type_results[model_type] = model_car_data_list
            print(f"Successfully processed {len(model_car_data_list)} events for {model_type}")
    
    # Create subplots - use standard 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    # Set up the plot
    relative_days = np.arange(-days_before, days_after + 1)
    
    # Define colors for different model types
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    # Plot each model type in its own subplot
    for i, (model_type, car_data_list) in enumerate(model_type_results.items()):
        if i >= 5:  # Only use first 5 subplots
            break
            
        ax = axes[i]
        
        if car_data_list:
            # Get all individual model directories for this type
            model_type_path = os.path.join(results_base, model_type)
            model_dirs = []
            for subitem in os.listdir(model_type_path):
                subitem_path = os.path.join(model_type_path, subitem)
                if os.path.isdir(subitem_path):
                    model_dirs.append(subitem_path)
            
            # Calculate mean CAR for each individual model
            individual_model_means = []
            for results_dir in model_dirs:
                model_name, car_data_list, total_predictions = process_model_results_from_cache(results_dir, eps_car_data)
                if model_name and car_data_list:
                    model_cars = [data['car'] for data in car_data_list if data is not None]
                    if model_cars:
                        # Calculate mean CAR for this individual model
                        model_mean_car = np.mean(model_cars, axis=0)
                        individual_model_means.append(model_mean_car)
            
            if individual_model_means:
                # Calculate mean and std of the individual model means
                individual_model_means = np.array(individual_model_means)
                model_type_mean_car = np.mean(individual_model_means, axis=0)
                model_type_std_car = np.std(individual_model_means, axis=0)
                
                color = colors[i % len(colors)]
                
                # Plot mean CAR
                ax.plot(relative_days, model_type_mean_car, 
                        color=color, linewidth=3, marker='o', markersize=6)
                
                # Add shaded area for standard deviation (spread)
                ax.fill_between(relative_days, 
                                model_type_mean_car - model_type_std_car, 
                                model_type_mean_car + model_type_std_car, 
                                alpha=0.3, 
                                color=color)
                
                # Add EPS surprises baseline if available
                if eps_car_data:
                    eps_cars = [data['car'] for data in eps_car_data if data is not None]
                    if eps_cars:
                        avg_eps_car = np.mean(eps_cars, axis=0)
                        ax.plot(relative_days, avg_eps_car, 
                                color='red', linewidth=2, marker='^', markersize=4, linestyle='--', alpha=0.8)
                
                # Add test set average CAR
                test_avg_car, test_relative_days = calculate_test_set_average_car()
                if test_avg_car is not None and test_relative_days is not None:
                    ax.plot(test_relative_days, test_avg_car, 
                            color='orange', linewidth=2, marker='d', markersize=4, linestyle=':', alpha=0.8)
                
                # Add vertical line at event date
                ax.axvline(x=0, color='black', linestyle='--', alpha=0.7)
                
                # Add horizontal line at zero
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                
                # Customize the subplot
                ax.set_xlabel(r'Days Relative to Event ($t$)', fontsize=12)
                ax.set_ylabel(r'Cumulative Abnormal Returns (CAR)', fontsize=12)
                
                # Create title in "Model X" format
                model_number = model_type.replace('heterognn', '')
                if model_number == '':
                    title = 'Model 1'
                else:
                    title = f'Model {model_number}'
                ax.set_title(title, fontsize=14, fontweight='bold')
                
                ax.grid(True, alpha=0.3)
                
                # Add shaded areas for different periods
                ax.axvspan(0, MID_POINT_START, alpha=0.1, color='blue')
                ax.axvspan(MID_POINT_START, days_after, alpha=0.1, color='green')
                
                # Print summary statistics for this model type
                pre_event_car = model_type_mean_car[DAYS_BEFORE]
                post_event_car = model_type_mean_car[-1]
                car_change = post_event_car - pre_event_car
                
                print(f"\n{model_type} Statistics ({len(individual_model_means)} models):")
                print(f"  Mean Pre-Event CAR (Day 0): {pre_event_car:.6f}")
                print(f"  Mean Post-Event CAR (Final): {post_event_car:.6f}")
                print(f"  Mean Total CAR Change: {car_change:.6f}")
    
    # Use the 6th subplot space for the legend
    legend_ax = axes[5]
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    legend_ax.set_frame_on(False)
    
    # Minimize whitespace for tighter layout
    plt.subplots_adjust(left=0.06, right=0.94, top=0.94, bottom=0.1, wspace=0.15, hspace=0.25)
    
    # Add overall title
    fig.suptitle(r"$\text{Mean CAR Analysis: All Models - Individual Subplots}$", fontsize=16, fontweight='bold')
    
    # Add comprehensive legend explaining all elements
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    
    # Create legend elements
    legend_elements = [
        # Model lines
        Line2D([0], [0], color='gray', linewidth=3, marker='o', markersize=6, label='Model Mean CAR'),
        Line2D([0], [0], color='gray', alpha=0.3, linewidth=10, label='Model +/-1 Standard Deviation'),
        
        # Baseline lines
        Line2D([0], [0], color='red', linewidth=2, linestyle='--', marker='^', markersize=4, label='Positive EPS Surprises Average'),
        Line2D([0], [0], color='orange', linewidth=2, linestyle=':', marker='d', markersize=4, label='Test Set Average'),
        
        # Period shading
        Patch(facecolor='blue', alpha=0.1, label='Early Post-Event Period (0 to 15 days)'),
        Patch(facecolor='green', alpha=0.1, label='Late Post-Event Period (15 to 40 days)'),
        
        # Event markers
        Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='Event Date (t = 0)'),
        Line2D([0], [0], color='black', linestyle='-', linewidth=1, alpha=0.3, label='Zero Baseline')
    ]
    
    # Add legend in the 6th subplot space
    legend_ax.legend(handles=legend_elements, loc='center', fontsize=16, frameon=True, fancybox=True, shadow=True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"All models subplots saved to {save_path}")
    else:
        plt.savefig("Analysis/all_models_subplots.png", dpi=300, bbox_inches='tight')
        print("All models subplots saved to Analysis/all_models_subplots.png")
    
    plt.show()

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
    # print(f"\n" + "="*80)
    # print("GENERATING MEAN CAR ANALYSIS PLOT")
    # print("="*80)
    
    # plot_mean_car_with_shred(model_results, eps_car_data, DAYS_BEFORE, DAYS_AFTER, SAVE_PLOT_PATH)
    
    # Plot all model types separately
    # plot_all_model_types_separate(eps_car_data, DAYS_BEFORE, DAYS_AFTER, SAVE_PLOT_PATH)
    
    # Plot all models as subplots
    plot_all_models_subplots(eps_car_data, DAYS_BEFORE, DAYS_AFTER, SAVE_PLOT_PATH)
    
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