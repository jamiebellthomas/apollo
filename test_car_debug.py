#!/usr/bin/env python3
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, timedelta

# Test parameters
ticker = "AAPL"
event_date = "2023-12-31"  # Use a date that should have data
start_day = 15
end_day = 40

print(f"Testing CAR calculation for {ticker} on {event_date}")
print(f"Period: day {start_day} to day {end_day}")

# Convert event date to datetime
event_dt = datetime.strptime(event_date, '%Y-%m-%d')

# Calculate date range
start_date = event_dt - timedelta(days=10)
end_date = event_dt + timedelta(days=end_day + 15)

print(f"Data range: {start_date} to {end_date}")

# Connect to database
conn = sqlite3.connect("Data/momentum_data.db")

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
benchmark_df = pd.read_sql_query(benchmark_query, conn, params=["SPY", start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')])

conn.close()

print(f"Ticker data points: {len(ticker_df)}")
print(f"Benchmark data points: {len(benchmark_df)}")

if ticker_df.empty or benchmark_df.empty:
    print("No data found!")
    exit()

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

print(f"Common trading days: {len(common_dates)}")

# Find event date index
event_idx = None
event_dt_date = event_dt.date()

print(f"Looking for event date: {event_dt_date}")

# First try exact match
for i, date in enumerate(common_dates):
    if date == event_dt_date:
        event_idx = i
        print(f"Found exact match at index {i}")
        break

# If no exact match, find the closest date within 5 days
if event_idx is None:
    print("No exact match, looking for closest date...")
    min_diff = float('inf')
    for i, date in enumerate(common_dates):
        diff = abs((date - event_dt_date).days)
        if diff < min_diff and diff <= 5:
            min_diff = diff
            event_idx = i
            print(f"Found closest date {date} at index {i} (diff: {diff} days)")

if event_idx is None:
    print("Could not find event date!")
    exit()

print(f"Event index: {event_idx}")

# Check if we have enough data after the event
if event_idx + end_day >= len(common_dates):
    print(f"Not enough data after event. Need {end_day} days, have {len(common_dates) - event_idx - 1}")
    exit()

# Calculate abnormal returns
abnormal_returns = ticker_returns - benchmark_returns

# Extract window from start_day to end_day after event
start_idx = event_idx + start_day
end_idx = event_idx + end_day + 1

print(f"Window: {start_idx} to {end_idx}")

# Check if we have enough data for the window
if start_idx >= len(abnormal_returns) or end_idx > len(abnormal_returns):
    print("Not enough data for window!")
    exit()

window_ar = abnormal_returns.iloc[start_idx:end_idx]

# Check if we have the expected number of days
expected_days = end_day - start_day + 1
if len(window_ar) < expected_days:
    print(f"Not enough days in window. Expected {expected_days}, got {len(window_ar)}")
    exit()

# Calculate cumulative abnormal returns
car = np.cumsum(window_ar.values)
relative_days = np.arange(start_day, end_day + 1)

print(f"Success! CAR calculated for {len(car)} days")
print(f"CAR values: {car[:5]}... (first 5 values)")
print(f"Relative days: {relative_days[:5]}... (first 5 values)") 