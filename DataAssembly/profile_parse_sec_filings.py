import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
import psutil
import time
import pandas as pd
from parse_sec_filings import extract_eps

def profile_function(func, *args, **kwargs):
    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss / (1024 * 1024)  # in MB
    start_cpu = time.process_time()

    _,_ = func(*args, **kwargs)

    end_cpu = time.process_time()
    end_mem = process.memory_info().rss / (1024 * 1024)  # in MB

    print(f"CPU time used: {end_cpu - start_cpu:.2f}s")
    print(f"Memory used: {end_mem - start_mem:.2f} MB")

    cpu_usage = end_cpu - start_cpu
    memory_usage = end_mem - start_mem
    return cpu_usage, memory_usage

def main():
    """
    Main function to profile the EPS extraction from SEC filings.
    """

    df = pd.read_csv(config.EPS_DATA_CSV)
    df_AAPL = df[df['Ticker'] == 'AAPL']

    if df_AAPL.empty:
        print("[INFO] No EPS data found for AAPL.")
        return
    # Extract 'Query' column from the DataFrame, this is the arg for extract_eps function that we are profiling
    queries = df_AAPL['Query'].tolist()
    print(f"[INFO] Extracting EPS data for {len(queries)} queries...")
    # Profile the extract_eps function
    total_cpu_usage = 0
    total_memory_usage = 0
    for query in queries:
        cpu_usage, memory_usage = profile_function(extract_eps, query)
        total_cpu_usage += cpu_usage
        total_memory_usage += memory_usage
        print(f"[INFO] CPU usage for query '{query}': {cpu_usage:.2f}s, Memory usage: {memory_usage:.2f} MB")

    average_cpu_usage = total_cpu_usage / len(queries)
    average_memory_usage = total_memory_usage / len(queries)
    print(f"[INFO] Average CPU usage: {average_cpu_usage:.2f}s, Average Memory usage: {average_memory_usage:.2f} MB")

    # Calculate the total expected CPU usage based on the average CPU usage and the number of queries in the original dataframe. 
    # Not just the AAPL queries.
    total_expected_cpu_usage = average_cpu_usage * len(df['Query'].tolist())
    print(f"[INFO] Total expected CPU usage for all queries: {total_expected_cpu_usage:.2f}s")
    total_expected_memory_usage = average_memory_usage * len(df['Query'].tolist())
    print(f"[INFO] Total expected Memory usage for all queries: {total_expected_memory_usage:.2f} MB")

if __name__ == "__main__":
    main()
    print("[INFO] Profiling complete.")




