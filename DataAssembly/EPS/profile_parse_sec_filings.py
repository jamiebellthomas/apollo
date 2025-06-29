import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import config
import psutil
import time
import pandas as pd
from parse_sec_filings import extract_eps
import shutil
import tracemalloc

def profile_function_batch(func, query_list):
    process = psutil.Process(os.getpid())

    # Start measuring
    tracemalloc.start()
    start_rss = process.memory_info().rss / (1024 * 1024)  # MB
    start_cpu = time.process_time()
    start_wall = time.time()

    # Run the batch
    for index,query in enumerate(query_list):
        print(f"[INFO] Processing query {index + 1}/{len(query_list)}: {query}")
        try:
            func(query)
        except Exception as e:
            print(f"[WARN] Failed on query: {e}")

    end_wall = time.time()
    end_cpu = time.process_time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end_rss = process.memory_info().rss / (1024 * 1024)  # MB

    return {
        "cpu_time": end_cpu - start_cpu,
        "wall_time": end_wall - start_wall,
        "rss_memory_used": end_rss - start_rss,
        "peak_memory_used": peak / (1024 * 1024),  # MB
    }

def get_disk_usage(path="."):
    usage = shutil.disk_usage(path)
    return usage.used / (1024 * 1024 * 1024)  # GB

def main():
    df = pd.read_csv(config.EPS_DATA_CSV)
    df_subset = df[df['Ticker'] == 'AAPL']

    if df_subset.empty:
        print("[INFO] No data for AAPL.")
        return

    queries_subset = df_subset['Query'].tolist()
    total_rows = len(df)
    subset_size = len(queries_subset)

    print(f"[INFO] Profiling {subset_size} queries (subset of {total_rows})...")

    disk_before = get_disk_usage()
    metrics = profile_function_batch(extract_eps, queries_subset)
    disk_after = get_disk_usage()

    disk_used = disk_after - disk_before
    scale_factor = total_rows / subset_size if subset_size else 0

    print("\n[SUBSET RESULTS]")
    print(f"Subset CPU time:            {metrics['cpu_time']:.2f} s")
    print(f"Subset wall time:           {metrics['wall_time']:.2f} s")
    print(f"Subset RSS memory used:     {metrics['rss_memory_used']:.2f} MB")
    print(f"Subset Peak memory used:    {metrics['peak_memory_used']:.2f} MB")
    print(f"Subset Disk usage:          {disk_used:.2f} GB")

    print("\n[PROJECTED TOTAL USAGE]")
    print(f"Total CPU time:             {metrics['cpu_time'] * scale_factor:.2f} s")
    print(f"Total wall time:            {metrics['wall_time'] * scale_factor:.2f} s")
    print(f"Total RSS memory used:      {metrics['rss_memory_used'] * scale_factor:.2f} MB")
    print(f"Total Peak memory used:     {metrics['peak_memory_used'] * scale_factor:.2f} MB")
    print(f"Total Disk usage:           {disk_used * scale_factor:.2f} GB")

if __name__ == "__main__":
    main()
