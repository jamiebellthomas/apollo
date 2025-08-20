#!/usr/bin/env python3
"""
Script to cache the processed dataset for fast loading
"""
import os
import pickle
import time
import sys
import tempfile
import shutil
import fcntl
import hashlib
import pandas as pd
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from SubGraphDataLoader import SubGraphDataLoader
from run import encode_all_to_heterodata, attach_y_and_meta

def get_cache_filename(n_facts, limit):
    """Generate consistent cache filename"""
    limit_str = limit if limit is not None else 'all'
    return f"cached_dataset_nf{n_facts}_lim{limit_str}.pkl"

def atomic_write_pickle(data, filepath):
    """Atomically write pickle data to file using temporary file"""
    # Create temporary file in same directory
    temp_file = tempfile.NamedTemporaryFile(
        mode='wb', 
        dir=os.path.dirname(filepath), 
        delete=False,
        suffix='.tmp'
    )
    
    try:
        # Write to temporary file
        pickle.dump(data, temp_file)
        temp_file.flush()
        os.fsync(temp_file.fileno())  # Ensure data is written to disk
        
        # Atomically move temporary file to target location
        shutil.move(temp_file.name, filepath)
    except Exception as e:
        # Clean up temporary file on error
        try:
            os.unlink(temp_file.name)
        except:
            pass
        raise e
    finally:
        temp_file.close()

def safe_read_pickle(filepath, timeout=30):
    """Safely read pickle file with file locking and timeout"""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            with open(filepath, 'rb') as f:
                # Try to acquire a shared lock (read-only)
                fcntl.flock(f.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                try:
                    data = pickle.load(f)
                    return data
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except (IOError, OSError) as e:
            if "Resource temporarily unavailable" in str(e):
                # File is locked by another process, wait and retry
                time.sleep(0.1)
                continue
            else:
                raise
        except Exception as e:
            # Other errors (corrupted file, etc.)
            print(f"Error reading cache file {filepath}: {e}")
            return None
    
    raise TimeoutError(f"Could not acquire lock on {filepath} within {timeout} seconds")

def validate_cache_data(cache_data):
    """Validate that cached data has expected structure"""
    required_keys = ['graphs', 'raw_sg', 'n_facts', 'limit', 'timestamp', 'graph_count']
    
    if not isinstance(cache_data, dict):
        return False
    
    for key in required_keys:
        if key not in cache_data:
            return False
    
    # Check that graphs and raw_sg have same length
    if len(cache_data['graphs']) != len(cache_data['raw_sg']):
        return False
    
    # Check that graph_count matches actual count
    if cache_data['graph_count'] != len(cache_data['graphs']):
        return False
    
    return True

def get_cache_dir():
    """Get the cache directory path relative to this script's location"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "dataset_cache")

def cache_dataset(n_facts=25, limit=None, cache_dir=None):
    """Cache the processed dataset for fast loading"""
    if cache_dir is None:
        cache_dir = get_cache_dir()
    
    print(f"Caching dataset with n_facts={n_facts}, limit={limit}")
    
    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache filename
    cache_filename = get_cache_filename(n_facts, limit)
    cache_path = os.path.join(cache_dir, cache_filename)
    
    # Check if cache already exists and is valid
    if os.path.exists(cache_path):
        print(f"Cache already exists at {cache_path}")
        try:
            # Try to validate existing cache
            cache_data = safe_read_pickle(cache_path)
            if cache_data and validate_cache_data(cache_data):
                response = input("Cache appears valid. Do you want to regenerate it? (y/N): ")
                if response.lower() != 'y':
                    print("Using existing cache.")
                    return cache_path
            else:
                print("Existing cache appears corrupted. Will regenerate.")
        except Exception as e:
            print(f"Error reading existing cache: {e}. Will regenerate.")
    
    print("Loading and processing dataset...")
    start_time = time.time()
    
    # Load subgraphs
    subgraphs_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), config.SUBGRAPHS_JSONL)
    loader = SubGraphDataLoader(min_facts=n_facts, limit=limit, jsonl_path=subgraphs_path)
    
    # Encode to HeteroData - now returns separate training and testing data
    training_graphs, testing_graphs, training_raw_sg, testing_raw_sg = encode_all_to_heterodata(loader)
    
    # Attach labels and metadata
    attach_y_and_meta(training_graphs, training_raw_sg)
    attach_y_and_meta(testing_graphs, testing_raw_sg)
    
    # Create separate training and testing cache files (like run.py expects)
    base_cache_filename = get_cache_filename(n_facts, limit)
    training_cache_filename = f"training_{base_cache_filename}"
    testing_cache_filename = f"testing_{base_cache_filename}"
    
    training_cache_path = os.path.join(cache_dir, training_cache_filename)
    testing_cache_path = os.path.join(cache_dir, testing_cache_filename)
    
    # Prepare training cache data
    training_cache_data = {
        'graphs': training_graphs,
        'raw_sg': training_raw_sg,
        'n_facts': n_facts,
        'limit': limit,
        'timestamp': time.time(),
        'graph_count': len(training_graphs)
    }
    
    # Prepare testing cache data
    testing_cache_data = {
        'graphs': testing_graphs,
        'raw_sg': testing_raw_sg,
        'n_facts': n_facts,
        'limit': limit,
        'timestamp': time.time(),
        'graph_count': len(testing_graphs)
    }
    
    # Atomically write both cache files
    print(f"Writing training cache to {training_cache_path}...")
    atomic_write_pickle(training_cache_data, training_cache_path)
    
    print(f"Writing testing cache to {testing_cache_path}...")
    atomic_write_pickle(testing_cache_data, testing_cache_path)
    
    elapsed_time = time.time() - start_time
    print(f"✅ Dataset cached successfully!")
    print(f"   Training cache: {training_cache_path}")
    print(f"   Testing cache: {testing_cache_path}")
    print(f"   Training graphs: {len(training_graphs)}")
    print(f"   Testing graphs: {len(testing_graphs)}")
    print(f"   Time taken: {elapsed_time:.2f} seconds")
    
    return training_cache_path, testing_cache_path

def load_cached_dataset(n_facts=25, limit=None, cache_dir=None):
    """Load cached dataset if available with error handling"""
    if cache_dir is None:
        cache_dir = get_cache_dir()
    
    cache_filename = get_cache_filename(n_facts, limit)
    cache_path = os.path.join(cache_dir, cache_filename)
    
    if not os.path.exists(cache_path):
        print(f"❌ No cache found for n_facts={n_facts}, limit={limit}")
        print(f"   Expected cache file: {cache_path}")
        print("Run cache_dataset() first to create the cache.")
        return None, None
    
    print(f"Loading cached dataset from {cache_path}")
    start_time = time.time()
    
    try:
        # Safely read cache with file locking
        cache_data = safe_read_pickle(cache_path)
        
        if not cache_data:
            print("❌ Failed to read cache file")
            return None, None
        
        # Validate cache data
        if not validate_cache_data(cache_data):
            print("❌ Cache data validation failed - file may be corrupted")
            return None, None
        
        elapsed_time = time.time() - start_time
        print(f"✅ Cached dataset loaded in {elapsed_time:.2f} seconds")
        print(f"   Graphs: {cache_data['graph_count']}")
        print(f"   Cached on: {time.ctime(cache_data['timestamp'])}")
        
        return cache_data['graphs'], cache_data['raw_sg']
        
    except TimeoutError as e:
        print(f"❌ Timeout waiting for cache file lock: {e}")
        print("Another process may be writing to the cache file.")
        return None, None
    except Exception as e:
        print(f"❌ Error loading cache: {e}")
        return None, None

if __name__ == "__main__":
    # Cache the dataset with current parameters
    cache_dataset(n_facts=35, limit=None) 
    
    # HARDCODED TEST SET PROCESSING LOOP
    print("\n" + "="*80)
    print("HARDCODED TEST SET PROCESSING")
    print("="*80)
    
    # Read the test set CSV
    test_set_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Data", "test_set.csv")
    print(f"Reading test set from: {test_set_path}")
    
    test_df = pd.read_csv(test_set_path)
    print(f"Test set contains {len(test_df)} entries")
    
    # Create a set of (ticker, date) tuples for fast lookup
    test_set_entries = set()
    for _, row in test_df.iterrows():
        ticker = row['Ticker']
        date = row['Reported_Date']
        test_set_entries.add((ticker, date))
    
    print(f"Unique (ticker, date) pairs in test set: {len(test_set_entries)}")
    
    # Load all subgraphs
    print("Loading all subgraphs...")
    subgraphs_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), config.SUBGRAPHS_JSONL)
    loader = SubGraphDataLoader(min_facts=35, limit=None, jsonl_path=subgraphs_path, split_data=False)
    
    # Filter subgraphs to only include test set entries
    print("Filtering subgraphs to test set entries...")
    test_subgraphs = []
    all_items = loader.training_items  # Since split_data=False, all items are in training_items
    
    for sg in all_items:
        sg_ticker = sg.primary_ticker
        sg_date = sg.reported_date
        
        # Convert date to string format if needed
        if hasattr(sg_date, 'strftime'):
            sg_date_str = sg_date.strftime('%Y-%m-%d')
        else:
            sg_date_str = str(sg_date)
        
        if (sg_ticker, sg_date_str) in test_set_entries:
            test_subgraphs.append(sg)
    
    print(f"Found {len(test_subgraphs)} subgraphs matching test set entries")
    
    if len(test_subgraphs) == 0:
        print("No matching subgraphs found! Check the date formats and ticker names.")
        sys.exit(1)
    
    # Separate test and training subgraphs
    print("Separating test and training subgraphs...")
    training_subgraphs = []
    
    for sg in all_items:
        sg_ticker = sg.primary_ticker
        sg_date = sg.reported_date
        
        # Convert date to string format if needed
        if hasattr(sg_date, 'strftime'):
            sg_date_str = sg_date.strftime('%Y-%m-%d')
        else:
            sg_date_str = str(sg_date)
        
        if (sg_ticker, sg_date_str) in test_set_entries:
            test_subgraphs.append(sg)
        else:
            training_subgraphs.append(sg)
    
    print(f"Found {len(test_subgraphs)} test subgraphs and {len(training_subgraphs)} training subgraphs")
    
    if len(test_subgraphs) == 0:
        print("No matching test subgraphs found! Check the date formats and ticker names.")
        sys.exit(1)
    
    if len(training_subgraphs) == 0:
        print("No training subgraphs found! This would leave no data for training.")
        sys.exit(1)
    
    # Create a custom loader with test and training subgraphs
    class HardcodedSplitLoader:
        def __init__(self, training_subgraphs, test_subgraphs):
            self.training_subgraphs = training_subgraphs
            self.test_subgraphs = test_subgraphs
        
        def get_training_items(self):
            return self.training_subgraphs
        
        def get_testing_items(self):
            return self.test_subgraphs
    
    hardcoded_loader = HardcodedSplitLoader(training_subgraphs, test_subgraphs)
    
    # Encode both training and test subgraphs
    print("Encoding training and test subgraphs...")
    start_time = time.time()
    
    # Use the existing encoding function with our custom loader
    training_graphs, testing_graphs, training_raw_sg, testing_raw_sg = encode_all_to_heterodata(hardcoded_loader)
    
    # Attach labels and metadata
    attach_y_and_meta(training_graphs, training_raw_sg)
    attach_y_and_meta(testing_graphs, testing_raw_sg)
    
    elapsed_time = time.time() - start_time
    print(f"✅ Encoding completed in {elapsed_time:.2f} seconds")
    print(f"   Training graphs: {len(training_graphs)}")
    print(f"   Test graphs: {len(testing_graphs)}")
    
    # Cache both training and test sets
    cache_dir = get_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    
    # Training cache
    training_cache_filename = "training_hardcoded_nf35_limall.pkl"
    training_cache_path = os.path.join(cache_dir, training_cache_filename)
    
    training_cache_data = {
        'graphs': training_graphs,
        'raw_sg': training_raw_sg,
        'n_facts': 35,
        'limit': None,
        'timestamp': time.time(),
        'graph_count': len(training_graphs),
        'split_type': 'hardcoded_from_csv'
    }
    
    # Test cache
    test_cache_filename = "test_hardcoded_nf35_limall.pkl"
    test_cache_path = os.path.join(cache_dir, test_cache_filename)
    
    test_cache_data = {
        'graphs': testing_graphs,
        'raw_sg': testing_raw_sg,
        'n_facts': 35,
        'limit': None,
        'timestamp': time.time(),
        'graph_count': len(testing_graphs),
        'split_type': 'hardcoded_from_csv',
        'test_set_source': 'Data/test_set.csv'
    }
    
    # Write both cache files
    print(f"Writing training cache to {training_cache_path}...")
    atomic_write_pickle(training_cache_data, training_cache_path)
    
    print(f"Writing test cache to {test_cache_path}...")
    atomic_write_pickle(test_cache_data, test_cache_path)
    
    print(f"✅ Hardcoded split cached successfully!")
    print(f"   Training cache: {training_cache_path}")
    print(f"   Test cache: {test_cache_path}")
    print(f"   Training graphs: {len(training_graphs)}")
    print(f"   Test graphs: {len(testing_graphs)}")
    print(f"   Original test set entries: {len(test_df)}")
    print(f"   Matched test subgraphs: {len(test_subgraphs)}")
    print(f"   Training subgraphs: {len(training_subgraphs)}")
    print(f"   Successfully encoded training: {len(training_graphs)}")
    print(f"   Successfully encoded test: {len(testing_graphs)}")
    
    # Print some sample matches for verification
    print("\nSample test set matches:")
    for i, sg in enumerate(test_subgraphs[:5]):
        print(f"  {i+1}. {sg.primary_ticker} - {sg.reported_date} (label: {getattr(sg, 'y', 'N/A')})")
    
    print("\nSample training set matches:")
    for i, sg in enumerate(training_subgraphs[:5]):
        print(f"  {i+1}. {sg.primary_ticker} - {sg.reported_date} (label: {getattr(sg, 'y', 'N/A')})")
    
    print("\n" + "="*80)
    print("HARDCODED SPLIT PROCESSING COMPLETE")
    print("="*80) 