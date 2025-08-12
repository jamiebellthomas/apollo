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
    
    # Encode to HeteroData
    graphs, raw_sg = encode_all_to_heterodata(loader)
    
    # Attach labels and metadata
    attach_y_and_meta(graphs, raw_sg)
    
    # Prepare cache data
    cache_data = {
        'graphs': graphs,
        'raw_sg': raw_sg,
        'n_facts': n_facts,
        'limit': limit,
        'timestamp': time.time(),
        'graph_count': len(graphs)
    }
    
    # Atomically write cache data
    print(f"Writing cache to {cache_path}...")
    atomic_write_pickle(cache_data, cache_path)
    
    elapsed_time = time.time() - start_time
    print(f"✅ Dataset cached successfully!")
    print(f"   Cache file: {cache_path}")
    print(f"   Graphs: {len(graphs)}")
    print(f"   Time taken: {elapsed_time:.2f} seconds")
    
    return cache_path

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
    cache_dataset(n_facts=25, limit=None) 