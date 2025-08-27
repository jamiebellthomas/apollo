#!/usr/bin/env python3
"""
Batch processing script to run attention_explainability.py on all HeteroGNN5 results folders.
Updated for top 25 facts.
"""

import os
import subprocess
import sys
from pathlib import Path

def main():
    # Get all heterognn5 folders
    results_dir = Path("../Results/heterognn5")
    if not results_dir.exists():
        print("Error: ../Results/heterognn5 directory not found")
        sys.exit(1)
    
    heterognn5_folders = [f for f in results_dir.iterdir() if f.is_dir()]
    heterognn5_folders.sort()
    
    print(f"Found {len(heterognn5_folders)} HeteroGNN5 results folders")
    
    # Check which ones already have the output file
    processed_folders = []
    unprocessed_folders = []
    
    for folder in heterognn5_folders:
        output_file = folder / "samples_top_25_facts.json"
        if output_file.exists():
            processed_folders.append(folder.name)
        else:
            unprocessed_folders.append(folder.name)
    
    print(f"Already processed: {len(processed_folders)} folders")
    print(f"Need to process: {len(unprocessed_folders)} folders")
    
    if not unprocessed_folders:
        print("All folders already processed!")
        return
    
    # Process unprocessed folders
    for i, folder_name in enumerate(unprocessed_folders, 1):
        folder_path = results_dir / folder_name
        print(f"\n[{i}/{len(unprocessed_folders)}] Processing {folder_name}...")
        
        try:
            result = subprocess.run([
                "python", "attention_explainability.py", str(folder_path)
            ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
            
            if result.returncode == 0:
                print(f"✅ Successfully processed {folder_name}")
            else:
                print(f"❌ Failed to process {folder_name}")
                print(f"Error: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Timeout processing {folder_name}")
        except Exception as e:
            print(f"❌ Exception processing {folder_name}: {e}")
    
    print(f"\nBatch processing completed!")
    print(f"Successfully processed: {len(processed_folders) + len([f for f in unprocessed_folders if (results_dir / f / 'samples_top_25_facts.json').exists()])} folders")

if __name__ == "__main__":
    main()
