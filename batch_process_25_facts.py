#!/usr/bin/env python3
"""
Batch processing script to run attention_explainability.py (25 facts version) on all HeteroGNN5 results folders.
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
    
    # Check which ones already have the new 25-facts output file
    processed_folders = []
    unprocessed_folders = []
    
    for folder in heterognn5_folders:
        output_file = folder / "samples_top_25_facts.json"
        if output_file.exists():
            processed_folders.append(folder.name)
        else:
            unprocessed_folders.append(folder.name)
    
    print(f"Already processed (25 facts): {len(processed_folders)} folders")
    print(f"Need to process: {len(unprocessed_folders)} folders")
    
    if not unprocessed_folders:
        print("All folders already processed with 25 facts!")
        return
    
    print("\nUnprocessed folders:")
    for folder in unprocessed_folders:
        print(f"  - {folder}")
    
    # Process each unprocessed folder
    print(f"\nStarting batch processing of {len(unprocessed_folders)} folders...")
    
    for i, folder_name in enumerate(unprocessed_folders, 1):
        folder_path = f"../Results/heterognn5/{folder_name}"
        print(f"\n[{i}/{len(unprocessed_folders)}] Processing {folder_name}...")
        
        try:
            # Run the attention explainability script
            result = subprocess.run([
                sys.executable, "attention_explainability.py", folder_path
            ], capture_output=True, text=True, timeout=300)  # 5 minute timeout per folder
            
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
    
    # Final summary
    final_processed = []
    final_unprocessed = []
    
    for folder in heterognn5_folders:
        output_file = folder / "samples_top_25_facts.json"
        if output_file.exists():
            final_processed.append(folder.name)
        else:
            final_unprocessed.append(folder.name)
    
    print(f"\nFinal Summary:")
    print(f"Successfully processed (25 facts): {len(final_processed)} folders")
    if final_unprocessed:
        print(f"Failed/Unprocessed: {len(final_unprocessed)} folders")
        for folder in final_unprocessed:
            print(f"  - {folder}")

if __name__ == "__main__":
    main()
