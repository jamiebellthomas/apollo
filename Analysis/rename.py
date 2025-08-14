#!/usr/bin/env python3
"""
Script to rename experiment folders in Results directory.
Renames folders starting with "2" to "seed=[SEED]" format where SEED is extracted from hyperparameters.txt
"""

import os
import re
import shutil
from pathlib import Path

def extract_seed_from_hyperparameters(hyperparams_file):
    """
    Extract the seed value from hyperparameters.txt file.
    
    Args:
        hyperparams_file (str): Path to hyperparameters.txt file
        
    Returns:
        int: The seed value, or None if not found
    """
    try:
        with open(hyperparams_file, 'r') as f:
            content = f.read()
            
        # Look for seed: <number> pattern
        match = re.search(r'seed:\s*(\d+)', content)
        if match:
            return int(match.group(1))
        else:
            print(f"Warning: No seed found in {hyperparams_file}")
            return None
    except FileNotFoundError:
        print(f"Error: Could not find {hyperparams_file}")
        return None
    except Exception as e:
        print(f"Error reading {hyperparams_file}: {e}")
        return None

def rename_experiment_folders(results_dir="Results/heterognn"):
    """
    Rename experiment folders that start with "2" to "seed=[SEED]" format.
    
    Args:
        results_dir (str): Path to the Results directory
    """
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Error: Results directory '{results_dir}' does not exist")
        return
    
    # Get all folders that start with "2"
    folders_to_rename = []
    for folder in results_path.iterdir():
        if folder.is_dir() and folder.name.startswith("2"):
            folders_to_rename.append(folder)
    
    print(f"Found {len(folders_to_rename)} folders starting with '2'")
    
    # Process each folder
    renamed_count = 0
    skipped_count = 0
    
    for folder in folders_to_rename:
        print(f"\nProcessing: {folder.name}")
        
        # Check if hyperparameters.txt exists
        hyperparams_file = folder / "hyperparameters.txt"
        if not hyperparams_file.exists():
            print(f"  Skipping: No hyperparameters.txt found in {folder.name}")
            skipped_count += 1
            continue
        
        # Extract seed value
        seed = extract_seed_from_hyperparameters(hyperparams_file)
        if seed is None:
            print(f"  Skipping: Could not extract seed from {folder.name}")
            skipped_count += 1
            continue
        
        # Create new folder name
        new_name = f"seed={seed}"
        new_path = results_path / new_name
        
        # Check if target name already exists
        if new_path.exists():
            print(f"  Skipping: Target folder '{new_name}' already exists")
            skipped_count += 1
            continue
        
        # Rename the folder
        try:
            folder.rename(new_path)
            print(f"  Renamed: {folder.name} -> {new_name}")
            renamed_count += 1
        except Exception as e:
            print(f"  Error renaming {folder.name}: {e}")
            skipped_count += 1
    
    print(f"\nSummary:")
    print(f"  Successfully renamed: {renamed_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Total processed: {len(folders_to_rename)}")

def main():
    """Main function to run the renaming script."""
    print("=== Experiment Folder Renaming Script ===")
    print("This script will rename folders in Results/ that start with '2'")
    print("to 'seed=[SEED]' format based on hyperparameters.txt files.\n")
    
    # Ask for confirmation
    response = input("Do you want to proceed? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Operation cancelled.")
        return
    
    # Run the renaming
    rename_experiment_folders()
    
    print("\nScript completed!")

if __name__ == "__main__":
    main() 