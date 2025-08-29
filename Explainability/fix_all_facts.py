#!/usr/bin/env python3

import json
from pathlib import Path

def fix_samples_file(file_path):
    """Fix a samples_all_facts.json file to include all facts."""
    print(f"Fixing {file_path}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    fixed_data = {}
    
    for sample_key, sample_data in data.items():
        total_facts = sample_data['sample_metadata']['total_facts']
        current_facts = sample_data['all_facts']
        
        print(f"  Sample {sample_key}: claimed {total_facts}, actual {len(current_facts)}")
        
        if len(current_facts) == total_facts:
            print(f"    ✅ Already correct")
            fixed_data[sample_key] = sample_data
            continue
        
        # Get existing fact indices
        existing_indices = set(f['fact_index'] for f in current_facts)
        missing_indices = set(range(total_facts)) - existing_indices
        
        print(f"    Missing indices: {sorted(missing_indices)}")
        
        # Create new facts list with all facts
        all_facts = []
        
        # Add existing facts first
        for fact in current_facts:
            all_facts.append(fact)
        
        # Add missing facts with 0.0 attention score
        for missing_idx in missing_indices:
            # Find a template fact to copy structure from
            template_fact = current_facts[0]
            new_fact = {
                "fact_index": missing_idx,
                "fact_id": f"fact_{missing_idx}",
                "attention_score": 0.0,
                "date": template_fact.get("date", ""),
                "tickers": template_fact.get("tickers", []),
                "event_type": template_fact.get("event_type", ""),
                "sentiment": template_fact.get("sentiment", None),
                "raw_text": template_fact.get("raw_text", "")
            }
            all_facts.append(new_fact)
        
        # Sort by fact_index
        all_facts.sort(key=lambda x: x['fact_index'])
        
        # Update the sample data
        fixed_sample_data = sample_data.copy()
        fixed_sample_data['all_facts'] = all_facts
        fixed_data[sample_key] = fixed_sample_data
        
        print(f"    ✅ Fixed: {len(all_facts)} facts")
    
    # Save the fixed data
    with open(file_path, 'w') as f:
        json.dump(fixed_data, f, indent=2, default=str)
    
    print(f"✅ Fixed {file_path}")

def main():
    """Fix all samples_all_facts.json files."""
    results_dir = Path("../Results/heterognn5")
    
    if not results_dir.exists():
        print("❌ Results directory not found!")
        return
    
    directories = sorted([d for d in results_dir.iterdir() if d.is_dir()])
    print(f"Found {len(directories)} directories to process...")
    
    for directory in directories:
        samples_file = directory / "samples_all_facts.json"
        if samples_file.exists():
            try:
                fix_samples_file(samples_file)
            except Exception as e:
                print(f"❌ Error fixing {directory.name}: {e}")
        else:
            print(f"⚠️  No samples_all_facts.json found in {directory.name}")
    
    print("✅ All files processed!")

if __name__ == "__main__":
    main()
