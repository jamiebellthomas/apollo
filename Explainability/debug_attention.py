#!/usr/bin/env python3

import json
from pathlib import Path

# Test with a single directory
test_dir = Path("../Results/heterognn5/20250820_145105")
samples_file = test_dir / "samples_all_facts.json"

print(f"Checking {samples_file}")

with open(samples_file, 'r') as f:
    data = json.load(f)

print(f"Total samples: {len(data)}")

# Check first sample
first_sample_key = list(data.keys())[0]
first_sample = data[first_sample_key]

print(f"First sample key: {first_sample_key}")
print(f"First sample metadata: {first_sample['sample_metadata']}")
print(f"Number of facts in first sample: {len(first_sample['all_facts'])}")

# Check the fact indices
fact_indices = [f['fact_index'] for f in first_sample['all_facts']]
print(f"Fact indices: {fact_indices}")
print(f"Min fact index: {min(fact_indices)}")
print(f"Max fact index: {max(fact_indices)}")

# Check if we have all indices from 0 to total_facts-1
total_facts_claimed = first_sample['sample_metadata']['total_facts']
expected_indices = set(range(total_facts_claimed))
actual_indices = set(fact_indices)
missing_indices = expected_indices - actual_indices
print(f"Missing fact indices: {sorted(missing_indices)}")
print(f"Number of missing indices: {len(missing_indices)}")

# Check if there are any facts with 0.0 attention scores
zero_attention_facts = [f for f in first_sample['all_facts'] if f['attention_score'] == 0.0]
print(f"Facts with 0.0 attention score: {len(zero_attention_facts)}")

# Check if there are any facts with non-zero attention scores
non_zero_attention_facts = [f for f in first_sample['all_facts'] if f['attention_score'] > 0.0]
print(f"Facts with non-zero attention score: {len(non_zero_attention_facts)}")

# Check the total_facts field
print(f"Total facts claimed in metadata: {total_facts_claimed}")

# Check if this matches the actual number of facts
if len(first_sample['all_facts']) == total_facts_claimed:
    print("✅ Number of facts matches the claimed total")
else:
    print(f"❌ MISMATCH: {len(first_sample['all_facts'])} facts vs {total_facts_claimed} claimed")

# Check a few more samples
for i, (sample_key, sample_data) in enumerate(list(data.items())[:3]):
    claimed = sample_data['sample_metadata']['total_facts']
    actual = len(sample_data['all_facts'])
    fact_indices = [f['fact_index'] for f in sample_data['all_facts']]
    min_idx = min(fact_indices)
    max_idx = max(fact_indices)
    print(f"Sample {i}: claimed {claimed}, actual {actual}, indices {min_idx}-{max_idx}")

print("\n" + "="*50)
print("DEBUGGING THE LOGIC")
print("="*50)

# Let me check what the logic should be doing
print("The logic should:")
print("1. Get fact_scores from aggregate_fact_attention_for_sample")
print("2. Create all_facts_with_scores by iterating from 0 to len(facts)-1")
print("3. For each fact index fi:")
print("   - If fi in fact_scores: use fact_scores[fi]")
print("   - Else: use 0.0")
print("4. This should give us ALL facts from 0 to len(facts)-1")

print("\nBut we're only getting 25 facts instead of all facts.")
print("This suggests the logic is not working as expected.")
