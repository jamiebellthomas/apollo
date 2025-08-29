#!/usr/bin/env python3

print("Testing debug output...")

# Simulate the logic
fact_scores = {1: 0.5, 24: 0.3, 6: 0.7, 4: 0.2, 27: 0.8, 26: 0.4, 7: 0.6, 8: 0.1, 38: 0.9, 32: 0.3, 17: 0.5, 33: 0.2, 18: 0.7, 19: 0.4, 31: 0.6, 13: 0.3, 22: 0.8, 12: 0.1, 16: 0.5, 21: 0.4, 15: 0.7, 0: 0.2, 2: 0.6, 3: 0.3, 5: 0.8}
len_facts = 40

print(f"fact_scores has {len(fact_scores)} entries")
print(f"fact_scores keys: {sorted(list(fact_scores.keys()))}")

# Include ALL facts - those with attention scores and those without
all_facts_with_scores = []

# First, add all facts that have attention scores
for fi, score in fact_scores.items():
    all_facts_with_scores.append((fi, score))

# Then, add all remaining facts with 0.0 attention score
scored_fact_indices = set(fact_scores.keys())
for fi in range(len_facts):
    if fi not in scored_fact_indices:
        all_facts_with_scores.append((fi, 0.0))

print(f"all_facts_with_scores has {len(all_facts_with_scores)} entries")
print(f"all_facts_with_scores indices: {sorted([fi for fi, _ in all_facts_with_scores])}")
print(f"Expected range: 0 to {len_facts-1}")
print(f"Missing indices: {sorted(set(range(len_facts)) - set([fi for fi, _ in all_facts_with_scores]))}")

# Check if we have all facts
if len(all_facts_with_scores) == len_facts:
    print("✅ SUCCESS: We have all facts!")
else:
    print(f"❌ FAILURE: Expected {len_facts} facts, got {len(all_facts_with_scores)}")
