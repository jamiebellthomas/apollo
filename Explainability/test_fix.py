#!/usr/bin/env python3

print("Testing the fix logic...")

# Simulate the exact scenario from the real data
fact_scores = {1: 0.5, 24: 0.3, 6: 0.7, 4: 0.2, 27: 0.8, 26: 0.4, 7: 0.6, 8: 0.1, 38: 0.9, 32: 0.3, 17: 0.5, 33: 0.2, 18: 0.7, 19: 0.4, 31: 0.6, 13: 0.3, 22: 0.8, 12: 0.1, 16: 0.5, 21: 0.4, 15: 0.7, 0: 0.2, 2: 0.6, 3: 0.3, 5: 0.8}
len_facts = 40

print(f"fact_scores has {len(fact_scores)} entries")
print(f"fact_scores keys: {sorted(list(fact_scores.keys()))}")

# Test the OLD logic (what's currently happening)
print("\n=== OLD LOGIC ===")
old_all_facts_with_scores = []
for fi in range(len_facts):
    if fi in fact_scores:
        old_all_facts_with_scores.append((fi, fact_scores[fi]))
    else:
        old_all_facts_with_scores.append((fi, 0.0))

print(f"OLD: all_facts_with_scores has {len(old_all_facts_with_scores)} entries")
print(f"OLD: all_facts_with_scores indices: {sorted([fi for fi, _ in old_all_facts_with_scores])}")

# Test the NEW logic (what should be happening)
print("\n=== NEW LOGIC ===")
new_all_facts_with_scores = []

# First, add all facts that have attention scores
for fi, score in fact_scores.items():
    new_all_facts_with_scores.append((fi, score))

# Then, add all remaining facts with 0.0 attention score
scored_fact_indices = set(fact_scores.keys())
for fi in range(len_facts):
    if fi not in scored_fact_indices:
        new_all_facts_with_scores.append((fi, 0.0))

print(f"NEW: all_facts_with_scores has {len(new_all_facts_with_scores)} entries")
print(f"NEW: all_facts_with_scores indices: {sorted([fi for fi, _ in new_all_facts_with_scores])}")

# Compare the results
print("\n=== COMPARISON ===")
if len(old_all_facts_with_scores) == len(new_all_facts_with_scores):
    print("✅ Both logics produce the same number of facts")
else:
    print(f"❌ Different number of facts: OLD={len(old_all_facts_with_scores)}, NEW={len(new_all_facts_with_scores)}")

old_indices = set([fi for fi, _ in old_all_facts_with_scores])
new_indices = set([fi for fi, _ in new_all_facts_with_scores])

if old_indices == new_indices:
    print("✅ Both logics produce the same fact indices")
else:
    print(f"❌ Different fact indices")
    print(f"OLD missing: {sorted(set(range(len_facts)) - old_indices)}")
    print(f"NEW missing: {sorted(set(range(len_facts)) - new_indices)}")

# Check if either logic produces all facts
expected_indices = set(range(len_facts))
if old_indices == expected_indices:
    print("✅ OLD logic produces all facts")
else:
    print(f"❌ OLD logic missing: {sorted(expected_indices - old_indices)}")

if new_indices == expected_indices:
    print("✅ NEW logic produces all facts")
else:
    print(f"❌ NEW logic missing: {sorted(expected_indices - new_indices)}")
