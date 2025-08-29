#!/usr/bin/env python3
"""
Script to add cluster_id to each fact in samples_all_facts.json files.
Uses the same clustering protocol as the original analysis.
"""

import json
import os
import warnings
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np

# Global cache for transformer
_CACHED_TRANSFORMER = None

def get_cached_transformer():
    """Get cached sentence transformer model."""
    global _CACHED_TRANSFORMER
    if _CACHED_TRANSFORMER is None:
        print("[transformer] Loading sentence transformer model from local cache...")
        try:
            cache_dir = os.path.join(os.path.dirname(__file__), '../KG/model_cache')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _CACHED_TRANSFORMER = SentenceTransformer(
                    "all-mpnet-base-v2",
                    cache_folder=cache_dir
                )
            print("[transformer] Model loaded successfully from local cache")
        except Exception as e:
            print(f"[transformer] Error loading model: {e}")
            raise
    return _CACHED_TRANSFORMER

def canonicalize_event_text(et: str, mode: str = "spaces") -> str:
    """Canonicalize event type text for consistent encoding."""
    if mode == "spaces":
        # Replace underscores with spaces
        return et.replace("_", " ")
    elif mode == "underscores":
        # Replace spaces with underscores
        return et.replace(" ", "_")
    else:
        return et

def load_cluster_centroids():
    """Load cluster centroids from the JSONL file."""
    centroids_file = Path("../Data/cluster_centroids.jsonl")
    
    if not centroids_file.exists():
        raise FileNotFoundError(f"Cluster centroids file not found: {centroids_file}")
    
    centroids = []
    with open(centroids_file, 'r') as f:
        for line in f:
            if line.strip():
                centroids.append(json.loads(line))
    
    print(f"[clusters] Loaded {len(centroids)} cluster centroids")
    return centroids

def find_nearest_cluster(event_type: str, centroids, transformer):
    """Find the nearest cluster for a given event type."""
    if not event_type or not event_type.strip():
        return None
    
    # Canonicalize the event type (replace underscores with spaces)
    canonical_event = canonicalize_event_text(event_type, mode="spaces")
    
    # Encode the event type
    try:
        embedding = transformer.encode([canonical_event])[0]
    except Exception as e:
        print(f"[warning] Failed to encode event type '{event_type}': {e}")
        return None
    
    # Find nearest centroid
    min_distance = float('inf')
    nearest_cluster = None
    
    for centroid_data in centroids:
        centroid_embedding = np.array(centroid_data['centroid'])
        distance = np.linalg.norm(embedding - centroid_embedding)
        
        if distance < min_distance:
            min_distance = distance
            nearest_cluster = centroid_data['cluster_id']
    
    return nearest_cluster

def add_cluster_ids_to_file(file_path: Path, centroids, transformer):
    """Add cluster_id to each fact in a samples_all_facts.json file."""
    print(f"[processing] {file_path.name}")
    
    # Load the file
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    total_facts = 0
    facts_with_clusters = 0
    
    # Process each sample
    for sample_key, sample_data in data.items():
        if 'all_facts' in sample_data:
            facts = sample_data['all_facts']
            
            for fact in facts:
                total_facts += 1
                event_type = fact.get('event_type', '')
                
                if event_type:
                    cluster_id = find_nearest_cluster(event_type, centroids, transformer)
                    if cluster_id is not None:
                        fact['cluster_id'] = cluster_id
                        facts_with_clusters += 1
                    else:
                        fact['cluster_id'] = None
                else:
                    fact['cluster_id'] = None
    
    # Save the modified file
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"  ✅ Added cluster_id to {facts_with_clusters}/{total_facts} facts")
    return facts_with_clusters, total_facts

def main():
    """Main function to process all samples_all_facts.json files."""
    print("🔍 Adding cluster IDs to all facts...")
    
    # Load transformer and centroids
    transformer = get_cached_transformer()
    centroids = load_cluster_centroids()
    
    # Find all samples_all_facts.json files
    results_dir = Path("../Results/heterognn5")
    files = list(results_dir.glob("*/samples_all_facts.json"))
    
    if not files:
        print("❌ No samples_all_facts.json files found!")
        return
    
    print(f"Found {len(files)} files to process...")
    
    total_facts_processed = 0
    total_facts_with_clusters = 0
    
    # Process each file
    for file_path in files:
        try:
            facts_with_clusters, total_facts = add_cluster_ids_to_file(file_path, centroids, transformer)
            total_facts_processed += total_facts
            total_facts_with_clusters += facts_with_clusters
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
    
    print(f"\n{'='*50}")
    print(f"CLUSTERING SUMMARY")
    print(f"{'='*50}")
    print(f"Files processed: {len(files)}")
    print(f"Total facts processed: {total_facts_processed}")
    print(f"Facts with cluster_id: {total_facts_with_clusters}")
    print(f"Facts without cluster_id: {total_facts_processed - total_facts_with_clusters}")
    print(f"Success rate: {total_facts_with_clusters/total_facts_processed*100:.1f}%")
    print(f"\n✅ Clustering completed successfully!")

if __name__ == "__main__":
    main()
