#!/usr/bin/env python3
"""
Attention-based Explainability Analysis for GNN Models

This script analyzes attention weights from a trained GNN model to identify
the top 25 most important facts for each prediction and creates a hierarchical JSON structure.
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

print("Script is starting...")

print("Imports completed")

# Add the KG directory to the path to import SubGraph
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'KG'))
from SubGraph import SubGraph

print("SubGraph imported")

def run_explainability_analysis(model_dir):
    """Run the attention-based explainability analysis"""
    print(f"Running explainability analysis for model: {model_dir}")
    
    # Import here to avoid circular imports
    sys.path.append('..')
    from replicate_eval import replicate_evaluation
    
    print("replicate_eval imported")
    
    # Run the evaluation to get attention weights
    print("Running model evaluation to capture attention weights...")
    results = replicate_evaluation(model_dir)
    
    print("Evaluation completed")
    
    # Get y_pred for predicted labels
    y_pred = results.get("y_pred", [])
    if hasattr(y_pred, "tolist"):
        y_pred = y_pred.tolist()
    print(f"Got y_pred with {sum(y_pred)} positive predictions")
    
    # Generate hierarchical diagnostics
    attention_weights = results.get('attention_weights', [])
    generate_hierarchical_diagnostics(attention_weights, model_dir, y_pred)

def aggregate_fact_attention_for_sample(sample_layers, target_edge=('fact','mentions','company')):
    """
    sample_layers: list over layers
      each layer: dict keyed by edge-type tuple -> {edge_index: [2,E], alpha: [E] or [E,H], ...}
    returns: dict fact_local_index -> score
    """
    from collections import defaultdict
    scores = defaultdict(list)

    for layer in sample_layers:
        if target_edge not in layer:
            continue
        payload = layer[target_edge]
        eidx = payload["edge_index"]   # [2,E]
        alpha = payload["alpha"]       # [E] or [E,H]

        # per-edge score
        if alpha.ndim == 2:
            edge_scores = alpha.mean(axis=1)
        else:
            edge_scores = alpha

        src_nodes = eidx[0]            # type-local fact ids
        for i, src in enumerate(src_nodes):
            scores[int(src)].append(float(edge_scores[i]))

    # reduce (mean over occurrences across layers)
    return {fi: float(np.mean(vals)) for fi, vals in scores.items()}


def generate_hierarchical_diagnostics(attention_weights, model_dir, y_pred):
    """
    Generate hierarchical diagnostics from attention weights.
    
    Args:
        attention_weights: List of attention data from replicate_evaluation (per-sample list-of-layers)
        results: Full evaluation results containing predictions
        model_dir: Path to model directory for output
    """
    print("Generating hierarchical diagnostics...")
    
    # Load raw_sg data from cached dataset
    print("Loading raw_sg data from cached dataset...")
    cache_file = Path("../KG/dataset_cache/testing_cached_dataset_nf35_limall.pkl")
    
    if not cache_file.exists():
        print(f"Error: Cache file not found at {cache_file}")
        return
    
    with open(cache_file, 'rb') as f:
        cache_data = pickle.load(f)
    
    # Fix: Load the correct structure from cache
    if isinstance(cache_data, dict) and 'raw_sg' in cache_data:
        raw_sg_data = cache_data['raw_sg']
    else:
        raw_sg_data = cache_data  # fallback for old format
    
    print(f"Loaded {len(raw_sg_data)} raw_sg samples")
    
    # Initialize hierarchical structure
    samples_data = {}
    
    print(f"Processing {len(attention_weights)} attention samples...")
    
    # attention_weights is list aligned with samples; each item is list-of-layers
    for sample_idx, sample_layers in enumerate(attention_weights):
        print(f"Processing sample {sample_idx}...")
        
        # Get raw_sg data for this sample
        if sample_idx < len(raw_sg_data):
            raw_sg = raw_sg_data[sample_idx]
            
            # Fix: Use correct attribute names
            if hasattr(raw_sg, 'fact_list'):
                facts = raw_sg.fact_list
            else:
                facts = raw_sg.facts  # fallback
            
            # Get sample metadata with correct field names
            primary_ticker = getattr(raw_sg, 'primary_ticker', 'Unknown')
            sample_date = getattr(raw_sg, 'reported_date', getattr(raw_sg, 'date', 'Unknown'))
            sample_label = getattr(raw_sg, 'graph_label', getattr(raw_sg, 'label', None))
            
            # Get predicted label from evaluation results
            
            print(f"Sample {sample_idx} - Primary Ticker: {primary_ticker}, Date: {sample_date}")
            print(f"Found {len(facts)} facts for sample {sample_idx}")
            
            # Aggregate attention scores for fact->company edges only
            fact_scores = aggregate_fact_attention_for_sample(
                sample_layers,
                target_edge=('fact','mentions','company')
            )
            
            print(f"[sample {sample_idx}] layers={len(sample_layers)}, "
                  f"scored_facts={len(fact_scores)}, total_facts={len(facts)}")
            
            # Get predicted label for this sample
            predicted_label = None
            if sample_idx < len(y_pred):
                predicted_label = int(y_pred[sample_idx])
            
            # Ensure exactly 25 facts per sample
            target_facts = min(25, len(facts))
            ranked = sorted(fact_scores.items(), key=lambda x: x[1], reverse=True)

            top = [(fi, sc) for fi, sc in ranked[:target_facts]]
            if len(top) < target_facts:
                # pad with unscored facts
                used = {fi for fi, _ in top}
                for fi in range(len(facts)):
                    if fi not in used:
                        top.append((fi, 0.0))
                        if len(top) == target_facts:
                            break

            # build JSON entry
            sample_facts = []
            for fi, sc in top:
                if fi < len(facts):
                    fact = facts[fi]
                    sample_facts.append({
                        "fact_index": fi,
                        "fact_id": fact.get("fact_id", f"fact_{fi}"),
                        "attention_score": float(sc),
                        "date": fact.get("date", ""),
                        "tickers": fact.get("tickers", []),
                        "event_type": fact.get("event_type", ""),
                        "sentiment": fact.get("sentiment", None),
                        "raw_text": fact.get("raw_text", "")
                    })
            
            samples_data[sample_idx] = {
                "sample_metadata": {
                    "sample_idx": sample_idx,
                    "primary_ticker": primary_ticker,
                    "sample_date": sample_date,
                    "sample_label": sample_label,
                    "predicted_label": predicted_label,
                    "total_facts": len(facts)
                },
                "all_facts": sample_facts
            }
        else:
            print(f"Warning: No raw_sg data found for sample {sample_idx}")
    
    # Save hierarchical JSON structure
    output_file = Path(model_dir) / "samples_all_facts.json"
    
    # Delete existing file if it exists
    if output_file.exists():
        output_file.unlink()
    
    # Save as JSON
    with open(output_file, 'w') as f:
        json.dump(samples_data, f, indent=2, default=str)
    
    print(f"Hierarchical samples data saved to: {output_file}")
    print(f"Total samples processed: {len(samples_data)}")
    print("✅ Hierarchical diagnostics generated successfully!")

def main():
    print("Main function called")
    parser = argparse.ArgumentParser(description='Run attention-based explainability analysis')
    parser.add_argument('model_dir', help='Path to the model directory containing results')
    
    args = parser.parse_args()
    
    print(f"Model directory: {args.model_dir}")
    
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory {args.model_dir} does not exist")
        return
    
    run_explainability_analysis(args.model_dir)

if __name__ == "__main__":
    print("Script main block reached")
    main()
    print("Script completed")
