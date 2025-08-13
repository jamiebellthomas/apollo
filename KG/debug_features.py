#!/usr/bin/env python3
"""
Debug script to test feature generation and see what's happening with node features.
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

from SubGraphDataLoader import SubGraphDataLoader
from SubGraph import SubGraph
from run import get_cached_transformer

def debug_features():
    """Debug the feature generation process."""
    
    print("=== DEBUGGING FEATURE GENERATION ===")
    
    # Load a single subgraph
    print("1. Loading subgraph...")
    subgraphs_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), config.SUBGRAPHS_JSONL)
    loader = SubGraphDataLoader(min_facts=35, limit=1, jsonl_path=subgraphs_path)
    
    if len(loader.items) == 0:
        print("❌ No subgraphs found!")
        return
    
    sg = loader.items[0]
    print(f"✅ Loaded subgraph: {sg.primary_ticker} on {sg.reported_date}")
    print(f"   Facts: {len(sg.fact_list)}")
    
    # Test encode method
    print("\n2. Testing encode() method...")
    text_encoder = get_cached_transformer()
    
    encoded = sg.encode(
        text_encoder=text_encoder,
        fuse="concat",
        l2_normalize=True
    )
    
    print(f"   fact_text_embeddings shape: {encoded['fact_text_embeddings'].shape}")
    print(f"   fact_event_embeddings shape: {encoded['fact_event_embeddings'].shape}")
    print(f"   fact_features shape: {encoded['fact_features'].shape if encoded['fact_features'] is not None else 'None'}")
    print(f"   edge sentiment shape: {encoded['edge']['sentiment'].shape}")
    print(f"   edge weighting shape: {encoded['edge']['weighting'].shape}")
    
    # Test get_fact_node_features
    print("\n3. Testing get_fact_node_features()...")
    fact_features = sg.get_fact_node_features(
        text_encoder=text_encoder,
        fuse="concat",
        l2_normalize=True
    )
    print(f"   fact_features shape: {fact_features.shape}")
    print(f"   fact_features sample: {fact_features[:2] if fact_features.size > 0 else 'empty'}")
    
    # Test get_ticker_node_features
    print("\n4. Testing get_ticker_node_features()...")
    ticker_features, ticker_index = sg.get_ticker_node_features()
    print(f"   ticker_features shape: {ticker_features.shape}")
    print(f"   ticker_index: {ticker_index}")
    print(f"   ticker_features sample: {ticker_features[:2] if ticker_features.size > 0 else 'empty'}")
    
    # Test to_numpy_graph
    print("\n5. Testing to_numpy_graph()...")
    np_graph = sg.to_numpy_graph(
        text_encoder=text_encoder,
        fuse="concat",
        l2_normalize=True
    )
    
    print(f"   np_graph keys: {list(np_graph.keys())}")
    print(f"   fact_features shape: {np_graph['fact_features'].shape}")
    print(f"   ticker_features shape: {np_graph['ticker_features'].shape}")
    print(f"   edge_index shape: {np_graph['edge_index'].shape}")
    print(f"   edge_attr shape: {np_graph['edge_attr'].shape}")
    
    # Test to_pyg_data
    print("\n6. Testing to_pyg_data()...")
    pyg_data = sg.to_pyg_data(
        text_encoder=text_encoder,
        fuse="concat",
        l2_normalize=True
    )
    
    print(f"   pyg_data keys: {list(pyg_data.keys())}")
    print(f"   pyg_data metadata: {pyg_data.metadata()}")
    
    # Check node features
    node_types = pyg_data.metadata()[0]
    for node_type in node_types:
        if node_type in pyg_data:
            if hasattr(pyg_data[node_type], 'x'):
                features = pyg_data[node_type].x
                print(f"   {node_type}.x shape: {features.shape}")
                print(f"   {node_type}.x sample: {features[:2] if features.numel() > 0 else 'empty'}")
            else:
                print(f"   {node_type} has no .x attribute")
        else:
            print(f"   {node_type} not in pyg_data")
    
    # Check edges
    edge_types = pyg_data.metadata()[1]
    for edge_type in edge_types:
        if edge_type in pyg_data:
            edge_data = pyg_data[edge_type]
            print(f"   {edge_type} edge_index shape: {edge_data.edge_index.shape}")
            print(f"   {edge_type} edge_attr shape: {edge_data.edge_attr.shape}")
        else:
            print(f"   {edge_type} not in pyg_data")
    
    # Test batching
    print("\n7. Testing batching...")
    from torch_geometric.loader import DataLoader
    
    # Create a small dataset and batch
    graphs = [pyg_data]
    dataloader = DataLoader(graphs, batch_size=1, shuffle=False)
    batch = next(iter(dataloader))
    
    print(f"   Batch keys: {list(batch.keys())}")
    print(f"   Batch metadata: {batch.metadata()}")
    
    # Check what's in the 'x' attribute
    if hasattr(batch, 'x') and batch.x is not None:
        print(f"   ✅ Batch has 'x' features: {batch.x.shape}")
        print(f"   Sample 'x' values: {batch.x[:2] if batch.x.numel() > 0 else 'empty'}")
    else:
        print(f"   ❌ Batch has no 'x' features")
    
    # Check if batch has fact features
    if hasattr(batch, 'fact') and hasattr(batch.fact, 'x'):
        print(f"   ✅ Batch has fact features: {batch.fact.x.shape}")
    else:
        print(f"   ❌ Batch has no fact features")
    
    # Check all attributes
    print(f"   All batch attributes:")
    for attr in dir(batch):
        if not attr.startswith('_'):
            try:
                value = getattr(batch, attr)
                if hasattr(value, 'shape'):
                    print(f"     {attr}: {value.shape}")
                else:
                    print(f"     {attr}: {type(value)}")
            except:
                pass
    
    # Check node stores
    print(f"   Node stores: {batch.node_stores}")
    for i, store in enumerate(batch.node_stores):
        print(f"     Store {i}: {store}")
        if hasattr(store, 'x'):
            print(f"       Store {i} has x: {store.x.shape}")
        else:
            print(f"       Store {i} has no x")
    
    # Check edge stores
    print(f"   Edge stores: {batch.edge_stores}")
    for i, store in enumerate(batch.edge_stores):
        print(f"     Edge store {i}: {store}")
    
    # Try to access features through node stores
    if len(batch.node_stores) > 0:
        first_store = batch.node_stores[0]
        if hasattr(first_store, 'x'):
            print(f"   ✅ Found features in first node store: {first_store.x.shape}")
        else:
            print(f"   ❌ No features in first node store")
    
    print("\n=== DEBUG COMPLETE ===")

if __name__ == "__main__":
    debug_features() 