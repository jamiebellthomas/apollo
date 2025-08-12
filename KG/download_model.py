#!/usr/bin/env python3
"""
Script to download the sentence transformer model locally
"""
import os
import warnings
import logging
from sentence_transformers import SentenceTransformer

# Suppress all warnings
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

def download_model():
    """Download the sentence transformer model locally"""
    print("Downloading sentence transformer model locally...")
    
    # Create a local cache directory
    cache_dir = os.path.join(os.path.dirname(__file__), "model_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        # Download the model to local cache
        model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            cache_folder=cache_dir
        )
        
        # Test the model
        test_text = ["This is a test sentence."]
        embeddings = model.encode(test_text)
        print(f"✅ Model downloaded successfully!")
        print(f"✅ Test embedding shape: {embeddings.shape}")
        print(f"✅ Model cache location: {cache_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False

if __name__ == "__main__":
    download_model() 