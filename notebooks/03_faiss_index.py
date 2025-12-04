
import pandas as pd
import numpy as np
import pickle
import faiss
import os

def build_faiss_index(embeddings_path: str, output_dir: str = './data/processed'):
    """
    Build FAISS index from embeddings
    
    Input: ./data/processed/embeddings.pkl
    Output: ./data/processed/faiss_index.bin
    """
    
    print("\n" + "="*80)
    print(" STEP 3: BUILD FAISS INDEX")
    print("="*80 + "\n")
    
    # Load embeddings
    print(f" Loading embeddings from: {embeddings_path}")
    with open(embeddings_path, 'rb') as f:
        embeddings = pickle.load(f)
    print(f" Loaded: shape {embeddings.shape}")
    
    # Create FAISS index
    print("\n Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    
    # Convert to float32
    embeddings_f32 = embeddings.astype('float32')
    index.add(embeddings_f32)
    
    print(f" Index created with {index.ntotal} vectors")
    
    # Save index
    index_file = f"{output_dir}/faiss_index.bin"
    print(f"\n Saving index to: {index_file}")
    os.makedirs(output_dir, exist_ok=True)
    
    faiss.write_index(index, index_file)
    print(" Saved successfully!")
    
    return index

if __name__ == "__main__":
    build_faiss_index('./data/processed/embeddings.pkl')
