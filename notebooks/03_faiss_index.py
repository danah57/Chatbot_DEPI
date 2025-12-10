
import pandas as pd
import numpy as np
import pickle
import faiss
import os

def build_faiss_index(embeddings_path: str, output_dir: str = './data/processed'):
    import numpy as np, os, pickle, faiss

    with open(embeddings_path, 'rb') as f:
        embeddings = pickle.load(f)
    print(f"Loaded: shape {embeddings.shape}")

    def normalize(vectors):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    print("Building FAISS cosine index (IndexFlatIP + normalized vectors)...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)

    embeddings_f32 = embeddings.astype('float32')
    embeddings_norm = normalize(embeddings_f32)

    index.add(embeddings_norm)
    print(f"Index created with {index.ntotal} vectors")

    os.makedirs(output_dir, exist_ok=True)
    index_file = f"{output_dir}/faiss_index.bin"
    faiss.write_index(index, index_file)
    print("Saved successfully!")
    return index

if _name_ == "_main_":
    build_faiss_index('./data/processed/embeddings.pkl')
