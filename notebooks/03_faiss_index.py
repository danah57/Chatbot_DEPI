
import pandas as pd
import numpy as np
import pickle
import faiss
import os

def build_faiss_index(embeddings_path: str, output_dir: str = './data/processed'):
    # Load embeddings
    with open(embeddings_path, 'rb') as f:
        embeddings = pickle.load(f)
    print(f"Loaded embeddings: shape {embeddings.shape}")

    # Normalize vectors for cosine similarity
    def normalize(vectors):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    embeddings_f32 = embeddings.astype('float32')
    embeddings_norm = normalize(embeddings_f32)

    # Create FAISS index for inner product (cosine similarity with normalized vectors)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_norm)
    print(f"Index created with {index.ntotal} vectors")

    # Save index
    os.makedirs(output_dir, exist_ok=True)
    index_file = os.path.join(output_dir, "faiss_index.bin")
    faiss.write_index(index, index_file)
    print(f"Index saved successfully at {index_file}")

    return index, index_file

if __name__ == "__main__":
    build_faiss_index('./data/processed/embeddings.pkl')
