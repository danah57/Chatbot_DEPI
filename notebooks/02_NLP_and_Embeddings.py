import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import time
import os

def create_embeddings(data_path: str, output_dir: str = './data/processed'):
    """
    Create embeddings for all programs
    
    Input: ./data/all_programs_cleaned.xlsx
    Output: ./data/processed/embeddings.pkl
    """
    
    print("\n" + "="*80)
    print(" STEP 2: CREATE EMBEDDINGS (Hugging Face)")
    print("="*80 + "\n")
    
    # Load data
    print(f"Loading data from: {data_path}")
    data = pd.read_excel(data_path)
    print(f"Loaded: {len(data)} records")
    
    # Initialize model
    print("\n Loading SentenceTransformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    print(" Model loaded!")
    
    # Create descriptions
    print("\n Creating text descriptions...")
    descriptions = []
    for idx, row in data.iterrows():
        desc = (
            f"{row['program']} "
            f"at {row['university_name']} "
            f"duration {row['duration']} "
            f"fees {row['fees']} "
            f"ielts {row['ielts']} "
            f"toefl {row['toefl']}"
        )
        descriptions.append(desc)
    print(f" Created {len(descriptions)} descriptions")
    
    # Create embeddings
    print("\n Creating embeddings (this may take 1-2 minutes)...")
    start_time = time.time()
    
    embeddings = model.encode(
        descriptions,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    elapsed = time.time() - start_time
    print(f" Embeddings created in {elapsed:.1f}s")
    print(f"   Shape: {embeddings.shape}")
    
    # Save embeddings
    output_file = f"{output_dir}/embeddings.pkl"
    print(f"\n Saving embeddings to: {output_file}")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, 'wb') as f:
        pickle.dump(embeddings, f)
    print(" Saved successfully!")
    
    return embeddings

if __name__ == "__main__":
    create_embeddings('./data/all_programs_cleaned.xlsx')