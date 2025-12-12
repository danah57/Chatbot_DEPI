import argparse
import logging
import os
import time
from typing import Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def setup_logging(level: int = logging.INFO):
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        level=level,
    )


def load_table(path: str) -> pd.DataFrame:
    """Load a table from Excel or CSV into a DataFrame."""
    ext = os.path.splitext(path)[1].lower()
    if ext in {'.xlsx', '.xls'}:
        return pd.read_excel(path)
    elif ext == '.csv':
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def build_descriptions(df: pd.DataFrame) -> pd.Series:
    """Create a text description column used for embeddings."""
    def row_to_text(row):
        parts = [
            str(row.get('program', '')).strip(),
            'at',
            str(row.get('university_name', '')).strip(),
        ]
        # include other relevant fields if present
        for field in ('duration', 'fees', 'ielts', 'toefl'):
            val = row.get(field)
            if pd.notna(val) and str(val).strip() != 'nan':
                parts.append(f"{field} {val}")
        return ' '.join([p for p in parts if p])

    return df.apply(row_to_text, axis=1)


def create_embeddings(
    data_path: str,
    output_dir: str = './data/processed',
    model_name: str = 'all-MiniLM-L6-v2',
    device: str = 'cpu',
    batch_size: int = 64,
    create_faiss: bool = False,
    normalize_embeddings: bool = True,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Create and save embeddings and metadata for retrieval.

    Produces the following files in `output_dir`:
    - `embeddings.npy`: float32 array shape (N, D)
    - `metadata.csv`: original table with added `description` column
    - `embeddings.pkl` (optional): pickle of embeddings (for compatibility)
    - `index.faiss` (optional): faiss index file if `create_faiss=True` and faiss available

    Returns (embeddings, metadata_df)
    """
    os.makedirs(output_dir, exist_ok=True)
    logging.info("Loading data from: %s", data_path)
    df = load_table(data_path)
    logging.info("Loaded %d rows", len(df))

    # Build descriptions
    logging.info("Building textual descriptions for embedding...")
    df = df.copy()
    df['description'] = build_descriptions(df)
    texts = df['description'].fillna('').tolist()

    # Initialize model
    logging.info("Loading SentenceTransformer model '%s' on device '%s'", model_name, device)
    model = SentenceTransformer(model_name, device=device)

    logging.info("Creating embeddings (this may take some minutes)...")
    start = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    elapsed = time.time() - start
    logging.info("Embeddings created in %.1fs — shape: %s", elapsed, embeddings.shape)

    # Optionally normalize embeddings for cosine similarity
    if normalize_embeddings:
        logging.info("L2-normalizing embeddings for cosine similarity retrieval")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

    # Save artifacts
    emb_path = os.path.join(output_dir, 'embeddings.npy')
    meta_path = os.path.join(output_dir, 'metadata.csv')
    pickle_path = os.path.join(output_dir, 'embeddings.pkl')

    logging.info("Saving embeddings to: %s", emb_path)
    np.save(emb_path, embeddings.astype(np.float32))

    logging.info("Saving metadata to: %s", meta_path)
    df.to_csv(meta_path, index=False)

    # Also keep a pickle for compatibility (small overhead)
    try:
        import pickle

        with open(pickle_path, 'wb') as f:
            pickle.dump(embeddings, f)
        logging.debug("Saved pickle backup to: %s", pickle_path)
    except Exception:
        logging.debug("Could not save pickle backup for embeddings")

    # Optional: build Faiss index if requested
    if create_faiss:
        try:
            import faiss

            dim = embeddings.shape[1]
            # Use inner product on normalized vectors === cosine similarity
            logging.info("Building Faiss IndexFlatIP with dimension %d", dim)
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings.astype(np.float32))
            index_path = os.path.join(output_dir, 'index.faiss')
            faiss.write_index(index, index_path)
            logging.info("Faiss index saved to: %s", index_path)
        except Exception as e:  # pragma: no cover - optional dependency may not be present
            logging.warning("Faiss index not created: %s", e)

    logging.info("All artifacts written to: %s", output_dir)
    return embeddings, df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Create embeddings and metadata for retrieval')
    p.add_argument('data_path', nargs='?', default='./data/all_programs_cleaned.xlsx',
                   help=r"Path to input table (.xlsx or .csv). Default: ./data/all_programs_cleaned.xlsx")
    p.add_argument('--output-dir', default='./data/processed', help='Directory to write artifacts')
    p.add_argument('--model', default='all-MiniLM-L6-v2', help='SentenceTransformer model name')
    p.add_argument('--device', default='cpu', help="Device for model (e.g., 'cpu' or 'cuda')")
    p.add_argument('--batch-size', type=int, default=64, help='Encode batch size')
    p.add_argument('--no-normalize', dest='normalize', action='store_false', help='Do not normalize embeddings')
    p.add_argument('--faiss', action='store_true', help='Create a Faiss index (if faiss installed)')
    p.add_argument('--debug', action='store_true', help='Enable debug logging')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    setup_logging(logging.DEBUG if args.debug else logging.INFO)

    # If user asked for 'auto' device, try to select GPU if available
    device = args.device
    if device == 'auto':
        try:
            import torch

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logging.info("Auto device selected: %s", device)
        except Exception:
            device = 'cpu'

    # If provided path does not exist, try sensible fallbacks inside the workspace `./data` folder
    if not os.path.exists(args.data_path):
        logging.warning("Input file not found: %s", args.data_path)
        # try common filenames in ./data
        candidates = [
            './data/all_programs_cleaned.xlsx',
            './data/all_programs.xlsx',
            './data/college_data.csv',
        ]
        found = None
        for c in candidates:
            if os.path.exists(c):
                found = c
                break
        if found:
            logging.info("Falling back to detected file: %s", found)
            args.data_path = found
        else:
            # list files in ./data to help user
            try:
                available = os.listdir('./data')
            except Exception:
                available = []
            logging.error(
                "No input file found. Checked: %s. Files in ./data: %s",
                args.data_path,
                available,
            )
            raise FileNotFoundError(f"Input file not found: {args.data_path}")

    create_embeddings(
        data_path=args.data_path,
        output_dir=args.output_dir,
        model_name=args.model,
        device=device,
        batch_size=args.batch_size,
        create_faiss=args.faiss,
        normalize_embeddings=args.normalize,
    )