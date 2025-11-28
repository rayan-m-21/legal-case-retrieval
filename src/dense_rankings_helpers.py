import json
import numpy as np
import faiss
from pathlib import Path

from .config import (
    get_labels_file,
    RANKINGS_DIR,
    EMBEDDINGS_DIR,
    ensure_output_dirs,
)

def load_npz_embeddings(path: str):
    data = np.load(path, allow_pickle=True)
    filenames = data['keys']
    vectors   = data['embeddings']  # (N, 768), float32
    return dict(zip(filenames, vectors)), vectors, filenames

def build_faiss_index(vectors: np.ndarray):
    dim = vectors.shape[1]  # 768
    vectors = vectors.copy().astype('float32')
    faiss.normalize_L2(vectors)  # REQUIRED for cosine similarity
    index = faiss.IndexFlatIP(dim)  # Inner Product = cosine (after L2 norm)
    index.add(vectors)
    return index

def search_faiss_by_id(index: faiss.Index, vector_id: int, k: int = 5):
    # Reconstruct the stored vector
    query_vec = index.reconstruct(vector_id)                # (d,)
    query = query_vec.reshape(1, -1)
    faiss.normalize_L2(query)

    # Search the *whole* index (or at least k+1)
    search_k = min(k + 1, index.ntotal)
    D, I = index.search(query, search_k)                    # (1, search_k)

    D, I = D[0], I[0]

    # Drop the query itself
    query_id_ind = np.where(I == vector_id)[0][0]
    I = np.delete(I, query_id_ind)
    D = np.delete(D, query_id_ind)

    return D, I

def output_embedding_rankings(embedding_file_name, output_file_name=None, train: bool = True):
    """
    Generate rankings based on dense embeddings using FAISS cosine similarity.
    
    Args:
        embedding_file_name: Path to the .npz embeddings file (can be str or Path)
        output_file_name: Optional output path. If None, auto-generates based on embedding name.
        train: If True, use training labels; otherwise use test labels.
    
    Returns:
        Path to the saved rankings file.
    """
    ensure_output_dirs()
    
    label_file = get_labels_file(train)
    embedding_path = Path(embedding_file_name)

    with open(label_file, 'r') as f:
        labels = json.load(f)

    outputs = {}

    # Get embeddings
    emb_dict, vectors, filenames = load_npz_embeddings(str(embedding_path))

    filename_to_id = {name: i for i, name in enumerate(filenames)}

    # Build the FAISS index for cosine similarity
    index = build_faiss_index(vectors)

    # Loop through each query case
    for query_file, noted_cases in labels.items():
        q_id = filename_to_id[query_file]

        # Retrieve closest in order
        distances, indices = search_faiss_by_id(index, q_id, k=index.ntotal)
        files_retrieved = [str(filenames[ind]) for ind in indices]

        outputs[query_file] = files_retrieved

    # Determine output path
    if output_file_name is None:
        output_file_name = RANKINGS_DIR / embedding_path.name.replace('.npz', '.json')
    else:
        output_file_name = Path(output_file_name)
    
    output_file_name.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file_name, 'w') as f:
        json.dump(outputs, f, indent=4)
    
    print(f"Rankings saved -> {output_file_name}")
    return output_file_name


if __name__ == '__main__':
    # Example usage
    example_embedding = EMBEDDINGS_DIR / 'train_BERT-base_cls_only_stride=256_dim=128.npz'
    if example_embedding.exists():
        output_embedding_rankings(example_embedding, train=True)
