import os
import pathlib
from rank_bm25 import BM25Okapi
import json
import time

from .config import (
    get_labels_file, get_clean_files_dir,
    RANKINGS_DIR, ensure_output_dirs,
)

def bm25_rank(query_file: str, corpus_folder: str) -> list:
    """
    For a given query file, rank all other files in the corpus_folder by BM25 relevance score.
    Returns a list of (filename, score) tuples, sorted descending by score.
    """
    # 1. Load query text
    query_path = pathlib.Path(query_file)
    query_text = query_path.read_text(encoding="utf-8", errors="ignore")
    tokenized_query = query_text.split()  # Simple whitespace tokenization (customize if needed)

    # 2. Load corpus (all files except query)
    corpus_texts = []
    corpus_filenames = []
    query_name = query_path.name
    for f in os.listdir(corpus_folder):
        if f == query_name:
            continue  # Exclude self

        path = pathlib.Path(corpus_folder) / f
        text = path.read_text(encoding="utf-8", errors="ignore")
        tokenized_text = text.split()
        corpus_texts.append(tokenized_text)
        corpus_filenames.append(f)

    # 3. Build BM25 index
    bm25 = BM25Okapi(corpus_texts)

    # 4. Get scores
    scores = bm25.get_scores(tokenized_query)

    # 5. Rank and return (descending score)
    ranked = sorted(zip(corpus_filenames, scores), key=lambda x: x[1], reverse=True)
    return ranked

def output_bm25_rankings(train: bool = True, debug: bool = False):
    """
    Generate BM25 rankings for all query files and save to output.
    
    Args:
        train: If True, use training data; otherwise use test data.
        debug: If True, print progress information.
    """
    ensure_output_dirs()
    
    label_file = get_labels_file(train)
    corpus_folder = get_clean_files_dir(train)

    with open(label_file, 'r') as f:
        labels = json.load(f)

    outputs = {}
    filenames = sorted(os.listdir(corpus_folder))
    n = len(filenames)
    start = time.time()
    target = 0

    for i, (query_file, noted_cases) in enumerate(labels.items()):
        if debug:
            if i == target:
                print(f"{(i/n)*100:5.1f}% – {(time.time()-start)/60:.2f} min")
                target += max(1, n // 10)

        query_file_path = corpus_folder / query_file
        file_score_pairs = bm25_rank(str(query_file_path), str(corpus_folder))
        files_retrieved = [x[0] for x in file_score_pairs]

        outputs[query_file] = files_retrieved

    # Save rankings
    split_name = 'train' if train else 'test'
    output_file = RANKINGS_DIR / f'bm25_{split_name}.json'
    
    with open(output_file, 'w') as f:
        json.dump(outputs, f, indent=4)
    
    print(f"BM25 rankings saved -> {output_file}")
    return output_file


if __name__ == "__main__":
    output_bm25_rankings(train=True, debug=True)