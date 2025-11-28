"""
Legal Case Retrieval - Source Package

This package contains modules for:
- preprocess: Text cleaning and preprocessing
- bm25: Sparse retrieval using BM25
- dense_embeddings_helpers: BERT-based dense embeddings
- dense_rankings_helpers: FAISS-based similarity search
- evaluate_rankings: Evaluation metrics (MRR, P@k, R@k, F1@k)
"""

from . import config
