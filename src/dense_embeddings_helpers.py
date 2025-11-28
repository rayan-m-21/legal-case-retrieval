import os
import pathlib
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import time
from sklearn.decomposition import PCA
import joblib

from .config import (
    MODELS as models,
    MAX_CHUNK_SIZE as max_chunk_size,
    POOLING_STRATEGIES,
    get_clean_files_dir,
    EMBEDDINGS_DIR,
    ensure_output_dirs,
)

# CACHE FOR MODELS
_tokenizers = {}
_models     = {}

def _get_model_and_tokenizer(name: str):
    """Lazy-load the HF model & tokenizer."""
    if name not in _tokenizers:
        tokenizer = AutoTokenizer.from_pretrained(name)
        model     = AutoModel.from_pretrained(name)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        _tokenizers[name] = tokenizer
        _models[name] = model
    return _tokenizers[name], _models[name]

def embed_text(text, model='BERT-base', stride=256, pool_strategy='cls_only'):
    """
    Returns the embedding of the text
    """
    tokenizer, bert = _get_model_and_tokenizer(models[model])

    # chunk size = max - 2 (one for [CLS] and [SEP])
    chunk_size = max_chunk_size[model] - 2

    # tokenize the text once
    tokens = tokenizer.encode(
        text,
        add_special_tokens=False,      # we add [CLS]/[SEP] per chunk
        truncation=False,
        return_overflowing_tokens=False,
        max_length=100_000
    )

    # initialise chunks
    chunks = []
    for i in range(0, len(tokens), stride):
        chunk = tokens[i: i + chunk_size]

        input_ids = [tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id]
        chunks.append(torch.tensor(input_ids, dtype=torch.long))

    if not chunks:
        # happens when whole text is too small
        input_ids = [tokenizer.cls_token_id] + tokens[:chunk_size] + [tokenizer.sep_token_id]
        chunks.append(torch.tensor(input_ids, dtype=torch.long))

    # pad and batch the chunks
    batch = torch.nn.utils.rnn.pad_sequence(chunks, batch_first=True, padding_value=tokenizer.pad_token_id)

    attn_mask = (batch != tokenizer.pad_token_id).long()


    if torch.cuda.is_available():
        batch = batch.cuda()
        attn_mask = attn_mask.cuda()

    # run through bert model
    with torch.no_grad():
        torch.cuda.empty_cache()
        outputs = bert(batch, attention_mask=attn_mask)
        hidden = outputs.last_hidden_state

    # # pool based on the specific strategy
    pad_id = tokenizer.pad_token_id
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    if pool_strategy == "cls_only":
        embeddings = hidden[:, 0, :]
    elif pool_strategy == "tokens":
        # mask out PAD, CLS, SEP
        mask = (batch != pad_id) & (batch != cls_id) & (batch != sep_id)   # (B, L)
        masked = hidden * mask.unsqueeze(-1).float()            # (B, L, 768)
        sums   = masked.sum(dim=1)                              # (B, 768)
        counts = mask.sum(dim=1, keepdim=True).clamp(min=1.0)   # (B, 1)
        embeddings = sums / counts
    else:
        raise ValueError(f"Unknown pool strategy={pool_strategy}")

    overall_emb = embeddings.mean(dim=0)
    return overall_emb.cpu().numpy()              # FAISS expects numpy

def embed_dataset(
    model, pool_strategy, stride, dataset_path=None, embeddings_save_dir=None, train=True, debug=False
):
    """
    Generate embeddings for all files in a dataset.
    
    Args:
        model: Model name key (e.g., 'BERT-base', 'LegalBERT')
        pool_strategy: Pooling strategy ('cls_only' or 'tokens')
        stride: Stride for chunking long documents
        dataset_path: Optional custom path to dataset folder
        embeddings_save_dir: Optional custom path for saving embeddings
        train: If True, use training data; otherwise use test data
        debug: If True, print progress information
    
    Returns:
        dict: {filename: reduced_vector}
    
    Saves .npz (keys + embeddings) and a separate _pca.pkl when PCA is applied.
    """
    ensure_output_dirs()
    
    if dataset_path:
        folder = pathlib.Path(dataset_path)
    else:
        folder = get_clean_files_dir(train)

    if debug:
        print(f"Processing: {folder}")

    if not embeddings_save_dir:
        embeddings_save_dir = EMBEDDINGS_DIR

    embeddings = {}                     # raw 768-dim vectors

    files = sorted(os.listdir(folder))
    n = len(files)

    if debug:
        print(f"\n=== TRAIN {model} | {pool_strategy} | stride={stride}")
        start = time.time()
        target=0

    for i, f in enumerate(files):
        if debug and i == target:
            print(f"{(i/n)*100:5.1f}% – {(time.time()-start)/60:.2f} min")
            target += n // 10

        path = pathlib.Path(folder) / f
        txt = path.read_text(encoding="utf-8", errors="ignore")
        vec = embed_text(txt, model=model, pool_strategy=pool_strategy, stride=stride)
        embeddings[f] = vec

    # ---- PCA (global, fitted once) --------------------------------
    corpus = np.stack(list(embeddings.values()))          # (N, 768)

    for dim in [768, 348, 128]:
        train_or_test_str = 'train' if train else 'test'
        embeddings_save_dir = pathlib.Path(embeddings_save_dir)
        embeddings_save_path = embeddings_save_dir / f'{train_or_test_str}_{model}_{pool_strategy}_stride={stride}_dim={dim}.npz'

        does_pca = dim < 768

        if does_pca:
            pca = PCA(n_components=dim, random_state=42)
            pca.fit(corpus)
            reduced = pca.transform(corpus).astype(np.float32)
        else:
            pca = None
            reduced = corpus.astype(np.float32)

        reduced_dict = {f: reduced[i] for i, f in enumerate(embeddings)}

        # Save embeddings
        embeddings_save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(embeddings_save_path),
            keys=np.array(list(reduced_dict.keys()), dtype=object),
            embeddings=np.stack(list(reduced_dict.values()))
        )

        # Save projection if necessary
        if does_pca:
            pca_path = str(embeddings_save_path).replace('.npz', '_pca.pkl')
            joblib.dump(pca, pca_path)

        if debug:
            print(f"Saved -> {embeddings_save_path}")
            if does_pca:
                print(f"   PCA -> {pca_path} (var={pca.explained_variance_ratio_.sum():.4f})")
    
    return reduced_dict
