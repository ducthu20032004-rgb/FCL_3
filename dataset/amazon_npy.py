# tools/make_amazon5_npy.py
# Converts HuggingFace "amazon_reviews_multi" (English) to per-class 384-D embeddings.
# Requires: pip install datasets sentence-transformers torch torchvision

import os
from pathlib import Path
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

OUT_TRAIN = Path("datalist/amazon5-train")  # temp shard dir
OUT_TEST  = Path("datalist/amazon5-test")
NPY_TRAIN = Path("dataset/amazon5-classes")
NPY_TEST  = Path("dataset/amazon5-test-classes")
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384-d


"""
Necessary functions
"""

def ensure_dirs():
    for p in [OUT_TRAIN, OUT_TEST, NPY_TRAIN, NPY_TEST]:
        p.mkdir(parents=True, exist_ok=True)

def collect_texts(split: str):
    ds = load_dataset("amazon_polarity")  # alternative: "amazon_reviews_multi" but is very large
    # For 5-class rating, use "amazon_polarity"? It's 2-class.
    # If you want 5-class, use "amazon_reviews_multi": ds = load_dataset("amazon_reviews_multi", "en")
    # and then map rating 1..5 -> labels 0..4, texts = ds[split]["review_body"]
    # For demonstration, let's stick with 2-class (negative/positive). If you need 5-class, see note below.
    if split == "train":
        subset = ds["train"]
    else:
        subset = ds["test"]
    texts = [ex["content"] for ex in subset]
    labels = np.array([ex["label"] for ex in subset], dtype=np.int64)  # 0 or 1
    return texts, labels

def encode_texts(texts, model):
    # returns np.array of shape (N, 384)
    vecs = model.encode(texts, batch_size=256, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=False)
    return vecs.astype("float32")

def dump_per_class(vecs: np.ndarray, labels: np.ndarray, out_dir: Path, num_classes: int):
    buckets = [vecs[labels == k] for k in range(num_classes)]
    out_dir.mkdir(parents=True, exist_ok=True)
    for k in range(num_classes):
        np.save(out_dir / f"{k}.npy", buckets[k])


"""
Main structure
1. Find a way to load the dataset (2-class or 5-class)
2. Encode all texts to embeddings
3. Save per-class .npy files
"""


ensure_dirs()
print("Loading SentenceTransformer:", EMB_MODEL)
model = SentenceTransformer(EMB_MODEL)

# Choose dataset variant:
USE_FIVE_CLASS = False  # set True if you want amazon_reviews_multi (5 classes)

if USE_FIVE_CLASS:
    from datasets import load_dataset
    ds_tr = load_dataset("amazon_reviews_multi", "en", split="train")
    ds_te = load_dataset("amazon_reviews_multi", "en", split="test")
    # labels are 1..5 → shift to 0..4
    tr_texts = ds_tr["review_body"]
    tr_labels = np.array(ds_tr["stars"], dtype=np.int64) - 1
    te_texts = ds_te["review_body"]
    te_labels = np.array(ds_te["stars"], dtype=np.int64) - 1
    num_classes = 5
else:
    # 2-class polarity (0=negative,1=positive)
    tr_texts, tr_labels = collect_texts("train")
    te_texts, te_labels = collect_texts("test")
    num_classes = 2

print("Encoding train texts...")
tr_vecs = encode_texts(tr_texts, model)
print("Encoding test texts...")
te_vecs = encode_texts(te_texts, model)

# Save per class
dump_per_class(tr_vecs, tr_labels, NPY_TRAIN, num_classes)
dump_per_class(te_vecs,  te_labels, NPY_TEST,  num_classes)
print("Saved per-class .npy to", NPY_TRAIN, "and", NPY_TEST)
