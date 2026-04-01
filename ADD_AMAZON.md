# Add “Amazon Reviews (5-class)” as a new dataset

We’ll use the **HuggingFace `amazon_reviews_multi`** dataset (5 star ratings) and the small Sentence-BERT encoder `sentence-transformers/all-MiniLM-L6-v2` (384-dim embeddings). We’ll export **per-class `.npy`** feature matrices for train/test, hook them into your **current** and **mine** partitioners, register a lightweight **Text-MLP** model, and provide run commands.

> Resulting file layout:
> 
> 
> ```
> dataset/
>   amazon5-classes/
>     0.npy 1.npy 2.npy 3.npy 4.npy        # train, shape ≈ (N_k, 384)
>   amazon5-test-classes/
>     0.npy 1.npy 2.npy 3.npy 4.npy        # test, shape ≈ (M_k, 384)
> tools/
>   make_amazon5_npy.py                    # preprocessing script
> system/
>   utils/
>     data_utils.py                        # + read_client_data_FCL_amazon5 (current)
>     data_utils_mine.py                   # + read_client_data_FCL_amazon5 (mine)
>   trainmodel/
>     text_models.py                       # TextMLP (simple MLP classifier)
> 
> ```
> 
> **Model input** will be `float32` tensors of shape `[batch, 384]`. You’ll run with `--model TEXTMLP` and `--input_dim 384`.
> 

---

## 1) Preprocess Amazon Reviews → per-class `.npy` feature blocks

Create `dataset/mamazon5_npy.py`:

```python
# dataset/amazon5_npy.py
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
```

> Note: The code above defaults to 2-class (amazon_polarity). If you want 5-class, set USE_FIVE_CLASS = True. The rest of the integration is identical; just set num_classes=5 in the hparams.
> 

Run once:

```bash
pip install datasets sentence-transformers torch torchvision
python dataset/amazon5_npy.py

```

You should now have:

```
dataset/amazon5-classes/{0..4}.npy        # or amazon2-classes/{0,1}.npy if USE_FIVE_CLASS=False
dataset/amazon5-test-classes/{0..4}.npy   # (or amazon2-test-classes)

```

Each `.npy` is `float32` of shape `(N_k, 384)`.

---

## 2) Register the dataset in your partitioners

### 2.1 `system/utils/data_utils.py` (for **current**)

Add a root mapping:

```python
def _dataset_root_and_K(dataset: str):
    ds = dataset.lower()
    # ...
    if ds == "amazon5":      # 5-class star ratings
        return Path("dataset/amazon5-classes"), 5
    if ds == "amazon2":      # polarity (optional)
        return Path("dataset/amazon2-classes"), 2
    # ...

```

Add a reader for the current partition (slice per-client class order, then build a `TensorDataset` from feature arrays):

```python
import numpy as np
import torch
from torch.utils.data import TensorDataset

def read_client_data_FCL_amazon5(client_id: int, task: int, classes_per_task: int, count_labels: bool=False):
    root, K = _dataset_root_and_K("AMAZON5")
    # Load client-specific class order like CIFAR:
    class_order = _load_or_make_class_order(K, num_clients=getattr(args, "num_clients", 10))  # mimic cifar path
    cli_order = class_order[client_id]  # length K, permutation of [0..K-1]

    # Classes for this task = consecutive chunk
    s = task * classes_per_task
    e = min(s + classes_per_task, K)
    assigned = [int(k) for k in cli_order[s:e]]

    # Load features for those classes and build TensorDataset [N, D] features
    feats = []
    labels = []
    for k in assigned:
        npy = np.load((root / f"{k}.npy"))
        if npy.ndim != 2:
            # if someone saved as (N, H, W, C), flatten last 3 dims
            if npy.ndim == 4:
                N, H, W, C = npy.shape
                npy = npy.reshape(N, -1)
        yk = np.full((npy.shape[0],), k, dtype=np.int32)
        feats.append(npy)
        labels.append(yk)
    if len(feats) == 0:
        X = np.zeros((0, 384), dtype=np.float32)  # match your embedding dim
        Y = np.zeros((0,), dtype=np.int64)
    else:
        X = np.concatenate(feats, axis=0).astype("float32")
        Y = np.concatenate(labels).astype("int64")

    X_tensor = torch.from_numpy(X)  # shape [N, D], float32
    Y_tensor = torch.from_numpy(Y)
    ds = TensorDataset(X_tensor, Y_tensor)

    label_info = {"labels": assigned, "assigned_labels": assigned}
    if count_labels:
        uniq, cnt = np.unique(Y, return_counts=True)
        label_counts = {int(k): int(v) for k, v in zip(uniq.tolist(), cnt.tolist())}
        label_info["counts"] = label_counts
        label_info["label_count"] = dict(label_counts)
    return ds, label_info

```

> The helper _load_or_make_class_order is the same you use for CIFAR; reuse your existing function to ensure consistent task slicing.
> 

### 2.2 `system/utils/data_utils_mine.py` (for **mine**)

Add the root mapping if not present:

```python
def _dataset_root_and_K(dataset: str) -> Tuple[Path, int]:
    ds = dataset.lower()
    if ds == "amazon5":
        return Path("dataset/amazon5-classes"), 5
    if ds == "amazon2":
        return Path("dataset/amazon2-classes"), 2
    # ...

```

Add the wrapper (the generic `_assemble` already supports non-image features since it only concatenates arrays and uses `_to_tensordataset` — update that helper to accept 2D `[N, D]` by skipping the HWC→CHW transpose when `arr.ndim == 2`):

```python
def _to_tensordataset_generic(arr: np.ndarray, labels: np.ndarray) -> TensorDataset:
    import torch, numpy as np
    x = arr.astype(np.float32)
    if x.ndim == 4:  # HWC → CHW
        x = np.transpose(x, (0, 3, 1, 2)).astype(np.float32)
    X = torch.tensor(x)
    Y = torch.tensor(labels.astype(np.int64))
    return TensorDataset(X, Y)

def read_client_data_FCL_amazon5(client_id: int, task: int, classes_per_task: int, count_labels: bool=False):
    root, K = _dataset_root_and_K("AMAZON5")
    return _assemble("AMAZON5", root, K, client_id, task, classes_per_task, count_labels)

```

Inside `_assemble(...)`, when you load each class `arr = _load_class_np(root, k)`, just ensure `arr` can be `(N, D)` or `(N, H, W, C)`. The `_load_class_np` today assumes images; extend it:

```python
def _load_class_np(root: Path, cls: int) -> np.ndarray:
    import numpy as np
    p = root / f"{cls}.npy"
    arr = np.load(p, allow_pickle=True)
    if arr.ndim == 2:   # e.g., (N, D) features from text embeddings
        return arr.astype(np.float32)
    if arr.ndim == 4 and arr.shape[-1] == 3:
        return arr.astype(np.uint8)   # image case, handled downstream
    if arr.ndim == 4 and arr.shape[1] == 3:  # CHW as npy
        return np.transpose(arr, (0,2,3,1))
    raise ValueError(f"Unexpected shape for {p}: {arr.shape}")

```

Then in `_assemble`, when concatenating features, call `_to_tensordataset_generic`:

```python
# after stacking X (np.array) and Y (np.array)
ds = _to_tensordataset_generic(X, Y)

```

Now your “mine” partition works with SVHN, Amazon, etc.

---

## 3) Add a lightweight **TextMLP** model

Create `system/trainmodel/text_models.py`:

```python
# system/trainmodel/text_models.py
import torch
import torch.nn as nn

class TextMLP(nn.Module):
    """
    Simple MLP for fixed-size text embeddings (e.g., 384-d).
    Expects input shape [B, D]. Outputs [B, num_classes].
    """
    def __init__(self, input_dim: int, num_classes: int, hidden=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        # x: [B, D]
        return self.net(x)

```

Register it in your model factory (where you create models from `args.model`). In `system/trainmodel/models.py` (or where you switch on `args.model`):

```python
from .text_models import TextMLP

# ...
elif args.model.upper() == "TEXTMLP":
    input_dim = getattr(args, "input_dim", 384)
    model = TextMLP(input_dim=input_dim, num_classes=args.num_classes)

```

> Your training loops already do logits = model(x) and CrossEntropyLoss; this will Just Work with [B, D] float features.
> 

---

## 4) Hparams for Amazon Reviews

Create `hparams/amazon5/FedAvg.json`:

```json
{
  "optimizer": "sgd",
  "device": "cuda",
  "device_id": "0",
  "dataset": "AMAZON5",
  "num_classes": 5,
  "model": "TEXTMLP",
  "input_dim": 384,
  "batch_size": 256,
  "local_learning_rate": 0.03,
  "global_rounds": 40,
  "local_epochs": 5,
  "algorithm": "FedAvg",
  "join_ratio": 1.0,
  "random_join_ratio": false,
  "num_clients": 20,
  "eval_gap": 2,
  "out_folder": "out",
  "client_drop_rate": 0.0,
  "time_threthold": 10000
}

```

For other algorithms, copy and change `"algorithm"` accordingly.

---

## 5) Run it

**Current** (codebase split):

```bash
python system/main.py --cfp ./hparams/amazon5/FlexCNN.json \
  --partition_options current --cpt 1 --nt 5 \
  --log True --offlog True

```

**Mine** (task-level permutation):

```bash
python system/main.py --cfp ./hparams/amazon5/FedAvg.json \
  --partition_options mine --cpt 1 --nt 5 \
  --mine_task_disorder 0.6 --mine_alpha 0.3 --mine_seed 123 \
  --log True --offlog True

```

> With 5 classes and --cpt 1, you’ll get 5 tasks (one per star rating). For a 2-class polarity variant, use dataset="AMAZON2", num_classes=2, --cpt 1 --nt 2.
> 

---

## 6) Evaluation criteria for text datasets

- **Per-task global accuracy**: Use the same micro average as your image tasks but restrict the global test set to the **global class block** (each “task” is exactly one rating class when `cpt=1`), i.e.
    
    (A_t = \frac{\sum_c #\text{correct on class set }Y_t}{\sum_c #\text{samples in }Y_t}).
    
- **Average Accuracy (AA)**: mean of (A_t) over all tasks at the end of training.
- **Per-client AA/Forgetting**: For each client (c), compute accuracy on the **global** test set restricted to that client’s assigned class set per task and average over tasks up to (t) (for AA) and use the standard (F_{t,k}^{(c)} = \max_{\tau \le t} A^{(c)}*{\tau,k} - A^{(c)}*{t,k}) for forgetting.
- **Text-specific**: Record **average review length** and **class balance** per client (histogram), since Dirichlet splits can introduce heavy skew; log these alongside AA/Forgetting.

---

## 7) Common pitfalls & checks

- **No double evaluation**: Build the global per-class counts once **per eval round** (your `_get_or_build_global_counts`), then reuse to compute AA/Forgetting for all clients — otherwise you’ll re-scan the test set per client.
- **Ensure global labels**: Your `.npy` files already bake in class IDs (0..4). If you ever construct a `DataLoader` directly from raw text, make sure you don’t remap labels per task.
- **Model input shape**: For TEXTMLP, pass `[B, D]` tensors. Don’t run image transforms or `permute(0,3,1,2)` on these features.
- **Resource planning**: `sentence-transformers` embedding can be done offline once; cache the `.npy` and don’t recompute every run.

---

## 8) Extending to other non-image datasets

The same pattern works for **AG News**, **Yahoo Answers**, **DBPedia**, **20 Newsgroups**, etc.:

1. Use a loader (`datasets` or your own script) to get raw text + labels.
2. Precompute embeddings to fixed dimension (e.g., 384).
3. Dump per-class `.npy` blocks for train/test.
4. Add `read_client_data_FCL_<name>` in both partitioners that returns `TensorDataset(torch.from_numpy(X).float(), y)`.
5. Use `TEXTMLP` (or a `TextCNN`/BERT encoder if you add one) and run with `-model TEXTMLP --input_dim <emb_dim>`.

If you share your `clientTARGET.train(...)` or `FedWeIT` assumptions about input shapes, I can include the minimal model and loader tweaks needed to keep everything consistent across image + text.

This approach keeps your **FCL** machinery unchanged while opening the door to a broad set of **non-vision** tasks.