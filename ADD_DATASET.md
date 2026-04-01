# Adding a New Dataset (and Its Evaluation Criteria)

This guide shows how to add a **new image classification dataset** to the FCL pipeline in this branch, with **both partitions**:

* `tuan` (codebase split, class-order slicing)
* `hetero` (task-level permutation: Dirichlet class skew across clients + task order disorder)

It also standardizes **evaluation criteria** so your results remain apples-to-apples across datasets.

---

## Quick checklist

1. **Global class IDs:** decide class count `K` and global label IDs `0..K-1`.
2. **Per-class arrays:** create per-class `.npy` folders for **train** and **test**:

   ```
   dataset/<dataset>-classes/<0..K-1>.npy          # train
   dataset/<dataset>-test-classes/<0..K-1>.npy     # test (global)
   ```
3. **Wire roots + K:** add the dataset to:

   * `system/utils/data_utils.py` (tuan's partition)
   * `system/utils/data_utils_mine.py` (hetero partition) via `_dataset_root_and_K`.
4. **Loaders:** ensure **global test loader** returns labels in `0..K-1`. If not, add a mapping attribute `label_to_global` on its dataset.
5. **Transforms & normalization:** set dataset-specific input size, mean/std, and augmentations consistently for train/test.
6. **Hparams:** add a `./hparams/<dataset>/<Algo>.json` (or reuse an existing one and change `"dataset"` and `"num_classes"`).
7. **Sanity:** run the partition viz and the “global per-task vector” evaluation to confirm columns fill for all tasks.

---

## 1. Per-class `.npy` format

Each `.npy` must contain either `(N, H, W, 3)` **uint8** or `(N, 3, H, W)` numeric arrays for the class ID equal to the filename:

```
dataset/<dataset>-classes/
  0.npy  1.npy  ...  K-1.npy

dataset/<dataset>-test-classes/
  0.npy  1.npy  ...  K-1.npy
```

**Why:** the partitioners (both `tuan` and `hetero`) and the global test evaluator assume **global labels** by filename, so remapping is unnecessary and mistakes are visible.

### Template converter (from images to per-class `.npy`)

```python
# tools/make_npy_per_class.py
from pathlib import Path
import numpy as np
from PIL import Image

def to_hwc_uint8(p):
    img = Image.open(p).convert("RGB")
    arr = np.array(img)  # H,W,3 uint8
    return arr

def pack(folder, out_npy):
    ims = [to_hwc_uint8(p) for p in sorted(Path(folder).glob("*"))]
    X = np.stack(ims, axis=0).astype(np.uint8)  # N,H,W,3
    np.save(out_npy, X)

# Example:
# for k in range(K): pack(f"raw/train/class_{k}/", f"dataset/mydata-classes/{k}.npy")
# for k in range(K): pack(f"raw/test/class_{k}/",  f"dataset/mydata-test-classes/{k}.npy")
```

---

## 2. Register dataset roots and `K`

In both files below, add your dataset name (case-insensitive) to return `(root_path, K)`:

* `system/utils/data_utils.py`
* `system/utils/data_utils_mine.py`

Example:

```python
def _dataset_root_and_K(dataset: str):
    ds = dataset.lower()
    if ds == "mydata":
        return Path("dataset/mydata-classes"), 200   # train root, K
    # keep existing branches (cifar10, cifar100, imagenet1k, ...)
```

**Global test set root** (if you build a separate loader from `.npy`) should be:

```
dataset/mydata-test-classes/
```

---

## 3. Current partition wiring

Add a `read_client_data_FCL_mydata(...)` in `system/utils/data_utils.py`, mirroring CIFAR-10/100:

* Load class shards from `dataset/<dataset>-classes`.
* Respect `client_id`, `task`, `classes_per_task (cpt)` with your **precomputed class order** files or the internal logic already used for other datasets.
* Ensure the returned dataset is a `TensorDataset` with **CHW float32 [0,1]** and labels **0..K-1**.

If your dataset needs specific normalizations/augmentations, set them near where CIFAR ones are set (keep eval transforms light and deterministic).

---

## 4. Hetero partition wiring

In `system/utils/data_utils_mine.py`:

* Ensure `_dataset_root_and_K("mydata")` returns the **train** root and `K`.
* The existing implementation already supports:

  * Dirichlet per-class split across clients (`--alpha`)
  * Task pool from the master order chunked by `--cpt`
  * Task-order disorder (`--task_disorder` and optional per-client overrides)
  * Seed control (`--seed`)

No code change is required beyond registering the root and `K`.

---

## 5. Global test loader (recommended)

For consistent evaluation (per-task curves; AA; forgetting), build one **global** test loader that yields labels `0..K-1`:

```python
# serverbase.py (example builder from per-class npy)
def _build_global_test_from_npy(self, name="mydata"):
    import numpy as np, torch, os
    from torch.utils.data import TensorDataset, DataLoader

    root = f"dataset/{name}-test-classes"
    K = int(self.args.num_classes)
    xs, ys = [], []
    for k in range(K):
        p = os.path.join(root, f"{k}.npy")
        if not os.path.exists(p): continue
        arr = np.load(p, allow_pickle=True)  # (N,H,W,3) or (N,3,H,W)
        if arr.ndim == 4 and arr.shape[-1] == 3:
            pass
        elif arr.ndim == 4 and arr.shape[1] == 3:
            arr = np.transpose(arr, (0,2,3,1))
        else:
            continue
        xs.append(arr)
        ys.append(np.full((arr.shape[0],), k, dtype=np.int64))
    X = np.concatenate(xs, axis=0)                  # HWC uint8
    Y = np.concatenate(ys, axis=0)                  # global IDs
    X = torch.tensor(np.transpose(X, (0,3,1,2)), dtype=torch.float32) / 255.0
    Y = torch.tensor(Y, dtype=torch.long)
    ds = TensorDataset(X, Y)
    bs = int(getattr(self.args, "batch_size", 128) or 128)
    self._global_test_dataset = ds
    self._global_test_loader  = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=2)
    self._global_test_K = K
    return self._global_test_loader
```

If you instead use torchvision (e.g., for CIFAR-like datasets), make sure transforms are compatible with your training normalization.

---

## 6. Evaluation criteria (dataset-specific)

Define per-dataset choices here so experiments are consistent:

* **Input size:** e.g., `32×32` (CIFAR), `64×64` (Tiny), `224×224` (ImageNet-style).
* **Normalization:** (mean, std) over **train** set; apply same to test.
* **Augmentations (train):** random crop/flip; avoid stochastic transforms in **evaluation**.
* **Per-task global test subsets:** use the **global task pool** ({\mathcal{Y}_k}) to define `D_k^{test}`; do **not** depend on client order.
* **Report** (at minimum):

  * **Per-task curves:** (A_k(r) = \frac{1}{|D_k^{test}|}\sum \mathbf{1}[f^{(r)}(x)=y]) for each (k).
  * **Per-client Average Accuracy after task (t):** (\mathrm{AA}*c(\le t) = \frac{1}{t+1}\sum*{s=0}^t A_{c,s}(r_t)), where (A_{c,s}) uses the global subset for (\mathcal{Y}_{\pi_c(s)}).
  * **Per-client Average Forgetting:** (\overline{F}*c(r_t) = \frac{1}{t}\sum*{s=0}^{t-1}\big(\max_{\tau\in{r_s,\dots,r_t}}A_{c,s}(\tau) - A_{c,s}(r_t)\big)).
* **Optional metrics:** AULC per task, FWT/BWT, ECE (calibration), client-fairness dispersion (std or Gini of ({\mathrm{AA}_c})).

> When you add a dataset, document the **input size**, **mean/std**, and **any constraints** (e.g., grayscale → stack to 3 channels; class imbalance → note in readme).

---

## 7. Hparams and CLI

1. Add a config in `./hparams/<dataset>/<Algo>.json`:

```json
{
  "optimizer": "sgd",
  "device": "cuda",
  "device_id": "0",
  "dataset": "MyData",
  "num_classes": 200,
  "model": "ResNet18",
  "model_str": "ResNet18",
  "batch_size": 128,
  "local_learning_rate": 0.01,
  "global_rounds": 100,
  "local_epochs": 5,
  "algorithm": "FedAvg",
  "join_ratio": 1.0,
  "random_join_ratio": false,
  "num_clients": 10,
  "eval_gap": 1,
  "out_folder": "out",
  "client_drop_rate": 0.0,
  "time_threthold": 10000
}
```

2. Run examples:

* **tuan**

```bash
python system/main.py --cfp ./hparams/mydata/FedAvg.json \
  --partition_options current --cpt 5 --nt 40 --log True --offlog True
```

* **hetero**

```bash
python system/main.py --cfp ./hparams/mydata/FedAvg.json \
  --partition_options hetero --cpt 5 --nt 40 \
  --task_disorder 0.6 --alpha 0.3 --seed 123 \
  --log True --offlog True
```

---

## 8. Sanity + debugging

* **Partition plan (text/CSV/heatmaps):** run your `visualize_and_print_partition(server, args)` before training to confirm:

  * assigned labels per (client, task) (from global task pool),
  * present labels (after Dirichlet split),
  * missing labels (assigned but zero after split).
* **Global per-task vector:** after finishing task (t), the row ([A_{t,0},\dots,A_{t,T-1}]) should fill **all columns** (no all-zeros beyond `task_0`). If not:

  * Check the global test loader returns labels in **0..K-1**;
  * If you use the union-of-clients fallback, ensure task-local labels are remapped to **global** using the global task pool block for each loader index.

---

## 9. Common pitfalls

* **Task-local labels at eval:** if a client’s test loader produces `0..cpt-1`, you must remap to **global** with a reliable mapping (`label_to_global` or your per-task label list). Otherwise per-task columns except `task_0` will appear as zeros.
* **Transforms mismatch:** eval transforms must not differ across datasets in a way that changes results (e.g., extra random crop at test time).
* **Inconsistent `K`/roots:** `_dataset_root_and_K` must return the correct train root and `K`, and your global test loader must agree on the same `K`.

---

## 10. What to add to the main README after integrating a dataset

* Add a row to the dataset table (name, #classes, input size, train/test counts, per-class root folders).
* Add two one-liners to “Running one experiment” (tuan & hetero).
* Add dataset-specific normalization and any special notes (e.g., grayscale, long-tail).

---

### Appendix: minimal dataset class for torchvision (optional)

If your dataset already exists in torchvision (or you write one), make sure its `__getitem__` returns `(x, y)` with **`y` in 0..K-1 global IDs**. If it doesn’t, expose a vector `label_to_global` on the dataset so the evaluator can remap.

```python
class MyDataTorch(Dataset):
    def __init__(self, root, split, transform=None):
        ...
        self.label_to_global = np.arange(K, dtype=np.int64)  # or your mapping
    def __getitem__(self, i):
        x = ...  # PIL or tensor
        y = ...  # GLOBAL id in 0..K-1
        if self.transform: x = self.transform(x)
        return x, y
```