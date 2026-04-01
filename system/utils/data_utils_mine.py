# system/utils/data_utils_mine.py
# -*- coding: utf-8 -*-
"""
Mine partition (task-level permutation):

Knobs:
  - ALPHA: Dirichlet alpha to split each class's samples ACROSS clients (standard FL).
  - classes_per_task (cpt): provided by the caller; defines a GLOBAL TASK POOL by chunking
    the master class order [0, 1, ..., K-1] into consecutive, disjoint groups of size cpt.
  - TASK_DISORDER in [0,1]: permutes the ORDER OF TASKS (not the classes inside a task)
    for each client with a "noisy sort":
        score_t = (1 - d) * t + d * U(0, T); order = argsort(score_t)
    d=0 → same as master, d=1 → fully random permutation; 0<d<1 → mild disorder.

API compatibility:
  read_client_data_FCL_{cifar10,cifar100,imagenet1k}(client_id, task, classes_per_task, count_labels=False)
    -> (TensorDataset, label_info)
label_info includes 'labels' (list of class IDs), 'counts' if requested, client task order,
and the global task pool used.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import math
import numpy as np
import torch
from torch.utils.data import TensorDataset

# =======================
# >>>>> HYPER-PARAMS <<<<<
# =======================
SEED: int = 7                 # global seed for the partitioner
TOTAL_CLIENTS: int = 10       # MUST equal your hparams["num_clients"]
ALPHA: float = 0.3            # Dirichlet alpha (per-class -> clients). Smaller => more skew.
TASK_DISORDER: float = 0    # in [0,1]; 0: same task order as master, 1: random permutation

# Optional per-client override; length must be TOTAL_CLIENTS if used
CLIENT_TASK_DISORDER: List[float] | None = None


# ======================
# Utilities
# ======================
def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))

def _dataset_root_and_K(dataset: str) -> Tuple[Path, int]:
    ds = dataset.lower()
    if ds == "cifar10":  return Path("dataset/cifar10-classes"), 10
    if ds == "cifar100": return Path("dataset/cifar100-classes"), 100
    return Path("dataset/imagenet1k-classes"), 1000  # default

def _glob_npy_for_class(root: Path, cls: int) -> Path:
    p = root / f"{cls}.npy"
    if p.exists(): return p
    cands = sorted(root.glob(f"{cls}_*.npy"))
    if not cands:
        raise FileNotFoundError(f"[mine] Missing .npy for class {cls} under {root}")
    return cands[0]

def _load_class_np(root: Path, cls: int) -> np.ndarray:
    arr = np.load(_glob_npy_for_class(root, cls), allow_pickle=True)
    # standardize to (N, H, W, 3) uint8
    if arr.ndim == 4 and arr.shape[-1] == 3:
        pass
    elif arr.ndim == 4 and arr.shape[1] == 3:
        arr = np.transpose(arr, (0, 2, 3, 1))
    elif arr.ndim == 3:
        if   arr.shape[-1] == 3: arr = arr[None, ...]
        elif arr.shape[0]  == 3: arr = np.transpose(arr, (1, 2, 0))[None, ...]
        else: raise ValueError(f"[mine] Unexpected array shape for class {cls}: {arr.shape}")
    else:
        raise ValueError(f"[mine] Unexpected array shape for class {cls}: {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr

def _to_tensordataset(images_hwc_uint8: np.ndarray, labels_int: np.ndarray) -> TensorDataset:
    x = torch.tensor(np.transpose(images_hwc_uint8, (0, 3, 1, 2)), dtype=torch.float32) / 255.0
    y = torch.tensor(labels_int.astype(np.int64), dtype=torch.long)
    return TensorDataset(x, y)


# ======================
# Plan (cached)
# ======================
class _Plan:
    """
    Builds once and caches:
      - Global master class order: [0, 1, ..., K-1]
      - GLOBAL TASK POOL: consecutive disjoint chunks of size cpt
      - Per-class Dirichlet allocations across clients (indices per client)
      - Per-client permutation over TASK INDICES via task-level "noisy sort"
    """
    def __init__(self, dataset: str, K: int, cpt: int, seed: int = SEED, total_clients: int = TOTAL_CLIENTS, task_disorder: float = TASK_DISORDER):
        if total_clients <= 0:
            raise ValueError("TOTAL_CLIENTS must be > 0")
        if not (0.0 <= task_disorder <= 1.0):
            raise ValueError("TASK_DISORDER must be in [0,1]")

        self.dataset = dataset
        self.K = int(K)
        self.cpt = int(cpt)
        self.C = int(total_clients)
        self.T = math.ceil(self.K / max(1, self.cpt))

        self.rng = _rng(seed)

        # Master class order (identity, as requested) and global task pool
        self.master_classes: List[int] = list(range(self.K))
        self.master_task_pool: List[List[int]] = self._make_task_pool(self.master_classes, self.cpt, self.T)

        # Per-client task orders (permutation of {0..T-1})
        self._client_task_order: Dict[int, List[int]] = {}

        # Per-class Dirichlet split across clients (indices per client)
        self._class_alloc_idx: Dict[int, List[np.ndarray]] = {}

    @staticmethod
    def _make_task_pool(master_classes: List[int], cpt: int, T: int) -> List[List[int]]:
        pool: List[List[int]] = []
        for t in range(T):
            s = t * cpt
            e = min(len(master_classes), (t + 1) * cpt)
            if s >= len(master_classes):
                pool.append([])
            else:
                pool.append(master_classes[s:e])
        return pool

    def _client_disorder(self, cid: int, task_disorder: int = TASK_DISORDER) -> float:
        if cid == 0:
            return 0.0  # client 0 is the master order by definition
        if CLIENT_TASK_DISORDER is not None and 0 <= cid < len(CLIENT_TASK_DISORDER):
            return float(CLIENT_TASK_DISORDER[cid])
        return float(task_disorder)

    def task_order_for_client(self, cid: int,
                              seed: int = SEED, task_disorder: float = TASK_DISORDER,
                              ) -> List[int]:
        if cid in self._client_task_order:
            return self._client_task_order[cid]

        d = self._client_disorder(cid, task_disorder)
        base = np.arange(self.T, dtype=np.int64)

        if d <= 0.0 or self.T <= 1:
            order = base.tolist()
        elif d >= 1.0:
            r = _rng(seed + 111_111 * (cid + 1))
            order = r.permutation(base).tolist()
        else:
            # task-level noisy sort: score_t = (1-d)*t + d*U(0,T)
            r = _rng(seed + 111_111 * (cid + 1))
            noise = r.random(self.T) * self.T
            ranks = base.astype(float)
            scores = (1.0 - d) * ranks + d * noise
            order = np.argsort(scores, kind="mergesort").astype(int).tolist()

        self._client_task_order[cid] = order
        return order

    def _ensure_alloc_for_class(self, k: int, root: Path, alpha: float = ALPHA):
        if k in self._class_alloc_idx:
            return
        # Load class k to know N_k
        arr = _load_class_np(root, k)
        N = int(arr.shape[0])
        # Dirichlet weights across clients
        w = self.rng.dirichlet(alpha=np.ones(self.C, dtype=np.float64) * max(1e-6, float(alpha)))
        raw = w * N
        counts = np.floor(raw).astype(int)
        rem = N - int(counts.sum())
        if rem > 0:
            frac = raw - counts.astype(float)
            order = np.argsort(-frac)
            counts[order[:rem]] += 1
        # materialize indices
        perm_idx = self.rng.permutation(N)
        splits: List[np.ndarray] = []
        cursor = 0
        for c in range(self.C):
            take = int(counts[c])
            if take <= 0:
                splits.append(np.empty((0,), dtype=np.int64))
            else:
                splits.append(perm_idx[cursor:cursor + take])
            cursor += take
        self._class_alloc_idx[k] = splits

    def indices_for(self, k: int, cid: int, root: Path, alpha: float = ALPHA) -> np.ndarray:
        self._ensure_alloc_for_class(k, root, alpha)
        return self._class_alloc_idx[k][cid]



def _get_plan(dataset: str, K: int, cpt: int,
              seed: int = SEED, alpha: float = ALPHA, total_clients: int = TOTAL_CLIENTS, task_disorder: float = TASK_DISORDER,
              ) -> _Plan:
    
    # Cache of plans keyed by (dataset, K, cpt, C, SEED, ALPHA, TASK_DISORDER)
    _PLAN_CACHE: Dict[Tuple[str, int, int, int, int, float, float], _Plan] = {}

    key = (dataset.upper(), int(K), int(cpt), int(total_clients), int(seed), float(alpha), float(task_disorder))
    if key not in _PLAN_CACHE:
        _PLAN_CACHE[key] = _Plan(dataset.upper(), K, cpt, seed, total_clients, task_disorder)
    return _PLAN_CACHE[key]


# ======================
# Assemble per (client, task)
# ======================
def _assemble(dataset: str, root: Path, K: int,
              client_id: int, task: int, cpt: int, count_labels: bool,
              seed: int, alpha: float, total_clients: int, task_disorder: float,
              ) -> Tuple[TensorDataset, Dict[str, Any]]:
    if client_id < 0 or client_id >= total_clients:
        raise ValueError(f"[mine] client_id {client_id} out of range [0,{total_clients-1}] — set TOTAL_CLIENTS.")
    if cpt <= 0:
        raise ValueError("[mine] classes_per_task (cpt) must be > 0")

    plan = _get_plan(dataset, K, cpt, seed, alpha, total_clients, task_disorder)
    T = plan.T
    if task < 0 or task >= T:
        raise ValueError(f"[mine] task {task} out of range [0,{T-1}] (T inferred from K and cpt).")

    # pick the client's task index in the global pool
    order_tasks = plan.task_order_for_client(client_id, seed, task_disorder)     # permutation of [0..T-1]
    t_true = int(order_tasks[task])                         # index into master_task_pool
    labels = list(plan.master_task_pool[t_true])            # immutable class group (disjoint inside client)

    xs, ys = [], []
    counts_map: Dict[int, int] = {}

    for k in labels:
        idx = plan.indices_for(k, client_id, root, alpha)          # this client's share for class k
        if idx.size > 0:
            arr = _load_class_np(root, k)
            xs.append(arr[idx])
            ys.append(np.full((idx.size,), fill_value=int(k), dtype=np.int64))
        counts_map[int(k)] = int(idx.size)

    if xs:
        X = np.concatenate(xs, axis=0)
        Y = np.concatenate(ys, axis=0)
        ds = _to_tensordataset(X, Y)
    else:
        ds = TensorDataset(torch.zeros((0, 3, 32, 32), dtype=torch.float32),
                           torch.zeros((0,), dtype=torch.long))

    present_labels = [int(k) for k in labels if counts_map.get(int(k), 0) > 0]
    missing_labels = [int(k) for k in labels if counts_map.get(int(k), 0) == 0]

    info: Dict[str, Any] = {
        "labels": list(labels),                 # assigned labels (from master task pool)
        "classes": list(labels),                # alias
        "counts": counts_map if count_labels else {},
        "label_count": dict(counts_map) if count_labels else {},
        "task_index_in_master": t_true,
        "task_order_pi_tasks": list(order_tasks),
        "master_task_pool": [list(g) for g in plan.master_task_pool],
        "assigned_labels": list(labels),        # <--- explicit
        "present_labels": present_labels,       # <--- explicit (non-zero)
        "missing_labels": missing_labels        # <--- explicit (zero count)
    }

    return ds, info


# ======================
# Public API (unchanged names/signatures)
# ======================
def read_client_data_FCL_cifar10(client_id: int, task: int, classes_per_task: int, count_labels: bool=False,
                                 seed: int = SEED, alpha: float = ALPHA, total_clients: int = TOTAL_CLIENTS, task_disorder: float = TASK_DISORDER):
    root, K = _dataset_root_and_K("CIFAR10")
    return _assemble("CIFAR10", root, K, client_id, task, classes_per_task, count_labels, seed, alpha, total_clients, task_disorder)

def read_client_data_FCL_cifar100(client_id: int, task: int, classes_per_task: int, count_labels: bool=False,
                                  seed: int = SEED, alpha: float = ALPHA, total_clients: int = TOTAL_CLIENTS, task_disorder: float = TASK_DISORDER):
    root, K = _dataset_root_and_K("CIFAR100")
    return _assemble("CIFAR100", root, K, client_id, task, classes_per_task, count_labels, seed, alpha, total_clients, task_disorder)

def read_client_data_FCL_imagenet1k(client_id: int, task: int, classes_per_task: int, count_labels: bool=False,
                                    seed: int = SEED, alpha: float = ALPHA, total_clients: int = TOTAL_CLIENTS, task_disorder: float = TASK_DISORDER):
    root, K = _dataset_root_and_K("IMAGENET1K")
    return _assemble("IMAGENET1K", root, K, client_id, task, classes_per_task, count_labels, seed, alpha, total_clients, task_disorder)
