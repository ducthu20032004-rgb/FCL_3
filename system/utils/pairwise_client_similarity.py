"""
Pairwise client similarity measurement for FedAvg (Federated Continual Learning).

HOW TO USE
----------
1. Copy `_measure_pairwise_client_similarity` and `get_shared_probe_dataset` into your
   FedAvg (or Server base) class.

2. At the END of each task loop in `train()`, right after
   `self.dump_global_task_accuracy_csv(...)`, add:

       self._measure_pairwise_client_similarity(
           task=task,
           glob_iter=glob_iter,
           csv_path=getattr(self.args, "pairwise_sim_csv",
                            "./pairwise_similarity.csv"),
           layers=["block1", "block2", "block3", "block4"],
       )

3. Adjust `datadir` / `images_per_class` inside `get_shared_probe_dataset` to match
   your local dataset layout.
"""

import os
import csv
import itertools
from typing import List, Optional

import numpy as np
import torch
import wandb

# ── re-use helpers already in your codebase ──────────────────────────────────
# compute_feature_resnet18_wrap  → extracts (N, D) numpy array for a given block
# compute_eps                    → ε-similarity (drift) between two feature arrays
# compute_alignment_from_arrays  → CKNNA / mutual-kNN alignment
# get_shared_probe_dataset       → returns a torch.utils.data.Dataset
#   (copy the function below if not already defined)
# Transform_dataset              → your TensorDataset-like wrapper


# ─────────────────────────────────────────────────────────────────────────────
#  Optional: define get_shared_probe_dataset here if not already in your repo
# ─────────────────────────────────────────────────────────────────────────────

def get_shared_probe_dataset(
    datadir: str = "./dataset/cifar10-classes/",
    classes: List[int] = list(range(10)),
    images_per_class: int = 75,
    train_images_per_class: int = 5000,
):
    """
    Load a fixed global probe dataset covering all label classes.

    Args:
        datadir:               Folder containing per-class .npy files named `{cls}.npy`.
                               Each file shape: (N, H, W, C) float32 in [0,255] or [0,1].
        classes:               Which class indices to include.
        images_per_class:      How many test images per class to keep (750 total for 10 classes).
        train_images_per_class: How many images are in the train split (images after this
                                index are test images).

    Returns:
        A torch.utils.data.TensorDataset with (x, y) where
            x: (N, C, H, W)  float32, already channel-first
            y: (N,)           int64
    """
    from torch.utils.data import TensorDataset

    x_list, y_list = [], []
    for cls in classes:
        data_file = os.path.join(datadir, f"{cls}.npy")
        data = np.load(data_file)                                  # (N, H, W, C) or (N, C, H, W)
        test_data = data[train_images_per_class:
                         train_images_per_class + images_per_class]

        # Convert to channel-first if needed  (H, W, C) → (C, H, W)
        if test_data.ndim == 4 and test_data.shape[-1] in (1, 3):
            test_data = test_data.transpose(0, 3, 1, 2)

        x_list.append(test_data.astype(np.float32))
        y_list.append(np.full(images_per_class, cls, dtype=np.int64))

    x = torch.tensor(np.concatenate(x_list))   # (N, C, H, W)
    y = torch.tensor(np.concatenate(y_list))   # (N,)
    return TensorDataset(x, y)


# ─────────────────────────────────────────────────────────────────────────────
#  Main method — paste this into your FedAvg (or Server) class
# ─────────────────────────────────────────────────────────────────────────────

def _measure_pairwise_client_similarity(
    self,
    task: int,
    glob_iter: int,
    csv_path: str = "./pairwise_similarity_hete01.csv",
    layers: Optional[List[str]] = None,
    topk_cknna: int = 10,
    probe_datadir: str = "./dataset/cifar10-classes/",
    images_per_class: int = 75,
    log_wandb: bool = True,
):
    """
    After each task ends, measure pairwise feature similarity between ALL clients
    on a fixed global probe dataset covering all 10 CIFAR-10 classes.

    Metrics per (client_i, client_j, layer):
      - eps_sim   : ε-similarity / drift  (lower = more diverged)
      - cknna     : mutual-kNN alignment  (higher = more similar representations)

    Results are written to `csv_path` and (optionally) logged to W&B.

    Signature compatible with FedAvg.train() — call as:
        self._measure_pairwise_client_similarity(task=task, glob_iter=glob_iter)
    """
    if layers is None:
        layers = ["block1", "block2", "block3", "block4"]

    # ── 1. Build probe dataset (once; cached on self) ─────────────────────
    if not hasattr(self, "_probe_dataset") or self._probe_dataset is None:
        try:
            self._probe_dataset = get_shared_probe_dataset(
                datadir=probe_datadir,
                images_per_class=images_per_class,
            )
            n = len(self._probe_dataset)
            print(f"[PairwiseSim] Probe dataset ready: {n} samples "
                  f"({images_per_class} per class × 10 classes)")
        except Exception as e:
            print(f"[PairwiseSim] Could not build probe dataset: {e}")
            return

    probe = self._probe_dataset

    # ── 2. Extract features for every client × every layer ───────────────
    # features[client_id][layer] = np.ndarray  (N, D)
    client_list = self.clients        # all clients, not just selected
    features = {}

    print(f"\n[PairwiseSim] Task {task} | extracting features for "
          f"{len(client_list)} clients × {len(layers)} layers …")

    for client in client_list:
        cid = getattr(client, "id", id(client))
        features[cid] = {}
        for layer in layers:
            try:
                feat = compute_feature_resnet18_wrap(      # already in your codebase
                    _model=client.model,
                    _model_task_index=cid,
                    _dataset=probe,
                    _target_layer_index=layer,
                    seed=getattr(self.args, "seed", 0),
                    args=self.args,
                )
                features[cid][layer] = feat               # (N, D) numpy
            except Exception as e:
                print(f"[PairwiseSim] client={cid} layer={layer} error: {e}")
                features[cid][layer] = None

    # ── 3. Pairwise loop: client_i × client_j (i < j) ────────────────────
    client_ids = [getattr(c, "id", id(c)) for c in client_list]
    pairs = list(itertools.combinations(client_ids, 2))

    results = []   # list of dicts for CSV

    for (ci, cj) in pairs:
        for layer in layers:
            fi = features[ci].get(layer)
            fj = features[cj].get(layer)
            if fi is None or fj is None:
                continue

            # ε-similarity (symmetric mean of both directions)
            try:
                eps_ij = compute_eps(fi, fj)               # already in your codebase
                eps_ji = compute_eps(fj, fi)
                eps_sim = float((eps_ij + eps_ji) / 2.0)
            except Exception as e:
                print(f"[PairwiseSim] eps error {ci}↔{cj} {layer}: {e}")
                eps_sim = float("nan")

            # mutual-kNN alignment (CKNNA)
            try:
                cknna, _ = compute_alignment_from_arrays(   # already in your codebase
                    fi, fj,
                    "mutual_knn",
                    topk=topk_cknna,
                    precise=True,
                )
                cknna = float(cknna)
            except Exception as e:
                print(f"[PairwiseSim] cknna error {ci}↔{cj} {layer}: {e}")
                cknna = float("nan")

            row = {
                "glob_iter": glob_iter,
                "task": task,
                "client_i": ci,
                "client_j": cj,
                "layer": layer,
                "eps_sim": round(eps_sim, 6),
                "cknna": round(cknna, 6),
            }
            results.append(row)

            print(
                f"  [pair {ci}↔{cj}] {layer:7s} | "
                f"eps_sim={eps_sim:.4f}  cknna={cknna:.4f}"
            )

    # ── 4. Write CSV ──────────────────────────────────────────────────────
    fieldnames = ["glob_iter", "task", "client_i", "client_j",
                  "layer", "eps_sim", "cknna"]
    write_header = not os.path.exists(csv_path)

    try:
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(results)
        print(f"[PairwiseSim] Wrote {len(results)} rows → {csv_path}")
    except Exception as e:
        print(f"[PairwiseSim] CSV write error: {e}")

    # ── 5. Log to W&B ─────────────────────────────────────────────────────
    if log_wandb and getattr(self.args, "wandb", False):
        wb_dict = {"glob_iter": glob_iter, "task": task}
        for row in results:
            key_prefix = (
                f"pairwise/c{row['client_i']}_c{row['client_j']}"
                f"/{row['layer']}"
            )
            wb_dict[f"{key_prefix}/eps_sim"] = row["eps_sim"]
            wb_dict[f"{key_prefix}/cknna"]   = row["cknna"]
        try:
            wandb.log(wb_dict)
        except Exception as e:
            print(f"[PairwiseSim] wandb log error: {e}")