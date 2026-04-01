# system/utils/partition_viz.py
# -*- coding: utf-8 -*-
"""
Partition visualization & reporting for FCL runs.
- Introspects the server's clients to recover the Task -> Class mapping they've been assigned.
- Prints a readable table before training starts.
- Saves heatmaps (presence & counts), CSV, TXT and JSON into ./figures (or custom fig_dir).

Usage in system/main.py (before server.train()):
    from system.utils.partition_viz import visualize_and_print_partition
    visualize_and_print_partition(server, args, fig_dir="figures")

This file is read-only with respect to training; it won’t modify datasets or clients.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

# Headless-safe matplotlib import
import matplotlib
matplotlib.use("Agg")  # no GUI required
import matplotlib.pyplot as plt
import numpy as np

try:
    import torch  # only used for label extraction if tensors are returned
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


# ---------- Utilities ----------

def _ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _to_int(x: Any) -> Optional[int]:
    if isinstance(x, (int, np.integer)):
        return int(x)
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor) and x.numel() == 1:
        return int(x.item())
    return None

def _dataset_label_array(ds: Any) -> Optional[np.ndarray]:
    """
    Try common ways to get labels from a dataset object without iterating:
    - ds.targets / ds.labels / ds.y / ds.ys
    - ds.tensors (TensorDataset) -> second tensor
    Returns np.ndarray of ints if found, else None.
    """
    for attr in ("targets", "labels", "y", "ys"):
        if hasattr(ds, attr):
            arr = getattr(ds, attr)
            try:
                arr = np.array(arr)
                if arr.ndim == 2 and arr.shape[1] == 1:
                    arr = arr[:, 0]
                return arr.astype(int)
            except Exception:
                pass
    if hasattr(ds, "tensors"):
        try:
            tens = ds.tensors
            if isinstance(tens, (tuple, list)) and len(tens) >= 2:
                y = tens[1]
                if TORCH_AVAILABLE and isinstance(y, torch.Tensor):
                    return y.detach().cpu().numpy().astype(int)
        except Exception:
            pass
    return None

def _dataset_unique_labels(ds: Any, scan_cap: int = 50000) -> Tuple[List[int], Optional[Dict[int, int]]]:
    """
    Return (sorted_unique_labels, optional_counts) for a dataset.
    Tries zero-copy attributes first; otherwise iterates items (capped).
    Counts dict may be None if we can't cheaply compute counts.
    """
    arr = _dataset_label_array(ds)
    if arr is not None:
        # Fast path: we have all labels
        uniq, counts = np.unique(arr.astype(int), return_counts=True)
        return uniq.tolist(), {int(k): int(v) for k, v in zip(uniq, counts)}

    # Slow path: iterate items
    label_list: List[int] = []
    n = len(ds) if hasattr(ds, "__len__") else scan_cap
    n = min(n, scan_cap)
    for i in range(n):
        try:
            item = ds[i]
        except Exception:
            break
        lbl = None
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            lbl = _to_int(item[1])
        else:
            lbl = _to_int(item)
        if lbl is not None:
            label_list.append(lbl)
    if not label_list:
        return [], None
    uniq, counts = np.unique(np.array(label_list, dtype=int), return_counts=True)
    return uniq.tolist(), {int(k): int(v) for k, v in zip(uniq, counts)}

def _looks_like_dataloader(x: Any) -> bool:
    return hasattr(x, "dataset") and hasattr(x, "__iter__")

def _looks_like_dataset(x: Any) -> bool:
    return hasattr(x, "__getitem__") and hasattr(x, "__len__")

def _extract_task_datasets_from_client(client: Any) -> List[Any]:
    """
    Heuristics to collect per-task datasets (train side) from a client object.
    Checks common attribute names first; otherwise scans attributes for lists/dicts of datasets or dataloaders.
    Returns a list ordered by task index (if dicts, sorted by key).
    """
    # 1) Known attribute names (ordered)
    candidate_attr_names = [
        "task_train_loaders", "train_loaders", "trainloaders",
        "task_train_datasets", "train_datasets", "train_sets",
        "train_data", "train_data_local_dict"
    ]
    for name in candidate_attr_names:
        if hasattr(client, name):
            obj = getattr(client, name)
            # dict -> sort by key
            if isinstance(obj, dict):
                items = [obj[k] for k in sorted(obj)]
            elif isinstance(obj, (list, tuple)):
                items = list(obj)
            else:
                items = [obj]
            # map dataloaders -> datasets
            ds_list = []
            for it in items:
                if _looks_like_dataloader(it):
                    ds_list.append(getattr(it, "dataset"))
                elif _looks_like_dataset(it):
                    ds_list.append(it)
            if ds_list:
                return ds_list

    # 2) Fallback: scan all attributes for plausible lists/dicts
    ds_list = []
    for k, v in vars(client).items():
        if isinstance(v, dict):
            vals = list(v.values())
        elif isinstance(v, (list, tuple)):
            vals = list(v)
        else:
            vals = [v]
        tmp = []
        for it in vals:
            if _looks_like_dataloader(it):
                tmp.append(getattr(it, "dataset"))
            elif _looks_like_dataset(it):
                tmp.append(it)
        # Heuristic: if it looks like multiple datasets and len > 1, consider it tasks
        if len(tmp) > 1:
            ds_list = tmp
            break
    return ds_list

def _dataset_name_map(dataset_name: str) -> Dict[int, str]:
    name = (dataset_name or "").lower()
    if "cifar10" in name:
        return {0:"airplane",1:"automobile",2:"bird",3:"cat",4:"deer",5:"dog",6:"frog",7:"horse",8:"ship",9:"truck"}
    # CIFAR-100 / ImageNet: names are long; default to "class_{id}"
    return {}

def _fmt_label_list(ids: List[int], name_map: Dict[int, str]) -> str:
    if not ids:
        return "[]"
    pieces = []
    for k in ids:
        nm = name_map.get(int(k))
        pieces.append(f"{k}" if nm is None else f"{k}:{nm}")
    return "[" + ", ".join(pieces) + "]"


# ---------- Main API ----------

def visualize_and_print_partition(server: Any, args: Any, fig_dir: str = "figures") -> Dict[str, Any]:
    """
    Build a full partition report from the actual datasets held by each client.
    Prints a colorized table to console and saves figures & files to fig_dir.
    Returns the report dict for further use.

    Console colors (with rich):
      - Header: yellow bold
      - "Client i:": cyan bold
      - assigned=[...]  (cyan)
      - present=[...(count)]  (green)
      - missing=[...]  (red)
      - (master_task=...)  (magenta)
    """

    # ---------- color console (rich) ----------
    try:
        from rich.console import Console
        _RICH = True
        console = Console()
    except Exception:
        _RICH = False
        console = None

    # ---- helper to fetch per-task meta stuffed by data_utils_mine.py ----
    def _get_task_meta(client: Any, tid: int) -> Optional[Dict[str, Any]]:
        for name in ("task_info", "task_meta", "task_label_info"):
            if hasattr(client, name):
                d = getattr(client, name)
                try:
                    return d.get(tid)
                except Exception:
                    pass
        if hasattr(client, "task_dict"):
            try:
                assigned = client.task_dict.get(tid)
                if assigned:
                    return {"assigned_labels": list(map(int, assigned))}
            except Exception:
                pass
        return None

    out_dir = _ensure_dir(fig_dir)
    num_clients = len(getattr(server, "clients", []))
    if num_clients == 0:
        print("[PartitionViz] No clients found on server; skipping.")
        return {}

    dataset_name = getattr(args, "dataset", "")
    name_map = _dataset_name_map(dataset_name)
    num_classes = getattr(args, "num_classes", None)

    report: Dict[str, Any] = {"dataset": dataset_name, "num_clients": num_clients, "clients": {}}
    max_tasks = 0
    global_class_ids: set = set()
    any_assigned_meta = False

    for cid, client in enumerate(server.clients):
        ds_list = _extract_task_datasets_from_client(client)
        if not ds_list:
            print(f"[PartitionViz] WARNING: Could not locate task datasets for client {cid}.")
        client_entry = {"tasks": {}}
        for tid, ds in enumerate(ds_list):
            present, counts = _dataset_unique_labels(ds)
            present = sorted(set(int(k) for k in present))
            for k in present: global_class_ids.add(int(k))

            meta = _get_task_meta(client, tid) or {}
            assigned = meta.get("assigned_labels") or meta.get("labels")
            if assigned is not None:
                assigned = list(map(int, assigned))
                any_assigned_meta = True
                missing = sorted(set(assigned) - set(present))
            else:
                missing = []

            client_entry["tasks"][tid] = {
                "present_labels": present,
                "assigned_labels": assigned,
                "missing_labels": missing,
                "counts": counts,
                "task_index_in_master": meta.get("task_index_in_master"),
            }
            max_tasks = max(max_tasks, tid + 1)
        report["clients"][str(cid)] = client_entry

    if num_classes is None:
        num_classes = (max(global_class_ids) + 1) if global_class_ids else 0
    report["num_classes_inferred"] = int(num_classes)

    # ---------- pretty formatting helpers ----------
    def _fmt_label_list_simple(lbls: List[int]) -> str:
        return "[" + ", ".join(str(int(k)) for k in (lbls or [])) + "]"

    def _fmt_labels_with_counts(lbls: List[int], cnts: Optional[Dict[int, int]]) -> str:
        if not lbls: return "[]"
        parts = []
        for k in lbls:
            k = int(k)
            c = cnts.get(k, 0) if cnts else None
            parts.append(f"{k}({c})" if c is not None else f"{k}")
        return "[" + ", ".join(parts) + "]"

    # ---------- 2) Colorized console + plain TXT ----------
    lines = []
    header = f"=== Data Partition Plan (dataset={dataset_name}, clients={num_clients}, classes={num_classes}) ==="
    if _RICH: console.print(f"[bold yellow]{header}[/]")
    else: print(header)
    lines.append(header)

    for cid in range(num_clients):
        if _RICH: console.print(f"[bold cyan]Client {cid}:[/]")
        else: print(f"Client {cid}:")
        lines.append(f"Client {cid}:")

        tasks = report["clients"].get(str(cid), {}).get("tasks", {})
        if not tasks:
            if _RICH: console.print("  (no tasks discovered)")
            else: print("  (no tasks discovered)")
            lines.append("  (no tasks discovered)")
            continue

        for tid in sorted(tasks):
            info = tasks[tid]
            present = info["present_labels"]
            counts  = info.get("counts") or {}
            assigned = info.get("assigned_labels")
            missing  = info.get("missing_labels") or []
            tm = info.get("task_index_in_master")

            if assigned is None:
                # no meta -> old behavior (present only)
                present_txt = _fmt_labels_with_counts(present, counts)
                if _RICH:
                    console.print(f"  Task {tid}: |present|={len(present)}  [green]{present_txt}[/]")
                else:
                    print(f"  Task {tid}: |present|={len(present)}  {present_txt}")
                lines.append(f"  Task {tid}: |present|={len(present)}  {present_txt}")
            else:
                assigned_txt = _fmt_label_list_simple(assigned)
                present_txt  = _fmt_labels_with_counts(present, counts)
                missing_txt  = _fmt_label_list_simple(missing)
                tm_txt = f"(master_task={tm}) " if tm is not None else ""

                # colored console
                if _RICH:
                    console.print(
                        f"  Task {tid}: [magenta]{tm_txt}[/]"
                        f"assigned=[cyan]{assigned_txt}[/]  |  "
                        f"present=[green]{present_txt}[/]  |  "
                        f"missing=[red]{missing_txt}[/]"
                    )
                else:
                    print(f"  Task {tid}: {tm_txt}assigned={assigned_txt}  |  present={present_txt}  |  missing={missing_txt}")

                # plain text line for file
                lines.append(
                    f"  Task {tid}: {tm_txt}assigned={assigned_txt}  |  present={present_txt}  |  missing={missing_txt}"
                )

    (out_dir / "partition_table.txt").write_text("\n".join(lines), encoding="utf-8")

    # ---------- 3) CSV (include ALL assigned labels with count=0 when present) ----------
    csv_lines = ["client,task,class_id,count,is_assigned,is_present,master_task_index"]
    for cid in range(num_clients):
        tasks = report["clients"].get(str(cid), {}).get("tasks", {})
        for tid, info in tasks.items():
            present = set(info["present_labels"])
            counts  = info.get("counts") or {}
            assigned = info.get("assigned_labels")
            tm = info.get("task_index_in_master")
            if assigned is None:
                for k in sorted(present):
                    csv_lines.append(f"{cid},{tid},{int(k)},{int(counts.get(int(k),0))},0,1,{'' if tm is None else tm}")
            else:
                for k in sorted(assigned):
                    cnt = int(counts.get(int(k), 0))
                    is_pres = 1 if int(k) in present else 0
                    csv_lines.append(f"{cid},{tid},{int(k)},{cnt},1,{is_pres},{'' if tm is None else tm}")
    (out_dir / "partition_table.csv").write_text("\n".join(csv_lines), encoding="utf-8")

    # ---------- 4) JSON ----------
    with open(out_dir / "partition_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # ---------- 5) Heatmaps (unchanged) ----------
    if num_classes and max_tasks:
        H = num_clients * max_tasks
        presence_present = np.zeros((H, num_classes), dtype=np.float32)
        presence_assigned = np.zeros((H, num_classes), dtype=np.float32) if any_assigned_meta else None
        counts_mat = np.zeros((H, num_classes), dtype=np.float32)
        ytick_labels = []
        row = 0
        for cid in range(num_clients):
            tasks = report["clients"].get(str(cid), {}).get("tasks", {})
            for tid in range(max_tasks):
                info = tasks.get(tid)
                ytick_labels.append(f"C{cid}-T{tid}")
                if info:
                    for k in info["present_labels"]:
                        if 0 <= int(k) < num_classes:
                            presence_present[row, int(k)] = 1.0
                    if presence_assigned is not None and info.get("assigned_labels"):
                        for k in info["assigned_labels"]:
                            if 0 <= int(k) < num_classes:
                                presence_assigned[row, int(k)] = 1.0
                    if info.get("counts"):
                        for k, v in info["counts"].items():
                            if 0 <= int(k) < num_classes:
                                counts_mat[row, int(k)] = float(v)
                row += 1

        def _save_heatmap(M: np.ndarray, title: str, path: Path, vmin: float, vmax: float):
            plt.figure(figsize=(min(18, 1 + num_classes * 0.25), min(2 + H * 0.25, 16)))
            plt.imshow(M, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax, cmap="Greys")
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.yticks(np.arange(H), ytick_labels, fontsize=6)
            if num_classes <= 50:
                plt.xticks(np.arange(num_classes), [str(i) for i in range(num_classes)], rotation=90, fontsize=6)
            else:
                step = max(1, num_classes // 50)
                ticks = np.arange(0, num_classes, step)
                plt.xticks(ticks, [str(i) for i in ticks], rotation=90, fontsize=6)
            plt.title(title); plt.tight_layout(); plt.savefig(path, dpi=250); plt.close()

        _save_heatmap(presence_present, "PRESENT: class presence by (client, task)",
                      out_dir / "partition_presence_present.png", 0.0, 1.0)
        if presence_assigned is not None:
            _save_heatmap(presence_assigned, "ASSIGNED: class presence by (client, task) from global task pool",
                          out_dir / "partition_presence_assigned.png", 0.0, 1.0)
        if counts_mat.max() > 0:
            norm_counts = counts_mat / (counts_mat.max(axis=1, keepdims=True) + 1e-8)
            _save_heatmap(norm_counts, "Per-(client,task) class counts (row-normalized)",
                          out_dir / "partition_counts_heatmap.png", 0.0, 1.0)
        else:
            _save_heatmap(presence_present, "Per-(client,task) class counts (binary only)",
                          out_dir / "partition_counts_heatmap.png", 0.0, 1.0)

    print(f"[PartitionViz] Wrote partition artifacts to: {out_dir.resolve()}")
    return report
