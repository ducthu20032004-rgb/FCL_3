"""
FedALA server with the SAME progress/metrics printing/tracking as FedAvg,
while preserving FedALA's original training algorithm (local init + ALA train).

- Uses RichRoundLogger if available (same look as FedAvg).
- Shows per-round header, per-client table (loss/acc/time/samples), and summary panel.
- Computes per-client AA (average accuracy up to current task) on the GLOBAL test pool.
- Tracks per-client forgetting with metric_average_forgetting (like FedAvg).
- Writes the global per-task accuracy row once per task end (if helper exists).

Paste over your existing FedALA in: system/flcore/servers/serverala.py
"""

import os
import time
import copy
import inspect

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import numpy as np
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from system.utils.data_utils import read_client_data_FCL_cifar100, read_client_data_FCL_imagenet1k, read_client_data_FCL_cifar10
from system.measure_gpu1 import get_resnet18_blocks,DEVICE,compute_eps, compute_cka, compute_alignment_from_arrays
from system.flcore.clients.clientala import clientALA

from system.flcore.servers.serverbase import Server

from system.flcore.metrics.average_forgetting import (
    metric_average_forgetting
)

# ---------- Pretty logger (safe if not installed) ----------
try:
    from system.utils.rich_progress import RichRoundLogger
except Exception:
    RichRoundLogger = None
def _make_loader(dataset, batch_size: int = 256):
    """
    Tao DataLoader an toan, xu ly moi kieu tra ve cua read_client_data_FCL_cifar10:

      Case 1 - torch.utils.data.Dataset chuan  -> dung truc tiep
      Case 2 - tuple/list 2 phan tu (X, Y) voi X,Y la array/tensor (N,...) -> TensorDataset
      Case 3 - list of (x_i, y_i) sample tuples -> stack roi TensorDataset
    num_workers=0 de tranh loi pickle / seek khi data da duoc load san vao RAM.
    """
    from torch.utils.data import TensorDataset, Dataset

    # Case 1: torch.utils.data.Dataset chuan
    if isinstance(dataset, Dataset):
        return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                          num_workers=0, pin_memory=(DEVICE.type == 'cuda'))

    # Case 2: (X, Y) - moi phan tu la array/tensor ca batch
    # Nhan dien: co dung 2 phan tu va phan tu dau co >= 2 chieu (batch dim + feature dims)
    if (isinstance(dataset, (tuple, list))
            and len(dataset) == 2
            and hasattr(dataset[0], 'shape')
            and len(np.shape(dataset[0])) >= 2):
        X, Y = dataset
        xs = torch.as_tensor(np.array(X, dtype=np.float32))
        ys = torch.as_tensor(np.array(Y)).long()
        return DataLoader(TensorDataset(xs, ys), batch_size=batch_size, shuffle=False,
                          num_workers=0, pin_memory=(DEVICE.type == 'cuda'))

    # Case 3: list of (x_i, y_i) sample tuples
    xs, ys = [], []
    for x, y in dataset:
        xs.append(torch.as_tensor(np.array(x, dtype=np.float32)))
        ys.append(torch.as_tensor(np.array(y)).long())
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    return DataLoader(TensorDataset(xs, ys), batch_size=batch_size, shuffle=False,
                      num_workers=0, pin_memory=(DEVICE.type == 'cuda'))
def compute_feature_resnet18_wrap(_model, _model_task_index, _dataset, _target_layer_index: str, seed, args):
    """
    Trích xuất features trên GPU, trả về numpy array (N, D).
    Chấp nhận Dataset chuẩn hoặc list of (x, y) tuples.
    """

    # ===== unwrap BaseHeadSplit =====
    backbone = _model.base if hasattr(_model, "base") else _model

    blocks = get_resnet18_blocks(backbone)

    backbone.eval()

    outputs = []

    loader = _make_loader(_dataset, batch_size=256)

    with torch.no_grad():
        for features, targets in tqdm(
            loader,
            desc=f'Feature M_{_model_task_index}^{_target_layer_index}',
            disable=True
        ):
            features = features.to(DEVICE, non_blocking=True)

            for block_name, operations in blocks.items():
                features = operations(features)

                if block_name == _target_layer_index:
                    break

            outputs.append(torch.flatten(features, 1).cpu())

    return torch.cat(outputs, dim=0).numpy()
class FedALA(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_clients(clientALA)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        self.Budget: List[float] = []
        # Per-client accuracy matrix over time (rows appended at eval points)
        if not hasattr(self, "client_accuracy_matrix"):
            self.client_accuracy_matrix: Dict[int, List[List[float]]] = {}

        # Ensure pretty round logger exists (FedAvg-style)
        if not hasattr(self, "_roundlog") or self._roundlog is None:
            if RichRoundLogger is not None:
                fig_dir = getattr(self.args, "fig_dir", "figures")
                pretty_on = getattr(self.args, "pretty_log", True)
                self._roundlog = RichRoundLogger(self.args, fig_dir=fig_dir, enable=pretty_on)
            else:
                class _Dummy:
                    def start(self, total_rounds): print(f"[Progress] total planned rounds: {total_rounds}")
                    def round_start(self, round_idx, task_id, selected_clients):
                        print(f"\n=== Round {round_idx} | Task {task_id} | Clients {selected_clients} ===")
                    def clients_end(self, round_idx, client_summaries): pass
                    def round_end(self, round_idx, global_metrics, time_cost=None):
                        print(f"[Summary] round={round_idx} time={time_cost:.2f}s")
                    def finish(self): pass
                self._roundlog = _Dummy()

    # ------------ helpers (local to this class, mirror FedAvg.train) ------------
    def _cid(self, c, idx):
        return getattr(c, "id", getattr(c, "cid", idx))

    def _flatten_params(self, model: torch.nn.Module) -> Optional[torch.Tensor]:
        try:
            with torch.no_grad():
                return torch.cat([p.detach().float().view(-1).cpu() for p in model.parameters() if p is not None])
        except Exception:
            return None

    def _call_client_train(self, client, task: int, round_idx: int, glob_iter: int):
        """Signature-aware call to client.train(...), passing only supported kwargs."""
        fn = getattr(client, "train")
        sig = inspect.signature(fn)
        kwargs = {}
        if "task" in sig.parameters: kwargs["task"] = task
        if "round" in sig.parameters: kwargs["round"] = round_idx
        if "rnd" in sig.parameters:   kwargs["rnd"] = round_idx
        if "epoch" in sig.parameters: kwargs["epoch"] = round_idx
        if "glob_iter" in sig.parameters: kwargs["glob_iter"] = glob_iter
        return fn(**kwargs)

    def _acc_vec_for_client_from_counts(self, client, cc: np.ndarray, tt: np.ndarray) -> List[float]:
        """
        Build [A_{*,k}]_{k=0..T-1} for THIS client from per-class (cc,tt) counts.
        Uses the server's robust label resolver for (client, task_idx).
        """
        vec: List[float] = []
        for k in range(self.N_TASKS):
            labels = self._labels_for_client_task(client, k)  # global IDs
            if not labels:
                vec.append(0.0)
                continue
            idx = np.asarray(labels, dtype=np.int64)
            corr = int(cc[idx].sum())
            tot = int(tt[idx].sum())
            vec.append((corr / tot) if tot > 0 else 0.0)
        return vec

    # ------------ FedALA training with FedAvg-style logging/metrics ------------
    def train(self):
        # Total "progress bar" rounds = global_rounds * num_tasks
        num_tasks = int(getattr(self.args, "num_tasks", getattr(self.args, "nt", 1)))
        total_rounds = int(self.global_rounds) * num_tasks
        self._roundlog.start(total_rounds=total_rounds)
        self.Budget = []

        # ---- Task loop ----
        for task in range(num_tasks):
            self.current_task = task
            torch.cuda.empty_cache()

            # (A) Load/advance per-client task data (preserve original FedALA behavior)
            if task == 0:
                # First task: just propagate available labels as in your original code
                available_labels = set()
                available_labels_current = set()
                available_labels_past = set()
                for u in self.clients:
                    available_labels = available_labels.union(set(u.classes_so_far))
                    available_labels_current = available_labels_current.union(set(u.current_labels))
                for u in self.clients:
                    u.available_labels = list(available_labels)
                    u.available_labels_current = list(available_labels_current)
                    u.available_labels_past = list(available_labels_past)
            else:
                # Read next task data according to partition_options (current/mine)

                for i in range(len(self.clients)):
                    if self.args.partition_options == 'tuan':
                        #from system.utils.data_utils import read_client_data_FCL_cifar100, read_client_data_FCL_imagenet1k, read_client_data_FCL_cifar10
                        if self.args.dataset == 'IMAGENET1k':
                            train_data, label_info = read_client_data_FCL_imagenet1k(i, task=task, classes_per_task=self.args.cpt, count_labels=True)
                        elif self.args.dataset == 'CIFAR100':
                            train_data, label_info = read_client_data_FCL_cifar100(i, task=task, classes_per_task=self.args.cpt, count_labels=True)
                        elif self.args.dataset == 'CIFAR10':
                            train_data, label_info = read_client_data_FCL_cifar10(i, task=task, classes_per_task=self.args.cpt, count_labels=True)
                        else:
                            raise NotImplementedError("Not supported dataset")
                    elif self.args.partition_options == 'hetero':
                        from system.utils.data_utils_mine import read_client_data_FCL_cifar10, read_client_data_FCL_cifar100, read_client_data_FCL_imagenet1k
                        if self.args.dataset == 'IMAGENET1k':
                            train_data, label_info = read_client_data_FCL_imagenet1k(i, task=task, classes_per_task=self.args.cpt, count_labels=True,
                                                                                    seed = self.args.seed, alpha = self.args.alpha,
                                                                                    total_clients = self.args.num_clients,task_disorder = self.args.task_disorder)
                        elif self.args.dataset == 'CIFAR100':
                            train_data, label_info = read_client_data_FCL_cifar100(i, task=task, classes_per_task=self.args.cpt, count_labels=True,
                                                                                seed = self.args.seed, alpha = self.args.alpha,
                                                                                total_clients = self.args.num_clients,task_disorder = self.args.task_disorder)
                        elif self.args.dataset == 'CIFAR10':
                            train_data, label_info = read_client_data_FCL_cifar10(i, task=task, classes_per_task=self.args.cpt, count_labels=True,
                                                                                seed = self.args.seed, alpha = self.args.alpha,
                                                                                total_clients = self.args.num_clients,task_disorder = self.args.task_disorder)
                        else:
                            raise NotImplementedError("Not supported dataset")

                    # Update client for the new task
                    self.clients[i].next_task(train_data, label_info)

                # Update label availability trackers (as in original)
                available_labels = set()
                available_labels_current = set()
                available_labels_past = self.clients[0].available_labels
                for u in self.clients:
                    available_labels = available_labels.union(set(u.classes_so_far))
                    available_labels_current = available_labels_current.union(set(u.current_labels))
                for u in self.clients:
                    u.available_labels = list(available_labels)
                    u.available_labels_current = list(available_labels_current)
                    u.available_labels_past = list(available_labels_past)
            init_losses = []
            trained_losses = []
            # ---- Rounds inside this task ----
            for i in range(self.global_rounds):
                glob_iter = i + self.global_rounds * task  # 0-based global counter
                disp_round = glob_iter + 1                 # pretty 1-based
                self._round_tag = glob_iter
                t0_round = time.time()

                # (1) Select clients
                self.selected_clients = self.select_clients()
                sel_ids = [self._cid(c, j) for j, c in enumerate(self.selected_clients)]

                # (2) Round header (FedAvg-style)
                self._roundlog.round_start(round_idx=disp_round, task_id=task, selected_clients=sel_ids)

                # (3) Broadcast global model (FedALA uses local_initialization in send_models)
                if hasattr(self, "send_models"):
                    self.send_models()
                model_global_before = copy.deepcopy(self.global_model)  # for drift eval
                # (4) Optional global eval at gap (same as FedAvg)
                eval_gap = int(getattr(self, "eval_gap", 1) or 1)
                if i % eval_gap == 0 and hasattr(self, "eval"):
                    try:
                        #print("wandb step =", glob_iter)
                        self.eval(task=task, glob_iter=glob_iter, flag="local")
                    except TypeError:
                        try:
                            self.eval()
                        except Exception:
                            pass

                # Build per-class counts ONCE for this eval point (for AA/forgetting)
                # (works with back-compat shim in serverbase)
                try:
                    cc, tt = self._get_or_build_global_counts(self._round_tag)
                except Exception:
                    # Worst-case: compute fresh
                    cc, tt, _K = self._compute_global_per_class_counts(self.global_model)

                # (5) Local training on selected clients + metrics capture (FedAvg-style)
                client_summaries: List[Dict[str, Any]] = []
                for j, client in enumerate(self.selected_clients):
                    t0_cli = time.time()
                    # 🔴 loss sau ALA, trước local train
                    if hasattr(client, "init_local_loss") and client.init_local_loss is not None:
                        init_losses.append(client.init_local_loss)
                    # before-train snapshot for delta
                    before = self._flatten_params(client.model)

                    # train with signature-aware args
                    try:
                        _ = self._call_client_train(client, task=task, round_idx=i, glob_iter=glob_iter)
                        # ── Lưu checkpoint sau mỗi round ──────────────────────────
                        try:
                            client._model_after_local = copy.deepcopy(client.model)
                        except Exception:
                            pass
                        try:
                            cid = self._cid(client, j)
                            save_dir = "/home/ghostm211/Thu/FCL_3/ALA/weight_round/"
                            os.makedirs(save_dir, exist_ok=True)
                            save_path = os.path.join(
                                save_dir,
                                f"client_{cid}_task_{task}_round_{i}.pt"
                            )
                            torch.save(client.model.state_dict(), save_path)
                            print(f"[CKPT] client={cid} task={task} round={i} → {save_path}")
                        except Exception as save_err:
                            print(f"[CKPT ERROR] client={cid} task={task} round={i}: {save_err}")
                        # ──────────────────────────────────────────────────────────

                    except Exception as e:
                        _ = {"error": str(e)}
                        # 🟢 loss sau khi local train xong
                    if hasattr(client, "trained_local_loss") and client.trained_local_loss is not None:
                        trained_losses.append(client.trained_local_loss)
                    # after-train delta l2
                    delta_l2 = None
                    try:
                        after = self._flatten_params(client.model)
                        if before is not None and after is not None and after.shape == before.shape:
                            delta_l2 = float(torch.norm(after - before, p=2))
                    except Exception:
                        pass

                    # averaged training loss from client (if available)
                    train_loss = None
                    try:
                        tr_sum, tr_n = client.train_metrics(task=task)  # (sum_loss, num_samples)
                        train_loss = (tr_sum / max(1, tr_n)) if tr_n else None
                    except Exception:
                        pass

                    # Client AA up to current task (use counts-based vector)
                    acc_vec = self._acc_vec_for_client_from_counts(client, cc, tt)
                    # append to client matrix for forgetting metric
                    self.client_accuracy_matrix.setdefault(client.id, []).append(acc_vec)
                    # AA up to current task index
                    if task >= 0:
                        aa_pct = float(100.0 * (np.mean(acc_vec[:task + 1]) if (task + 1) > 0 else 0.0))
                    else:
                        aa_pct = float(100.0 * (np.mean(acc_vec) if len(acc_vec) else 0.0))
                    #print(f"Serverala : Client {client.id} AA% up to task {task}: {aa_pct}")
                    # Forgetting (uses your existing helper)
                    try:
                        cf = metric_average_forgetting(int(task % self.N_TASKS), self.client_accuracy_matrix[client.id])
                        forg_pct = float(100.0 * cf)
                    except Exception:
                        forg_pct = None
                    #print(f"Serverala : Client {client.id} Average Forgetting% so far: {forg_pct}")
                    client_summaries.append({
                        "client": sel_ids[j],
                        "loss": train_loss,
                        "acc": aa_pct,           # (%)
                        "forg": forg_pct,        # (%), Rich table may ignore if not configured
                        "time": time.time() - t0_cli,
                        "samples": getattr(client, "train_samples", None),
                        "delta_l2": delta_l2,
                    })

                # Show per-client table (FedAvg-style)
                self._roundlog.clients_end(round_idx=disp_round, client_summaries=client_summaries)
                # tự viết 
                if self.args.wandb and len(init_losses) > 0 and len(trained_losses) > 0:
                    avg_init_loss = float(sum(init_losses) / len(init_losses))
                    avg_trained_loss = float(sum(trained_losses) / len(trained_losses))

                    wandb.log({
                        "FedALA/Avg Init Local Loss (Before Train)": avg_init_loss,      # 🔴
                        "FedALA/Avg Trained Local Loss (After Train)": avg_trained_loss  # 🟢
                    }, step=glob_iter)

                # (6) Aggregate (FedALA sequence preserved)
                if hasattr(self, "receive_models"):
                    self.receive_models()
                if hasattr(self, "receive_grads"):
                    self.receive_grads()
                model_origin = copy.deepcopy(self.global_model)
                if hasattr(self, "aggregate_parameters"):
                    self.aggregate_parameters()
                elif hasattr(self, "aggregate"):
                    self.aggregate()
                model_global_after = self.global_model  # alias for eval hooks

                if getattr(self.args, "measure_drift", True):
                    print("Drift thanh cong")
                    try:
                        # dict lưu kết quả per client per block
                        drift_results = {}  # client.id -> {block_idx -> {metric: value}}

                        for j, client in enumerate(self.selected_clients):

                            if not hasattr(client, "_model_after_local"):
                                continue
                            if client.id in [5,6,7,8,9]:
                                continue
                            test_data = read_client_data_FCL_cifar10(
                                client.id, task=task,
                                classes_per_task=self.args.cpt,
                                count_labels=False, train=False
                            )

                            drift_results[client.id] = {}

                            for block_idx in [4]:
                                target_layer = f'block{block_idx}'
                                try:
                                    feat_global = compute_feature_resnet18_wrap(
                                        model_global_before, task, test_data, target_layer,
                                        self.args.seed, self.args)
                                    feat_local = compute_feature_resnet18_wrap(
                                        client._model_after_local, task, test_data, target_layer,
                                        self.args.seed, self.args)
                                    feat_aggre = compute_feature_resnet18_wrap(
                                        model_global_after, task, test_data, target_layer,
                                        self.args.seed, self.args)

                                    drift_trained  = compute_eps(feat_global, feat_local)
                                    drift_aggre    = compute_eps(feat_local,  feat_aggre)
                                    drift_global = compute_eps(feat_global,feat_aggre)
                                    # _,cka_trained    = compute_cka(feat_global, feat_local)
                                    # _,cka_aggre      = compute_cka(feat_local,  feat_aggre)
                                    #_,cka_global = compute_cka(feat_global,feat_aggre)
                                    cknna_trained,_  = compute_alignment_from_arrays(feat_global, feat_local, "mutual_knn", topk=10, precise=True)
                                    cknna_aggre ,_   = compute_alignment_from_arrays(feat_local,  feat_aggre, "mutual_knn", topk=10, precise=True)
                                    cknna_global,_ = compute_alignment_from_arrays(feat_global,feat_aggre, "mutual_knn", topk=10, precise=True)
                                    drift_results[client.id][block_idx] = {
                                        "drift_trained": drift_trained,
                                        "drift_aggre":   drift_aggre,
                                        "drift_global":drift_global,
                                        # "cka_trained":   cka_trained,
                                        # "cka_aggre":     cka_aggre,
                                        #"cka_global":cka_global,
                                        "cknna_global":cknna_global,
                                        "cknna_trained": cknna_trained,
                                        "cknna_aggre":   cknna_aggre,

                                    }

                                    # print per block per client
                                    print(
                                        f"[Drift] round={disp_round} task={task} client={client.id} {target_layer} | "
                                        # f"drift_trained={drift_trained:.4f} drift_aggre={drift_aggre:.4f} | drift_global = {drift_global} |"
                                        # f"cka_trained={cka_trained:.4f} cka_aggre={cka_aggre:.4f} | cka_global = {cka_global} |"
                                        f"cknna_trained={cknna_trained:.4f} cknna_aggre={cknna_aggre:.4f} cknna_global = {cknna_global}"
                                    )

                                except Exception as e:
                                    print(f"[Drift] client={client.id} block{block_idx} error: {e}")
                                    drift_results[client.id][block_idx] = None

                            # append vào client_summaries đúng client
                            block_summary = {}
                            for block_idx in [4]:
                                r = drift_results[client.id].get(block_idx)
                                if r is None:
                                    continue
                                for metric_name, val in r.items():
                                    block_summary[f"block{block_idx}/{metric_name}"] = val
                            client_summaries[j].update(block_summary)

                        # wandb log: per-client per block per metric
                        if getattr(self.args, "wandb", True):
                            # Determine whether to log this round
                            log_every_round = getattr(self.args, "wandb_drift_every_round", True)  # test mode
                            final_round_of_task = 24  # production: only log at round 24
                            
                            should_log = log_every_round or (i == final_round_of_task)
                            
                            if should_log and i == 0:  # keep your existing round==0 guard
                                for cid in drift_results:
                                    for block_idx in [4]:
                                        if drift_results[cid].get(block_idx) is None:
                                            continue
                                        wb_drift = {"round": disp_round, "task": task, "client": cid}
                                        for metric_name in ["drift_trained", "drift_aggre", "cka_trained", "cka_aggre"]:
                                            if metric_name in drift_results[cid][block_idx]:
                                                wb_drift[f"drift/client{cid}/block{block_idx}/{metric_name}"] = float(drift_results[cid][block_idx][metric_name])
                                        wandb.log(wb_drift)

                    except Exception as e:
                        print(f"[Drift measure] warning: {e}")

                        import traceback; traceback.print_exc()   
                    # === Ghi CSV (thay thế / bổ sung wandb) ===
                    import csv, os

                    csv_path = "/home/ghostm211/Thu/FCL_3/ALA/drift_results.csv"
                    # fieldnames = ["round", "task", "client", "block",
                    #             "drift_trained", "drift_aggre", "drift_global",
                    #             "cka_trained", "cka_aggre", "cka_global"]
                    fieldnames = ["round", "task", "client", "block",
                                  "cka_global","cknna_trained","cknna_aggre","cknna_global"]
                    write_header = not os.path.exists(csv_path)

                    with open(csv_path, "a", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        if write_header:
                            writer.writeheader()

                        for cid in drift_results:
                            for block_idx in [4]:
                                r = drift_results[cid].get(block_idx)
                                if r is None:
                                    continue
                                writer.writerow({
                                    "round":         disp_round,
                                    "task":          task,
                                    "client":        cid,
                                    "block":         block_idx,
                                    # "drift_trained": round(float(r.get("drift_trained", float("nan"))), 6),
                                    # "drift_aggre":   round(float(r.get("drift_aggre",   float("nan"))), 6),
                                    # "drift_global":  round(float(r.get("drift_global",  float("nan"))), 6),
                                    # "cka_trained":   round(float(r.get("cka_trained",   float("nan"))), 6),
                                    # "cka_aggre":     round(float(r.get("cka_aggre",     float("nan"))), 6),
                                    "cka_global":    round(float(r.get("cka_global",    float("nan"))), 6),
                                    "cknna_trained": round(float(r.get("cknna_trained", float("nan"))), 6),
                                    "cknna_aggre":   round(float(r.get("cknna_aggre",   float("nan"))), 6),
                                    "cknna_global":  round(float(r.get("cknna_global",  float("nan"))), 6)
                                })                    
                # (7) Extras (as in FedAvg / your original)
                if getattr(self.args, "seval", False) and hasattr(self, "spatio_grad_eval"):
                    try:
                        self.spatio_grad_eval(model_origin=model_origin, glob_iter=glob_iter)
                    except Exception:
                        pass
                if getattr(self.args, "pca_eval", False) and hasattr(self, "proto_eval") and getattr(self, "uploaded_models", None):
                    try:
                        self.proto_eval(global_model=self.global_model, local_model=self.uploaded_models[0], task=task, round=i)
                    except Exception:
                        pass

                # (8) End-of-round summary
                elapsed = time.time() - t0_round
                self.Budget.append(elapsed)
                g_metrics = {}
                if i % eval_gap == 0 and hasattr(self, "test"):
                    try:
                        m = self.test()
                        if isinstance(m, dict):
                            g_metrics = {
                                "test_loss": m.get("test_loss", m.get("loss")),
                                "test_acc": m.get("test_acc", m.get("acc")),
                            }
                    except Exception:
                        pass
                self._roundlog.round_end(round_idx=disp_round, global_metrics=g_metrics, time_cost=elapsed)

            # ---- End of task: dump global per-task accuracy row (once) ----
            try:
                if hasattr(self, "dump_global_task_accuracy_csv"):
                    self.dump_global_task_accuracy_csv(after_task=int(task), glob_iter=int(self.global_rounds * (task + 1) - 1))
                elif hasattr(self, "dump_client_task_accuracy_csv"):
                    # Backward compatibility: per-client vectors also acceptable
                    self.dump_client_task_accuracy_csv(after_task=int(task), glob_iter=int(self.global_rounds * (task + 1) - 1))
            except Exception as e:
                print(f"[dump task-acc] warning: {e}")

        # finish progress
        self._roundlog.finish()

    # -------- FedALA-specific: broadcast global model by local initialization --------
    def send_models(self):
        assert (len(self.clients) > 0)
        for client in self.clients:
            client.model.to(self.device)
            # FedALA's local initialization step
            client.local_initialization(self.global_model)