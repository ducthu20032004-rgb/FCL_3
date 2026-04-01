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

import time
import copy
import inspect
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import numpy as np
import wandb
from flcore.clients.clientala import clientALA
from flcore.servers.serverbase import Server
from flcore.metrics.average_forgetting import metric_average_forgetting

# Pretty logger (safe if not installed)
try:
    from utils.rich_progress import RichRoundLogger
except Exception:
    RichRoundLogger = None


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
                        from utils.data_utils import read_client_data_FCL_cifar100, read_client_data_FCL_imagenet1k, read_client_data_FCL_cifar10
                        if self.args.dataset == 'IMAGENET1k':
                            train_data, label_info = read_client_data_FCL_imagenet1k(i, task=task, classes_per_task=self.args.cpt, count_labels=True)
                        elif self.args.dataset == 'CIFAR100':
                            train_data, label_info = read_client_data_FCL_cifar100(i, task=task, classes_per_task=self.args.cpt, count_labels=True)
                        elif self.args.dataset == 'CIFAR10':
                            train_data, label_info = read_client_data_FCL_cifar10(i, task=task, classes_per_task=self.args.cpt, count_labels=True)
                        else:
                            raise NotImplementedError("Not supported dataset")
                    elif self.args.partition_options == 'hetero':
                        from utils.data_utils_mine import read_client_data_FCL_cifar10, read_client_data_FCL_cifar100, read_client_data_FCL_imagenet1k
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