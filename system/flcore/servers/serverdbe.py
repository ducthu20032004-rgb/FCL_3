"""
FedDBE server with FedAvg-style printing and tracking, while preserving DBE's
training algorithm:

  • Initialization: build clients, run a warmup train(task=0), compute global_mean,
    set client.global_mean (kept exactly).
  • Round loop: select_clients → send_models → client.train(task)
    → receive_models → receive_grads → aggregate_parameters (DBE) + optional diagnostics.

Adds:
  • RichRoundLogger (same look as FedAvg): per-round header, per-client table, summary panel
  • Per-client AA@t (average accuracy up to current task) on the GLOBAL test pool
  • Per-client Average Forgetting (so-far) with metric_average_forgetting
  • Optional delta-L2 of local updates for a quick sanity check
  • (Optional) dump a GLOBAL per-task accuracy row once per task end (if helper exists)

Drop this in as: system/flcore/servers/serverdbe.py
"""

import time
import copy
import numpy as np
import torch
import statistics
from typing import Any, Dict, List, Optional

from flcore.clients.clientdbe import clientDBE
from flcore.servers.serverbase import Server
from flcore.metrics.average_forgetting import metric_average_forgetting

# Pretty logger (safe if not installed)
try:
    from utils.rich_progress import RichRoundLogger
except Exception:
    RichRoundLogger = None


class FedDBE(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # ----- DBE initialization period (kept as original) -----
        self.set_clients(clientDBE)
        self.selected_clients = self.clients

        # warm-up train for initial statistics
        for client in self.selected_clients:
            client.train(task=0)  # no DBE

        self.uploaded_ids = []
        self.uploaded_weights = []
        tot_samples = 0
        for client in self.selected_clients:
            tot_samples += len(client.train_data)
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(len(client.train_data))
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / max(1, tot_samples)

        global_mean = 0
        for cid, w in zip(self.uploaded_ids, self.uploaded_weights):
            global_mean += self.clients[cid].running_mean * w
        print('>>>> global_mean <<<<', global_mean)
        for client in self.selected_clients:
            client.global_mean = global_mean.data.clone()

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget: List[float] = []
        print('featrue map shape: ', self.clients[0].client_mean.shape)
        print('featrue map numel: ', self.clients[0].client_mean.numel())

        # FedAvg-style tracking state
        if not hasattr(self, "client_accuracy_matrix"):
            self.client_accuracy_matrix: Dict[int, List[List[float]]] = {}

        # Pretty round logger
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

    # ---------------- helpers ----------------
    def _cid(self, c, idx):
        return getattr(c, "id", getattr(c, "cid", idx))

    def _flatten_params(self, model: torch.nn.Module) -> Optional[torch.Tensor]:
        try:
            with torch.no_grad():
                return torch.cat([p.detach().float().view(-1).cpu() for p in model.parameters() if p is not None])
        except Exception:
            return None

    def _acc_vec_for_client_from_counts(self, client, cc: np.ndarray, tt: np.ndarray) -> List[float]:
        """
        Build [A_{*,k}]_{k=0..T-1} for THIS client from per-class (cc,tt) counts,
        using the server's robust label resolver for (client, task_idx).
        """
        vec: List[float] = []
        for k in range(self.N_TASKS):
            labels = self._labels_for_client_task(client, k)  # GLOBAL class IDs
            if not labels:
                vec.append(0.0)
                continue
            idx = np.asarray(labels, dtype=np.int64)
            corr = int(cc[idx].sum())
            tot  = int(tt[idx].sum())
            vec.append((corr / tot) if tot > 0 else 0.0)
        return vec

    # ---------------- training with FedAvg-style logging ----------------
    def train(self):
        # if self.args.num_tasks % self.N_TASKS != 0:
        #     raise ValueError("Set num_task again")

        num_tasks    = int(self.args.num_tasks)
        total_rounds = int(self.global_rounds) * num_tasks
        eval_gap     = int(getattr(self, "eval_gap", 1) or 1)

        self._roundlog.start(total_rounds=total_rounds)

        for task in range(self.args.num_tasks):

            print(f"\n================ Current Task: {task} =================")
            if task == 0:
                # first task: propagate label info
                available_labels = set()
                available_labels_current = set()
                available_labels_past = set()
                for u in self.clients:
                    available_labels |= set(u.classes_so_far)
                    available_labels_current |= set(u.current_labels)
                for u in self.clients:
                    u.available_labels = list(available_labels)
                    u.available_labels_current = list(available_labels_current)
                    u.available_labels_past = list(available_labels_past)

            else:
                self.current_task = task
                torch.cuda.empty_cache()
                for i in range(len(self.clients)):
                    if self.args.partition_options == 'tuan':
                        from utils.data_utils import read_client_data_FCL_cifar100, read_client_data_FCL_imagenet1k
                        if self.args.dataset == 'IMAGENET1k':
                            train_data, label_info = read_client_data_FCL_imagenet1k(i, task=task, classes_per_task=self.args.cpt, count_labels=True)
                        elif self.args.dataset == 'CIFAR100':
                            train_data, label_info = read_client_data_FCL_cifar100(i, task=task, classes_per_task=self.args.cpt, count_labels=True)
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

                    self.clients[i].next_task(train_data, label_info)

                # update label info
                available_labels = set()
                available_labels_current = set()
                available_labels_past = self.clients[0].available_labels
                for u in self.clients:
                    available_labels |= set(u.classes_so_far)
                    available_labels_current |= set(u.current_labels)
                for u in self.clients:
                    u.available_labels = list(available_labels)
                    u.available_labels_current = list(available_labels_current)
                    u.available_labels_past = list(available_labels_past)

            # ============ train rounds =============
            for i in range(self.global_rounds):
                glob_iter  = i + self.global_rounds * task
                disp_round = glob_iter + 1
                self._round_tag = glob_iter
                s_t = time.time()

                # (1) select clients & header
                self.selected_clients = self.select_clients()
                sel_ids = [self._cid(c, j) for j, c in enumerate(self.selected_clients)]
                self._roundlog.round_start(round_idx=disp_round, task_id=task, selected_clients=sel_ids)

                # (2) broadcast global model
                self.send_models()

                # (3) optional global eval
                if i % eval_gap == 0 and hasattr(self, "eval"):
                    try:
                        self.eval(task=task, glob_iter=glob_iter, flag="global")
                    except TypeError:
                        try:
                            self.eval()
                        except Exception:
                            pass

                # Build global per-class counts ONCE for this eval point (AA/forgetting)
                try:
                    cc, tt = self._get_or_build_global_counts(self._round_tag)
                except Exception:
                    cc, tt, _K = self._compute_global_per_class_counts(self.global_model)

                # (4) local training for selected clients + metrics capture
                client_summaries: List[Dict[str, Any]] = []
                for client in self.selected_clients:
                    t0_cli = time.time()
                    before = self._flatten_params(client.model)

                    client.train(task=task)

                    delta_l2 = None
                    try:
                        after = self._flatten_params(client.model)
                        if before is not None and after is not None and after.shape == before.shape:
                            delta_l2 = float(torch.norm(after - before, p=2))
                    except Exception:
                        pass

                    # averaged training loss (if exposed)
                    train_loss = None
                    try:
                        tr_sum, tr_n = client.train_metrics(task=task)
                        train_loss = (tr_sum / max(1, tr_n)) if tr_n else None
                    except Exception:
                        pass

                    # AA up to task and forgetting
                    acc_vec = self._acc_vec_for_client_from_counts(client, cc, tt)
                    self.client_accuracy_matrix.setdefault(client.id, []).append(acc_vec)
                    aa_pct = float(100.0 * (np.mean(acc_vec[:task + 1]) if (task + 1) > 0 else 0.0))
                    try:
                        cf = metric_average_forgetting(int(task % self.N_TASKS), self.client_accuracy_matrix[client.id])
                        forg_pct = float(100.0 * cf)
                    except Exception:
                        forg_pct = None

                    client_summaries.append({
                        "client": client.id,
                        "loss":   train_loss,
                        "acc":    aa_pct,      # %
                        "forg":   forg_pct,    # %
                        "time":   time.time() - t0_cli,
                        "samples": getattr(client, "train_samples", None),
                        "delta_l2": delta_l2,
                    })

                # show per-client table
                self._roundlog.clients_end(round_idx=disp_round, client_summaries=client_summaries)

                # (5) aggregate (DBE sequence)
                self.receive_models()
                self.receive_grads()
                model_origin = copy.deepcopy(self.global_model)
                self.aggregate_parameters()

                # optional diagnostics
                if getattr(self.args, "seval", False) and hasattr(self, "spatio_grad_eval"):
                    try:
                        self.spatio_grad_eval(model_origin=model_origin, glob_iter=glob_iter)
                    except Exception:
                        pass
                if getattr(self.args, "pca_eval", False) and hasattr(self, "proto_eval"):
                    try:
                        self.proto_eval(model=self.global_model, task=task, round=i)
                    except Exception:
                        pass

                # end-of-round summary
                self.Budget.append(time.time() - s_t)
                g_metrics = {}
                if i % eval_gap == 0 and hasattr(self, "test"):
                    try:
                        m = self.test()
                        if isinstance(m, dict):
                            g_metrics = {
                                "test_loss": m.get("test_loss", m.get("loss")),
                                "test_acc":  m.get("test_acc",  m.get("acc")),
                            }
                    except Exception:
                        pass
                self._roundlog.round_end(round_idx=disp_round, global_metrics=g_metrics, time_cost=self.Budget[-1])

        self._roundlog.finish()
