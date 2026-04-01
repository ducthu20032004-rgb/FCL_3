"""
FedFCIL server with FedAvg-style printing and tracking, preserving FCIL's training algorithm.

What this adds (vs. your original FedFCIL):
  • RichRoundLogger (same look as FedAvg): per-round header, per-client table, summary panel.
  • Per-client AA@t (average accuracy up to current task) on the GLOBAL test pool.
  • Per-client Average Forgetting (so-far) with metric_average_forgetting.
  • Optional delta-L2 of local updates for quick sanity.
  • Dump a GLOBAL per-task accuracy row ONCE at the end of each task.

The FCIL training sequence (beforeTrain / update_new_set / client.train(ep_g, model_old),
proto_grad collection, aggregate_parameters, dataloader/monitor) is kept intact.
"""

import time
import copy
import inspect
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from flcore.clients.clientfcil import clientFCIL
from flcore.servers.serverbase import Server
from flcore.trainmodel.models import LeNet2, weights_init
from flcore.utils_core.fcil_utils import Proxy_Data
from utils.data_utils import get_unique_tasks
from flcore.metrics.average_forgetting import metric_average_forgetting

# Pretty logger (safe if not installed)
try:
    from utils.rich_progress import RichRoundLogger
except Exception:
    RichRoundLogger = None


class FedFCIL(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_clients(clientFCIL)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        self.Budget: List[float] = []

        # FCIL-specific state (kept from your original)
        self.pool_grad = None
        self.best_model_1 = None
        self.best_model_2 = None
        self.best_perf = 0
        self.unique_task: List[List[int]] = []
        self.old_unique_task: List[List[int]] = []
        self.encode_model = LeNet2(num_classes=self.num_classes)
        self.encode_model.apply(weights_init)
        self.cil = True

        # Per-client accuracy matrices over eval steps (rows appended at eval points)
        if not hasattr(self, "client_accuracy_matrix"):
            self.client_accuracy_matrix: Dict[int, List[List[float]]] = {}

        # Pretty round logger (FedAvg-style)
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

    # ----------------------- helpers (local) -----------------------
    def _cid(self, c, idx):
        return getattr(c, "id", getattr(c, "cid", idx))

    def _flatten_params(self, model: torch.nn.Module) -> Optional[torch.Tensor]:
        try:
            with torch.no_grad():
                return torch.cat([p.detach().float().view(-1).cpu() for p in model.parameters() if p is not None])
        except Exception:
            return None

    def _call_client_before_train(self, client, task_id: int, is_new_flag: int):
        """FCIL hook beforeTrain(task_id, flag) where flag=0/1 old/new client."""
        fn = getattr(client, "beforeTrain", None)
        if fn is None:
            return
        sig = inspect.signature(fn)
        kwargs = {}
        # FCIL uses (task_id, old_new_flag) positional; pass as such
        return fn(task_id, is_new_flag)

    def _call_client_update_new_set(self, client):
        fn = getattr(client, "update_new_set", None)
        if fn is not None:
            return fn()

    def _call_client_train(self, client, ep_g, model_old):
        """FCIL's client.train signature is (ep_g, model_old)."""
        fn = getattr(client, "train")
        return fn(ep_g, model_old)

    def _acc_vec_for_client_from_counts(self, client, cc: np.ndarray, tt: np.ndarray) -> List[float]:
        """
        Build [A_{*,k}]_{k=0..T-1} for THIS client from per-class (cc,tt) counts,
        using the server's robust label resolver for (client, task_idx).
        """
        vec: List[float] = []
        for k in range(self.N_TASKS):
            labels = self._labels_for_client_task(client, k)  # GLOBAL IDs
            if not labels:
                vec.append(0.0)
                continue
            idx = np.asarray(labels, dtype=np.int64)
            corr = int(cc[idx].sum())
            tot  = int(tt[idx].sum())
            vec.append((corr / tot) if tot > 0 else 0.0)
        return vec

    # ----------------------- training (FedAvg-like logging) -----------------------
    def train(self):
        """Preserve FCIL pipeline, add FedAvg-style progress/metrics."""
        if self.args.num_tasks % self.N_TASKS != 0:
            raise ValueError("Set num_task again")

        num_tasks = int(self.args.num_tasks)
        total_rounds = int(self.global_rounds) * num_tasks
        self._roundlog.start(total_rounds=total_rounds)
        self.Budget = []

        # old/new client bookkeeping from your original
        old_client_0: List[int] = []
        old_client_1: List[int] = [i for i in range(self.num_clients)]
        new_client: List[int] = []
        task_list: List[List[int]] = []

        for task in range(num_tasks):
            current_list, sofar_list = [], []
            print(f"\n================ Current Task: {task} =================")

            if task == 0:
                # initial label info
                available_labels = set()
                available_labels_current = set()
                available_labels_past = set()
                for u in self.clients:
                    available_labels |= set(u.classes_so_far)
                    available_labels_current |= set(u.current_labels)
                    sofar_list.append(u.classes_so_far)
                    current_list.append(u.current_labels)
                    task_list.append(u.current_labels)
                for u in self.clients:
                    u.available_labels = list(available_labels)
                    u.available_labels_current = list(available_labels_current)
                    u.available_labels_past = list(available_labels_past)
            else:
                # load next task data (FCIL uses read_client_data_FCL_* that returns (train, test, meta))
                torch.cuda.empty_cache()
                for i in range(len(self.clients)):
                    if self.args.dataset == 'IMAGENET1k':
                        train_data, test_data, label_info = read_client_data_FCL_imagenet1k(
                            i, task=task, classes_per_task=self.args.cpt, count_labels=True
                        )
                    elif self.args.dataset == 'CIFAR100':
                        train_data, test_data, label_info = read_client_data_FCL_cifar100(
                            i, task=task, classes_per_task=self.args.cpt, count_labels=True
                        )
                    else:
                        raise NotImplementedError("Not supported dataset")
                    # FCIL client.next_task expects (train, test, label_info)
                    self.clients[i].next_task(train_data, test_data, label_info)

                # update label tracking (as in your original)
                available_labels = set()
                available_labels_current = set()
                available_labels_past = self.clients[0].available_labels
                for u in self.clients:
                    available_labels |= set(u.classes_so_far)
                    available_labels_current |= set(u.current_labels)
                    sofar_list.append(u.classes_so_far)
                    current_list.append(u.current_labels)
                    task_list.append(u.current_labels)
                for u in self.clients:
                    u.available_labels = list(available_labels)
                    u.available_labels_current = list(available_labels_current)
                    u.available_labels_past = list(available_labels_past)

                # unique tasks & assignment (kept)
                self.old_unique_task = self.unique_task
                self.unique_task = get_unique_tasks(task_list)
                self.assign_unique_tasks()
                for u in self.clients:
                    u.assign_task_id(self.task_dict)

            # ----- inner rounds -----
            eval_gap = int(getattr(self, "eval_gap", 1) or 1)
            for i in range(self.global_rounds):
                glob_iter = i + self.global_rounds * task
                disp_round = glob_iter + 1
                self._round_tag = glob_iter
                t0_round = time.time()

                # FCIL originals
                pool_grad: List[Any] = []
                model_old = self.model_back()    # returns [best_model_1, best_model_2]
                task_id = task
                ep_g = (task * self.global_rounds + i)
                print('federated global round: {}, task_id: {}'.format(ep_g, task_id))

                # (1) select clients
                self.selected_clients = self.select_clients()
                sel_ids = [self._cid(c, j) for j, c in enumerate(self.selected_clients)]

                # round header
                self._roundlog.round_start(round_idx=disp_round, task_id=task, selected_clients=sel_ids)

                # (2) broadcast global model
                if hasattr(self, "send_models"):
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

                # build global per-class counts ONCE for this eval point (for AA/forgetting)
                try:
                    cc, tt = self._get_or_build_global_counts(self._round_tag)
                except Exception:
                    cc, tt, _K = self._compute_global_per_class_counts(self.global_model)

                # (4) local training for selected clients + metrics capture
                w_local = []
                client_summaries: List[Dict[str, Any]] = []
                for client in self.selected_clients:
                    t0_cli = time.time()

                    # before snapshot for delta
                    before = self._flatten_params(client.model)

                    # FCIL hooks
                    if client.id in old_client_0:
                        self._call_client_before_train(client, task_id, 0)
                    else:
                        self._call_client_before_train(client, task_id, 1)
                    self._call_client_update_new_set(client)

                    # FCIL local train
                    self._call_client_train(client, ep_g, model_old)

                    # collect local model
                    local_model = client.model.state_dict()
                    w_local.append(local_model)

                    # proto grad sharing (unchanged)
                    proto_grad = client.proto_grad_sharing()
                    if proto_grad is not None:
                        for grad_i in proto_grad:
                            pool_grad.append(grad_i)

                    # delta l2
                    delta_l2 = None
                    try:
                        after = self._flatten_params(client.model)
                        if before is not None and after is not None and after.shape == before.shape:
                            delta_l2 = float(torch.norm(after - before, p=2))
                    except Exception:
                        pass

                    # averaged training loss (if available)
                    train_loss = None
                    try:
                        tr_sum, tr_n = client.train_metrics(task=task)
                        train_loss = (tr_sum / max(1, tr_n)) if tr_n else None
                    except Exception:
                        pass

                    # AA up to task and forgetting (FedAvg-style)
                    acc_vec = self._acc_vec_for_client_from_counts(client, cc, tt)  # [A_k]
                    self.client_accuracy_matrix.setdefault(client.id, []).append(acc_vec)
                    aa_pct = float(100.0 * (np.mean(acc_vec[:task + 1]) if (task + 1) > 0 else 0.0))
                    try:
                        cf = metric_average_forgetting(int(task % self.N_TASKS), self.client_accuracy_matrix[client.id])
                        forg_pct = float(100.0 * cf)
                    except Exception:
                        forg_pct = None

                    client_summaries.append({
                        "client": client.id,
                        "loss": train_loss,
                        "acc": aa_pct,      # %
                        "forg": forg_pct,   # %
                        "time": time.time() - t0_cli,
                        "samples": getattr(client, "train_samples", None),
                        "delta_l2": delta_l2,
                    })

                # table
                self._roundlog.clients_end(round_idx=disp_round, client_summaries=client_summaries)

                # (5) aggregate parameters (FCIL sequence)
                if hasattr(self, "receive_models"): self.receive_models()
                w_g_last = copy.deepcopy(self.global_model)
                if hasattr(self, "aggregate_parameters"): self.aggregate_parameters()

                # (6) local eval at gap (as in your original)
                if i % eval_gap == 0:
                    try:
                        self.eval(task=task, glob_iter=glob_iter, flag="local")
                    except Exception:
                        pass

                # (7) FCIL dataloader/monitor steps
                self.dataloader(pool_grad)

                # (8) end-of-round summary
                elapsed = time.time() - t0_round
                self.Budget.append(elapsed)
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
                self._roundlog.round_end(round_idx=disp_round, global_metrics=g_metrics, time_cost=elapsed)

            # ---- end-of-task evals (keep your original) ----
            if int(task/self.N_TASKS) == int(self.args.num_tasks/self.N_TASKS-1):
                if self.args.offlog and not self.args.debug:
                    self.eval_task(task=task, glob_iter=glob_iter, flag="local")
            self.send_models()
            self.eval_task(task=task, glob_iter=glob_iter, flag="global")

            # ---- dump GLOBAL per-task accuracy row ONCE per task end ----
            try:
                if hasattr(self, "dump_global_task_accuracy_csv"):
                    self.dump_global_task_accuracy_csv(after_task=int(task), glob_iter=int(self.global_rounds * (task + 1) - 1))
            except Exception as e:
                print(f"[dump global_task_acc] warning: {e}")

        # finish progress
        self._roundlog.finish()

    # ---------------- FCIL original helpers (kept) ----------------
    def model_back(self):
        return [self.best_model_1, self.best_model_2]

    def dataloader(self, pool_grad):
        self.pool_grad = pool_grad
        if len(pool_grad) != 0:
            self.reconstruction()
        # (Optional) any monitoring your original pipeline expects
        # You can keep your monitor logic here if needed.