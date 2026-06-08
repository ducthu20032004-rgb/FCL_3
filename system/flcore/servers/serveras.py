# system/flcore/servers/serveras.py
# -*- coding: utf-8 -*-
"""
FedAS server with FedAvg-style printing and tracking, while preserving FedAS's
original training algorithm:

  select_clients → send_selected_models(selected_ids, epoch, task)
  → each client trains with flag (selected or not) → receive_models → receive_grads
  → aggregate_wrt_fisher (FIM-weighted) → (optional) eval & diagnostics.

Adds:
  • RichRoundLogger (same look as FedAvg): per-round header, per-client table, summary panel
  • Per-client AA@t (average accuracy up to current task) on the GLOBAL test pool
  • Per-client Average Forgetting (so-far) with metric_average_forgetting
  • Optional delta-L2 of local updates for a quick sanity check
  • (Optional) dump a GLOBAL per-task accuracy row once per task end (if helper exists)

Training behavior and aggregation remain unchanged.
"""

import time
import copy
import numpy as np
import torch
import statistics
from typing import Any, Dict, List, Optional
from torch.utils.data import DataLoader
from tqdm import tqdm
from system.utils.data_utils import read_client_data_FCL_cifar10
from system.flcore.clients.clientas import clientAS
from system.flcore.servers.serverbase import Server
from system.flcore.metrics.average_forgetting import metric_average_forgetting

# Pretty logger (safe if not installed)
try:
    from system.utils.rich_progress import RichRoundLogger
except Exception:
    RichRoundLogger = None
from system.measure_gpu1 import *
import statistics

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
class FedAS(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_clients(clientAS)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget: List[float] = []

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
    def all_clients(self):
        return self.clients

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
                vec.append(0.0); continue
            idx = np.asarray(labels, dtype=np.int64)
            corr = int(cc[idx].sum())
            tot  = int(tt[idx].sum())
            vec.append((corr / tot) if tot > 0 else 0.0)
        return vec

    # ---------------- FedAS-specific API (kept) ----------------
    def send_selected_models(self, selected_ids, epoch, task):
        assert (len(self.clients) > 0)
        for client in [client for client in self.clients if (client.id in selected_ids)]:
            start_time = time.time()
            progress = (epoch + 1) / self.global_rounds
            client.set_parameters(self.global_model, progress, task)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def aggregate_wrt_fisher(self):
        assert (len(self.uploaded_models) > 0)
        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for p in self.global_model.parameters():
            p.data.zero_()
        # FIM-based weights
        FIM_weight_list = [self.clients[id].fim_trace_history[-1] for id in self.uploaded_ids]
        FIM_weight_list = [w / (sum(FIM_weight_list) + 1e-12) for w in FIM_weight_list]
        for w, client_model in zip(FIM_weight_list, self.uploaded_models):
            self.add_parameters(w, client_model)

    # ---------------- training with FedAvg-style logging ----------------
    def train(self):

        if self.args.num_tasks % self.N_TASKS != 0:
            raise ValueError("Set num_task again")

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
                        from system.utils.data_utils import read_client_data_FCL_cifar100, read_client_data_FCL_imagenet1k
                        if self.args.dataset == 'IMAGENET1k':
                            train_data, label_info = read_client_data_FCL_imagenet1k(i, task=task, classes_per_task=self.args.cpt, count_labels=True)
                        elif self.args.dataset == 'CIFAR100':
                            train_data, label_info = read_client_data_FCL_cifar100(i, task=task, classes_per_task=self.args.cpt, count_labels=True)
                        else:
                            raise NotImplementedError("Not supported dataset")
                    # elif self.args.partition_options == 'hetero':
                    #     from system.utils.data_utils_mine import read_client_data_FCL_cifar10, read_client_data_FCL_cifar100, read_client_data_FCL_imagenet1k
                    #     if self.args.dataset == 'IMAGENET1k':
                    #         train_data, label_info = read_client_data_FCL_imagenet1k(i, task=task, classes_per_task=self.args.cpt, count_labels=True,
                    #                                                                 seed = self.args.seed, alpha = self.args.alpha,
                    #                                                                 total_clients = self.args.num_clients,task_disorder = self.args.task_disorder)
                    #     elif self.args.dataset == 'CIFAR100':
                    #         train_data, label_info = read_client_data_FCL_cifar100(i, task=task, classes_per_task=self.args.cpt, count_labels=True,
                    #                                                             seed = self.args.seed, alpha = self.args.alpha,
                    #                                                             total_clients = self.args.num_clients,task_disorder = self.args.task_disorder)
                    #     elif self.args.dataset == 'CIFAR10':
                    #         train_data, label_info = read_client_data_FCL_cifar10(i, task=task, classes_per_task=self.args.cpt, count_labels=True,
                    #                                                             seed = self.args.seed, alpha = self.args.alpha,
                    #                                                             total_clients = self.args.num_clients,task_disorder = self.args.task_disorder)
                    #     else:
                    #         raise NotImplementedError("Not supported dataset")

                    # update client loaders
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

            # ============ inner rounds =============
            for i in range(self.global_rounds):

                glob_iter  = i + self.global_rounds * task
                disp_round = glob_iter + 1
                self._round_tag = glob_iter
                s_t = time.time()

                # select clients and list of all clients
                self.selected_clients = self.select_clients()
                self.alled_clients = self.all_clients()

                selected_ids = [client.id for client in self.selected_clients]

                # header
                self._roundlog.round_start(round_idx=disp_round, task_id=task, selected_clients=selected_ids)
                model_global_before = copy.deepcopy(self.global_model)
                # send models only to selected clients
                self.send_selected_models(selected_ids, i, task)

                # optional global eval
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

                # local training: FedAS trains ALL clients each round; flag indicates selection
                client_summaries: List[Dict[str, Any]] = []
                for client in self.alled_clients:
                    # Only compute deltas/metrics for selected clients (to match FedAvg table)
                    compute_metrics = (client.id in selected_ids)

                    before = self._flatten_params(client.model) if compute_metrics else None

                    client.train(client.id in selected_ids, task)
                    try:                                                      # ← thêm
                        client._model_after_local = copy.deepcopy(client.model)
                    except Exception:
                        pass   
                    if client.id in [0, 1,2,3,4]:  
                        try:
                            save_dir =  "/C:/Thu/FCL/weight_fedas/weight_client_round/"  # Thay đổi đường dẫn theo nhu cầu
                            os.makedirs(save_dir, exist_ok=True)
                            save_path = f"{save_dir}client_{client.id}_task_{task}_round_{i}.pt"
                            
                            model_to_save = client.model
                            torch.save(model_to_save.state_dict(), save_path)
                            print(f"[SAVED] client={client.id} task={task} round={i} → {save_path}")
                        except Exception as save_err:
                            print(f"[ERROR] Failed to save client {client.id} task {task} round {i}: {save_err}")
                            import traceback; traceback.print_exc()                 
                    delta_l2 = None
                    if compute_metrics:
                        try:
                            after = self._flatten_params(client.model)
                            if before is not None and after is not None and after.shape == before.shape:
                                delta_l2 = float(torch.norm(after - before, p=2))
                        except Exception:
                            pass

                        # averaged training loss (if client exposes it)
                        train_loss = None
                        try:
                            tr_sum, tr_n = client.train_metrics(task=task)
                            train_loss = (tr_sum / max(1, tr_n)) if tr_n else None
                        except Exception:
                            pass

                        # AA up to current task and forgetting
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
                            "time":   None,        # (optional) you can time per-client if needed
                            "samples": getattr(client, "train_samples", None),
                            "delta_l2": delta_l2,
                        })

                # show per-client table for selected clients
                self._roundlog.clients_end(round_idx=disp_round, client_summaries=client_summaries)

                # aggregation (FedAS)
                self.receive_models()
                self.receive_grads()
                model_origin = copy.deepcopy(self.global_model)
                self.aggregate_wrt_fisher()
                model_global_after = copy.deepcopy(self.global_model)  

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


                    csv_path = "/C:/Thu/FCL/weight_fedas/fedas_drift_results.csv"
                    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
                    fieldnames = ["round", "task", "client", "block",
                                "drift_trained", "drift_aggre", "drift_global",
                                "cknna_trained", "cknna_aggre", "cknna_global"]

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
                                    "drift_trained": round(float(r.get("drift_trained", float("nan"))), 6),
                                    "drift_aggre":   round(float(r.get("drift_aggre",   float("nan"))), 6),
                                    "drift_global":  round(float(r.get("drift_global",  float("nan"))), 6),
                                    # "cka_trained":   round(float(r.get("cka_trained",   float("nan"))), 6),
                                    # "cka_aggre":     round(float(r.get("cka_aggre",     float("nan"))), 6),
                                    # "cka_global":    round(float(r.get("cka_global",    float("nan"))), 6),
                                    "cknna_trained": round(float(r.get("cknna_trained", float("nan"))), 6),
                                    "cknna_aggre":   round(float(r.get("cknna_aggre",   float("nan"))), 6),
                                    "cknna_global":  round(float(r.get("cknna_global",  float("nan"))), 6)
                                })             
                # optional diagnostics (FedAS)
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

                # optional local eval at gap
                if i % eval_gap == 0:
                    try:
                        self.eval(task=task, glob_iter=glob_iter, flag="local")
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

            # end-of-task reporting (keep your original)
            if int(task/self.N_TASKS) == int(self.args.num_tasks/self.N_TASKS-1):
                if self.args.offlog and not self.args.debug:
                    self.eval_task(task=task, glob_iter=glob_iter, flag="local")

                    # need eval before data update
                    self.send_selected_models(selected_ids, self.global_rounds-1, task)
                    self.eval_task(task=task, glob_iter=glob_iter, flag="global")

            # optional: dump GLOBAL per-task accuracy row once per task end
            try:
                if hasattr(self, "dump_global_task_accuracy_csv"):
                    self.dump_global_task_accuracy_csv(after_task=int(task),
                                                       glob_iter=int(self.global_rounds * (task + 1) - 1))
            except Exception as e:
                print(f"[dump global_task_acc] warning: {e}")

        self._roundlog.finish()
