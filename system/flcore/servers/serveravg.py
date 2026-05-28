import time
import copy
import inspect

from timm import task
import torch
import torch.nn as nn
import wandb
import numpy as np

from torch.nn.utils import (
    vector_to_parameters,
    parameters_to_vector
)

from torch.optim.lr_scheduler import StepLR

from system.flcore.clients.clientavg import clientAVG

from system.flcore.servers.serverbase import Server

from system.utils.data_utils import *
from system.utils.model_utils import ParamDict

from system.flcore.metrics.average_forgetting import (
    metric_average_forgetting
)

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
                          num_workers=4, pin_memory=(DEVICE.type == 'cuda'),
                          persistent_workers=True, prefetch_factor=2)

    # Case 2: (X, Y) - moi phan tu la array/tensor ca batch
    if (isinstance(dataset, (tuple, list))
            and len(dataset) == 2
            and hasattr(dataset[0], 'shape')
            and len(np.shape(dataset[0])) >= 2):
        X, Y = dataset
        xs = torch.as_tensor(np.array(X, dtype=np.float32))
        ys = torch.as_tensor(np.array(Y)).long()
        return DataLoader(TensorDataset(xs, ys), batch_size=batch_size, shuffle=False,
                          num_workers=4, pin_memory=(DEVICE.type == 'cuda'),
                          persistent_workers=True, prefetch_factor=2)

    # Case 3: list of (x_i, y_i) sample tuples
    xs, ys = [], []
    for x, y in dataset:
        xs.append(torch.as_tensor(np.array(x, dtype=np.float32)))
        ys.append(torch.as_tensor(np.array(y)).long())
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    return DataLoader(TensorDataset(xs, ys), batch_size=batch_size, shuffle=False,
                      num_workers=4, pin_memory=(DEVICE.type == 'cuda'),
                      persistent_workers=True, prefetch_factor=2)


def compute_feature_resnet18_wrap(_model, _model_task_index, _dataset, _target_layer_index: str, seed, args):
    """
    Trích xuất features trên GPU, trả về numpy array (N, D).
    Chấp nhận Dataset chuẩn hoặc list of (x, y) tuples.
    """
    # ===== unwrap BaseHeadSplit =====
    backbone = _model.base if hasattr(_model, "base") else _model
    blocks   = get_resnet18_blocks(backbone)
    backbone.eval()
    outputs  = []
    loader   = _make_loader(_dataset, batch_size=256)

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


# ---------- Temp-model helpers (save/load/delete on disk) ----------
import os

def _save_temp_model(model, path):
    torch.save(model.state_dict(), path)

def _load_temp_model(model_ref, path):
    """
    Load state_dict vào shallow copy của model_ref.
    map_location='cpu' để tiết kiệm VRAM.
    """
    m = copy.copy(model_ref)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m

def _delete_temp_model(path):
    try:
        os.remove(path)
    except Exception:
        pass


class FedAvg(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_clients(clientAVG)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def train(self):
        """
        Federated training with progress + reliable metrics capture.
        - Preserves original algorithmic steps.
        - Auto-detects client.train(...) signature and passes only supported kwargs.
        - Forces metrics measurement if client doesn't report them.
        - Logs per-client delta L2 to verify updates happened.
        """

        if not hasattr(self, "_roundlog"):
            if RichRoundLogger is not None:
                fig_dir  = getattr(self.args, "fig_dir", "figures")
                pretty_on = getattr(self.args, "pretty_log", True)
                self._roundlog = RichRoundLogger(self.args, fig_dir=fig_dir, enable=pretty_on)
            else:
                class _Dummy:
                    def start(self, total_rounds):
                        print(f"[Progress] total planned rounds: {total_rounds}")
                    def round_start(self, round_idx, task_id, selected_clients):
                        print(f"\n=== Round {round_idx} | Task {task_id} | Clients {selected_clients} ===")
                    def clients_end(self, round_idx, client_summaries): pass
                    def round_end(self, round_idx, global_metrics, time_cost=None):
                        print(f"[Summary] round={round_idx} time={time_cost:.2f}s")
                    def finish(self): pass
                self._roundlog = _Dummy()

        # ---------- Helpers ----------
        def _cid(c, idx):
            return getattr(c, "id", getattr(c, "cid", idx))

        def _flatten_params(model):
            with torch.no_grad():
                return torch.cat([p.detach().float().view(-1).cpu()
                                  for p in model.parameters() if p is not None])

        def _call_client_train(client, task, round_idx, glob_iter):
            fn  = getattr(client, "train")
            sig = inspect.signature(fn)
            kwargs = {}
            if "task"      in sig.parameters: kwargs["task"]      = task
            if "round"     in sig.parameters: kwargs["round"]     = round_idx
            if "rnd"       in sig.parameters: kwargs["rnd"]       = round_idx
            if "epoch"     in sig.parameters: kwargs["epoch"]     = round_idx
            if "glob_iter" in sig.parameters: kwargs["glob_iter"] = glob_iter
            return fn(**kwargs)

        # ---------- Main loop ----------
        total_rounds = int(self.global_rounds) * int(getattr(self.args, "num_tasks", getattr(self.args, "nt", 1)))
        self._roundlog.start(total_rounds=total_rounds)
        self.Budget  = getattr(self, "Budget", [])
        eval_gap     = int(getattr(self, "eval_gap", 1) or 1)
        num_tasks    = int(getattr(self.args, "num_tasks", getattr(self.args, "nt", 1)))
        tmp_dir      = "/kaggle/working/FCL_3/tmp_models"

        for task in range(num_tasks):
            self.current_task = task
            torch.cuda.empty_cache()

            start_round = getattr(self.args, "start_round", 0)
            print("start_round =", start_round)
            print("global_rounds =", self.global_rounds)

            for i in range(start_round, self.global_rounds):
                glob_iter  = i + self.global_rounds * task
                self._round_tag = glob_iter
                disp_round = glob_iter + 1
                t0_round   = time.time()

                # (1) Select clients
                self.selected_clients = self.select_clients()
                sel_ids = [_cid(c, k) for k, c in enumerate(self.selected_clients)]

                # (2) Header
                self._roundlog.round_start(round_idx=disp_round, task_id=task, selected_clients=sel_ids)

                # (3) Broadcast global model
                if hasattr(self, "send_models"):
                    self.send_models()
                model_global_before = copy.deepcopy(self.global_model)

                # (4) Optional global eval
                if i % eval_gap == 0 and hasattr(self, "eval"):
                    try:
                        self.eval(task=task, glob_iter=glob_iter, flag="local")
                    except TypeError:
                        try:
                            self.eval()
                        except Exception:
                            pass
                    self._get_or_build_global_counts(self._round_tag)

                # (5) Local training
                client_summaries    = []
                per_client_forgetting = {}
                self._round_tag     = glob_iter
                cc, tt = self._get_or_build_global_counts(self._round_tag)

                for j, client in enumerate(self.selected_clients):
                    per_t0 = time.time()

                    # snapshot before-train weights
                    try:
                        before = _flatten_params(client.model)
                    except Exception:
                        before = None

                    # train
                    ret = _call_client_train(client, task=task, round_idx=i, glob_iter=glob_iter)

                    # ----- Save local model sau train để đo drift round-to-round -----
                    os.makedirs(tmp_dir, exist_ok=True)
                    curr_path = f"{tmp_dir}/client_{client.id}_curr.pt"
                    _save_temp_model(client.model, curr_path)
                    client._model_curr_path = curr_path
                    # ------------------------------------------------------------------

                    # if client.id in [0, 1, 2, 3, 4]:
                    #     try:
                    #         save_dir  = "C:/Thu/FCL/checkpoints"
                    #         os.makedirs(save_dir, exist_ok=True)
                    #         save_path = f"{save_dir}/client_{client.id}_task_{task}_round_{i}.pt"
                    #         torch.save(client.model.state_dict(), save_path)
                    #         print(f"[SAVED] client={client.id} task={task} round={i} → {save_path}")
                    #     except Exception as save_err:
                    #         print(f"[ERROR] Failed to save client {client.id}: {save_err}")
                    #         import traceback; traceback.print_exc()

                    # delta L2
                    delta_l2 = None
                    try:
                        after = _flatten_params(client.model)
                        if before is not None and after.shape == before.shape:
                            delta_l2 = float(torch.norm(after - before, p=2))
                    except Exception:
                        pass

                    # train loss
                    try:
                        tr_sum, tr_n = client.train_metrics(task=task)
                        train_loss   = (tr_sum / max(1, tr_n)) if tr_n else None
                    except Exception:
                        train_loss, tr_n = None, None

                    # accuracy + forgetting
                    aa_pct   = self._client_AA_global_upto(client, upto_task=task)
                    acc_vec  = self._client_acc_vector_all_tasks_from_counts(client, cc, tt)
                    self.client_accuracy_matrix.setdefault(client.id, []).append(acc_vec)
                    cf       = metric_average_forgetting(int(task % self.N_TASKS),
                                                         self.client_accuracy_matrix[client.id])
                    per_client_forgetting[client.id] = float(cf)
                    cf_pct   = per_client_forgetting.get(client.id)

                    client_summaries.append({
                        "client":   sel_ids[j],
                        "loss":     train_loss,
                        "acc":      aa_pct,
                        "forg":     (100.0 * cf_pct) if (cf_pct is not None) else None,
                        "time":     time.time() - per_t0,
                        "samples":  getattr(client, "train_samples", None),
                        "delta_l2": delta_l2,
                    })

                # show per-client table
                self._roundlog.clients_end(round_idx=disp_round, client_summaries=client_summaries)

                # (6) Aggregate
                if hasattr(self, "receive_models"):
                    self.receive_models()
                if hasattr(self, "receive_grads"):
                    self.receive_grads()
                if hasattr(self, "aggregate_parameters"):
                    self.aggregate_parameters()
                elif hasattr(self, "aggregate"):
                    self.aggregate()
                model_global_after = copy.deepcopy(self.global_model)

                # ================================================================
                # === Đo drift + cka + cknna ===
                # ================================================================
                if getattr(self.args, "measure_drift", True):
                    print("Drift thanh cong")
                    try:
                        drift_results = {}  # cid -> {block_idx -> {metric: value}}

                        for j, client in enumerate(self.selected_clients):
                            if client.id in [5, 6, 7, 8, 9]:
                                continue

                            cid       = client.id
                            curr_path = getattr(client, "_model_curr_path", None)
                            prev_path = getattr(client, "_model_local_prev_path", None)
                            drift_results[cid] = {}

                            test_data = read_client_data_FCL_cifar10(
                                cid, task=task,
                                classes_per_task=self.args.cpt,
                                count_labels=False, train=False
                            )

                            # ----------------------------------------------------
                            # [INNER-ROUND] global_before vs local_after vs global_after
                            # Comment lại để dùng sau
                            # ----------------------------------------------------
                            # for block_idx in [4]:
                            #     target_layer = f'block{block_idx}'
                            #     try:
                            #         feat_global = compute_feature_resnet18_wrap(
                            #             model_global_before, task, test_data, target_layer,
                            #             self.args.seed, self.args)
                            #         feat_local  = compute_feature_resnet18_wrap(
                            #             client._model_after_local, task, test_data, target_layer,
                            #             self.args.seed, self.args)
                            #         feat_aggre  = compute_feature_resnet18_wrap(
                            #             model_global_after, task, test_data, target_layer,
                            #             self.args.seed, self.args)
                            #         _, cka_global    = compute_cka(feat_global, feat_aggre)
                            #         cknna_trained, _ = compute_alignment_from_arrays(feat_global, feat_local,  "mutual_knn", topk=10, precise=True)
                            #         cknna_aggre,   _ = compute_alignment_from_arrays(feat_local,  feat_aggre,  "mutual_knn", topk=10, precise=True)
                            #         cknna_global,  _ = compute_alignment_from_arrays(feat_global, feat_aggre,  "mutual_knn", topk=10, precise=True)
                            #         drift_results[cid][block_idx] = {
                            #             "cka_global":    cka_global,
                            #             "cknna_global":  cknna_global,
                            #             "cknna_trained": cknna_trained,
                            #             "cknna_aggre":   cknna_aggre,
                            #         }
                            #         print(
                            #             f"[Drift] round={disp_round} task={task} client={cid} {target_layer} | "
                            #             f"cknna_trained={cknna_trained:.4f} cknna_aggre={cknna_aggre:.4f} cknna_global={cknna_global:.4f}"
                            #         )
                            #     except Exception as e:
                            #         print(f"[Drift inner] client={cid} block{block_idx} error: {e}")
                            #         drift_results[cid][block_idx] = None
                            # ----------------------------------------------------

                            # ----------------------------------------------------
                            # [ROUND-TO-ROUND] local_after_{round i-1} vs local_after_{round i}
                            # ----------------------------------------------------
                            if curr_path is None or prev_path is None:
                                print(f"  [skip] client={cid}: chưa có đủ 2 round để so sánh")
                                # Round đầu tiên: promote curr → prev, chờ round sau
                                if curr_path is not None:
                                    client._model_local_prev_path = curr_path
                                    client._model_curr_path       = None
                                continue

                            for block_idx in [4]:
                                target_layer = f'block{block_idx}'
                                try:
                                    # Load tạm trên CPU để tiết kiệm VRAM
                                    prev_model = _load_temp_model(client.model, prev_path)
                                    curr_model = _load_temp_model(client.model, curr_path)

                                    feat_prev = compute_feature_resnet18_wrap(
                                        prev_model, task, test_data, target_layer,
                                        self.args.seed, self.args)
                                    feat_curr = compute_feature_resnet18_wrap(
                                        curr_model, task, test_data, target_layer,
                                        self.args.seed, self.args)

                                    # Giải phóng ngay sau khi extract xong
                                    del prev_model, curr_model
                                    torch.cuda.empty_cache()

                                    _, cka_rtr   = compute_cka(feat_prev, feat_curr)
                                    cknna_rtr, _ = compute_alignment_from_arrays(
                                        feat_prev, feat_curr, "mutual_knn", topk=10, precise=True)

                                    drift_results[cid][block_idx] = {
                                        "cka_rtr":   cka_rtr,
                                        "cknna_rtr": cknna_rtr,
                                    }

                                    print(
                                        f"[Drift rtr] round={disp_round} task={task} client={cid} {target_layer} | "
                                        f"cka_rtr={cka_rtr:.4f}  cknna_rtr={cknna_rtr:.4f}"
                                    )

                                except Exception as e:
                                    print(f"[Drift rtr] client={cid} block{block_idx} error: {e}")
                                    drift_results[cid][block_idx] = None

                            # Xóa prev (không cần nữa), promote curr → prev cho round tiếp theo
                            _delete_temp_model(prev_path)
                            client._model_local_prev_path = curr_path
                            client._model_curr_path       = None
                            # ----------------------------------------------------

                            # Append drift metrics vào client_summaries
                            block_summary = {}
                            for block_idx in [4]:
                                r = drift_results[cid].get(block_idx)
                                if r is None:
                                    continue
                                for metric_name, val in r.items():
                                    block_summary[f"block{block_idx}/{metric_name}"] = val
                            client_summaries[j].update(block_summary)

                        # --- W&B log drift ---
                        if getattr(self.args, "wandb", True):
                            log_every_round    = getattr(self.args, "wandb_drift_every_round", True)
                            final_round_of_task = 24
                            should_log = log_every_round or (i == final_round_of_task)
                            if should_log and i == 0:
                                for cid in drift_results:
                                    for block_idx in [4]:
                                        if drift_results[cid].get(block_idx) is None:
                                            continue
                                        wb_drift = {"round": disp_round, "task": task, "client": cid}
                                        for metric_name in ["cka_rtr", "cknna_rtr"]:
                                            if metric_name in drift_results[cid][block_idx]:
                                                wb_drift[f"drift/client{cid}/block{block_idx}/{metric_name}"] = \
                                                    float(drift_results[cid][block_idx][metric_name])
                                        wandb.log(wb_drift)

                    except Exception as e:
                        print(f"[Drift measure] warning: {e}")
                        import traceback; traceback.print_exc()

                    # --- Ghi CSV ---
                    import csv
                    csv_path    = "/kaggle/working/FCL_3/drift_local_round_to_round.csv"
                    fieldnames  = ["round", "task", "client", "block", "cka_rtr", "cknna_rtr"]
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
                                    "round":     disp_round,
                                    "task":      task,
                                    "client":    cid,
                                    "block":     block_idx,
                                    "cka_rtr":   round(float(r.get("cka_rtr",   float("nan"))), 6),
                                    "cknna_rtr": round(float(r.get("cknna_rtr", float("nan"))), 6),
                                })
                # ================================================================

                # ===== SAVE CHECKPOINT PER ROUND =====
                if getattr(self.args, "save_checkpoint", False) and glob_iter > 0 and ((glob_iter + 1) % 1 == 0):
                    self.save_checkpoint(glob_iter=glob_iter, tag="latest")

                # (7) Optional extras
                if getattr(self.args, "seval", False) and hasattr(self, "spatio_grad_eval"):
                    try:
                        model_origin = copy.deepcopy(self.global_model)
                        self.spatio_grad_eval(model_origin=model_origin, glob_iter=glob_iter)
                    except Exception:
                        pass
                if getattr(self.args, "pca_eval", False) and hasattr(self, "proto_eval") \
                        and getattr(self, "uploaded_models", None):
                    try:
                        self.proto_eval(global_model=self.global_model,
                                        local_model=self.uploaded_models[0],
                                        task=task, round=i)
                    except Exception:
                        pass

                # (8) End-of-round summary
                elapsed  = time.time() - t0_round
                self.Budget.append(elapsed)

                g_metrics = {}
                if i % eval_gap == 0:
                    try:
                        if hasattr(self, "test"):
                            m = self.test()
                            if isinstance(m, dict):
                                g_metrics = {
                                    "test_loss": m.get("test_loss", m.get("loss")),
                                    "test_acc":  m.get("test_acc",  m.get("acc")),
                                }
                    except Exception:
                        pass

                self._roundlog.round_end(round_idx=disp_round, global_metrics=g_metrics, time_cost=elapsed)

                # --- W&B: log global metrics per round ---
                if getattr(self.args, "wandb", False):
                    wb = {
                        "round":          disp_round,
                        "task":           task,
                        "time/round_sec": elapsed,
                    }
                    if "test_loss" in g_metrics and g_metrics["test_loss"] is not None:
                        wb["test/loss"] = float(g_metrics["test_loss"])
                    if "test_acc" in g_metrics and g_metrics["test_acc"] is not None:
                        wb["test/acc"] = float(g_metrics["test_acc"])

                    client_losses = [c["loss"] for c in client_summaries if c["loss"] is not None]
                    client_accs   = [c["acc"]  for c in client_summaries if c["acc"]  is not None]
                    client_forgs  = [c["forg"] for c in client_summaries if c["forg"] is not None]

                    if client_losses:
                        wb["train/loss"]     = float(np.mean(client_losses))
                        wb["train/loss_std"] = float(np.std(client_losses))
                    if client_accs:
                        wb["train/acc"] = float(np.mean(client_accs))
                    if client_forgs:
                        wb["forgetting/avg_pct"] = float(np.mean(client_forgs))

                    wandb.log(wb)

            try:
                self.dump_global_task_accuracy_csv(after_task=int(task), glob_iter=int(glob_iter))
            except Exception as e:
                print(f"[dump global_task_acc] warning: {e}")

            # ===== SAVE CHECKPOINT PER TASK =====
            if getattr(self.args, "save_checkpoint", False):
                self.save_checkpoint(glob_iter=glob_iter, tag=f"task{task}")

        self._roundlog.finish()


    # def train(self):
    #     for task in range(self.args.num_tasks):
    #         print(f"\n================ Current Task: {task} =================")
    #         if task == 0:
    #             available_labels = set()
    #             available_labels_current = set()
    #             available_labels_past = set()
    #             for u in self.clients:
    #                 available_labels = available_labels.union(set(u.classes_so_far))
    #                 available_labels_current = available_labels_current.union(set(u.current_labels))
    #             for u in self.clients:
    #                 u.available_labels = list(available_labels)
    #                 u.available_labels_current = list(available_labels_current)
    #                 u.available_labels_past = list(available_labels_past)
    #         else:
    #             self.current_task = task
    #             torch.cuda.empty_cache()
    #             for i in range(len(self.clients)):
    #                 if self.args.dataset == 'IMAGENET1k':
    #                     train_data, label_info = read_client_data_FCL_imagenet1k(i, task=task, classes_per_task=self.args.cpt, count_labels=True)
    #                 elif self.args.dataset == 'CIFAR100':
    #                     train_data, label_info = read_client_data_FCL_cifar100(i, task=task, classes_per_task=self.args.cpt, count_labels=True)
    #                 elif self.args.dataset == 'CIFAR10':
    #                     train_data, label_info = read_client_data_FCL_cifar10(i, task=task, classes_per_task=self.args.cpt, count_labels=True)
    #                 else:
    #                     raise NotImplementedError("Not supported dataset")
    #                 self.clients[i].next_task(train_data, label_info)
    #             available_labels = set()
    #             available_labels_current = set()
    #             available_labels_past = self.clients[0].available_labels
    #             for u in self.clients:
    #                 available_labels = available_labels.union(set(u.classes_so_far))
    #                 available_labels_current = available_labels_current.union(set(u.current_labels))
    #             for u in self.clients:
    #                 u.available_labels = list(available_labels)
    #                 u.available_labels_current = list(available_labels_current)
    #                 u.available_labels_past = list(available_labels_past)
    #         for i in range(self.global_rounds):
    #             glob_iter = i + self.global_rounds * task
    #             s_t = time.time()
    #             self.selected_clients = self.select_clients()
    #             self.send_models()
    #             if i % self.eval_gap == 0:
    #                 print(f"\n-------------Round number: {i}-------------")
    #                 self.eval(task=task, glob_iter=glob_iter, flag="global")
    #             for client in self.selected_clients:
    #                 client.train(task=task)
    #             self.receive_models()
    #             self.receive_grads()
    #             model_origin = copy.deepcopy(self.global_model)
    #             self.aggregate_parameters()
    #             if self.args.seval:
    #                 self.spatio_grad_eval(model_origin=model_origin, glob_iter=glob_iter)
    #             if self.args.pca_eval:
    #                 self.proto_eval(global_model=self.global_model,
    #                                 local_model=self.uploaded_models[0], task=task, round=i)
    #             self.Budget.append(time.time() - s_t)
    #             print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])