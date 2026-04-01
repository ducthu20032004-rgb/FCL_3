import time, copy, inspect
import torch
import torch.nn as nn
from flcore.clients.clientavg import clientAVG
from flcore.servers.serverbase import Server
from utils.data_utils import *
from utils.model_utils import ParamDict
from torch.nn.utils import vector_to_parameters, parameters_to_vector
import wandb
from flcore.metrics.average_forgetting import metric_average_forgetting

from torch.optim.lr_scheduler import StepLR
import numpy as np

import statistics


# ---------- Pretty logger (safe if not installed) ----------
try:
    from utils.rich_progress import RichRoundLogger
except Exception:
    RichRoundLogger = None


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

        # ---------- Helpers ----------
        def _cid(c, idx):
            return getattr(c, "id", getattr(c, "cid", idx))

        def _get_device_of(model):
            try:
                return next(model.parameters()).device
            except Exception:
                return torch.device("cpu")

        def _flatten_params(model):
            with torch.no_grad():
                return torch.cat([p.detach().float().view(-1).cpu() for p in model.parameters() if p is not None])

        def _get_current_train_loader(client, task):
            """
            Try common attribute names to recover the current train loader for this task.
            """
            # Single loader patterns
            for name in ("train_loader", "trainloader"):
                if hasattr(client, name) and getattr(client, name) is not None:
                    return getattr(client, name)
            # Per-task list patterns
            for name in ("task_train_loaders", "train_loaders", "trainloaders"):
                if hasattr(client, name):
                    arr = getattr(client, name)
                    try:
                        return arr[task]
                    except Exception:
                        # fallback: last loader if index missing
                        try:
                            return arr[-1]
                        except Exception:
                            pass
            # Dataset (without DataLoader): make a temporary loader
            for name in ("train_dataset", "train_data"):
                if hasattr(client, name) and getattr(client, name) is not None:
                    ds = getattr(client, name)
                    try:
                        from torch.utils.data import DataLoader
                        return DataLoader(ds, batch_size=128, shuffle=False, num_workers=4)# mặc định num_workers =  0
                    except Exception:
                        return None
            return None

        @torch.no_grad()
        def _quick_eval_loss_acc(model, loader, device, max_batches=20):
            """
            Lightweight eval on a few batches to derive loss/acc if the client doesn't report them.
            Uses CrossEntropyLoss by default; if your client has a custom criterion, plug it here.
            """
            if loader is None:
                return None, None
            model.eval()
            ce = nn.CrossEntropyLoss()
            n, correct, loss_sum, seen = 0, 0, 0.0, 0
            for b_idx, batch in enumerate(loader):
                if b_idx >= max_batches:
                    break
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    x, y = batch[0], batch[1]
                else:
                    # can't parse
                    continue
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)
                loss = ce(logits, y)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                loss_sum += float(loss.item()) * x.size(0)
                n += x.size(0)
                seen += 1
            if n == 0:
                return None, None
            return loss_sum / n, (100.0 * correct / n)

        def _call_client_train(client, task, round_idx, glob_iter):
            """
            Call client.train(...) with only the kwargs it supports (signature-aware).
            """
            fn = getattr(client, "train")
            sig = inspect.signature(fn)
            kwargs = {}
            if "task" in sig.parameters:      kwargs["task"] = task
            if "round" in sig.parameters:     kwargs["round"] = round_idx
            if "rnd" in sig.parameters:       kwargs["rnd"] = round_idx
            if "epoch" in sig.parameters:     kwargs["epoch"] = round_idx
            if "glob_iter" in sig.parameters: kwargs["glob_iter"] = glob_iter
            return fn(**kwargs)  # may or may not return metrics

        # ---------- Main loop ----------
        total_rounds = int(self.global_rounds) * int(getattr(self.args, "num_tasks", getattr(self.args, "nt", 1)))
        self._roundlog.start(total_rounds=total_rounds)
        self.Budget = getattr(self, "Budget", [])
        eval_gap = int(getattr(self, "eval_gap", 1) or 1)
        num_tasks = int(getattr(self.args, "num_tasks", getattr(self.args, "nt", 1)))

        for task in range(num_tasks):
            # (A) Task boundary behavior (keep your original pre-task logic if any)
            self.current_task = task
            torch.cuda.empty_cache()

            # (B) Rounds within task

            start_round = getattr(self.args, "start_round", 0)
            print("start_round =", start_round)
            print("global_rounds =", self.global_rounds)
            for i in range(start_round, self.global_rounds):
                glob_iter = i + self.global_rounds * task     # 0-based global counter

                self._round_tag = glob_iter
                
                disp_round = glob_iter + 1                    # pretty 1-based
                t0_round = time.time()

                # (1) Select clients and store to attr (required by receive_models())
                self.selected_clients = self.select_clients()
                sel_ids = [_cid(c, k) for k, c in enumerate(self.selected_clients)]

                # (2) Header
                self._roundlog.round_start(round_idx=disp_round, task_id=task, selected_clients=sel_ids)

                # (3) Broadcast global model
                if hasattr(self, "send_models"):
                    self.send_models()

                # (4) Optional global eval (lightweight)
                if i % eval_gap == 0 and hasattr(self, "eval"):
                    try:
                        self.eval(task=task, glob_iter=glob_iter, flag="global")
                    except TypeError:
                        # fall back if repo defines different signature
                        try:
                            self.eval()
                        except Exception:
                            pass

                    # self.dump_client_task_accuracy_csv(after_task=task, glob_iter=glob_iter)

                    self._get_or_build_global_counts(self._round_tag)

                # (5) Local training
                client_summaries = []

                per_client_forgetting = {}
                
                # tag this round; any monotonically increasing counter works
                self._round_tag = glob_iter  # or use your own global round index

                # 1) Build per-class counts once this round
                cc, tt = self._get_or_build_global_counts(self._round_tag)

                # 2) For every client, update its accuracy row vector and compute forgetting
                per_client_forgetting = {}  # cid -> float in [0,1]

                for j, client in enumerate(self.selected_clients):
                    per_t0 = time.time()

                    # snapshot before-train weights to measure delta
                    try:
                        before = _flatten_params(client.model)
                    except Exception:
                        before = None

                    # call train with supported kwargs
                    ret = None
                    
                    ret = _call_client_train(client, task=task, round_idx=i, glob_iter=glob_iter)

                    if client.id in [0, 1]:  # chỉ lưu 2 client
                        try:
                            save_dir = "/kaggle/working/checkpoints"
                            os.makedirs(save_dir, exist_ok=True)
                            save_path = f"{save_dir}/client_{client.id}_task_{task}_round_{i}.pt"
                            
                            model_to_save = client.model
                            torch.save(model_to_save.state_dict(), save_path)
                            print(f"[SAVED] client={client.id} task={task} round={i} → {save_path}")
                        except Exception as save_err:
                            print(f"[ERROR] Failed to save client {client.id} task {task} round {i}: {save_err}")
                            import traceback; traceback.print_exc()


                    # compute delta L2 of local update
                    delta_l2 = None
                    try:
                        after = _flatten_params(client.model)
                        if before is not None and after.shape == before.shape:
                            delta_l2 = float(torch.norm(after - before, p=2))
                    except Exception:
                        pass

                    # --- consistent metrics with W&B ---
                    try:
                        # averaged training loss on this client's train split
                        tr_sum, tr_n = client.train_metrics(task=task)   # returns (sum_ce_loss, num_samples)
                        train_loss = (tr_sum / max(1, tr_n)) if tr_n else None
                    except Exception:
                        train_loss, tr_n = None, None

                    # NEW: client AA on global test set up to current task
                    aa_pct = self._client_AA_global_upto(client, upto_task=task)
                    
                    acc_vec = self._client_acc_vector_all_tasks_from_counts(client, cc, tt)   # [A_k] for this eval point
                    self.client_accuracy_matrix.setdefault(client.id, []).append(acc_vec)     # append time row

                    # average forgetting up to current task index
                    cf = metric_average_forgetting(int(task % self.N_TASKS), self.client_accuracy_matrix[client.id])
                    per_client_forgetting[client.id] = float(cf)  # keep as fraction

                    cf_pct = per_client_forgetting.get(client.id)

                    client_summaries.append({
                        "client": sel_ids[j],
                        "loss": train_loss,
                        "acc": aa_pct,
                        "forg": (100.0 * cf_pct) if (cf_pct is not None) else None,  # NEW: client forgetting (%)
                        "time": time.time() - per_t0,
                        "samples": getattr(client, "train_samples", None),
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
                # ===== SAVE CHECKPOINT PER ROUND (mỗi 10 round) =====
                if getattr(self.args, "save_checkpoint", False) and glob_iter > 0 and ((glob_iter+1) % 5 == 0):
                    self.save_checkpoint(glob_iter=glob_iter, tag="latest")
                # (7) Optional extras (preserve if available)
                if getattr(self.args, "seval", False) and hasattr(self, "spatio_grad_eval"):
                    try:
                        model_origin = copy.deepcopy(self.global_model)
                        self.spatio_grad_eval(model_origin=model_origin, glob_iter=glob_iter)
                    except Exception:
                        pass
                if getattr(self.args, "pca_eval", False) and hasattr(self, "proto_eval") and getattr(self, "uploaded_models", None):
                    try:
                        self.proto_eval(global_model=self.global_model,
                                        local_model=self.uploaded_models[0],
                                        task=task, round=i)
                    except Exception:
                        pass

                # (8) End-of-round summary (time + optional quick global metrics)
                elapsed = time.time() - t0_round
                self.Budget.append(elapsed)

                # If you want to show acc/loss in the panel too, compute on eval_gap:
                g_metrics = {}
                if i % eval_gap == 0:
                    try:
                        # if your server exposes a light "test()" returning dict
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
                # --- W&B: log GLOBAL metrics per round ---
                if getattr(self.args, "wandb", False):
                    wb = {
                        "round": disp_round,
                        "task": task,
                        "time/round_sec": elapsed,
                    }

                    # Global test metrics (nếu có)
                    if "test_loss" in g_metrics and g_metrics["test_loss"] is not None:
                        wb["test/loss"] = float(g_metrics["test_loss"])
                    if "test_acc" in g_metrics and g_metrics["test_acc"] is not None:
                        wb["test/acc"] = float(g_metrics["test_acc"])

                    # Global train metrics (mean over selected clients)
                    client_losses = [c["loss"] for c in client_summaries if c["loss"] is not None]
                    client_accs   = [c["acc"]  for c in client_summaries if c["acc"]  is not None]
                    client_forgs  = [c["forg"] for c in client_summaries if c["forg"] is not None]

                    if len(client_losses) > 0:
                        wb["train/loss"] = float(np.mean(client_losses))
                        wb["train/loss_std"] = float(np.std(client_losses))
                    if len(client_accs) > 0:
                        wb["train/acc"] = float(np.mean(client_accs))
                    if len(client_forgs) > 0:
                        wb["forgetting/avg_pct"] = float(np.mean(client_forgs))

                    wandb.log(wb)
            try:
                self.dump_global_task_accuracy_csv(after_task=int(task), glob_iter=int(glob_iter))
            except Exception as e:
                print(f"[dump global_task_acc] warning: {e}")
            # ===== SAVE CHECKPOINT PER TASK =====
            if getattr(self.args, "save_checkpoint", False):
                tag = f"task{task}"
                self.save_checkpoint(glob_iter=glob_iter, tag=tag)
        self._roundlog.finish()



    # def train(self):

    #     # if self.args.num_tasks % self.N_TASKS != 0:
    #     #     raise ValueError("Set num_task again")

    #     for task in range(self.args.num_tasks):

    #         print(f"\n================ Current Task: {task} =================")
    #         if task == 0:
    #              # update labels info. for the first task
    #             available_labels = set()
    #             available_labels_current = set()
    #             available_labels_past = set()
    #             for u in self.clients:
    #                 available_labels = available_labels.union(set(u.classes_so_far))
    #                 available_labels_current = available_labels_current.union(set(u.current_labels))
    #             # print("ahihi " + str(len(available_labels_current)))
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

    #                 # update dataset
    #                 self.clients[i].next_task(train_data, label_info) # assign dataloader for new data
    #                 # print(self.clients[i].task_dict)

    #             # update labels info.
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

    #         # ============ train ==============

    #         for i in range(self.global_rounds):

    #             glob_iter = i + self.global_rounds * task
    #             s_t = time.time()
    #             self.selected_clients = self.select_clients()
    #             self.send_models()

    #             if i%self.eval_gap == 0:
    #                 print(f"\n-------------Round number: {i}-------------")
    #                 self.eval(task=task, glob_iter=glob_iter, flag="global")

    #             for client in self.selected_clients:
    #                 client.train(task=task)

    #             # threads = [Thread(target=client.train)
    #             #            for client in self.selected_clients]
    #             # [t.start() for t in threads]
    #             # [t.join() for t in threads]

    #             self.receive_models()
    #             self.receive_grads()
    #             model_origin = copy.deepcopy(self.global_model)
    #             self.aggregate_parameters()

    #             if self.args.seval:
    #                 self.spatio_grad_eval(model_origin=model_origin, glob_iter=glob_iter)

    #             if self.args.pca_eval:
    #                 self.proto_eval(global_model=self.global_model,
    #                                 local_model=self.uploaded_models[0], task=task, round=i)

    #             # if i%self.eval_gap == 0:
    #             #     self.eval(task=task, glob_iter=glob_iter, flag="local")

    #             self.Budget.append(time.time() - s_t)
    #             print('-'*25, 'time cost', '-'*25, self.Budget[-1])

    #         # Comment for boosting speed for rebuttal run
            
    #         # if int(task/self.N_TASKS) == int(self.args.num_tasks/self.N_TASKS-1):
    #         #     if self.args.offlog == True and not self.args.debug:  
    #         #         self.eval_task(task=task, glob_iter=glob_iter, flag="local")

    #         #         # need eval before data update
    #         #         self.send_models()
    #         #         self.eval_task(task=task, glob_iter=glob_iter, flag="global")
