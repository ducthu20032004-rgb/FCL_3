"""
FedTARGET server with the SAME progress/metrics printing/tracking as FedAvg,
while preserving FedTARGET's original training algorithm (TARGET: exemplar-free
distillation with synthetic data generation).

What this adds (vs. your original):
  • RichRoundLogger (same look as FedAvg): per-round header, per-client table, summary panel.
  • Per-client AA@t (average accuracy up to current task) computed on the GLOBAL test pool.
  • Per-client Average Forgetting (so-far) with metric_average_forgetting.
  • Optional delta-L2 of local updates for quick sanity.
  • Dump a GLOBAL per-task accuracy row ONCE at the end of each task.

The training sequence (send_models → client.train → receive_models → receive_grads
→ aggregate_parameters → data_generation) is kept intact.
"""

import os
import time
import copy
import shutil
import inspect
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from flcore.clients.clienttarget import clientTARGET
from flcore.servers.serverbase import Server
from flcore.metrics.average_forgetting import metric_average_forgetting

# TARGET utils (kept as in your original)
from flcore.utils_core.target_utils import *

# Pretty logger (safe if not installed)
try:
    from utils.rich_progress import RichRoundLogger
except Exception:
    RichRoundLogger = None


class FedTARGET(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # TARGET-specific init (copied from your original)
        self.synthtic_save_dir = "dataset/synthetic_data"
        if os.path.exists(self.synthtic_save_dir):
            shutil.rmtree(self.synthtic_save_dir)
        self.nums = 8000
        self.total_classes = []
        self.syn_data_loader = None
        self.old_network = None
        self.kd_alpha = 25
        if "CIFAR100" in self.dataset:
            self.dataset_size = 50000
        elif "IMAGENET1k" in self.dataset:
            self.dataset_size = 1281167
        self.available_labels_current = None

        # clients
        self.set_clients(clientTARGET)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")
        self.Budget: List[float] = []

        # Per-client accuracy matrices over eval steps (rows appended at eval points)
        if not hasattr(self, "client_accuracy_matrix"):
            self.client_accuracy_matrix: Dict[int, List[List[float]]] = {}

        # Pretty logger (same look as FedAvg)
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

    # ----------------------- helpers (local to TARGET) -----------------------
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
            tot = int(tt[idx].sum())
            vec.append((corr / tot) if tot > 0 else 0.0)
        return vec

    # ----------------------- training (FedAvg-like logging) -----------------------
    def train(self):
        # sanity
        if self.args.num_tasks % self.N_TASKS != 0:
            raise ValueError("Set num_task again")

        num_tasks = int(self.args.num_tasks)
        total_rounds = int(self.global_rounds) * num_tasks
        self._roundlog.start(total_rounds=total_rounds)

        for task in range(num_tasks):
            print(f"\n================ Current Task: {task} =================")
            self.current_task = task
            torch.cuda.empty_cache()

            if task == 0:
                # initial label info (as in your original)
                available_labels = set()
                available_labels_current = set()
                available_labels_past = set()
                for u in self.clients:
                    available_labels |= set(u.classes_so_far)
                    available_labels_current |= set(u.current_labels)
                self.available_labels_current = available_labels_current
                for u in self.clients:
                    u.available_labels = list(available_labels)
                    u.available_labels_current = list(available_labels_current)
                    u.available_labels_past = list(available_labels_past)
            else:
                # Keep TARGET behavior: store old_network for KD, update each client
                self.old_network = copy.deepcopy(self.global_model)
                for p in self.old_network.parameters():
                    p.requires_grad = False

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
                    self.clients[i].old_network = copy.deepcopy(self.clients[i].model)
                    for p in self.clients[i].old_network.parameters():
                        p.requires_grad = False

                # update label availability trackers
                available_labels = set()
                available_labels_current = set()
                available_labels_past = self.clients[0].available_labels
                for u in self.clients:
                    available_labels |= set(u.classes_so_far)
                    available_labels_current |= set(u.current_labels)
                self.available_labels_current = available_labels_current
                for u in self.clients:
                    u.available_labels = list(available_labels)
                    u.available_labels_current = list(available_labels_current)
                    u.available_labels_past = list(available_labels_past)

            # ---------------- rounds within task ----------------
            eval_gap = int(getattr(self, "eval_gap", 1) or 1)
            for i in range(self.global_rounds):
                glob_iter = i + self.global_rounds * task
                disp_round = glob_iter + 1
                self._round_tag = glob_iter
                t0_round = time.time()

                # TARGET: at task > 0, prep synthetic data + KD teacher per client BEFORE training
                if task > 0:
                    for u in self.clients:
                        u.old_network = copy.deepcopy(self.old_network)
                        u.syn_data_loader = u.get_syn_data_loader()
                        u.it = DataIter(u.syn_data_loader)
                        u.old_network = u.old_network.to(self.device)

                # (1) select clients
                self.selected_clients = self.select_clients()
                sel_ids = [self._cid(c, j) for j, c in enumerate(self.selected_clients)]

                # header
                self._roundlog.round_start(round_idx=disp_round, task_id=task, selected_clients=sel_ids)

                # (2) broadcast global model
                if hasattr(self, "send_models"):
                    self.send_models()

                # (3) optional eval at gap
                # if i % eval_gap == 0 and hasattr(self, "eval"):
                #     try:
                #         self.eval(task=task, glob_iter=glob_iter, flag="global")
                #     except TypeError:
                #         try:
                #             self.eval()
                #         except Exception:
                #             pass

                # build global per-class counts ONCE for this eval point (AA/forgetting)
                try:
                    cc, tt = self._get_or_build_global_counts(self._round_tag)
                except Exception:
                    cc, tt, _K = self._compute_global_per_class_counts(self.global_model)

                # (4) local training + metrics capture
                client_summaries: List[Dict[str, Any]] = []
                for j, client in enumerate(self.selected_clients):
                    t0_cli = time.time()

                    # snapshot before
                    before = self._flatten_params(client.model)

                    # TARGET local train (keep algorithm)
                    try:
                        _ = self._call_client_train(client, task=task, round_idx=i, glob_iter=glob_iter)
                    except Exception as e:
                        _ = {"error": str(e)}

                    # delta L2
                    delta_l2 = None
                    try:
                        after = self._flatten_params(client.model)
                        if before is not None and after is not None and after.shape == before.shape:
                            delta_l2 = float(torch.norm(after - before, p=2))
                    except Exception:
                        pass

                    # averaged training loss if exposed
                    train_loss = None
                    try:
                        tr_sum, tr_n = client.train_metrics(task=task)
                        train_loss = (tr_sum / max(1, tr_n)) if tr_n else None
                    except Exception:
                        pass

                    # AA@t and forgetting (same as FedAvg)
                    acc_vec = self._acc_vec_for_client_from_counts(client, cc, tt)  # [A_k]
                    self.client_accuracy_matrix.setdefault(client.id, []).append(acc_vec)
                    aa_pct = float(100.0 * (np.mean(acc_vec[:task + 1]) if (task + 1) > 0 else 0.0))
                    try:
                        cf = metric_average_forgetting(int(task % self.N_TASKS), self.client_accuracy_matrix[client.id])
                        forg_pct = float(100.0 * cf)
                    except Exception:
                        forg_pct = None

                    client_summaries.append({
                        "client": sel_ids[j],
                        "loss": train_loss,
                        "acc": aa_pct,      # %
                        "forg": forg_pct,   # %
                        "time": time.time() - t0_cli,
                        "samples": getattr(client, "train_samples", None),
                        "delta_l2": delta_l2,
                    })

                # show per-client table
                self._roundlog.clients_end(round_idx=disp_round, client_summaries=client_summaries)

                # (5) aggregate (keep TARGET sequence)
                if hasattr(self, "receive_models"):
                    self.receive_models()
                if hasattr(self, "receive_grads"):
                    self.receive_grads()
                model_origin = copy.deepcopy(self.global_model)
                if hasattr(self, "aggregate_parameters"):
                    self.aggregate_parameters()
                elif hasattr(self, "aggregate"):
                    self.aggregate()

                # (6) optional diagnostics
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

                # (7) optional local eval at gap (as in your original)
                # if i % eval_gap == 0:
                #     try:
                #         self.eval(task=task, glob_iter=glob_iter, flag="local")
                #     except Exception:
                #         pass

                # (8) end-of-round summary
                elapsed = time.time() - t0_round
                self.Budget.append(elapsed)
                g_metrics = {}
                if i % eval_gap == 0:# and hasattr(self, "test"):
                    # try:
                    #     self.eval(task=task, glob_iter=glob_iter, flag="global")
                    # except Exception:
                    #     pass
                    
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

            # ---------------- TARGET synthetic data generation (preserve) ----------------
            self.data_generation(task=task, available_labels_current=self.available_labels_current)

            # ---------------- end-of-task evals (preserve) ----------------
            # if int(task/self.N_TASKS) == int(self.args.num_tasks/self.N_TASKS - 1):
            #     if self.args.offlog and not self.args.debug:
            #         self.eval_task(task=task, glob_iter=glob_iter, flag="global")
            self.send_models()  # as in your original
            self.eval_task(task=task, glob_iter=glob_iter, flag="global")

            # ---------------- dump GLOBAL per-task row ONCE per task end ----------------
            try:
                if hasattr(self, "dump_global_task_accuracy_csv"):
                    self.dump_global_task_accuracy_csv(after_task=int(task), glob_iter=int(self.global_rounds * (task + 1) - 1))
            except Exception as e:
                print(f"[dump global_task_acc] warning: {e}")

        # finish progress
        self._roundlog.finish()

    # ---------------------------- TARGET data gen ----------------------------
    def kd_train(self, student, teacher, criterion, optimizer, task):
        student.train()
        teacher.eval()
        loader = self.get_all_syn_data(task=task)
        data_iter = DataIter(loader)
        for i in range(kd_steps):
            images = data_iter.next().to(self.device)
            with torch.no_grad():
                t_out = teacher(images)
            s_out = student(images.detach())
            loss_s = criterion(s_out, t_out.detach())
            optimizer.zero_grad()
            loss_s.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0, norm_type=2)
            optimizer.step()
        return loss_s.item()

    def data_generation(self, task, available_labels_current):
        nz = 256
        img_size = 32
        img_shape = (3, 32, 32)
        generator = Generator(nz=nz, ngf=64, img_size=img_size, nc=3, device=self.device).to(self.device)
        student = copy.deepcopy(self.global_model)
        student.apply(weight_init)

        tmp_dir = os.path.join(self.synthtic_save_dir, f"task_{task}")
        os.makedirs(tmp_dir, exist_ok=True)

        synthesizer = GlobalSynthesizer(
            copy.deepcopy(self.global_model), student, generator,
            nz=nz, allowed_classes=available_labels_current, img_size=img_shape,
            init_dataset=None, save_dir=tmp_dir, transform=train_transform,
            normalizer=normalizer, synthesis_batch_size=synthesis_batch_size,
            sample_batch_size=sample_batch_size, iterations=g_steps, warmup=warmup,
            lr_g=lr_g, lr_z=lr_z, adv=adv, bn=bn, oh=oh, reset_l0=reset_l0,
            reset_bn=reset_bn, bn_mmt=bn_mmt, is_maml=is_maml, fabric=None, args=self.args
        )

        criterion = KLDiv(T=T)
        optimizer = torch.optim.SGD(student.parameters(), lr=0.002, weight_decay=0.0001, momentum=0.9)

        for it in range(syn_round):
            synthesizer.synthesize()
            if it >= warmup:
                loss = self.kd_train(student, self.global_model, criterion, optimizer, task)
                # (optional) print one line every few iters
                if (it + 1) % max(1, syn_round // 5) == 0:
                    print(f"Task {task}, Data Generation, Iter {it+1}/{syn_round} => Student loss: {loss:.2f}")
        print(f"For task {task}, data generation completed!")

    def get_all_syn_data(self, task):
        data_dir = os.path.join(self.synthtic_save_dir, f"task_{task}")
        syn_dataset = UnlabeledImageDataset(data_dir, transform=train_transform, nums=self.nums)
        loader = torch.utils.data.DataLoader(
            syn_dataset, batch_size=sample_batch_size, shuffle=True, num_workers=0, pin_memory=True, sampler=None
        )
        return loader