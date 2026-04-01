# system/flcore/servers/serveraffcl.py
# -*- coding: utf-8 -*-
"""
FedAFFCL server with FedAvg-style printing and tracking, while preserving AFFCL's
original training algorithm (send_parameters → client.train(task, glob_iter, global_classifier)
→ receive_models → receive_grads → aggregate_parameters_affcl).

Adds:
  • RichRoundLogger (same look as FedAvg): per-round header, per-client table, summary panel
  • Per-client AA@t (average accuracy up to current task) on the GLOBAL test pool
  • Per-client Average Forgetting (so-far) with metric_average_forgetting
  • Optional delta-L2 of local updates for a quick sanity check
  • (Optional) dump a GLOBAL per-task accuracy row once per task end (if helper exists)
"""

import time
import copy
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from flcore.clients.clientaffcl import ClientAFFCL
from flcore.servers.serverbase import Server
from flcore.metrics.average_forgetting import metric_average_forgetting

# Pretty logger (safe if not installed)
try:
    from utils.rich_progress import RichRoundLogger
except Exception:
    RichRoundLogger = None


class FedAFFCL(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.classifier_head_list = ['classifier.fc_classifier', 'classifier.fc2']

        self.set_clients(ClientAFFCL)

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

    # ----------------------- helpers -----------------------
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

    # ----------------------- training (FedAvg-style logging) -----------------------
    def train(self):

        if self.args.num_tasks % self.N_TASKS != 0:
            raise ValueError("Set num_task again")

        num_tasks     = int(self.args.num_tasks)
        total_rounds  = int(self.global_rounds) * num_tasks
        eval_gap      = int(getattr(self, "eval_gap", 1) or 1)

        self._roundlog.start(total_rounds=total_rounds)

        for task in range(num_tasks):

            print(f"\n================ Current Task: {task} =================")
            if task == 0:
                # update labels info. for the first task
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

                    # assign new dataloaders
                    self.clients[i].next_task(train_data, label_info)

                # update labels info.
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

            # ============ train ==============
            for i in range(self.global_rounds):

                glob_iter  = i + self.global_rounds * task
                disp_round = glob_iter + 1
                self._round_tag = glob_iter
                s_t = time.time()

                # (1) select clients & header
                self.selected_clients = self.select_clients()
                sel_ids = [self._cid(c, j) for j, c in enumerate(self.selected_clients)]
                self._roundlog.round_start(round_idx=disp_round, task_id=task, selected_clients=sel_ids)

                # (2) broadcast parameters (AFFCL uses 'send_parameters')
                self.send_parameters(mode='all', beta=1)

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

                # Obtain global classifier head (AFFCL uses this during client training)
                global_classifier = self.global_model.classifier
                global_classifier.eval()

                # (4) local training for selected clients + metrics capture
                client_summaries: List[Dict[str, Any]] = []
                for client in self.selected_clients:
                    t0_cli = time.time()

                    # snapshot before for delta-L2
                    before = self._flatten_params(client.model)

                    # AFFCL local training
                    verbose = False
                    client.train(task, glob_iter, global_classifier, verbose=verbose)

                    # delta L2
                    delta_l2 = None
                    try:
                        after = self._flatten_params(client.model)
                        if before is not None and after is not None and after.shape == before.shape:
                            delta_l2 = float(torch.norm(after - before, p=2))
                    except Exception:
                        pass

                    # averaged training loss (if exposed by client)
                    train_loss = None
                    try:
                        tr_sum, tr_n = client.train_metrics(task=task)
                        train_loss = (tr_sum / max(1, tr_n)) if tr_n else None
                    except Exception:
                        pass

                    # AA up to current task and forgetting (FedAvg-style)
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
                        "loss":   train_loss,
                        "acc":    aa_pct,      # %
                        "forg":   forg_pct,    # %
                        "time":   time.time() - t0_cli,
                        "samples": getattr(client, "train_samples", None),
                        "delta_l2": delta_l2,
                    })

                # show per-client table
                self._roundlog.clients_end(round_idx=disp_round, client_summaries=client_summaries)

                # (5) aggregate (AFFCL-specific)
                self.receive_models()
                self.receive_grads()
                model_origin = copy.deepcopy(self.global_model)
                self.aggregate_parameters_affcl()

                # diagnostics already in your class (angles/distances/norms)
                angle = [self.cos_sim(model_origin, self.global_model, models) for models in self.uploaded_models]
                distance = [self.distance(self.global_model, models) for models in self.uploaded_models]
                norm = [self.distance(model_origin, models) for models in self.uploaded_models]
                self.angle_value = float(np.mean(angle)) if len(angle) else 0.0
                self.distance_value = float(np.mean(distance)) if len(distance) else 0.0
                self.norm_value = float(np.mean(norm)) if len(norm) else 0.0
                angle_value = []
                for grad_i in self.grads:
                    for grad_j in self.grads:
                        angle_value.append(self.cosine_similarity(grad_i, grad_j))
                self.grads_angle_value = float(np.mean(angle_value)) if len(angle_value) else 0.0
                print(f"grad angle: {self.grads_angle_value}")

                # (6) optional local eval
                if i % eval_gap == 0:
                    try:
                        self.eval(task=task, glob_iter=glob_iter, flag="local")
                    except Exception:
                        pass

                # (7) end-of-round summary
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

            # end-of-task reporting
            if int(task/self.N_TASKS) == int(self.args.num_tasks/self.N_TASKS-1):
                if self.args.offlog and not self.args.debug:
                    self.eval_task(task=task, glob_iter=glob_iter, flag="local")
                    # need eval before data update
                    self.send_models()
                    self.eval_task(task=task, glob_iter=glob_iter, flag="global")

            # optional: dump GLOBAL per-task accuracy row once per task end
            try:
                if hasattr(self, "dump_global_task_accuracy_csv"):
                    self.dump_global_task_accuracy_csv(after_task=int(task), glob_iter=int(self.global_rounds * (task + 1) - 1))
            except Exception as e:
                print(f"[dump global_task_acc] warning: {e}")

        # finish progress
        self._roundlog.finish()

    # ---------------- AFFCL aggregation kept ----------------
    def aggregate_parameters_affcl(self, class_partial=False):
        assert (self.selected_clients is not None and len(self.selected_clients) > 0)

        param_dict = {}
        for name, param in self.global_model.named_parameters():
            param_dict[name] = torch.zeros_like(param.data)

        total_train = 0
        for client in self.selected_clients:
            total_train += len(client.train_data)  # weighted by train size

        param_weight_sum = {}
        for client in self.selected_clients:
            for name, param in client.model.named_parameters():
                if ('fc_classifier' in name and class_partial):
                    class_available = torch.Tensor(client.classes_so_far).long()
                    param_dict[name][class_available] += param.data[class_available] * len(client.train_data) / total_train
                    add_weight = torch.zeros([param.data.shape[0]]).cuda()
                    add_weight[class_available] = len(client.train_data) / total_train
                else:
                    param_dict[name] += param.data * len(client.train_data) / total_train
                    add_weight = len(client.train_data) / total_train

                if name not in param_weight_sum.keys():
                    param_weight_sum[name] = add_weight
                else:
                    param_weight_sum[name] += add_weight

        for name, param in self.global_model.named_parameters():
            if 'fc_classifier' in name and class_partial:
                valid_class = (param_weight_sum[name] > 0)
                weight_sum = param_weight_sum[name][valid_class]
                if 'weight' in name:
                    weight_sum = weight_sum.view(-1, 1)
                param.data[valid_class] = param_dict[name][valid_class] / weight_sum
            else:
                param.data = param_dict[name] / param_weight_sum[name]

    # ---------------- client creation kept ----------------
    def set_clients(self, clientObj):
        for i in range(self.num_clients):
            if self.args.partition_options == 'tuan':
                from utils.data_utils import read_client_data_FCL_cifar100, read_client_data_FCL_imagenet1k
                if self.args.dataset == 'IMAGENET1k':
                    train_data, label_info = read_client_data_FCL_imagenet1k(i, task=0, classes_per_task=self.args.cpt, count_labels=True)
                elif self.args.dataset == 'CIFAR100':
                    train_data, label_info = read_client_data_FCL_cifar100(i, task=0, classes_per_task=self.args.cpt, count_labels=True)
                else:
                    raise NotImplementedError("Not supported dataset")
            elif self.args.partition_options == 'hetero':
                from utils.data_utils_mine import read_client_data_FCL_cifar10, read_client_data_FCL_cifar100, read_client_data_FCL_imagenet1k
                if self.args.dataset == 'IMAGENET1k':
                    train_data, label_info = read_client_data_FCL_imagenet1k(i, task=0, classes_per_task=self.args.cpt, count_labels=True,
                                                                            seed = self.args.seed, alpha = self.args.alpha,
                                                                            total_clients = self.args.num_clients,task_disorder = self.args.task_disorder)
                elif self.args.dataset == 'CIFAR100':
                    train_data, label_info = read_client_data_FCL_cifar100(i, task=0, classes_per_task=self.args.cpt, count_labels=True,
                                                                        seed = self.args.seed, alpha = self.args.alpha,
                                                                        total_clients = self.args.num_clients,task_disorder = self.args.task_disorder)
                elif self.args.dataset == 'CIFAR10':
                    train_data, label_info = read_client_data_FCL_cifar10(i, task=0, classes_per_task=self.args.cpt, count_labels=True,
                                                                        seed = self.args.seed, alpha = self.args.alpha,
                                                                        total_clients = self.args.num_clients,task_disorder = self.args.task_disorder)
                else:
                    raise NotImplementedError("Not supported dataset")

            client = clientObj(self.args, id=i, train_data=train_data, classifier_head_list=self.classifier_head_list)
            self.clients.append(client)

            # update classes so far & current labels
            client.classes_so_far.extend(label_info['labels'])
            client.current_labels.extend(label_info['labels'])
            client.task_dict[0] = label_info['labels']

    def send_parameters(self, mode='all', beta=1, selected=False):
        users = self.clients
        if selected:
            assert (self.selected_users is not None and len(self.selected_users) > 0)
            users = self.selected_users

        for user in users:
            if mode == 'all':  # share all parameters
                user.set_parameters_precise(self.global_model, beta=beta)
            else:  # share a part parameters
                user.set_shared_parameters(self.global_model, mode=mode)
