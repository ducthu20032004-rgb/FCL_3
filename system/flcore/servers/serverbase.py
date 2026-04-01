import os
import json
import shutil
import torch
import torch.nn.functional as F
import wandb
import numpy as np
import csv
import copy
import time
import random
from datetime import datetime
# from utils.data_utils import *
from flcore.metrics.average_forgetting import metric_average_forgetting

import time
from utils.rich_progress import RichRoundLogger

# --- at top of file ---
import torch
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, List, Tuple

import statistics

class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.time_threthold = args.time_threthold
        self.offlog = args.offlog
        self._roundlog = RichRoundLogger(args, fig_dir=getattr(args, "fig_dir", "figures"))

        self._aa_cache = {"round": -1, "cc": None, "tt": None}  # per-class counts cache
        self._round_tag = -1  # set this each round in train()

        self.save_folder = f"{args.out_folder}/{args.dataset}_{args.algorithm}_{args.model_str}_{args.optimizer}_round{args.global_rounds}_localep{args.local_epochs}_lr{args.local_learning_rate}_nclient{args.num_clients}_part{args.partition_options}_cpt{args.cpt}_alpha{args.alpha}_disorder{args.task_disorder}_ddmmyy{datetime.now().strftime('%d%m%y_%H%M%S')}_time{time.time()}"
        if self.offlog:    
            if os.path.exists(self.save_folder):
                shutil.rmtree(self.save_folder)
            os.makedirs(self.save_folder, exist_ok=True)

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate

        self.global_accuracy_matrix = []
        self.local_accuracy_matrix = []

        if self.args.dataset == 'IMAGENET1k':
            self.N_TASKS = 500
        elif self.args.dataset == 'CIFAR100':
            self.N_TASKS = 50
        elif self.args.dataset == 'CIFAR10':
            self.N_TASKS = 5
        if self.args.nt is not None:
            self.N_TASKS = self.args.num_classes // self.args.cpt

        # FCL
        self.task_dict = {}
        self.current_task = 0

        self.angle_value = 0
        self.grads_angle_value = 0
        self.distance_value = 0
        self.norm_value = 0

        self.client_accuracy_matrix = {}

        self.file_name = f"{self.args.algorithm}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    def save_checkpoint(self, glob_iter: int, tag: str = "latest"):
        """
        Save GLOBAL model checkpoint.
        """
        ckpt_dir = os.path.join("checkpoints", self.dataset, self.algorithm)
        os.makedirs(ckpt_dir, exist_ok=True)

        ckpt_path = os.path.join(ckpt_dir, f"{tag}_round_{glob_iter}.pth")

        torch.save({
            "round": glob_iter,
            "model_state": self.global_model.state_dict(),
            "args": vars(self.args),
            "timestamp": time.time(),
        }, ckpt_path)

        print(f"[CKPT] Saved: {ckpt_path}")
        return ckpt_path


    def load_checkpoint(self, ckpt_path: str):
        """
        Load GLOBAL model checkpoint.
        """
        assert os.path.isfile(ckpt_path), f"Checkpoint not found: {ckpt_path}"

        ckpt = torch.load(ckpt_path, map_location=self.device,weights_only=False)
        self.global_model.load_state_dict(ckpt["model_state"])
        self.global_model.to(self.device)
        self.global_model.eval()

        start_round = ckpt.get("round", 0)
        print(f"[CKPT] Loaded: {ckpt_path} (round={start_round})")
        return start_round
    def _cifar_transforms_for_eval(self):
        # keep it simple: ToTensor() -> [0,1]; matches your _to_tensordataset normalization
        from torchvision import transforms
        return transforms.Compose([transforms.ToTensor()])

    def _build_cifar_global_test(self, dataset_name: str):
        from torchvision import datasets
        tfm = self._cifar_transforms_for_eval()
        if dataset_name.upper() == "CIFAR10":
            ds = datasets.CIFAR10(root="/home/lucaznguyen/FCL/dataset", train=False, download=True, transform=tfm)
        elif dataset_name.upper() == "CIFAR100":
            ds = datasets.CIFAR100(root="/home/lucaznguyen/FCL/dataset", train=False, download=True, transform= tfm)
        else:
            raise ValueError("Use a dataset-specific global test loader for ImageNet1K.")
        return ds

    def _build_imagenet1k_global_test_from_npy(root_dir: str = "dataset/imagenet1k-val-classes"):
        """
        Optional: if you have per-class .npy for ImageNet-1K validation, build a flat TensorDataset here.
        Expected structure: root_dir/{0..999}.npy with HWC uint8.
        """
        import numpy as np
        import os
        xs, ys = [], []
        for k in range(1000):
            p = os.path.join(root_dir, f"{k}.npy")
            if not os.path.exists(p): continue
            arr = np.load(p, allow_pickle=True)  # (N,H,W,3) or (N,3,H,W)
            if arr.ndim == 4 and arr.shape[-1] == 3:
                pass
            elif arr.ndim == 4 and arr.shape[1] == 3:
                arr = np.transpose(arr, (0,2,3,1))
            else:
                continue
            xs.append(arr)
            ys.append(np.full((arr.shape[0],), k, dtype=np.int64))
        if not xs:
            raise FileNotFoundError("No ImageNet-1K val npy files found; please prepare them or implement your own loader.")
        import numpy as np
        import torch
        X = np.concatenate(xs, axis=0)              # HWC uint8
        Y = np.concatenate(ys, axis=0)              # labels
        X = torch.tensor(np.transpose(X, (0,3,1,2)), dtype=torch.float32) / 255.0
        Y = torch.tensor(Y, dtype=torch.long)
        from torch.utils.data import TensorDataset
        return TensorDataset(X, Y)
    
    def _global_task_pool_labels(self):
        """
        Return the GLOBAL task pool: a list of T lists, each containing cpt global class IDs.
        Uses the natural master order [0..K-1] chunked by cpt.
        """
        K   = int(self.args.num_classes)
        cpt = int(self.args.cpt)
        T   = (K + cpt - 1) // cpt
        Y_sets = [list(range(t*cpt, min((t+1)*cpt, K))) for t in range(T)]
        return Y_sets  # length T, each a list of global class IDs



    @torch.no_grad()
    def _eval_on_global_test_restricted(self, model, label_set: set[int], upto_task=None):
        """
        FAST: use per-class (correct,total) counts cached once per round.
        Returns (correct, total) for the given label_set.
        """
        if not label_set:
            return 0, 0
        # Make sure the cache is built for this round
        cc, tt = self._get_or_build_global_counts(self._round_tag)
        idx = np.fromiter((int(k) for k in label_set), dtype=np.int64)
        return int(cc[idx].sum()), int(tt[idx].sum())


    # # ---- Per-client task labels (robust getter) ----
    def _get_client_task_labels(self, client, task_idx: int):
        for name in ("task_info", "task_meta", "task_label_info"):
            d = getattr(client, name, None)
            if isinstance(d, dict) and task_idx in d and d[task_idx]:
                info = d[task_idx]
                if "assigned_labels" in info: return [int(x) for x in info["assigned_labels"]]
                if "labels" in info:          return [int(x) for x in info["labels"]]
        labels = client.task_dict.get(task_idx, [])
        if hasattr(client, "task_dict") and len(labels) > 0:
            return [int(x) for x in client.task_dict[task_idx]]
        return []

    def _client_AA_global_upto(self, client, upto_task: int) -> Optional[float]:
        """
        AA_c(<=t) = (1/(t+1)) * sum_{s=0..t} A_c,s
        A_c,s is GLOBAL (micro) accuracy on the separate global test set restricted to client c's task-s labels.
        Returns percentage in [0,100] or None if no data.
        """
        import numpy as np
        accs = []
        for s in range(upto_task + 1):
            labels_s = self._get_client_task_labels(client, s)
            if not labels_s: continue
            corr, tot = self._eval_on_global_test_restricted(self.global_model, set(labels_s))
            if tot > 0:
                accs.append(corr / tot)
        if not accs:
            return None
        return 100.0 * float(np.mean(accs))

    def _client_acc_vector_all_tasks_from_counts(self, client, counts_correct: np.ndarray, counts_total: np.ndarray):
        """
        Build A^{(c)}_{*,k} at the CURRENT eval point: a length-N_TASKS vector of per-task accuracies
        for this client, using global per-class (correct,total) counts and the client's task label sets.
        """
        vec = []
        for s in range(self.N_TASKS):
            labels_s = self._get_client_task_labels(client, s)
            if not labels_s:
                vec.append(0.0); continue
            idx = np.array(labels_s, dtype=np.int64)
            corr = int(counts_correct[idx].sum())
            tot  = int(counts_total[idx].sum())
            vec.append((corr / tot) if tot > 0 else 0.0)
        return vec

    # ====== NEW EVALUATION CORE (GLOBAL TEST, PER-CLIENT TASK VECTORS) ======

    # ---------- GLOBAL TEST LOADER (if available) ----------
    def _ensure_global_test_loader(self):
        """
        Try to build a single global test DataLoader with GLOBAL labels 0..K-1.
        If not available (e.g., ImageNet-1K without a prepared loader), return None;
        the fallback path will iterate the union of client test loaders with remapping.
        """
        if getattr(self, "_global_test_loader", None) is not None:
            return self._global_test_loader
        try:
            import torchvision, torchvision.transforms as T
            from torch.utils.data import DataLoader
            ds = str(getattr(self.args, "dataset", "")).upper()
            tfm = T.Compose([T.ToTensor()])
            if ds == "CIFAR10":
                testset = torchvision.datasets.CIFAR10(root="dataset", train=False, download=False, transform=tfm)
                K = 10
            elif ds == "CIFAR100":
                testset = torchvision.datasets.CIFAR100(root="dataset", train=False, download=False, transform=tfm)
                K = 100
            else:
                return None
            bs = int(getattr(self.args, "batch_size", 128) or 128)
            self._global_test_loader = DataLoader(testset, batch_size=bs, shuffle=False, num_workers=0, pin_memory=False)
            self._global_test_K = K
            return self._global_test_loader
        except Exception:
            return None

    @torch.no_grad()
    def _per_class_counts_from_loader(self, model, loader, K: int):
        """
        One pass: global labels (0..K-1) -> per-class (correct, total).
        """
        import numpy as np, torch
        dev = next(model.parameters()).device
        was_training = model.training
        model.eval()
        cc = np.zeros(K, dtype=np.int64)
        tt = np.zeros(K, dtype=np.int64)
        try:
            for batch in loader:
                x, y = (batch[0], batch[1]) if isinstance(batch, (list, tuple)) else batch
                x = x.to(dev, non_blocking=False)
                y = y.to(dev, non_blocking=False)
                pred = model(x).argmax(dim=1)
                y_np = y.detach().cpu().numpy()
                m_np = (pred == y).detach().cpu().numpy().astype(bool)
                cc[:] += np.bincount(y_np[m_np], minlength=K)[:K]
                tt[:] += np.bincount(y_np,       minlength=K)[:K]
        finally:
            if was_training: model.train()
        return cc, tt

    # ---------- FALLBACK: UNION-OF-CLIENTS WITH REMAP TO GLOBAL ----------
    def _labels_for_client_task(self, client, task_idx: int):
        """
        Robust resolver for GLOBAL class IDs at (client, task_idx).
        Uses cached meta when available; reconstructs for 'mine' if needed.
        """
        # try meta first
        for name in ("task_info", "task_meta", "task_label_info"):
            d = getattr(client, name, None)
            if isinstance(d, dict) and task_idx in d and d[task_idx]:
                info = d[task_idx]
                if "assigned_labels" in info: return [int(x) for x in info["assigned_labels"]]
                if "labels" in info:          return [int(x) for x in info["labels"]]
        labels = client.task_dict.get(task_idx, [])
        if hasattr(client, "task_dict") and len(labels) > 0:
            return [int(x) for x in client.task_dict[task_idx]]

        # reconstruct for 'mine'
        if getattr(self.args, "partition_options", "current") == "mine":
            import numpy as np
            K   = int(self.args.num_classes)
            cpt = int(self.args.cpt)
            T   = (K + cpt - 1) // cpt
            # global task pool from [0..K-1]
            Y_sets = [list(range(t*cpt, min((t+1)*cpt, K))) for t in range(T)]
            # disorder for this client (client 0 = 0)
            psi_default = float(getattr(self.args, "mine_task_disorder", 0.0))
            psi_c = 0.0 if client.id == 0 else psi_default
            # overrides
            ov = getattr(self.args, "mine_client_disorder", None)
            if isinstance(ov, str) and ov.strip():
                try:
                    vals = [float(x.strip()) for x in ov.strip().replace('[','').replace(']','').split(',') if x.strip()]
                    if client.id < len(vals):
                        psi_c = max(0.0, min(1.0, vals[client.id]))
                except Exception:
                    pass
            # adjacent-swap with seed
            seed = int(getattr(self.args, "mine_seed", 7))
            r = np.random.default_rng(seed + 997 * client.id)
            order = list(range(T))
            if psi_c > 0.0:
                for i in range(T-1):
                    if r.random() < psi_c:
                        order[i], order[i+1] = order[i+1], order[i]
            labels = Y_sets[order[task_idx]] if 0 <= task_idx < T else []
            try:
                if hasattr(client, "task_dict"):
                    client.task_dict[task_idx] = labels
            except Exception:
                pass
            return labels
        return []

    @torch.no_grad()
    def _per_class_counts_from_union_with_remap(self, model):
        """
        Iterate client test loaders; if a loader uses task-local labels (0..cpt-1),
        remap to GLOBAL class IDs using that client's task labels for the loader index.
        """
        import numpy as np, torch
        K = int(getattr(self.args, "num_classes", 100))
        cc = np.zeros(K, dtype=np.int64)
        tt = np.zeros(K, dtype=np.int64)
        dev = next(model.parameters()).device
        was_training = model.training
        model.eval()
        try:
            for cli in self.clients:
                # choose available structure
                if hasattr(cli, "task_test_loaders") and isinstance(cli.task_test_loaders, (list, tuple)):
                    loaders = list(cli.task_test_loaders)
                elif hasattr(cli, "test_loaders") and isinstance(cli.test_loaders, (list, tuple)):
                    loaders = list(cli.test_loaders)
                elif hasattr(cli, "test_loader") and cli.test_loader is not None:
                    loaders = [cli.test_loader]
                else:
                    loaders = []
                for s, ld in enumerate(loaders):
                    gl_labels = self._labels_for_client_task(cli, s)  # global IDs for this task
                    gl_arr = np.array(gl_labels, dtype=np.int64) if len(gl_labels) else None
                    for batch in ld:
                        x, y = (batch[0], batch[1]) if isinstance(batch, (list, tuple)) else batch
                        x = x.to(dev, non_blocking=False)
                        y = y.to(dev, non_blocking=False)
                        pred = model(x).argmax(dim=1)

                        y_local = y.detach().cpu().numpy()
                        if gl_arr is not None and gl_arr.size > 0:
                            y_local = np.clip(y_local, 0, gl_arr.size-1)
                            y_np = gl_arr[y_local]   # remapped to global IDs
                        else:
                            y_np = y_local           # best effort

                        p_np = pred.detach().cpu().numpy()
                        m_np = (p_np == y_np)
                        cc[:] += np.bincount(y_np[m_np], minlength=K)[:K]
                        tt[:] += np.bincount(y_np,       minlength=K)[:K]
        finally:
            if was_training: model.train()
        return cc, tt

    @torch.no_grad()
    def _compute_global_per_class_counts(self, model):
        """
        Build per-class (correct,total) counts on the GLOBAL test set.
        Prefer a clean global test loader (labels 0..K-1); if unavailable,
        fall back to the union of clients' test loaders and remap task-local labels.
        Returns (cc, tt, K).
        """
        import numpy as np, torch

        K = int(getattr(self.args, "num_classes", 100))
        cc = np.zeros(K, dtype=np.int64)
        tt = np.zeros(K, dtype=np.int64)

        # --- try a single global test loader if you have one
        loader = getattr(self, "_global_test_loader", None)
        if loader is None:
            try: loader = self._ensure_global_test_loader()
            except Exception: loader = None

        dev = next(model.parameters()).device
        was_training = model.training
        model.eval()

        def _accumulate_global(x, y):
            x = x.to(dev, non_blocking=False)
            y = y.to(dev, non_blocking=False)
            pred = model(x).argmax(dim=1)
            y_np = y.detach().cpu().numpy()
            m_np = (pred == y).detach().cpu().numpy().astype(bool)
            cc[:] += np.bincount(y_np[m_np], minlength=K)[:K]
            tt[:] += np.bincount(y_np,       minlength=K)[:K]

        def _accumulate_with_remap(x, y, gl_labels):
            # gl_labels: list of GLOBAL class IDs for this task (order matches local labels 0..len-1)
            import numpy as np
            x = x.to(dev, non_blocking=False)
            y = y.to(dev, non_blocking=False)
            pred = model(x).argmax(dim=1)

            y_local = y.detach().cpu().numpy()
            if gl_labels and len(gl_labels) > 0:
                gl = np.array(gl_labels, dtype=np.int64)
                y_local = np.clip(y_local, 0, gl.size - 1)
                y_np = gl[y_local]
            else:
                y_np = y_local  # best effort

            p_np = pred.detach().cpu().numpy()
            m_np = (p_np == y_np)
            cc[:] += np.bincount(y_np[m_np], minlength=K)[:K]
            tt[:] += np.bincount(y_np,       minlength=K)[:K]

        try:
            if loader is not None:
                # Path A: true global test loader with global labels 0..K-1
                for batch in loader:
                    x, y = (batch[0], batch[1]) if isinstance(batch, (list, tuple)) else batch
                    _accumulate_global(x, y)
            else:
                # Path B: union of clients (remap task-local -> global)
                Y_sets = self._global_task_pool_labels()  # use global pool for mapping by task index
                for cli in self.clients:
                    # pick structure
                    if hasattr(cli, "task_test_loaders") and isinstance(cli.task_test_loaders, (list, tuple)):
                        loaders = list(cli.task_test_loaders)
                    elif hasattr(cli, "test_loaders") and isinstance(cli.test_loaders, (list, tuple)):
                        loaders = list(cli.test_loaders)
                    elif hasattr(cli, "test_loader") and cli.test_loader is not None:
                        loaders = [cli.test_loader]
                    else:
                        loaders = []
                    for s, ld in enumerate(loaders):
                        gl_labels = Y_sets[s] if s < len(Y_sets) else []  # map local 0.. -> global IDs
                        for batch in ld:
                            x, y = (batch[0], batch[1]) if isinstance(batch, (list, tuple)) else batch
                            _accumulate_with_remap(x, y, gl_labels)
        finally:
            if was_training: model.train()

        return cc, tt, K
    
    def _global_task_accuracy_vector_from_counts(self, cc, tt):
        """
        From per-class counts, compute [A_{*,k}] for k=0..T-1 where each
        A_{*,k} = (sum_{y in Y_k} cc[y]) / (sum_{y in Y_k} tt[y]),
        and {Y_k} is the GLOBAL task pool.
        Returns a list of floats in [0,1].
        """
        import numpy as np
        Y_sets = self._global_task_pool_labels()
        vec = []
        for k, Yk in enumerate(Y_sets):
            if not Yk:
                vec.append(0.0); continue
            idx = np.array(Yk, dtype=np.int64)
            corr = int(cc[idx].sum())
            tot  = int(tt[idx].sum())
            vec.append((corr / tot) if tot > 0 else 0.0)
        return vec



    # ---------- BACK-COMPAT SHIM (what your code is still calling) ----------
    def _get_or_build_global_counts(self, round_tag: int):
        """
        Back-compat shim so old call sites keep working.
        Caches per-class counts for the given 'round_tag' and returns (cc, tt).
        """
        c = self._aa_cache
        if c["round"] != int(round_tag) or c["cc"] is None or c["tt"] is None:
            cc, tt, K = self._compute_global_per_class_counts(self.global_model)
            self._aa_cache = {"round": int(round_tag), "cc": cc, "tt": tt, "K": K}
        return self._aa_cache["cc"], self._aa_cache["tt"]


    def _labels_for_client_task(self, client, task_idx: int):
        """
        Robust resolver: return GLOBAL class IDs for (client, task_idx).
        Works for both 'current' and 'mine'. For 'mine' we reconstruct the task
        order from CLI knobs if not yet cached in client.task_dict.
        """
        # 1) Try client-side metadata first
        for name in ("task_info", "task_meta", "task_label_info"):
            d = getattr(client, name, None)
            if isinstance(d, dict) and task_idx in d and d[task_idx]:
                info = d[task_idx]
                if "assigned_labels" in info: return [int(x) for x in info["assigned_labels"]]
                if "labels" in info:          return [int(x) for x in info["labels"]]
        labels = client.task_dict.get(task_idx, [])
        if hasattr(client, "task_dict") and len(labels) > 0:
            return [int(x) for x in client.task_dict[task_idx]]

        # 2) Reconstruct for 'mine'
        if getattr(self.args, "partition_options", "current") == "mine":
            import numpy as np
            K   = int(self.args.num_classes)
            cpt = int(self.args.cpt)
            T   = (K + cpt - 1) // cpt

            # global task pool from master order [0..K-1], chunked by cpt
            Y_sets = [list(range(t*cpt, min((t+1)*cpt, K))) for t in range(T)]

            # disorder for this client (client 0 fixed to 0)
            psi_default = float(getattr(self.args, "mine_task_disorder", 0.0))
            psi_c = 0.0 if client.id == 0 else psi_default

            # optional overrides
            ov = getattr(self.args, "mine_client_disorder", None)
            if isinstance(ov, str) and ov.strip():
                s = ov.strip().replace('[','').replace(']','')
                try:
                    vals = [float(x.strip()) for x in s.split(',') if x.strip()!='']
                    if client.id < len(vals):
                        v = vals[client.id]
                        psi_c = max(0.0, min(1.0, v))
                except Exception:
                    pass

            # adjacent-swap permutation with seed
            seed = int(getattr(self.args, "mine_seed", 7))
            r = np.random.default_rng(seed + 997 * client.id)
            order = list(range(T))
            if psi_c > 0.0:
                for i in range(T-1):
                    if r.random() < psi_c:
                        order[i], order[i+1] = order[i+1], order[i]

            if 0 <= task_idx < T:
                labels = Y_sets[order[task_idx]]
            else:
                labels = []

            # cache back for later
            try:
                if hasattr(client, "task_dict"):
                    client.task_dict[task_idx] = labels
            except Exception:
                pass
            return labels

        # 3) Otherwise, unknown
        return []


    def _ensure_global_test_loader(self):
        """
        Build a SINGLE global test DataLoader with GLOBAL labels (0..K-1).
        Prefers torchvision test split; if unavailable, returns None and the caller
        should use the fallback that remaps union-of-clients loaders.
        """
        if getattr(self, "_global_test_loader", None) is not None:
            return self._global_test_loader

        try:
            import torchvision
            import torchvision.transforms as T
            from torch.utils.data import DataLoader

            ds = str(getattr(self.args, "dataset", "")).upper()
            tfm = T.Compose([T.ToTensor()])  # keep simple; eval-time
            if ds == "CIFAR10":
                testset = torchvision.datasets.CIFAR10(root="dataset", train=False, download=False, transform=tfm)
                K = 10
            elif ds == "CIFAR100":
                testset = torchvision.datasets.CIFAR100(root="dataset", train=False, download=False, transform=tfm)
                K = 100
            else:
                return None  # Not implemented here (e.g., ImageNet-1K); fallback path will handle.

            bs = int(getattr(self.args, "batch_size", 128) or 128)
            self._global_test_loader = DataLoader(testset, batch_size=bs, shuffle=False, num_workers=0, pin_memory=False)
            self._global_test_K = K
            return self._global_test_loader
        except Exception:
            return None


    @torch.no_grad()
    def _per_class_counts_from_loader(self, model, loader, K: int):
        """
        Generic per-class (correct,total) counts on a loader that yields GLOBAL labels 0..K-1.
        """
        import numpy as np
        import torch

        dev = next(model.parameters()).device
        was_training = model.training
        model.eval()

        cc = np.zeros(K, dtype=np.int64)
        tt = np.zeros(K, dtype=np.int64)

        try:
            for batch in loader:
                x, y = (batch[0], batch[1]) if isinstance(batch, (list, tuple)) else batch
                x = x.to(dev, non_blocking=False)
                y = y.to(dev, non_blocking=False)
                pred = model(x).argmax(dim=1)

                y_np = y.detach().cpu().numpy()
                m_np = (pred == y).detach().cpu().numpy().astype(np.bool_)  # bool
                cc[:] += np.bincount(y_np[m_np], minlength=K)[:K]
                tt[:] += np.bincount(y_np,       minlength=K)[:K]
        finally:
            if was_training:
                model.train()
        return cc, tt


    @torch.no_grad()
    def _per_class_counts_from_union_with_remap(self, model):
        """
        Fallback: iterate each client's test loaders. If a loader uses TASK-LOCAL labels (0..cpt-1),
        remap them to GLOBAL class IDs using the client’s task labels for that loader index.
        """
        import numpy as np
        import torch

        K = int(getattr(self.args, "num_classes", 100))
        cc = np.zeros(K, dtype=np.int64)
        tt = np.zeros(K, dtype=np.int64)

        dev = next(model.parameters()).device
        was_training = model.training
        model.eval()

        def _accumulate_with_gl_map(x, y, gl_labels):
            # gl_labels : list of global class IDs, in the same order as local labels 0..len-1
            x = x.to(dev, non_blocking=False)
            y = y.to(dev, non_blocking=False)
            pred = model(x).argmax(dim=1)

            y_local = y.detach().cpu().numpy()
            # map local -> global safely
            if len(gl_labels) > 0:
                gl = np.array(gl_labels, dtype=np.int64)
                y_local = np.clip(y_local, 0, gl.size - 1)
                y_np = gl[y_local]  # global ids
            else:
                y_np = y_local  # best effort, assume global already

            p_np = pred.detach().cpu().numpy()
            m_np = (p_np == y_np)
            cc[:] += np.bincount(y_np[m_np], minlength=K)[:K]
            tt[:] += np.bincount(y_np,       minlength=K)[:K]

        try:
            # Iterate per client, per task loader
            for cli in self.clients:
                # choose the available structure
                if hasattr(cli, "task_test_loaders") and isinstance(cli.task_test_loaders, (list, tuple)):
                    loaders = list(cli.task_test_loaders)
                elif hasattr(cli, "test_loaders") and isinstance(cli.test_loaders, (list, tuple)):
                    loaders = list(cli.test_loaders)
                elif hasattr(cli, "test_loader") and cli.test_loader is not None:
                    loaders = [cli.test_loader]
                else:
                    loaders = []

                for s, ld in enumerate(loaders):
                    # labels for this client's task 's' in GLOBAL id space
                    gl_labels = self._labels_for_client_task(cli, s)

                    for batch in ld:
                        x, y = (batch[0], batch[1]) if isinstance(batch, (list, tuple)) else batch
                        _accumulate_with_gl_map(x, y, gl_labels)
        finally:
            if was_training:
                model.train()

        return cc, tt


    def _compute_global_per_class_counts(self, model):
        """
        One entry-point: try a clean global test loader (global labels). If unavailable,
        fall back to union-of-clients + remap to GLOBAL labels. Returns (cc, tt, K).
        """
        loader = self._ensure_global_test_loader()
        if loader is not None:
            K = int(getattr(self, "_global_test_K", getattr(self.args, "num_classes", 100)))
            cc, tt = self._per_class_counts_from_loader(model, loader, K)
            return cc, tt, K
        else:
            cc, tt = self._per_class_counts_from_union_with_remap(model)
            K = int(getattr(self.args, "num_classes", 100))
            return cc, tt, K


    def compute_client_task_vectors(self, model):
        """
        Returns: dict cid -> list[N_TASKS] with plain accuracy per task (fraction in [0,1]),
        using the CURRENT 'model' and a GLOBAL test set (or remapped union).
        """
        import numpy as np
        cc, tt, K = self._compute_global_per_class_counts(model)

        T = int(self.N_TASKS)
        out = {}
        for cli in self.clients:
            vec = []
            for k in range(T):
                labels = self._labels_for_client_task(cli, k)
                if not labels:
                    vec.append(0.0); continue
                idx = np.array(labels, dtype=np.int64)
                corr = int(cc[idx].sum())
                tot  = int(tt[idx].sum())
                vec.append((corr / tot) if tot > 0 else 0.0)
            out[cli.id] = vec
        return out


    def dump_global_task_accuracy_csv(self, after_task: int, glob_iter: int):
        """
        Compute the GLOBAL per-task accuracy vector (current global model) and
        write it into each client's CSV (for convenience), and into a single
        long-format file. Values in [0,1], not %.
        """
        if glob_iter is None:
            glob_iter = getattr(self, "_round_tag", -1)
        import os, csv
        acc_vec = self._global_task_accuracy_vector_from_counts(*self._compute_global_per_class_counts(self.global_model)[:2])

        T = len(acc_vec)
        # Global long file
        global_dir = os.path.join(self.save_folder, "Global")
        os.makedirs(global_dir, exist_ok=True)
        long_csv = os.path.join(global_dir, "global_task_acc_long.csv")
        need_header_long = not os.path.exists(long_csv)
        with open(long_csv, "a", newline="") as f_long:
            w_long = csv.writer(f_long)
            if need_header_long:
                w_long.writerow(["after_task", "step", "eval_task", "acc"])  # global per-task
            for k, a in enumerate(acc_vec):
                w_long.writerow([int(after_task), int(glob_iter), int(k), f"{float(a):.6f}"])

        # Per-client wide (same vector for each client, since same global model)
        for cli in self.clients:
            cdir = os.path.join(self.save_folder, "Client_Global", f"Client_{cli.id}")
            os.makedirs(cdir, exist_ok=True)
            wide_csv = os.path.join(cdir, "per_task_acc.csv")
            need_header_wide = not os.path.exists(wide_csv)
            import csv
            with open(wide_csv, "a", newline="") as f_wide:
                w = csv.writer(f_wide)
                if need_header_wide:
                    w.writerow(["after_task", "step"] + [f"task_{k}" for k in range(T)])
                w.writerow([int(after_task), int(glob_iter)] + [f"{float(a):.6f}" for a in acc_vec])


    def set_clients(self, clientObj):
        for i in range(self.num_clients):
            print(f"Creating client {i} ...")

            if self.args.partition_options == "tuan":
                from utils.data_utils import read_client_data_FCL_cifar10, read_client_data_FCL_cifar100, read_client_data_FCL_imagenet1k

                if self.args.dataset == 'IMAGENET1k':
                    train_data, label_info = read_client_data_FCL_imagenet1k(i, task=0, classes_per_task=self.args.cpt, count_labels=True)
                elif self.args.dataset == 'CIFAR100':
                    train_data, label_info = read_client_data_FCL_cifar100(i, task=0, classes_per_task=self.args.cpt, count_labels=True)
                elif self.args.dataset == 'CIFAR10':
                    train_data, label_info = read_client_data_FCL_cifar10(i, task=0, classes_per_task=self.args.cpt, count_labels=True)
                else:
                    raise NotImplementedError("Not supported dataset")

            elif self.args.partition_options == "hetero":
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

            client = clientObj(self.args, id=i, train_data=train_data)
            self.clients.append(client)

            # update classes so far & current labels
            client.classes_so_far.extend(label_info['labels'])
            client.current_labels.extend(label_info['labels'])
            client.task_dict[0] = label_info['labels']
            client.file_name = self.file_name

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = sorted(
            random.sample(
                self.selected_clients, 
                int((1 - self.client_drop_rate) * self.current_num_join_clients)
            ), 
            key=lambda client: client.id
        )

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += len(client.train_data)
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(len(client.train_data))
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def receive_grads(self):

        self.grads = copy.deepcopy(self.uploaded_models)
        # This for copy the list to store all the gradient update value

        for model in self.grads:
            for param in model.parameters():
                param.data.zero_()

        for grad_model, local_model in zip(self.grads, self.uploaded_models):
            for grad_param, local_param, global_param in zip(grad_model.parameters(), local_model.parameters(),
                                                             self.global_model.parameters()):
                grad_param.data = local_param.data - global_param.data
        for w, client_model in zip(self.uploaded_weights, self.grads):
            self.mul_params(w, client_model)

    def mul_params(self, w, client_model):
        for param in client_model.parameters():
            param.data = param.data.clone() * w

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def test_metrics(self, task, glob_iter, flag):
        
        num_samples = []
        tot_correct = []

        # tag this round; any monotonically increasing counter works
        self._round_tag = glob_iter  # or use your own global round index

        # 1) Build per-class counts once this round
        cc, tt = self._get_or_build_global_counts(self._round_tag)

        per_client_forgetting = {}

        for c in self.clients:
            ct, ns = c.test_metrics(task=task)
            tot_correct.append(ct*1.0)
            num_samples.append(ns)

            # acc_vec = self._client_acc_vector_all_tasks_from_counts(c, cc, tt)   # [A_k] for this eval point
            # # self.client_accuracy_matrix.setdefault(c.id, []).append(acc_vec)     # append time row

            # # average forgetting up to current task index
            # cf = metric_average_forgetting(int(task % self.N_TASKS), self.client_accuracy_matrix[c.id])
            # per_client_forgetting[c.id] = float(cf)  # keep as fraction

            # test_acc = sum(tot_correct)*1.0 / sum(num_samples)
    
            # if flag != "off":
            #     if flag == "global":
            #         subdir = os.path.join(self.save_folder, f"Client_Global/Client_{c.id}")
            #         log_key = f"Client_Global/Client_{c.id}/Averaged Test Accurancy"
            #     elif flag == "local":
            #         subdir = os.path.join(self.save_folder, f"Client_Local/Client_{c.id}")
            #         log_key = f"Client_Local/Client_{c.id}/Averaged Test Accurancy"

            #     aa_pct = self._client_AA_global_upto(c, upto_task=task)

            #     # if self.args.wandb:
            #     #     wandb.log({log_key: test_acc}, step=glob_iter)

            #     # if self.args.wandb:
            #     #     if aa_pct is not None:
            #     #         print("function test_metrics : LOGGING STEP:", glob_iter, log_dict.keys())
            #     #         log_dict[f"Client_{c.id}/AA"] = aa_pct
            #     #         log_dict[f"Client_{c.id}/Forgetting"] = 100.0 * cf
            #     #     wandb.log(log_dict, step=glob_iter)
            #     if self.offlog:
            #         os.makedirs(subdir, exist_ok=True)

            #         file_path = os.path.join(subdir, "test_accuracy.csv")
            #         file_exists = os.path.isfile(file_path)

            #         with open(file_path, mode="w", newline="") as f:
            #             writer = csv.writer(f)
            #             if not file_exists:
            #                 writer.writerow(["Step", "Test Accuracy"])  
            #             writer.writerow([glob_iter, test_acc]) 

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct

    def train_metrics(self, task=None):

        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics(task=task)
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def eval(self, task, glob_iter, flag):
        stats = self.test_metrics(task, glob_iter, flag=flag)
        stats_train = self.train_metrics(task=task)
        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        if flag == "global":
            subdir = os.path.join(self.save_folder, "Global")
            log_keys = {
                "Global/Averaged Train Loss": train_loss,
                "Global/Averaged Test Accuracy": test_acc,
                # "Global/Averaged Angle": self.angle_value,
                "Global/Averaged Grads Angle": self.grads_angle_value,
                "Global/Averaged Distance": self.distance_value,
                "Global/Averaged GradNorm": self.norm_value,
            }
            if self.args.tgm:
                self.t_angle_after = statistics.mean(client.t_angle_after for client in self.selected_clients)

                log_keys.update({
                    "Global/Timestep Angle After": self.t_angle_after,
                })
                # print(log_keys)

        elif flag == "local":
            subdir = os.path.join(self.save_folder, "Local")
            log_keys = {
                "Local/Averaged Train Loss": train_loss,
                "Local/Averaged Test Accuracy": test_acc,
            }


        if self.args.log and flag == "global":
            # print(f"{sum(stats_train[2])}, {sum(stats_train[1])}")task_id
            print(f"Global Averaged Test Accuracy: {test_acc}")
            print(f"Global Averaged Test Loss: {train_loss}")
            # if self.args.wandb:
            #     wandb.log({
            #         "Global/Averaged Test Accuracy": test_acc,
            #         "Global/Averaged Test Loss": train_loss,
            #     },step = glob_iter)
                
        if self.args.log and flag == "local":
            # print(f"{sum(stats_train[2])}, {sum(stats_train[1])}")
            print(f"Local Averaged Test Accuracy: {test_acc}")
            print(f"Local Averaged Test Loss: {train_loss}")

        if self.args.wandb:
            wandb.log(log_keys, step=glob_iter)

        # if self.args.wandb:
        #     for cli in self.clients:
        #         aa_pct = self._client_AA_global_upto(cli, upto_task=task)
        #         print(f"Client {cli.id} AA% up to task {task}: {aa_pct}")
        #         wandb.log({f"Client_Global/Client_{cli.id}/AA_upto_t_global": aa_pct}, step=glob_iter)

        if self.offlog:
            os.makedirs(subdir, exist_ok=True)

            file_path = os.path.join(subdir, "metrics.csv")
            file_exists = os.path.isfile(file_path)

            with open(file_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Step", "Train Loss", "Test Accuracy"])  
                writer.writerow([glob_iter, train_loss, test_acc]) 
    def compute_bwt(self, accuracy_matrix: list, task: int) -> float:
        """
        BWT = (1 / task) * Σ_{s=0}^{task-1} [ A_{task,s} - A_{s,s} ]

        A_{task, s} = acc on task s AFTER learning task t  (current row, col s)
        A_{s,s}     = acc on task s RIGHT AFTER learning it (diagonal)

        BWT < 0  →  forgetting
        BWT > 0  →  positive backward transfer (rất hiếm)

        Args:
            accuracy_matrix : list of rows
            task            : task index hiện tại (>= 1 mới có ý nghĩa)

        Returns:
            float  (0.0 nếu task == 0)
        """
        if task == 0 or len(accuracy_matrix) < 2:
            return 0.0

        current_row = accuracy_matrix[task]           # A_{task, *}
        bwt_sum = 0.0
        count = 0
        for s in range(task):                         # s = 0 .. task-1
            a_ss  = accuracy_matrix[s][s]             # diagonal: acc on task s right after trained
            a_ts  = current_row[s]                    # current acc on task s
            bwt_sum += (a_ts - a_ss)
            count += 1

        return bwt_sum / count if count > 0 else 0.0


    def compute_fwt(self, accuracy_matrix: list, task: int,
                    random_baseline: float = None) -> float:
        """
        FWT = (1 / task) * Σ_{s=1}^{task} [ A_{s-1, s} - b_s ]

        A_{s-1, s} = acc on task s BEFORE it was trained  (zero-shot transfer)
        b_s        = random baseline = 1 / cpt  (hoặc truyền vào nếu có baseline riêng)

        FWT > 0  →  model generalise tốt sang tasks chưa học
        FWT < 0  →  model bị interference

        Args:
            accuracy_matrix  : list of rows
            task             : task index hiện tại (>= 1 mới có ý nghĩa)
            random_baseline  : override b_s; nếu None thì dùng 1/cpt

        Returns:
            float  (0.0 nếu task == 0)
        """
        if task == 0 or len(accuracy_matrix) < 2:
            return 0.0

        b_s = random_baseline if random_baseline is not None \
            else 1.0 / max(int(self.args.cpt), 1)

        fwt_sum = 0.0
        count = 0
        for s in range(1, task + 1):                  # s = 1 .. task
            prev_row_idx = s - 1
            if prev_row_idx >= len(accuracy_matrix):
                continue
            a_prev_s = accuracy_matrix[prev_row_idx][s]   # zero-shot acc on task s
            fwt_sum += (a_prev_s - b_s)
            count += 1

        return fwt_sum / count if count > 0 else 0.0
    # evaluate after end 1 task
    def eval_task(self, task, glob_iter, flag):
        accuracy_on_all_task = []

        for t in range(self.N_TASKS):
            stats = self.test_metrics(task=t, glob_iter=glob_iter, flag="off")
            test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
            accuracy_on_all_task.append(test_acc)

        if flag == "global":
            self.global_accuracy_matrix.append(accuracy_on_all_task)
            accuracy_matrix = self.global_accuracy_matrix
            subdir = os.path.join(self.save_folder, "Global")
            log_key = "Global/Averaged Forgetting"
        elif flag == "local":
            self.local_accuracy_matrix.append(accuracy_on_all_task)
            accuracy_matrix = self.local_accuracy_matrix
            subdir = os.path.join(self.save_folder, "Local")
            log_key = "Local/Averaged Forgetting"

        avg_acc = sum(accuracy_on_all_task[:task+1]) / (task+1)
        forgetting = metric_average_forgetting(int(task%self.N_TASKS), accuracy_matrix)
        fwt = self.compute_fwt(accuracy_matrix, task)
        bwt = self.compute_bwt(accuracy_matrix, task)
        print("\n========== DEBUG TASK EVAL ==========")

        for t_id, acc in enumerate(accuracy_on_all_task):
            print(f"Task {t_id} Accuracy: {acc:.4f}")

        print(f"\nAverage Accuracy up to Task {task}: {avg_acc:.4f}")
        print(f"Forgetting at Task {task}: {forgetting:.4f}")
        print(f"FWT at Task {task}: {fwt:.4f}")
        print(f"BWT at Task {task}: {bwt:.4f}")

        print("\nCurrent Accuracy Matrix:")
        for row_idx, row in enumerate(accuracy_matrix):
            formatted_row = ["{:.4f}".format(v) for v in row]
            print(f"After Task {row_idx}: {formatted_row}")

        print("======================================\n")

        if self.args.wandb:
            prefix = "Global" if flag == "global" else "Local"
            log_dict = {f"{prefix}/Averaged_Forgetting": forgetting}

            # log accuracy từng task
            for t_id, acc in enumerate(accuracy_on_all_task):
                print(f"{prefix}/Task {t_id} Accuracy: {acc}")
                log_dict[f"{prefix}/Task_{t_id}_Accuracy"] = acc

        #     # log Average Accuracy (AA)
        #     avg_acc = sum(accuracy_on_all_task[:task+1]) / (task+1)
            print(f"Average Accuracy up to task {task}: {avg_acc}")
            log_dict[f"{prefix}/Average_Accuracy"] = avg_acc
                # FWT/BWT chỉ có ý nghĩa từ task 1 trở đi
            if task > 0:
                log_dict[f"{prefix}/BWT"] = bwt  # backward: ảnh hưởng lên task cũ
                log_dict[f"{prefix}/FWT"] = fwt  # forward: khả năng học task mới
            wandb.log(log_dict, step=glob_iter)

            print(f"{log_key}: {forgetting:.4f}")

        if self.offlog:
            os.makedirs(subdir, exist_ok=True)

            csv_filename = os.path.join(subdir, f"{self.args.algorithm}_accuracy_matrix.csv")
            with open(csv_filename, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerows(accuracy_matrix)

    def assign_unique_tasks(self):
        # Convert lists to sets of tuples for easy comparison
        unique_set = {tuple(task) for task in self.unique_task}
        old_unique_set = {tuple(task) for task in self.old_unique_task}

        # Find new tasks by taking the difference
        new_tasks = unique_set - old_unique_set
        # print(f"new_tasks: {new_tasks}")
        # Loop over new tasks and assign them to task_dict
        for task in new_tasks:
            self.current_task += 1
            self.task_dict[self.current_task] = list(task)

    def cos_sim(self, prev_model, model1, model2):
        prev_param = torch.cat([p.data.view(-1) for p in prev_model.parameters()])
        params1 = torch.cat([p.data.view(-1) for p in model1.parameters()])
        params2 = torch.cat([p.data.view(-1) for p in model2.parameters()])

        grad1 = params1 - prev_param
        grad2 = params2 - prev_param

        cos_sim = torch.dot(grad1, grad2) / (torch.norm(grad1) * torch.norm(grad2))
        return cos_sim.item()

    def cosine_similarity(self, model1, model2):
        params1 = torch.cat([p.data.view(-1) for p in model1.parameters()])
        params2 = torch.cat([p.data.view(-1) for p in model2.parameters()])
        cos_sim = torch.dot(params1, params2) / (torch.norm(params1) * torch.norm(params2))
        return cos_sim.item()

    def distance(self, model1, model2):
        params1 = torch.cat([p.data.view(-1) for p in model1.parameters()])
        params2 = torch.cat([p.data.view(-1) for p in model2.parameters()])

        mse = F.mse_loss(params1, params2)
        return mse.item()

    def spatio_grad_eval(self, model_origin, glob_iter):
        angle = [self.cos_sim(model_origin, self.global_model, models) for models in self.uploaded_models]
        distance = [self.distance(self.global_model, models) for models in self.uploaded_models]
        norm = [self.distance(model_origin, models) for models in self.uploaded_models]
        self.angle_value = statistics.mean(angle)
        self.distance_value = statistics.mean(distance)
        self.norm_value = statistics.mean(norm)
        angle_value = []

        # for grad_i in self.grads:
        #     for grad_j in self.grads:
        #         angle_value.append(self.cosine_similarity(grad_i, grad_j))

        for i in range(len(self.grads)):
            for j in range(i + 1, len(self.grads)):
                angle_value.append(self.cosine_similarity(self.grads[i], self.grads[j]))

        cosine_to_client0 = {}
        count_positive = 0  # cosine > 0
        count_negative = 0  # cosine >= 0

        for i in range(1, len(self.grads)):
            sim = self.cosine_similarity(self.grads[0], self.grads[i])
            cosine_to_client0[f"{i}"] = sim

            if sim > 0:
                count_positive += 1
            if sim <= 0:
                count_negative += 1

        if self.args.wandb:
            wandb.log({f"cosine/{k}": v for k, v in cosine_to_client0.items()}, step=glob_iter)

            wandb.log({
                "cosine_count/positive (>0)": count_positive,
                "cosine_count/negative (<=0)": count_negative
            }, step=glob_iter)

        self.grads_angle_value = statistics.mean(angle_value)
        # print(f"grad angle: {self.grads_angle_value}")

    def proto_eval(self, global_model, local_model, task, round):
        # TODO save models to ./pca_eval/file_name/global
        model_filename = f"task_{task}_round_{round}.pth"
        save_dir = os.path.join("pca_eval", self.file_name, "global")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save model state_dict
        model_path = os.path.join(save_dir, model_filename)
        torch.save(global_model.state_dict(), model_path)

        save_dir = os.path.join("pca_eval", self.file_name, "local")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save model state_dict
        model_path = os.path.join(save_dir, model_filename)
        torch.save(local_model.state_dict(), model_path)
    