# system/flcore/clients/clientGLFC.py
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

from flcore.clients.clientbase import Client
from flcore.criterion.distillation import (
    kl_divergence_with_temperature as kd_kl,
    relation_mse
)
# from flcore.utils.memory import ExemplarMemory, class_frequency

class clientGLFC(Client):
    """
    GLFC (Global-Local Forgetting Compensation) – client side.
    - Class-aware gradient compensation: reweight CE by inverse class freq^alpha
    - Relation distillation + KD from server-provided teacher for old classes
    - Small exemplar memory with herding selection
    Paper: Dong et al., CVPR 2022. ArXiv:2203.11473
    """
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.temperature = getattr(args, "glfc_T", 2.0)
        self.alpha = getattr(args, "glfc_alpha", 0.5)
        self.kd_coef = getattr(args, "glfc_distill", 1.0)
        self.rel_coef = getattr(args, "glfc_relation", 1.0)
        self.num_classes = getattr(args, "num_classes", 100)
        self.use_memory = getattr(args, "use_memory", True)
        per_cls = getattr(args, "glfc_mem_per_class", 20)
        self.memory = ExemplarMemory(per_class=per_cls, num_classes=self.num_classes)

        self._teacher_model = None  # set by server each round if available
        self.upload_prototypes = None  # read by server after train()

    def set_teacher(self, model_or_none):
        self._teacher_model = model_or_none

    def _build_train_dataset(self):
        base_ds = self.trainloader.dataset  # assumes PFLlib client has .trainloader
        if not self.use_memory:
            return base_ds
        mem_ds = self.memory.as_dataset()
        return base_ds if mem_ds is None else ConcatDataset([base_ds, mem_ds])

    def _class_weights(self, dataset):
        # Estimate class freq from current dataset for compensation
        labels = []
        loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
        for _, y in loader:
            labels.append(y)
        labels = torch.cat(labels) if labels else torch.empty(0, dtype=torch.long)
        if labels.numel() == 0:
            return None
        freq = class_frequency(labels, self.num_classes) + 1e-6
        weights = (1.0 / (freq ** self.alpha))
        weights = weights / weights.mean()
        return weights.to(self.device)

    def train(self):
        self.model.to(self.device)
        self.model.train()
        dataset = self._build_train_dataset()
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, drop_last=False)
        class_w = self._class_weights(dataset)

        opt = self.optimizer

        for epoch in range(self.local_epochs):
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                logits = self.model(xb)
                # CE with optional class-aware weights
                ce = F.cross_entropy(logits, yb, weight=class_w)

                loss = ce
                if self._teacher_model is not None:
                    with torch.no_grad():
                        t_logits = self._teacher_model(xb)
                    kd = kd_kl(logits, t_logits, T=self.temperature)
                    rel = relation_mse(logits, t_logits)
                    loss = loss + self.kd_coef * kd + self.rel_coef * rel

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                opt.step()

        # Update memory using herding on the client's newly seen classes
        if self.use_memory:
            self.memory.update(self.model, self.device, self.trainloader.dataset)

        # Compute class prototypes on (train + memory) for server (privacy-preserving summary)
        self.upload_prototypes = self._compute_prototypes()

        # Return nothing—server will fetch state via base APIs; metrics are recorded by base.
        return

    @torch.no_grad()
    def _compute_prototypes(self):
        # Average normalized features per class using current model
        self.model.eval()
        loader = DataLoader(self.trainloader.dataset, batch_size=256, shuffle=False, num_workers=0)
        feats_sum = torch.zeros(self.num_classes, getattr(self.args, "feature_dim", 512), device=self.device)
        counts = torch.zeros(self.num_classes, device=self.device)
        for xb, yb in loader:
            xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)
            # try to fetch penultimate features
            try:
                if hasattr(self.model, "forward_features"):
                    f = self.model.forward_features(xb)
                else:
                    f = self.model(xb)
            except Exception:
                f = self.model(xb)
            f = F.normalize(f, dim=1)
            for c in yb.unique():
                idx = (yb == c)
                feats_sum[c] += f[idx].sum(dim=0)
                counts[c] += idx.sum()
        proto = feats_sum / (counts.unsqueeze(1) + 1e-9)
        return proto.detach().cpu()
