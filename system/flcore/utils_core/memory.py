# system/flcore/utils/memory.py
import torch
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
from .feature import penultimate_features

def class_frequency(labels, num_classes):
    freq = torch.bincount(labels, minlength=num_classes).float()
    return freq

def select_exemplars_herding(model, device, dataset, per_class, num_classes):
    """
    Simple herding: choose per_class samples nearest to class mean in feature space.
    Assumes dataset[i] -> (x, y). Returns dict{cls: (X,Y) tensors}
    """
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    feats, ys, xs = [], [], []
    model.eval()
    with torch.no_grad(), penultimate_features(model) as ffn:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            feat = ffn(x)
            if feat is None:  # fallback: use logits as features
                feat = model(x)
            feats.append(feat.cpu())
            ys.append(y.cpu())
            xs.append(x.cpu())  # keep on CPU
    feats = torch.cat(feats); ys = torch.cat(ys); xs = torch.cat(xs)

    out = {}
    for c in range(num_classes):
        idx = (ys == c).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        f_c = feats[idx]
        mean = f_c.mean(dim=0, keepdim=True)
        # distances to mean
        d = torch.norm(f_c - mean, dim=1)
        k = min(per_class, d.numel())
        sel = idx[torch.topk(-d, k).indices]  # smallest distance
        Xc = xs[sel]; Yc = ys[sel]
        out[c] = (Xc, Yc)
    return out

class ExemplarMemory:
    def __init__(self, per_class: int = 20, num_classes: int = 100):
        self.per_class = per_class
        self.num_classes = num_classes
        self.bank = {}  # class -> (X,Y)

    def update(self, model, device, dataset):
        selected = select_exemplars_herding(model, device, dataset, self.per_class, self.num_classes)
        for c, xy in selected.items():
            self.bank[c] = xy

    def as_dataset(self):
        parts = []
        for (X, Y) in self.bank.values():
            parts.append(TensorDataset(X, Y))
        if not parts:
            return None
        return ConcatDataset(parts)
