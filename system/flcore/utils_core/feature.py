# system/flcore/utils/feature.py
import torch
from contextlib import contextmanager

def _register_penultimate_hook(model, container_names=("classifier", "fc", "head")):
    """
    Best-effort hook to capture penultimate features without changing model code.
    Returns: feature_fn(x)->feat and a handle to remove later.
    """
    captured = {}

    # Try common classifier attributes
    tail = None
    for name in container_names:
        if hasattr(model, name):
            tail = getattr(model, name)
            break

    def hook(_, __, output):
        captured["feat"] = output.detach()

    handle = None
    if tail is not None and hasattr(tail, "register_forward_hook"):
        handle = tail.register_forward_hook(hook)

    def feature_fn(x, forward_callable=None):
        captured.clear()
        # If model exposes 'forward_features', prefer it
        if hasattr(model, "forward_features"):
            feat = model.forward_features(x)
            return feat
        # Else run a forward pass and get penultimate from hook
        _ = model(x)
        return captured.get("feat", None)

    return feature_fn, handle

@contextmanager
def penultimate_features(model):
    feat_fn, handle = _register_penultimate_hook(model)
    try:
        yield feat_fn
    finally:
        if handle is not None:
            handle.remove()
