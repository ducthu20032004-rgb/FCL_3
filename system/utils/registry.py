from __future__ import annotations

from typing import Callable, Dict

import torch.nn as nn

BACKBONE_REGISTRY: Dict[str, Callable] = {}


def register_backbone(name: str) -> Callable:
    def decorator(builder: Callable) -> Callable:
        if name in BACKBONE_REGISTRY:
            raise ValueError(f"Backbone '{name}' is already registered.")
        BACKBONE_REGISTRY[name] = builder
        return builder

    return decorator


def build_backbone(name: str, num_classes: int, args) -> nn.Module:
    if name not in BACKBONE_REGISTRY:
        raise KeyError(
            f"Unknown backbone '{name}'. Available backbones: {sorted(BACKBONE_REGISTRY)}"
        )
    return BACKBONE_REGISTRY[name](num_classes=num_classes, args=args)


def list_backbones() -> list[str]:
    return sorted(BACKBONE_REGISTRY)
