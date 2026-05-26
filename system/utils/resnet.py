from __future__ import annotations

import os
from collections import OrderedDict
from typing import Dict, Iterable, List, Tuple

import torch

import torch.nn as nn
from torchvision.models import resnet18

from .registry import register_backbone


class CIFARResNet18(nn.Module):
    """ResNet-18 wrapper exposing backbone/head utilities.

    These helpers keep the backbone registry simple while enabling methods that
    need decoupled backbone/head states (FedAS), feature extraction (FedAS),
    block-wise control (FedALA, FedL2P), and BN inspection (FedL2P).
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        base = resnet18(weights=None)
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.head = nn.Linear(base.fc.in_features, num_classes)

    def extract_features(self, x: torch.Tensor, return_block_outputs: bool = False):
        block_outputs: List[torch.Tensor] = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        block_outputs.append(x)

        x = self.layer1(x)
        block_outputs.append(x)
        x = self.layer2(x)
        block_outputs.append(x)
        x = self.layer3(x)
        block_outputs.append(x)
        x = self.layer4(x)
        block_outputs.append(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        block_outputs.append(x)

        if return_block_outputs:
            return x, block_outputs
        return x

    def forward_from_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        return self.forward_from_features(features)

    def backbone_state_dict(self) -> Dict[str, torch.Tensor]:
        modules = OrderedDict(
            conv1=self.conv1,
            bn1=self.bn1,
            layer1=self.layer1,
            layer2=self.layer2,
            layer3=self.layer3,
            layer4=self.layer4,
        )
        state: Dict[str, torch.Tensor] = {}
        for name, module in modules.items():
            for key, value in module.state_dict().items():
                state[f"{name}.{key}"] = value
        return state

    def head_state_dict(self) -> Dict[str, torch.Tensor]:
        return {f"head.{key}": value for key, value in self.head.state_dict().items()}

    def load_backbone_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True) -> None:
        grouped: Dict[str, Dict[str, torch.Tensor]] = {
            "conv1": {},
            "bn1": {},
            "layer1": {},
            "layer2": {},
            "layer3": {},
            "layer4": {},
        }
        for key, value in state_dict.items():
            prefix, rest = key.split(".", 1)
            if prefix in grouped:
                grouped[prefix][rest] = value
        self.conv1.load_state_dict(grouped["conv1"], strict=strict)
        self.bn1.load_state_dict(grouped["bn1"], strict=strict)
        self.layer1.load_state_dict(grouped["layer1"], strict=strict)
        self.layer2.load_state_dict(grouped["layer2"], strict=strict)
        self.layer3.load_state_dict(grouped["layer3"], strict=strict)
        self.layer4.load_state_dict(grouped["layer4"], strict=strict)

    def load_head_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True) -> None:
        head_state = {}
        for key, value in state_dict.items():
            if key.startswith("head."):
                head_state[key[len("head.") :]] = value
        self.head.load_state_dict(head_state, strict=strict)

    def named_backbone_parameters(self):
        for module_name in ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]:
            module = getattr(self, module_name)
            for name, param in module.named_parameters():
                yield f"{module_name}.{name}", param

    def named_head_parameters(self):
        for name, param in self.head.named_parameters():
            yield f"head.{name}", param

    def get_trainable_block_prefixes(self) -> List[str]:
        return ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "head"]

    def get_block_parameter_groups(self) -> List[Tuple[str, Iterable[nn.Parameter]]]:
        return [
            ("conv1", self.conv1.parameters()),
            ("bn1", self.bn1.parameters()),
            ("layer1", self.layer1.parameters()),
            ("layer2", self.layer2.parameters()),
            ("layer3", self.layer3.parameters()),
            ("layer4", self.layer4.parameters()),
            ("head", self.head.parameters()),
        ]

    def get_bn_layers(self) -> List[Tuple[str, nn.BatchNorm2d]]:
        layers: List[Tuple[str, nn.BatchNorm2d]] = []
        for name, module in self.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                layers.append((name, module))
        return layers


@register_backbone("cifar_resnet18")
def make_cifar_resnet18(num_classes: int, args=None) -> nn.Module:
    del args
    return CIFARResNet18(num_classes=num_classes)