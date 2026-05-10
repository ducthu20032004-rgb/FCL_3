from typing import List

import torch
from torch import nn as nn, Tensor
from torch.nn import ModuleDict, Linear

# from models.imagenet_based_models import PredictionLayerConfig
from system.utilities_probe.utils import xavier_uniform_initialize
from torchvision.models import ResNet, resnet18
import os
import logging
from tqdm import tqdm
import torch.nn.functional as F
from typing import List

import torch
from torch import nn as nn, Tensor
from torch.nn import ModuleDict, Linear
from torchvision.models import ResNet

from system.utilities_probe.utils import xavier_uniform_initialize


from collections import OrderedDict
import torch.nn as nn


class ResNet18ForProbe(nn.Module):

    def __init__(self, backbone):

        super().__init__()

        self.blocks = nn.ModuleDict(OrderedDict({

            "block0": nn.Sequential(
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool
            ),

            "block1": backbone.layer1,

            "block2": backbone.layer2,

            "block3": backbone.layer3,

            "block4": backbone.layer4,

        }))

        self.block_output_size = {
            "block0": 64 ,
            "block1": 64 ,
            "block2": 128 ,
            "block3": 256 ,
            "block4": 512 ,
        }


class LinearProbeCIFAR10(nn.Module):
    def __init__(self, under_investigation_model, intended_block, num_tasks=5):
        super().__init__()
        self.intended_block = intended_block
        self.in_channel = under_investigation_model.block_output_size[intended_block]
        self.under_investigation_blocks = under_investigation_model.blocks
        
        for block_values in self.under_investigation_blocks.values():
            for parameters in block_values.parameters():
                parameters.requires_grad = False

        # Tạo head riêng cho từng task
        self.heads = nn.ModuleDict()
        for i in range(num_tasks):
            head = nn.Linear(self.in_channel, 10)  # 10 classes/task với CIFAR10 FCL
            xavier_uniform_initialize(head)
            self.heads[f'task_{i}'] = head

    def forward(self, features: Tensor, task_id):
        for block_name, operations in self.under_investigation_blocks.items():
            features = operations(features)
            if block_name == self.intended_block:
                break
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        
        task_key = f'task_{task_id}' if not str(task_id).startswith('task_') else task_id
        return self.heads[task_key](features)

class LinearProbeCIFAR100(nn.Module):
    def __init__(
        self,
        under_investigation_model: nn.Module,
        intended_block: str,
    ):
        super(LinearProbeCIFAR100, self).__init__()
        self.intended_block = intended_block
        self.in_channel = under_investigation_model.block_output_size[self.intended_block]
        self.under_investigation_blocks = under_investigation_model.blocks
        # Freezing the blocks
        for block_values in self.under_investigation_blocks.values():
            for parameters in block_values.parameters():
                parameters.requires_grad = False

        self.projectors = nn.ModuleDict()
        for i in range(20):
            task_projector = nn.Linear(self.in_channel, 5).to('cuda')
            xavier_uniform_initialize(task_projector)
            self.projectors[f'Task_{i}'] = task_projector

    def forward(self, features: Tensor, task_id):
        for block_name, operations in self.under_investigation_blocks.items():
            features = operations(features)
            
            if block_name == self.intended_block:
                break
        features = torch.flatten(features, 1).detach()
        features = self.projectors[f'Task_{task_id}'](features)
        return features