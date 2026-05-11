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
        self._printed_task = set()
        self.fc_task0 = nn.Linear(self.in_channel, 10)
        self.fc_task1 = nn.Linear(self.in_channel, 10)
        self.fc_task2 = nn.Linear(self.in_channel, 10)
        self.fc_task3 = nn.Linear(self.in_channel, 10)
        self.fc_task4 = nn.Linear(self.in_channel, 10)
        xavier_uniform_initialize(self.fc_task0)
        xavier_uniform_initialize(self.fc_task1)   
        xavier_uniform_initialize(self.fc_task2)
        xavier_uniform_initialize(self.fc_task3)
        xavier_uniform_initialize(self.fc_task4)

    def forward(self, features: Tensor, task_id):
        for block_name, operations in self.under_investigation_blocks.items():
            features = operations(features)
            if block_name == self.intended_block:
                break
        features = F.adaptive_avg_pool2d(features, (1, 1))
        features = torch.flatten(features, 1)
        
        if task_id == "0":
            if "0" not in self._printed_task:
                print("Using head for task 0")
                self._printed_task.add("0")
            features = self.fc_task0(features)
        elif task_id == "1":
            if "1" not in self._printed_task:
                print("Using head for task 1")
                self._printed_task.add("1")
            features = self.fc_task1(features)
        elif task_id == "2":
            if "2" not in self._printed_task:
                print("Using head for task 2")
                self._printed_task.add("2")
            features = self.fc_task2(features)
        elif task_id == "3":
            if "3" not in self._printed_task:
                print("Using head for task 3")
                self._printed_task.add("3")
            features = self.fc_task3(features)
        elif task_id == "4":
            if "4" not in self._printed_task:
                print("Using head for task 4")
                self._printed_task.add("4")
            features = self.fc_task4(features)
        else:
            raise ValueError(f"Invalid task_id: {task_id}. Expected values are between 0 and 4.")
        return features

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