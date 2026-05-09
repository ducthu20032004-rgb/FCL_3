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
            "block0": 64 * 8 * 8,
            "block1": 64 * 8 * 8,
            "block2": 128 * 4 * 4,
            "block3": 256 * 2 * 2,
            "block4": 512 * 1 * 1,
        }


class LinearProbeCIFAR10(nn.Module):
    def __init__(
        self,
        under_investigation_model: nn.Module,
        intended_block: str,
    ):
        super(LinearProbeCIFAR10, self).__init__()
        self.intended_block = intended_block
                # Map named blocks to their corresponding layers
        
        self.in_channel = under_investigation_model.block_output_size[self.intended_block]
        self.under_investigation_blocks = under_investigation_model.blocks
        # Freezing the blocks
        for block_values in self.under_investigation_blocks.values():
            for parameters in block_values.parameters():
                parameters.requires_grad = False

        self.fc_task1 = nn.Linear(self.in_channel, 10)
        # self.fc_task2 = nn.Linear(self.in_channel, 5)
        xavier_uniform_initialize(self.fc_task1)
        # xavier_uniform_initialize(self.fc_task2)

    def forward(self, features: Tensor, task_id: int):
        for block_name, operations in self.under_investigation_blocks.items():
            features = operations(features)
            if block_name == self.intended_block:
                break
        
        features = torch.flatten(features, 1)
        return self.fc_task1(features)
        # if task_id == 1:
        #     features = self.fc_task1(features)
        # else:
        #     features = self.fc_task2(features)
        # return features
