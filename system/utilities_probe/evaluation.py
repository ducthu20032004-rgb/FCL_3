from typing import Dict, Sequence, Union, Optional

import numpy as np
import torch

from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import torch
from torch import nn as nn, Tensor
from torch.nn import functional as F

from torchvision.models import ResNet, resnet18
from system.task_data_loader.scenarios import Scenario
from system.utilities_probe.probes import ResNet18ForProbe
from system.utilities_probe.metrics import PredictionBasedMetric, RepresentationBasedMetric
from system.utilities_probe.utils import gpu_information_summary, merge, to_numpy
from tqdm import tqdm
class PredictionBasedEvaluator:
    def __init__(self, metrics: Sequence[PredictionBasedMetric], batch_size: int = 32, num_workers: int = 0, num_classes: int = 10):
        self.batch_size = batch_size
        self.num_workers = num_workers
        n_gpu, self.device = gpu_information_summary(show=False)
        self.metrics = metrics
        self.num_classes = num_classes
    def eval_all_tasks(self, model: Union[ResNet18ForProbe], data_stream: Scenario):
        task_evaluation = dict()
        for task in tqdm(data_stream.tasks, desc=f'Evaluating tasks...'):
            task_eval_metrics = self.eval_one_task(
                model=model, task=task.test, task_id=task.id, nb_classes=self.num_classes
            )
            for metric_name, metric_dict_value in task_eval_metrics.items():
                task_eval_metrics[metric_name] = {
                    f"task_{task.id}_{key}": value for key, value in metric_dict_value.items()
                }
            task_evaluation = merge(task_evaluation, task_eval_metrics)
        return task_evaluation

    def eval_one_task(
        self,
        model: Union[ResNet18ForProbe],
        task: Union[Dataset],
        task_id: Optional[str] = None,
        nb_classes: int = 10,
    ) -> Dict[str, Dict[str, float]]:
        self.before_eval_one_task(task=task, nb_classes=nb_classes)
        model.to(self.device)
        model.eval()
        eval_loader = torch.utils.data.DataLoader(
            task, batch_size=self.batch_size, 
            shuffle=False, 
            # pin_memory=True
            pin_memory=False,
            #  generator=torch.Generator(device='cuda'),
        )
        with torch.inference_mode():
            for batch_number, evaluation_instance in enumerate(eval_loader):
                evaluation_features, evaluation_targets = evaluation_instance
                evaluation_features = evaluation_features.to(self.device)
                evaluation_targets = evaluation_targets.numpy()
                logits = to_numpy(
                    model(evaluation_features, task_id) if task_id is not None else model(evaluation_features)
                )
                # print(f'evaluation_targets : {evaluation_targets}')
                # print(f'logits : {logits.shape}')
                self.eval_one_batch(logits=logits, targets=evaluation_targets)
                # print(f'estimates : {np.argmax(logits, 1)}')
                # raise KeyError()

        return self.compute_eval_one_task()

    def eval_one_batch(self, logits: np.ndarray, targets: np.ndarray) -> None:
        for metric in self.metrics:
            metric.eval_one_batch(logits=logits, targets=targets)
            #print(f'metric.compute_metric() : {metric.compute_metric()}')

    def compute_eval_one_task(self) -> Dict[str, Dict[str, float]]:
        metric_evaluation = dict()
        for metric in self.metrics:
            metric_evaluation[type(metric).__name__] = metric.compute_metric()

        return metric_evaluation

    def before_eval_one_task(self, task: Union[Dataset], nb_classes: int = -1) -> None:
        for metric in self.metrics:
            metric.initialize_metric(task=task, nb_classes=nb_classes)


class RepresentationBasedEvaluator:
    def __init__(self, metrics: Sequence[RepresentationBasedMetric], batch_size: int = 32, num_workers: int = 0):
        self.batch_size = batch_size
        self.num_workers = num_workers
        _, self.device = gpu_information_summary(show=False)
        self.metrics = metrics

    def record_original_representations(
        self, model: ResNet18ForProbe, task: Union[Dataset], task_id: str = "."
    ) -> None:
        model.to(self.device)
        model.eval()
        eval_loader = torch.utils.data.DataLoader(
            task, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )
        with torch.inference_mode():
            for batch_number, evaluation_instance in enumerate(eval_loader):
                evaluation_features, _ = evaluation_instance
                evaluation_features = evaluation_features.to(self.device)
                block_reps = model.block_forward(evaluation_features, task_id)
                if batch_number == 0:
                    self.initialize_memory(representation_blocks=block_reps, is_old_blocks=True)
                else:
                    self.aggregate_batches(representation_blocks=block_reps, is_old_blocks=True)

    def record_updated_representations(
        self, model: ResNet18ForProbe, task: Union[Dataset], task_id: str = "."
    ) -> None:
        model.to(self.device)
        model.eval()
        eval_loader = torch.utils.data.DataLoader(
            task, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True
        )
        with torch.inference_mode():
            for batch_number, evaluation_instance in enumerate(eval_loader):
                evaluation_features, _ = evaluation_instance
                evaluation_features = evaluation_features.to(self.device)
                block_reps = model.block_forward(evaluation_features, task_id)
                if batch_number == 0:
                    self.initialize_memory(representation_blocks=block_reps, is_old_blocks=False)
                else:
                    self.aggregate_batches(representation_blocks=block_reps, is_old_blocks=False)

    def compute_metrics(self) -> Dict[str, Dict[str, float]]:
        metric_evaluation = dict()
        for metric in self.metrics:
            metric_evaluation[type(metric).__name__] = metric.compute_metric()
        return metric_evaluation

    def initialize_memory(self, representation_blocks: Dict[str, torch.Tensor], is_old_blocks: bool = True):
        for metric in self.metrics:
            metric.initialize_memory(representation_blocks=representation_blocks, is_old_blocks=is_old_blocks)

    def aggregate_batches(self, representation_blocks: Dict[str, torch.Tensor], is_old_blocks: bool = True):
        for metric in self.metrics:
            metric.aggregate_batches(representation_blocks=representation_blocks, is_old_blocks=is_old_blocks)