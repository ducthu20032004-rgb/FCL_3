from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import ttach as tta

from system.flcore.grad_cam.utils_cam.activation_and_gradients import ActivationsAndGradients
from system.flcore.grad_cam.utils_cam.model_targets import ClassifierOutputTarget
from system.flcore.grad_cam.utils_cam.svd_on_activations import get_2d_projection

class BaseCAM:
    def __init__(
        self,
        model: torch.nn.Module,
        target_layers: List[torch.nn.Module],
        reshape_transform: Callable = None,
        compute_input_gradient: bool = False,
        uses_gradients: bool = True,
        tta_transforms: Optional[tta.Compose] = None,
        detach: bool = True,
    ) -> None:
        self.model = model.eval()
        self.target_layers = target_layers

        # Use the same device as the model.
        self.device = next(self.model.parameters()).device
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients
        if tta_transforms is None:
            self.tta_transforms = tta.Compose(
                [
                    tta.HorizontalFlip(),
                    tta.Multiply(factors=[0.9, 1, 1.1]),
                ]
            )
        else:
            self.tta_transforms = tta_transforms

        self.detach = detach
        self.activations_and_grads = ActivationsAndGradients(self.model, target_layers, reshape_transform, self.detach)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    def get_cam_weights(
        self,
        input_tensor: torch.Tensor,
        target_layers: List[torch.nn.Module],
        targets: List[torch.nn.Module],
        activations: torch.Tensor,
        grads: torch.Tensor,
    ) -> np.ndarray:
        #2D image 
        if len(grads.shape) == 4:
            return np.mean(grads, axis=(2, 3))
        
        # 3D image
        elif len(grads.shape) == 5:
            return np.mean(grads, axis=(2, 3, 4))
        
        else:
            raise ValueError("Invalid grads shape." 
                             "Shape of grads should be 4 (2D image) or 5 (3D image).")
    def get_cam_image(
        self,
        input_tensor: torch.Tensor,
        target_layer: torch.nn.Module,
        targets: List[torch.nn.Module],
        activations: torch.Tensor,
        grads: torch.Tensor,
        eigen_smooth: bool = False,
    ) -> np.ndarray:
        weights = self.get_cam_weights(input_tensor, target_layer, targets, activations, grads)
        if isinstance(activations, torch.Tensor):
            activations = activations.cpu().detach().numpy()
        # 2D conv
        if len(activations.shape) == 4:
            weighted_activations = weights[:, :, None, None] * activations
        # 3D conv
        elif len(activations.shape) == 5:
            weighted_activations = weights[:, :, None, None, None] * activations
        else:
            raise ValueError(f"Invalid activation shape. Get {len(activations.shape)}.")

        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)
        return cam

    def forward(
        self, input_tensor: torch.Tensor, targets: List[torch.nn.Module], eigen_smooth: bool = False
    ) -> np.ndarray:
        input_tensor = input_tensor.to(self.device)

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)

        self.outputs = outputs = self.activations_and_grads(input_tensor)

        # if targets is None:
        #     target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
        #     targets = [ClassifierOutputTarget(category) for category in target_categories]

        if self.uses_gradients:
            self.model.zero_grad()
            loss = sum([target(output) for target, output in zip(targets, outputs)])
            if self.detach:
                loss.backward(retain_graph=True)
            else:
                # keep the computational graph, create_graph = True is needed for hvp
                torch.autograd.grad(loss, input_tensor, retain_graph = True, create_graph = True)
                # When using the following loss.backward() method, a warning is raised: "UserWarning: Using backward() with create_graph=True will create a reference cycle"
                # loss.backward(retain_graph=True, create_graph=True)
        
        # Lay grads + activation
        grads = self.activations_and_grads.gradients[-1]
        activations = self.activations_and_grads.activations[-1]

        # Tinh weights
        weights = grads.mean(dim = (2,3)) # (B,C)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        #cam_per_layer = self.compute_cam_per_layer(input_tensor, targets, eigen_smooth)
        return weights 
    # =========================
    # DATASET LEVEL (GPU optimized)
    # =========================
    @torch.no_grad()
    def get_importance(self, loader, target_layer):
        all_weights = []

        for images, _ in loader:
            images = images.to(self.device)
            
            # Forward để lấy activations
            _ = self.activations_and_grads(images)
            
            # Lấy activation của layer cuối, mean theo spatial → (B, C)
            activations = self.activations_and_grads.activations[-1]  # (B, C, H, W)
            weights = activations.mean(dim=(2, 3))  # (B, C)
            weights = weights - weights.mean(dim = 1,keepdim = True)
            all_weights.append(weights.detach().cpu())

        all_weights = torch.cat(all_weights, dim=0)  # (N, C)
        importance = all_weights.mean(dim=0)          # (C,)
        return importance

    # =========================
    # FULL DISTRIBUTION
    # =========================
    def get_distribution(self, loader,target_layer):

        all_weights = []

        for images, _ in loader:
            # Lấy activation của layer cuối, mean theo spatial → (B, C)
            activations = self.activations_and_grads.activations[-1]  # (B, C, H, W)
            weights = activations.mean(dim=(2, 3))  # (B, C)
            weights = weights - weights.mean(dim = 1,keepdim = True)
            all_weights.append(weights)

        all_weights = torch.cat(all_weights, dim=0)  # (N,C)

        return all_weights.cpu()  # chỉ chuyển 1 lần