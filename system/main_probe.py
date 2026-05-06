import os

import torch
import torchvision
import tqdm

from system.utils.data_utils import *
from torch.utils.data import DataLoader
from system.utils.nflows import nn
from system.utils.probes import LinearProbeCIFAR10

BLOCK_NAMES = ["layer1", "layer2", "layer3", "layer4"]

def save_path(task_id, block_idx, folder="probe"):
    return f"{folder}/head_task{task_id}_block{block_idx}.pt"

def train_all_probes(
    client_index: int = 0,
    classes_per_task: int = 2,
    num_tasks: int = 5,
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 128,
    folder: str = "/kaggle/working",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    os.makedirs(folder, exist_ok=True)

    block_output_size = {
        "layer1": 64 * 8 * 8,
        "layer2": 128 * 4 * 4,
        "layer3": 256 * 2 * 2,
        "layer4": 512 * 1 * 1,
    }

    # Load data trước, dùng lại cho tất cả block
    print("Loading data...")
    all_loaders = {}
    for task_id in range(num_tasks):
        all_loaders[task_id] = {
            "train": DataLoader(
                read_client_data_FCL_cifar10(client_index, task=task_id, classes_per_task=classes_per_task, train=True),
                batch_size=batch_size, shuffle=True, num_workers=2
            ),
            "val": DataLoader(
                read_client_data_FCL_cifar10(client_index, task=task_id, classes_per_task=classes_per_task, train=False),
                batch_size=batch_size, shuffle=False, num_workers=2
            ),
        }

    criterion = nn.CrossEntropyLoss()

    for block_idx, block_name in enumerate(BLOCK_NAMES):
        in_features = block_output_size[block_name]

        # Build backbone 1 lần cho mỗi block
        resnet = torchvision.resnet18(pretrained=True)
        resnet.block_output_size = block_output_size
        resnet.blocks = nn.ModuleDict({
            "layer1": nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1),
            "layer2": resnet.layer2,
            "layer3": resnet.layer3,
            "layer4": nn.Sequential(resnet.layer4, resnet.avgpool),
        })

        for task_id in range(num_tasks):
            print(f"\n{'='*50}")
            print(f"  block{block_idx} ({block_name}) | task{task_id}")
            print(f"{'='*50}")

            # Train probe mới hoàn toàn cho mỗi (block, task)
            model = LinearProbeCIFAR10(resnet, block_name).to(device)
            head  = model.fc_task1
            optimizer = torch.optim.Adam(head.parameters(), lr=lr)

            for epoch in range(epochs):
                # Train
                model.train()
                total_loss, correct, total = 0, 0, 0
                for x, y in tqdm(all_loaders[task_id]["train"], desc=f"  Epoch {epoch+1}/{epochs}"):
                    x, y = x.to(device), y.to(device).long()
                    optimizer.zero_grad()
                    out = model(x, task_id=1)
                    loss = criterion(out, y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * y.size(0)
                    correct    += (out.argmax(1) == y).sum().item()
                    total      += y.size(0)

                # Val
                model.eval()
                val_correct, val_total = 0, 0
                with torch.no_grad():
                    for x, y in all_loaders[task_id]["val"]:
                        x, y = x.to(device), y.to(device).long()
                        out = model(x, task_id=1)
                        val_correct += (out.argmax(1) == y).sum().item()
                        val_total   += y.size(0)

                print(f"  Epoch {epoch+1:2d} | Loss: {total_loss/total:.4f} | "
                      f"Train: {correct/total:.3f} | Val: {val_correct/val_total:.3f}")

            # Lưu riêng từng file
            path = save_path(task_id, block_idx, folder)
            torch.save({
                "block_name": block_name,
                "block_idx": block_idx,
                "task_id": task_id,
                "in_features": in_features,
                "classes_per_task": classes_per_task,
                "head": head.state_dict(),
            }, path)
            print(f"  Saved → {path}")