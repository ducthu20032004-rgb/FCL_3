
# import torch
# from torchvision.models import resnet18
# from torch.utils.data import DataLoader
# from system.measure import load_model_with_head
# from system.utils.data_utils import read_client_data_FCL_cifar10

# test_data_t = read_client_data_FCL_cifar10(
#                 0, task=1,
#                 classes_per_task=2,
#                 count_labels=False, train=False
#             )

# raw_sd = torch.load('weight-client-avg-resnet18/client_0_task_1.pt', map_location='cpu')
# print('head.weight shape:', raw_sd['head.weight'].shape)  # expect (10, 512)

# # Load model task 1 và thử predict trên 1 batch của task 0
# model_debug = load_model_with_head('weight-client-avg-resnet18/client_0_task_1.pt', num_classes=10)
# loader_debug = DataLoader(test_data_t, batch_size=10, shuffle=False)
# x, y = next(iter(loader_debug))
# with torch.no_grad():
#     out = model_debug(x)
# print('True labels (task 0):', y)
# print('Predicted labels:', torch.argmax(out, dim=1))
# print('Logits:\n', out)

# import torch


# sd1 = torch.load('weight-client-avg-resnet18/client_0_task_1.pt', map_location='cpu')
# sd2 = torch.load('weight-client-avg-resnet18/client_1_task_1.pt', map_location='cpu')

# for k in sd1:
#     if not torch.allclose(sd1[k], sd2[k]):
#         print("DIFFERENT at", k)
#         break
# else:
#     print("IDENTICAL")

import matplotlib.pyplot as plt
import numpy as np

# --- Dữ liệu từ log bạn gửi (block 0 → block 4, round 0 → 9) ---
blocks = [0, 1, 2, 3, 4]

# epsilon từng round, block 0→4
epsilon = [
    [0.1612, 0.0896, 10.5388, 42.6045, 46.2603],
    [0.0912, 0.0789, 9.7629, 40.9637, 39.4238],
    [0.1174, 0.0661, 11.0996, 36.7517, 33.9683],
    [0.1048, 0.0880, 8.5887, 35.3171, 33.1160],
    [0.0998, 0.0744, 9.3891, 35.6355, 34.2299],
    [0.1025, 0.0793, 8.7048, 36.9261, 32.0563],
    [0.1179, 0.0778, 9.5368, 32.9956, 30.6864],
    [0.0931, 0.0686, 8.3490, 30.8989, 30.9740],
    [0.1336, 0.0883, 8.9007, 35.6457, 31.2280],
    [0.0924, 0.0715, 9.6322, 32.1583, 30.0]  # block4 ε approx
]

# CKA từng round, block 0→4
cka = [
    [0.0184, 0.0241, 0.0369, 0.0670, 0.1643],
    [0.0198, 0.0281, 0.0451, 0.0874, 0.2011],
    [0.0205, 0.0313, 0.0478, 0.0901, 0.2117],
    [0.0201, 0.0304, 0.0511, 0.0967, 0.2100],
    [0.0208, 0.0303, 0.0502, 0.0969, 0.2187],
    [0.0210, 0.0309, 0.0498, 0.1019, 0.2328],
    [0.0214, 0.0312, 0.0486, 0.0989, 0.2320],
    [0.0220, 0.0323, 0.0522, 0.1047, 0.2431],
    [0.0217, 0.0309, 0.0487, 0.1023, 0.2442],
    [0.0219, 0.0317, 0.0508, 0.1077, 0.245]  # block4 CKA approx
]

# Forgetting theo round (chỉ đo block cuối, block4)
forgetting = [88.2, 88.2, 90.0, 90.6, 90.5, 90.8, 90.65, 90.6, 91.0, 90.75]

# --- Vẽ epsilon và CKA theo block ---
plt.figure(figsize=(12,5))
for r in range(len(epsilon)):
    plt.plot(blocks, epsilon[r], label=f'ε round {r}', linestyle='--', alpha=0.4, color='red')
    plt.plot(blocks, cka[r], label=f'CKA round {r}', linestyle='-', alpha=0.4, color='blue')

plt.xlabel('Block')
plt.ylabel('Value')
plt.title('Evolution of ε (red dashed) and CKA (blue) across blocks')
plt.legend([],[], frameon=False)  # ẩn legend dài dòng
plt.grid(True)
plt.show()

# --- Vẽ Forgetting theo round ---
plt.figure(figsize=(8,4))
plt.plot(range(len(forgetting)), forgetting, marker='o', color='purple')
plt.xlabel('Round')
plt.ylabel('Forgetting (%)')
plt.title('Catastrophic Forgetting over Rounds (block4)')
plt.grid(True)
plt.show()