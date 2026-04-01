
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

import torch


sd1 = torch.load('weight-client-avg-resnet18/client_0_task_1.pt', map_location='cpu')
sd2 = torch.load('weight-client-avg-resnet18/client_1_task_1.pt', map_location='cpu')

for k in sd1:
    if not torch.allclose(sd1[k], sd2[k]):
        print("DIFFERENT at", k)
        break
else:
    print("IDENTICAL")