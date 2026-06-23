from asyncio import current_task
import os
import pandas as pd
import itertools
import logging
import argparse
import pickle
from xmlrpc import client
import math
import torch.nn as nn
import numpy as np
import torch
import torchvision
import wandb
from tqdm import tqdm
from system.flcore.metrics.average_forgetting import metric_average_forgetting
from torchvision.models import resnet18,resnet34,resnet50
from torchvision.models.resnet import BasicBlock
from sklearn.linear_model import LinearRegression
from system.measure_alignment import compute_alignment, compute_alignment_from_arrays
from system.measure_alignment import compute_alignment
from system.utils.CKA import TorchCKA, hsic
from system.utils.data_utils import *
from torch.utils.data import DataLoader
import sys
from system.flcore.grad_cam.base_cam import *
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load class_order
all_class_orders = np.load('./dataset/class_order/class_order_cifar10.npy', allow_pickle=True)

os.makedirs('./outputs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(stream=open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)),
        logging.FileHandler('./outputs/drift_run.log', mode='w', encoding='utf-8'),
    ]
)
logger = logging.getLogger(__name__)
logger.info(f'🖥️  Running on: {DEVICE}' + (f' ({torch.cuda.get_device_name(0)})' if DEVICE.type == 'cuda' else ''))

import os
import csv

class ScatterLogger:
    def __init__(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.files = {}

    def log_pair(self, name, x, y):
        path = os.path.join(self.save_dir, f"{name}.csv")

        # nếu chưa mở file thì tạo + header
        if name not in self.files:
            f = open(path, "w", newline="")
            writer = csv.writer(f)
            writer.writerow(["x", "y"])
            self.files[name] = (f, writer)

        _, writer = self.files[name]
        writer.writerow([x, y])

    def close(self):
        for f, _ in self.files.values():
            f.close()
# ─────────────────────────────────────────────────────────────────────────────
# Đường dẫn checkpoint
# ─────────────────────────────────────────────────────────────────────────────
def get_model_path(saving_dir: str, client_id: int, task: int, round: int) -> str:
    return os.path.join(saving_dir, f'client_{client_id}_task_{task}_round_{round}.pt')

def get_model_path_no_round(saving_dir: str, client_id: int, task: int) -> str:
    return os.path.join(saving_dir, f'client_{client_id}_task_{task}.pt')
def get_model_path_probe(saving_dir: str, client_id: int, task: int, intended_block: int) -> str:
    return os.path.join(
        saving_dir,
        f"head_task{task}_block{intended_block}.pt"
    )
# ─────────────────────────────────────────────────────────────────────────────
# Các hàm đo
# ─────────────────────────────────────────────────────────────────────────────
def mask_classes(outputs: torch.Tensor,class_order : np.array, classes_per_task: int, task_index: int) -> torch.Tensor:
    """
    Mask đúng các class thuộc task_index theo class_order thực tế.
    :param class_order: class_order của client đó, shape (num_classes,)
                        ví dụ [3, 7, 0, 5, 1, 9, 2, 4, 6, 8]
    :param task_index:  task đang test (0, 1, 2, ...)
    """
    actual_classes = class_order[task_index * classes_per_task : 
                                 (task_index + 1) * classes_per_task].tolist()
    
    mask = torch.full_like(outputs, float('-inf'))
    mask[:, actual_classes] = 0.0
    return outputs + mask

def get_resnet18_blocks(_model):
    return {
        'block0': torch.nn.Sequential(_model.conv1, _model.bn1, _model.relu, _model.maxpool),
        'block1': _model.layer1,
        'block2': _model.layer2,
        'block3': _model.layer3,
        'block4': _model.layer4,
    }
def get_vit_blocks(model):
    """Lấy các block của ViT-Ti từ timm"""
    return {
        f'block{i}': model.blocks[i]
        for i in range(len(model.blocks))  # ViT-Ti: 12 blocks
    }

def compute_feature_vit(model, task_index, dataset, 
                         target_block_idx: int, args):
    """
    Trích features tại block target_block_idx của ViT.
    Hook vào output của blocks[target_block_idx].
    """
    model.eval()
    features_out = []

    # Đăng ký forward hook
    def hook_fn(module, input, output):
        # output shape: (B, seq_len, dim)
        # Lấy CLS token (index 0) hoặc mean pooling
        features_out.append(output[:, 0, :].detach().cpu())  # CLS token

    handle = model.blocks[target_block_idx].register_forward_hook(hook_fn)

    loader = _make_loader(dataset, batch_size=256)
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(DEVICE)
            model(x)   # forward pass kích hoạt hook

    handle.remove()
    return torch.cat(features_out, dim=0).numpy()  # (N, D)


def load_vit_from_checkpoint(ckpt_path: str, num_classes: int):
    import timm
    raw_sd = torch.load(ckpt_path, map_location='cpu')

    # Strip prefix 'base.' nếu có (do BaseHeadSplit)
    new_sd = {}
    for k, v in raw_sd.items():
        if k.startswith('base.'):
            new_sd[k[len('base.'):]] = v
        elif k.startswith('head.'):
            new_sd[k] = v
        else:
            new_sd[k] = v

    # ── Tự detect img_size và patch_size từ checkpoint ──────────────────
    # pos_embed shape: (1, num_patches + 1, embed_dim)
    # num_patches = (img_size / patch_size) ** 2
    pos_embed_shape = new_sd.get('pos_embed', raw_sd.get('pos_embed')).shape
    num_patches = pos_embed_shape[1] - 1   # trừ CLS token

    patch_proj_shape = new_sd.get(
        'patch_embed.proj.weight',
        raw_sd.get('patch_embed.proj.weight')
    ).shape
    patch_size = patch_proj_shape[-1]      # (out_ch, in_ch, kH, kW) → kH

    import math
    img_size = int(math.sqrt(num_patches)) * patch_size

    logger.info(
        f'  [load_vit] detected: img_size={img_size} '
        f'patch_size={patch_size} num_patches={num_patches}'
    )

    # ── Tạo model đúng config ────────────────────────────────────────────
    model = timm.create_model(
        'vit_tiny_patch16_224',   # tên base, img_size/patch_size sẽ bị override
        pretrained=False,
        num_classes=num_classes,
        img_size=img_size,
        patch_size=patch_size,
    )

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    real_missing = [
        k for k in missing
        if 'num_batches_tracked' not in k and not k.startswith('head.')
    ]
    if real_missing:
        logger.warning(f'  [WARN] Missing keys: {real_missing}')

    model.to(DEVICE).eval()
    return model

def compute_width(_model, _target_layer: int):
    layers = [_model.layer1, _model.layer2, _model.layer3, _model.layer4]
    if not (0 <= _target_layer < len(layers)):
        raise IndexError(f"_target_layer phải trong [0, 3], nhận được: {_target_layer}")
    layer = layers[_target_layer]
    if isinstance(layer, torch.nn.Sequential):
        block = layer[-1]
        if isinstance(block, BasicBlock):
            return torch.norm(block.conv2.weight, p='fro').item()
    raise TypeError(f"Unexpected type: {type(layer)}")
def compute_bwt(accuracy_matrix: list, task: int) -> float:
        """
        BWT = (1 / task) * Σ_{s=0}^{task-1} [ A_{task,s} - A_{s,s} ]

        A_{task, s} = acc on task s AFTER learning task t  (current row, col s)
        A_{s,s}     = acc on task s RIGHT AFTER learning it (diagonal)

        BWT < 0  →  forgetting
        BWT > 0  →  positive backward transfer (rất hiếm)

        Args:
            accuracy_matrix : list of rows
            task            : task index hiện tại (>= 1 mới có ý nghĩa)

        Returns:
            float  (0.0 nếu task == 0)
        """
        if task == 0 or len(accuracy_matrix) < 2:
            return 0.0

        current_row = accuracy_matrix[task]           # A_{task, *}
        bwt_sum = 0.0
        count = 0
        for s in range(task):                         # s = 0 .. task-1
            a_ss  = accuracy_matrix[s][s]             # diagonal: acc on task s right after trained
            a_ts  = current_row[s]                    # current acc on task s
            bwt_sum += (a_ts - a_ss)
            count += 1

        return bwt_sum / count if count > 0 else 0.0

def compute_bwt(accuracy_matrix: list, task: int) -> float:
    """
    BWT = (1 / task) * Σ_{i=0}^{task-1} [ A_{current, i} - A_{i, i} ]

    A_{current, i} = acc on task i tại round hiện tại (dùng model mới nhất)
    A_{i, i}       = acc on task i ngay khi vừa học xong task i (best at time)

    BWT < 0 → catastrophic forgetting
    BWT > 0 → backward transfer tích cực (hiếm)
    """
    if task == 0 or len(accuracy_matrix) < 2:
        return 0.0

    # A_{current, i}: acc on task i ở row cuối cùng
    current_row = accuracy_matrix[-1]

    bwt_sum = 0.0
    count = 0

    for i in range(task):  # i = 0 .. task-1
        # A_{i, i}: tìm row đầu tiên mà task i xuất hiện (ngay khi học xong task i)
        a_ii = None
        for row in accuracy_matrix:
            if i in row:
                a_ii = row[i]
                break  # lấy lần đầu tiên task i được đánh giá

        a_current_i = current_row.get(i, None)

        if a_ii is not None and a_current_i is not None:
            bwt_sum += (a_current_i - a_ii)
            count += 1

    return bwt_sum / count if count > 0 else 0.0
# evaluate after end 1 task
def load_resnet18_from_checkpoint(ckpt_path: str, load_head: bool = False, num_classes: int = 10) -> torch.nn.Module:
    model  = resnet18(weights=None)
    raw_sd = torch.load(ckpt_path, map_location='cpu')

    new_sd = {}
    for k, v in raw_sd.items():
        if k.startswith('base.'):
            new_sd[k[len('base.'):]] = v
        elif k.startswith('head.') and load_head:
            new_sd[k] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)

    real_missing = [
        k for k in missing
        if 'num_batches_tracked' not in k and not k.startswith('fc.')
        and (load_head or not k.startswith('head.'))
    ]
    if real_missing:
        logger.warning(f'  [WARN] Missing backbone keys: {real_missing}')

    model.to(DEVICE)
    model.eval()
    return model

def load_model_with_head(ckpt_path: str, num_classes: int) -> torch.nn.Module:
    return 
    raw_sd = torch.load(ckpt_path, map_location='cpu')

    head_keys = [k for k in raw_sd.keys() if k.startswith('head.')]
    # print(f'Head keys: {head_keys}')

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    new_sd = {}
    for k, v in raw_sd.items():
        if k.startswith('base.'):
            new_sd[k[len('base.'):]] = v
        elif k == 'head.weight':
            new_sd['fc.weight'] = v
        elif k == 'head.bias':
            new_sd['fc.bias'] = v
        elif k == 'head.fc.weight':
            new_sd['fc.weight'] = v
        elif k == 'head.fc.bias':
            new_sd['fc.bias'] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    real_missing = [k for k in missing if 'num_batches_tracked' not in k]
    # print([k for k in raw_sd.keys() if 'head' in k])
    # print('fc.weight loaded:', 'fc.weight' in new_sd)
    if real_missing:
        logger.warning(f'  [WARN] Missing keys: {real_missing}')

    model.to(DEVICE)
    model.eval()
    return model

def _make_loader(dataset, batch_size: int = 256):
    """
    Tao DataLoader an toan, xu ly moi kieu tra ve cua read_client_data_FCL_cifar10:

      Case 1 - torch.utils.data.Dataset chuan  -> dung truc tiep
      Case 2 - tuple/list 2 phan tu (X, Y) voi X,Y la array/tensor (N,...) -> TensorDataset
      Case 3 - list of (x_i, y_i) sample tuples -> stack roi TensorDataset
    num_workers=0 de tranh loi pickle / seek khi data da duoc load san vao RAM.
    """
    from torch.utils.data import TensorDataset, Dataset

    # Case 1: torch.utils.data.Dataset chuan
    if isinstance(dataset, Dataset):
        return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                          num_workers=0, pin_memory=(DEVICE.type == 'cuda'))

    # Case 2: (X, Y) - moi phan tu la array/tensor ca batch
    # Nhan dien: co dung 2 phan tu va phan tu dau co >= 2 chieu (batch dim + feature dims)
    if (isinstance(dataset, (tuple, list))
            and len(dataset) == 2
            and hasattr(dataset[0], 'shape')
            and len(np.shape(dataset[0])) >= 2):
        X, Y = dataset
        xs = torch.as_tensor(np.array(X, dtype=np.float32))
        ys = torch.as_tensor(np.array(Y)).long()
        return DataLoader(TensorDataset(xs, ys), batch_size=batch_size, shuffle=False,
                          num_workers=0, pin_memory=(DEVICE.type == 'cuda'))

    # Case 3: list of (x_i, y_i) sample tuples
    xs, ys = [], []
    for x, y in dataset:
        xs.append(torch.as_tensor(np.array(x, dtype=np.float32)))
        ys.append(torch.as_tensor(np.array(y)).long())
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    return DataLoader(TensorDataset(xs, ys), batch_size=batch_size, shuffle=False,
                      num_workers=0, pin_memory=(DEVICE.type == 'cuda'))



def compute_feature_resnet18(_model, _model_task_index, _dataset, _target_layer_index: str, seed, args):
    """
    Trích xuất features trên GPU, trả về numpy array (N, D).
    Chấp nhận Dataset chuẩn hoặc list of (x, y) tuples.
    """
    blocks = get_resnet18_blocks(_model)
    _model.eval()
    outputs = []

    loader = _make_loader(_dataset, batch_size=256)

    with torch.no_grad():
        for features, targets in tqdm(loader,
                                      desc=f'Feature M_{_model_task_index}^{_target_layer_index}',
                                      disable=True):
            features = features.to(DEVICE, non_blocking=True)  # (B, C, H, W)

            for block_name, operations in blocks.items():
                features = operations(features)
                if block_name == _target_layer_index:
                    break

            # Flatten → (B, D), trả về CPU ngay để giải phóng VRAM
            outputs.append(torch.flatten(features, 1).cpu())

    return torch.cat(outputs, dim=0).numpy()  # (N, D)

def test_metrics(model, testloader,class_order,task_index):
    model.eval()
    test_acc = 0
    test_num = 0

    with torch.no_grad():
        for x, y in testloader:
            if isinstance(x, list):
                x[0] = x[0].to(DEVICE, non_blocking=True)
            else:
                x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            output = model(x)
            output = mask_classes(output,class_order= class_order,classes_per_task= args.cpt, task_index = task_index)
            accuracy_matrix = (torch.argmax(output, dim=1) == y).cpu().numpy()
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            test_num += y.shape[0]

    return test_acc / test_num if test_num > 0 else 0.0

def compute_forgetting(
    acc_matrix: list,
    current_task: int,
) -> tuple:  # (FM, per_task_forgetting_dict, per_task_max_acc_dict)
    if current_task == 0:
        return float('nan'), {}, {}

    round_global_now = len(acc_matrix) - 1
    f_list = []
    per_task_forgetting = {}
    per_task_max_acc    = {}

    for old_task in range(current_task):
        past_acc_on_old_task = [
            acc_matrix[r][old_task]
            for r in range(round_global_now)
            if old_task in acc_matrix[r]
        ]
        if not past_acc_on_old_task:
            continue

        max_acc             = max(past_acc_on_old_task)
        current_acc_on_old  = acc_matrix[round_global_now].get(old_task, None)
        if current_acc_on_old is None:
            continue

        forgetting_on_old_task          = max_acc - current_acc_on_old
        per_task_forgetting[old_task]   = forgetting_on_old_task
        per_task_max_acc[old_task]      = max_acc
        f_list.append(forgetting_on_old_task)

        logger.info(
            f'    [FM] old_task={old_task} | '
            f'max_past={max_acc*100:.2f}% | '
            f'current={current_acc_on_old*100:.2f}% | '
            f'f={forgetting_on_old_task*100:.2f}%'
        )

    FM = float(np.mean(f_list)) if f_list else float('nan')
    return FM, per_task_forgetting, per_task_max_acc

def compute_eta(_feature_t: np.ndarray):
    feat = torch.from_numpy(_feature_t).to(DEVICE)
    feature_dim = feat.shape[-1]
    norms = torch.linalg.norm(feat, ord=2, dim=-1)
    mn, mx = norms.min().item(), norms.max().item()
    sq = float(feature_dim) ** 0.5
    return mn, mx, mn / sq, mx / sq

def compute_sigma(_feature_t: np.ndarray, _feature_tprime: np.ndarray):
    ft  = torch.from_numpy(_feature_t).to(DEVICE)
    ftp = torch.from_numpy(_feature_tprime).to(DEVICE)
    # Nếu kích thước khác nhau (khác số sample), dùng min
    n = min(ft.shape[0], ftp.shape[0])
    diff = ft[:n] - ftp[:n]
    return torch.linalg.norm(diff, ord=2, dim=-1).max().item()

def compute_eps(_feature_t: np.ndarray, _feature_tprime: np.ndarray):
    # LinearRegression vẫn dùng CPU/numpy (sklearn)
    n = min(_feature_t.shape[0], _feature_tprime.shape[0])
    ft, ftp = _feature_t[:n], _feature_tprime[:n]
    reg = LinearRegression(fit_intercept=False).fit(ftp, ft)
    ftp_transformed = reg.predict(ftp)

    # Tính norm trên GPU
    diff = torch.from_numpy(ft - ftp_transformed).to(DEVICE)
    return torch.linalg.norm(diff, ord=2, dim=-1).max().item()

def compute_cka(feat_a: np.ndarray, feat_b: np.ndarray):
    """CKA tính hoàn toàn trên GPU."""
    cka_obj = TorchCKA(device=DEVICE)
    ta = torch.from_numpy(feat_a).float().to(DEVICE)
    tb = torch.from_numpy(feat_b).float().to(DEVICE)

    hsic_ab = cka_obj.linear_HSIC(ta, tb)
    hsic_aa = cka_obj.linear_HSIC(ta, ta)
    hsic_bb = cka_obj.linear_HSIC(tb, tb)
    cka = hsic_ab / (torch.sqrt(hsic_aa) * torch.sqrt(hsic_bb))
    return hsic_ab, cka

def to_float(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().item()
    return float(x)
#─────────────────────────────────────────────────────
# Hàm đo chính
# ─────────────────────────────────────────────────────────────────────────────
def measure_all_representation_drift(args):
    if args.kaggle == False:
        root = 'C:/Thu/FCL/outputs'
    else :
        root = 'kaggle/working'
    
    output_file = (
        f'{root}/representation_drift_temporal_5tasks_resnet18'
        f'-{args.partition_options}-{args.backbone}.csv'
    )

    os.makedirs(root, exist_ok=True)
    with open(output_file, 'w', newline='') as f:
        f.write(
            'client,block,t,tprime,'
            'sigma_old,eps_old,'
            'linear_cka_old,align10\n'
        )

    task_pairs = list(itertools.combinations(range(args.num_tasks), 2))
    num_blocks  = 5
    total       = args.num_clients * len(task_pairs) * num_blocks
    done        = 0

    # FIX: init acc_history_old ở đây, ngoài tất cả vòng lặp
    # key = (t, tprime, block_idx) → value = old_test_acc của round trước
    acc_history_old: dict = {}

    for client_id in range(args.num_clients):
        logger.info('=' * 60)
        logger.info(
            f'  CLIENT {client_id:>2} / {args.num_clients - 1}'
            f'   ({len(task_pairs)} task-pairs × {num_blocks} blocks)'
        )
        logger.info('=' * 60)
        args.client = client_id
        client_class_order = all_class_orders[client_id][:10]
        list_acc_matrix = []
        for (t, tprime) in task_pairs:

            for round_idx in [24]:
                logger.info(f'  ┌── Task pair ({t}, {tprime}), round {round_idx}')

                #ckpt_t  = get_model_path(args.saving_dir, client_id, t,      round_idx)
                ckpt_t  = get_model_path_no_round(args.saving_dir, client_id, t)
                ckpt_tp  = get_model_path_no_round(args.saving_dir, client_id, tprime)
                ckpt_eps_t = get_model_path_no_round(args.saving_dir, client_id, t)
                #ckpt_tp = get_model_path(args.saving_dir, client_id, tprime, round_idx)
                #ckpt_eps_t = get_model_path(args.saving_dir, client_id, t, 24)


                # model_t      = load_resnet18_from_checkpoint(ckpt_t,  load_head=False)
                # model_eps_t = load_resnet18_from_checkpoint(ckpt_eps_t, load_head=False)
                # model_tprime = load_resnet18_from_checkpoint(ckpt_tp, load_head=False)
                model_t = load_vit_from_checkpoint(ckpt_t,num_classes=10)
                model_eps_t = load_vit_from_checkpoint(ckpt_eps_t,num_classes=10)
                model_tprime = load_vit_from_checkpoint(ckpt_tp,num_classes=10)

                logger.info(f'  │  model_t  ← {ckpt_t}')
                logger.info(f'  │  model_t\' ← {ckpt_tp}')

                test_data_t      = read_client_data_FCL_cifar100(
                    client_id, task=t,      classes_per_task=args.cpt,
                    count_labels=False, train=False)
                test_data_tprime = read_client_data_FCL_cifar100(
                    client_id, task=tprime, classes_per_task=args.cpt,
                    count_labels=False, train=False)

                loader_t      = _make_loader(test_data_t)
                loader_tprime = _make_loader(test_data_tprime)

                # model_head_t  = load_model_with_head(ckpt_t,  num_classes=args.classes)
                # model_head_tp = load_model_with_head(ckpt_tp, num_classes=args.classes)

                # # ── Logits & cosine similarity (head level) ──────────────────
                # logits_t_list      = []
                # logits_tprime_list = []

                # for x_t, _ in loader_t:
                #     logits_t_list.append(
                #         model_head_t(x_t.to(DEVICE)).detach().cpu()
                #     )
                # for x_tp, _ in loader_tprime:
                #     logits_tprime_list.append(
                #         model_head_tp(x_tp.to(DEVICE)).detach().cpu()
                #     )

                # logits_t      = torch.cat(logits_t_list,      dim=0)
                # logits_tprime = torch.cat(logits_tprime_list, dim=0)

                # # cos_sin: cosine giữa logit của model_tp trên data_t vs data_tp
                # # NOTE: 2 loader khác nhau nên shape có thể khác nhau
                # # → dùng logits của cùng loader_t cho cả 2 model để so sánh fair
                # logits_tp_on_t_list = []
                # for x_t, _ in loader_t:
                #     logits_tp_on_t_list.append(
                #         model_head_tp(x_t.to(DEVICE)).detach().cpu()
                #     )
                # logits_tp_on_t = torch.cat(logits_tp_on_t_list, dim=0)

                # # cosine giữa logit của model_t và model_tp, cùng trên data_t
                # cos_sin = torch.nn.functional.cosine_similarity(
                #     logits_t, logits_tp_on_t, dim=1
                # )

                # # # ── Accuracy ─────────────────────────────────────────────────
                # acc_t_on_head    = test_metrics(model_head_t,  loader_t,class_order=client_class_order, task_index=t)
                # current_test_acc = test_metrics(model_head_tp, loader_tprime,class_order=client_class_order, task_index=tprime)
                # old_test_acc     = test_metrics(model_head_tp, loader_t,class_order=client_class_order, task_index=t)
                # list_acc_matrix.append(acc_t_on_head)
                # drop_acc = max(list_acc_matrix) - old_test_acc
                # # ── Neuron importance (layer4[-1], fixed) ────────────────────
                # target_layer_tp = [model_tprime.layer4[-1]]
                # model_cam_curr  = BaseCAM(model_tprime, target_layer_tp)
                # neuron_curr     = model_cam_curr.get_importance(loader_t, target_layer_tp)

                # target_layer_t = [model_t.layer4[-1]]
                # model_cam_prev = BaseCAM(model_t, target_layer_t)
                # neuron_prev    = model_cam_prev.get_importance(loader_t, target_layer_t)

                # # Drift (L2)
                # drift_neuron = torch.norm(neuron_curr - neuron_prev).item()

                # # Cosine similarity của importance vector
                # cosine_neuron = torch.nn.functional.cosine_similarity(
                #     neuron_curr.unsqueeze(0),
                #     neuron_prev.unsqueeze(0)
                # ).item()

                # # Top-k overlap
                # k_top    = 50
                # top_curr = torch.topk(neuron_curr, k_top).indices
                # top_prev = torch.topk(neuron_prev, k_top).indices
                # overlap  = len(
                #     set(top_curr.tolist()) & set(top_prev.tolist())
                # ) / k_top

                # ── Per-block loop ───────────────────────────────────────────
                for block_idx in range(12):
                    target_layer = f'block{block_idx}'


                    try:
                        feat_t_old = compute_feature_vit(
                                model_eps_t, t, test_data_t, block_idx, args
                            )

                        feat_tp_old = compute_feature_vit(
                            model_tprime, tprime, test_data_t, block_idx, args
                        )
                        
                        # eta_min, eta_max, eta_min_n, eta_max_n = compute_eta(feat_t_old)
                        sigma_old         = compute_sigma(feat_t_old, feat_tp_old)
                        eps_old           = compute_eps(feat_t_old, feat_tp_old)
                        hsic_val, cka_old = compute_cka(feat_t_old, feat_tp_old)

                        feat_t_tensor_old  = torch.from_numpy(feat_t_old).float().to(DEVICE)
                        feat_tp_tensor_old = torch.from_numpy(feat_tp_old).float().to(DEVICE)
                        cka_obj        = TorchCKA(device=DEVICE)
                        linear_cka_old     = cka_obj.linear_CKA(feat_t_tensor_old, feat_tp_tensor_old)
                        kernel_cka_old     = cka_obj.kernel_CKA(feat_t_tensor_old, feat_tp_tensor_old, sigma=None)



                        
                        topk_list  = [10]
                        align_score = {}
                        for k in topk_list:
                            align_score[k], _ = compute_alignment_from_arrays(
                                feat_t_old, feat_tp_old, "mutual_knn", topk=k, precise=True)

                        # # acc_drop_rate: so với round trước của cùng (t, tprime, block)
                        # prev_key     = (t, tprime, block_idx)
                        # prev_acc_old = acc_history_old.get(prev_key, float('nan'))
                        # acc_drop_rate = (
                        #     (prev_acc_old - old_test_acc)
                        #     if not (prev_acc_old != prev_acc_old)  # not isnan
                        #     else float('nan')
                        # )
                        # acc_history_old[prev_key] = old_test_acc

                        done    += 1
                        progress = f'[{done}/{total}]'

                        logger.info(
                            f'  │  {progress} {target_layer} | '
                        #     f'cos_sin_logit={cos_sin.mean():.4f}  '
                        #     f'drift_neuron={drift_neuron:.4f}  '
                        #     f'cosine_neuron={cosine_neuron:.4f}  '
                        #     f'overlap@{k_top}={overlap:.4f}  '
                             f'σ_old={sigma_old:.4f}   '
                           f'ε_old={eps_old:.4f}'
                           f'linear_CKA_old={float(linear_cka_old):.4f}  '
                        #     f'kernel_CKA_old={float(kernel_cka_old):.4f}  '

                        #     f'forgetting={drop_acc*100:.4f}%'

                        #     f'align@10={align_score[10]:.4f}  '
                        #     f'ACC({tprime})={current_test_acc*100:.2f}%  '
                        #     f'ACC_old({t})={old_test_acc*100:.2f}%  '
                        #     f'ACC_t_real={acc_t_on_head*100:.2f}%  '
                        #    f'acc_drop={acc_drop_rate*100 if acc_drop_rate==acc_drop_rate else float("nan"):.2f}%'
                        )



                        # ── WandB ─────────────────────────────────────────────
                        print(">>> WANDB LOGGING", round_idx)
                        if args.use_wandb :
                            prefix = f'block{block_idx}/pair_{t}_{tprime}'

                            wandb.log({
                                # index
                                'round': round_idx,
                                'client': client_id,

                                # representation
                                # f'{client_id}/{prefix}/cka': float(cka_old),
                                # f'{client_id}/{prefix}/linear_cka': float(linear_cka_old),
                                # f'{client_id}/{prefix}/kernel_cka': float(kernel_cka_old),
                                # f'{client_id}/{prefix}/align10': align_score[10],
                                # # f'{client_id}/{prefix}/eta_min_norm': eta_min_n,
                                # # f'{client_id}/{prefix}/eta_max_norm': eta_max_n,

                                # # f'{client_id}/{prefix}/ratio_feature': ratio_feature,
                                # # f'{client_id}/{prefix}/width_t': width_t,
                                # # f'{client_id}/{prefix}/width_tprime': width_tp,
                                # f'{client_id}/{prefix}/sigma_old': sigma_old,
                                # f'{client_id}/{prefix}/eps_old': eps_old,
                                # f'{client_id}/{prefix}/accuracy_tprime': current_test_acc * 100,
                                # f'{client_id}/{prefix}/accold_taskpair_{t}_{tprime}': old_test_acc * 100,
                                # f'{client_id}/{prefix}/acc_t_on_head': acc_t_on_head * 100,
                                # f'{client_id}/{prefix}/acc_drop_rate': acc_drop_rate * 100 if acc_drop_rate == acc_drop_rate else float('nan'),
                                # f'{client_id}/{prefix}/cos_sin_logit': cos_sin.mean().item(),
                                # f'{client_id}/{prefix}/forgetting_drop': drop_acc * 100,
                                # # neuron-level
                                # f'{client_id}/{prefix}/drift_neuron': drift_neuron,
                                # f'{client_id}/{prefix}/cosine_neuron': cosine_neuron,
                                # f'{client_id}/{prefix}/overlap_at50': overlap,



                            })

                        # ── CSV write ─────────────────────────────────────────
                        try:
                            line = (
                                f'{client_id},{block_idx},{t},{tprime},'                                
                                f'{sigma_old},{eps_old},'
                                f'{float(linear_cka_old)},{align_score[10]}\n'
                            )

                            if 'nan' in line.lower():
                                logger.warning(
                                    f'[NaN DETECTED] client={client_id} '
                                    f'block={block_idx} t={t} tp={tprime} '
                                    f'round={round_idx}'
                                )

                            with open(output_file, 'a') as f:
                                f.write(line)
                                f.flush()

                            if done % 50 == 0:
                                logger.debug(f'[SAMPLE LINE] {line.strip()}')

                        except Exception as e:
                            logger.error(
                                f'[WRITE FAIL] client={client_id} '
                                f'block={block_idx} t={t} tp={tprime} | {e}'
                            )

                    except Exception as e:
                        logger.error(
                            f'  │  [SKIP] client={client_id} {target_layer} '
                            f't={t} tp={tprime} round={round_idx} | {e}'
                        )
                        continue

            # # FIX: close scatter SAU khi hết vòng round, không phải trong vòng round
            # for block_idx in range(num_blocks):
            #     scatters[block_idx].close()
            # logger.info(f'  └── Task pair ({t}, {tprime}) done')

    logger.info(f'\n  Hoàn thành! CSV → {output_file}')

def get_shared_probe_dataset(
    datadir='./dataset/cifar10-classes/',
    classes=list(range(10)),
    images_per_class=75,
    train_images_per_class=5000,
):
    x_list, y_list = [], []
    for cls in classes:
        data_file = datadir + str(cls) + '.npy'
        data = np.load(data_file)
        # Lấy từ phần test (sau train split)
        test_data = data[train_images_per_class : train_images_per_class + images_per_class]
        x_list.append(test_data)
        y_list.append(np.full(images_per_class, cls, dtype=np.int64))

    x = torch.tensor(np.concatenate(x_list), dtype=torch.float32)
    y = torch.tensor(np.concatenate(y_list), dtype=torch.long)
    return Transform_dataset(x, y)

def get_shared_probe_dataset_task(
    task_id,
    classes_per_task=2,
    datadir='./dataset/cifar10-classes/',
    images_per_class=1000,
    train_images_per_class=5000,
    split="test",
):
    start_cls = task_id * classes_per_task
    end_cls = (task_id + 1) * classes_per_task
    classes = list(range(start_cls, end_cls))

    x_list, y_list = [], []

    for cls in classes:
        data_file = os.path.join(datadir, f"{cls}.npy")
        data = np.load(data_file)

        if split == "test":
            start = train_images_per_class
        elif split == "train":
            start = 0
        else:
            raise ValueError(f"Unknown split={split}")

        end = start + images_per_class
        selected = data[start:end]

        print(
            f"[PROBE] task={task_id} class={cls} "
            f"split={split} idx=[{start}:{end}] "
            f"shape={selected.shape}"
        )

        x_list.append(selected)
        y_list.append(np.full(len(selected), cls, dtype=np.int64))

    x = torch.tensor(np.concatenate(x_list, axis=0), dtype=torch.float32)
    y = torch.tensor(np.concatenate(y_list, axis=0), dtype=torch.long)

    print(f"[PROBE] unique labels = {torch.unique(y).tolist()} | total={len(y)}")

    return Transform_dataset(x, y)
def measure_all_drift_follow_task_client_pair(args):
    output_file = f'/home/ubuntu/thu.td/FCL_3/outputs/client_representation_drift_{args.num_tasks}_{args.num_clients}Client_-{args.partition_options}-{args.backbone}.csv'

    if not os.path.isfile(output_file):
        with open(output_file, 'w') as f:
            f.write('block_idx,client1,client2,t,'
                    'cka,sigma,eps,'
                    'align@10,align@20\n')

    client_pairs = list(itertools.combinations(range(args.num_clients), 2))
    
    num_blocks   = 5
    topk_list    = [10, 20]

    total = args.num_tasks * len(client_pairs) * num_blocks
    done  = 0

    root = "result_client_pair1"
    os.makedirs(root, exist_ok=True)
    print(f"Root directory for results: {root}")

    for task_id in range(args.num_tasks):
        logger.info('=' * 60)
        logger.info(f'  TASK {task_id:>2} / {args.num_tasks - 1}'
                    f'   ({len(client_pairs)} client-pairs × {num_blocks} blocks)')
        logger.info('=' * 60)
        args.task = task_id

        pair_scatters = {
            (c, cp): {
                b: ScatterLogger(f"{root}/block{b}/clientpair_{c}_{cp}")
                for b in range(num_blocks)
            }
            for (c, cp) in client_pairs
        }

        for (client, client_prime) in client_pairs:
            scatters = pair_scatters[(client, client_prime)]
            logger.info(f'  ┌── Client pair ({client}, {client_prime})')

            ckpt_client       = get_model_path_no_round(args.saving_dir, client,       task_id)
            #ckpt_client = get_model_path(args.saving_dir, client,       task_id, 24)
            #ckpt_client_prime = get_model_path(args.saving_dir, client_prime, task_id, 24)
            ckpt_client_prime = get_model_path_no_round(args.saving_dir, client_prime, task_id)

            skip = False
            for ckpt in [ckpt_client, ckpt_client_prime]:
                if not os.path.isfile(ckpt):
                    logger.error(f'  [MISSING] {ckpt}')
                    skip = True
            if skip:
                continue

            model_c      = load_resnet18_from_checkpoint(ckpt_client,       load_head=False)
            model_cprime = load_resnet18_from_checkpoint(ckpt_client_prime, load_head=False)
            #model_c = load_vit_from_checkpoint(ckpt_client,num_classes=10)
            #model_cprime = load_vit_from_checkpoint(ckpt_client_prime,num_classes=10)
            logger.info(f'  │  model_c  ← {ckpt_client}')
            logger.info(f'  │  model_c\' ← {ckpt_client_prime}')

            # shared_probe = get_shared_probe_dataset_task(
            #     task_id=task_id,
            #     classes_per_task=args.cpt,
            #     images_per_class=250,
            #     train_images_per_class=5000,
            #     split="test",
            # )
            shared_probe = get_shared_probe_dataset()
            # model_head_c  = load_model_with_head(ckpt_client,       num_classes=args.classes)
            # model_head_cp = load_model_with_head(ckpt_client_prime, num_classes=args.classes)
            loader_c      = _make_loader(shared_probe)
            loader_cp     = _make_loader(shared_probe)

            # logits_c_list  = []
            # logits_cp_list = []
            # for x, _ in loader_c:
            #     logits_c_list.append(model_head_c(x.to(DEVICE)).detach().cpu())
            # for x, _ in loader_cp:
            #     logits_cp_list.append(model_head_cp(x.to(DEVICE)).detach().cpu())

            # logits_c  = torch.cat(logits_c_list,  dim=0)
            # logits_cp = torch.cat(logits_cp_list, dim=0)
            # cos_sim   = torch.nn.functional.cosine_similarity(logits_c, logits_cp, dim=1).mean().item()

            for num_block in range(num_blocks):
                target_layer = f'block{num_block}'
                scatter      = scatters[num_block]
                try:
                    logger.info(f'  computing feat_c block{num_block}...')
                    feat_c  = compute_feature_resnet18(model_c,      task_id, shared_probe, target_layer, args.seed, args)
                    logger.info(f'  computing feat_cp block{num_block}...')
                    feat_cp = compute_feature_resnet18(model_cprime, task_id, shared_probe, target_layer, args.seed, args)
                    logger.info(f'  computing sigma...')
                    sigma        = compute_sigma(feat_c, feat_cp)
                    logger.info(f'  computing eps...')
                    eps          = compute_eps(feat_c, feat_cp)
                    logger.info(f'  computing cka...')
                    _, cka       = compute_cka(feat_c, feat_cp)

                    align_scores = {}
                    for k in topk_list:
                        align_scores[k], _ = compute_alignment_from_arrays(
                            feat_c, feat_cp, "mutual_knn", topk=k, precise=True)

                    done += 1
                    logger.info(
                        f'  │  [{done}/{total}] {target_layer} | '
                        f'  σ={sigma:.4f}  '
                        f'ε={eps:.4f}  CKA={cka:.4f}  '
                        f'align@10={align_scores[10]:.4f}  align@20={align_scores[20]:.4f}'
                    )



                    line = (
                        f'{num_block},{client},{client_prime},{task_id},'
                        f'{cka:.6f},{sigma:.6f},{eps:.6f},'
                        f'{align_scores[10]:.4f}\n'
                    )
                    with open(output_file, 'a') as f:
                        f.write(line)
                        f.flush()

                    if args.use_wandb:
                        wandb.log({
                            'client':       client,
                            'client_prime': client_prime,
                            'block':        num_block,
                            't':            task_id,
                            'sigma':        sigma,
                            'eps':          eps,
                            'cka':          cka,
                            'align@10':     align_scores[10],
                            'align@20':     align_scores[20],
                            'pair':         f'({client},{client_prime})',
                        })

                except Exception as e:
                    logger.error(
                        f'  │  [SKIP] pair=({client},{client_prime}) '
                        f'{target_layer} t={task_id} | {e}'
                    )
                    continue

            logger.info(f'  └── Client pair ({client}, {client_prime}) done')

    logger.info(f'\n✅  Hoàn thành! CSV → {output_file}')

    
def measure_follow_training(args):

    if args.kaggle == False:
        root = './outputs'
    else :
        root = '/kaggle/working'
    output_file = (
        f'{root}/representation_drift_temporal_252_4'
        f'-{args.partition_options}-{args.backbone}.csv'
    )

    header = (
        'client,block,task,round,'
        'eps_current,cka_curr,align10_curr\n'
    )

    write_header = (
        not os.path.isfile(output_file)
        or os.path.getsize(output_file) == 0
    )

    if write_header:
        with open(output_file, 'a') as f:
            f.write(header)

    output_file_v2 = (
        f'{root}/forgetting_detail_client9'
        f'-{args.partition_options}-{args.backbone}.csv'
    )

    header_v2 = (
        'client,task_old,task_current,round,'
        'acc_curr_on_curr,acc_curr_on_old,'
        'forgetting_per_task,FM,bwt,max_past_acc\n'
    )

    write_header_v2 = (
        not os.path.isfile(output_file_v2)
        or os.path.getsize(output_file_v2) == 0
    )
    with open(output_file_v2, 'a') as f:
        if write_header_v2:
            f.write(header_v2)
    num_blocks = 5

    for client_id in [0]:
        logger.info('=' * 60)
        logger.info(f'  CLIENT {client_id:>2} / {args.num_clients - 1}')
        logger.info('=' * 60)

        acc_matrix   = []
        round_global = 0
        client_class_order = all_class_orders[client_id][:10]
        # ── TRƯỚC vòng for task — init bảng theo block ──────────────────────
        # if args.use_wandb:
        #     block_tables = {
        #         block_idx: {
        #             "table_eps":                  wandb.Table(columns=["epsilon",  "accuracy",   "task"], log_mode="MUTABLE"),
        #             "table_sigma":                wandb.Table(columns=["sigma",    "accuracy",   "task"], log_mode="MUTABLE"),
        #             "table_cka":                  wandb.Table(columns=["cka",      "accuracy",   "task"], log_mode="MUTABLE"),
        #             "table_align":                wandb.Table(columns=["align150", "accuracy",   "task"], log_mode="MUTABLE"),
        #             "table_cosine":               wandb.Table(columns=["cosine",   "accuracy",   "task"], log_mode="MUTABLE"),
        #             "table_eps_forgetting":       wandb.Table(columns=["epsilon",  "forgetting", "task"], log_mode="MUTABLE"),
        #             "table_sigma_forgetting":     wandb.Table(columns=["sigma",    "forgetting", "task"], log_mode="MUTABLE"),
        #             "table_cka_forgetting":       wandb.Table(columns=["cka",      "forgetting", "task"], log_mode="MUTABLE"),
        #             "table_align_forgetting":     wandb.Table(columns=["align150", "forgetting", "task"], log_mode="MUTABLE"),
        #             "table_cosine_forgetting":    wandb.Table(columns=["cosine",   "forgetting", "task"], log_mode="MUTABLE"),
        #             "table_eps_old":              wandb.Table(columns=["epsilon",  "accuracy",   "task"], log_mode="MUTABLE"),
        #             "table_sigma_old":            wandb.Table(columns=["sigma",    "accuracy",   "task"], log_mode="MUTABLE"),
        #             "table_cka_old":              wandb.Table(columns=["cka",      "accuracy",   "task"], log_mode="MUTABLE"),
        #             "table_eps_forgetting_old":   wandb.Table(columns=["epsilon",  "forgetting", "task"], log_mode="MUTABLE"),
        #             "table_sigma_forgetting_old": wandb.Table(columns=["sigma",    "forgetting", "task"], log_mode="MUTABLE"),
        #             "table_cka_forgetting_old":   wandb.Table(columns=["cka",      "forgetting", "task"], log_mode="MUTABLE"),
        #             "table_old_curr_eps_acc":     wandb.Table(columns=["epsilon",  "acc",        "type", "round"], log_mode="MUTABLE"),
        #             "table_old_curr_eps_fgt":     wandb.Table(columns=["epsilon",  "forgetting", "type", "round"], log_mode="MUTABLE"),
        #             "table_kernel_old":           wandb.Table(columns=["kernel_cka", "accuracy", "task"], log_mode="MUTABLE"),
        #             "table_nl_cka_old":           wandb.Table(columns=["nl_cka", "accuracy", "task"], log_mode="MUTABLE")
        #         }
        #         for block_idx in range(num_blocks)
        #     }
        for task in range(0, 5):
            logger.info(f'  ── Task {task}')

            # scatters = {
            #     block_idx: ScatterLogger(
            #         f"23_Follow_training_logs/client_{client_id}/block{block_idx}/task_{task}"
            #     )
            #     for block_idx in range(num_blocks)
            # }

            # Load old loaders 1 lần cho cả task
            old_loaders = {}
            for old_task in range(task):
                test_data_old_task = read_client_data_FCL_cifar10(
                    client_id, task=old_task,
                    classes_per_task=args.cpt,
                    count_labels=False, train=False,
                )
                old_loaders[old_task] = _make_loader(test_data_old_task)

            # ── Vòng round ──────────────────────────────────────────────────
            for round_idx in range(25):

                if round_idx == 0 and task == 0:
                    logger.info(f'  │  [SKIP] task=0 round=0 — no previous checkpoint')
                    continue

                if round_idx == 0:
                    ckpt_curr = get_model_path(args.saving_dir, client_id, task,     0)
                    ckpt_prev = get_model_path(args.saving_dir, client_id, task - 1, 0)
                    logger.info(
                        f'  │  [cross-task] task={task} round=0 '
                        f'← task={task-1} round=0 as baseline'
                    )
                else:
                    ckpt_curr = get_model_path(args.saving_dir, client_id, task, round_idx)
                    ckpt_prev = get_model_path(args.saving_dir, client_id, task, round_idx - 1)
                if task > 0: 
                    ckpt_prev_task = get_model_path(args.saving_dir,client_id,task -1, round_idx)
                missing = [c for c in [ckpt_curr, ckpt_prev] if not os.path.isfile(c)]
                if missing:
                    for m in missing:
                        logger.error(f'  │  [MISSING] {m}')
                    continue

                model_curr      = load_resnet18_from_checkpoint(ckpt_curr, load_head=False)
                model_prev      = load_resnet18_from_checkpoint(ckpt_prev, load_head=False)
                # model_head_curr = load_model_with_head(ckpt_curr, num_classes=10)
                # model_head_prev = load_model_with_head(ckpt_prev, num_classes=10)
                # if task > 0:
                #     model_prev_task = load_resnet18_from_checkpoint(ckpt_prev_task,load_head=False)
                logger.info(f'  │  model_curr ← {ckpt_curr}')
                logger.info(f'  │  model_prev ← {ckpt_prev}')

                test_data_curr = read_client_data_FCL_cifar10(
                    client_id, task=task, classes_per_task=args.cpt,
                    count_labels=False, train=False
                )
                loader_curr = _make_loader(test_data_curr)
                all_labels = []
                for _, y in loader_curr:
                    all_labels.append(y)

                all_labels = torch.cat(all_labels)

                print("Unique labels:", torch.unique(all_labels))
                has_old = task > 0
                # if has_old:
                #     test_data_old = read_client_data_FCL_cifar10(
                #         client_id, task=task - 1, classes_per_task=args.cpt,
                #         count_labels=False, train=False
                #     )
                #     loader_old = _make_loader(test_data_old)
                #     all_labels = []

                    # for _, y in loader_old:
                    #     all_labels.append(y)

                    # all_labels = torch.cat(all_labels)

                    # print("Unique labels:", torch.unique(all_labels))

                # logits_curr_list, logits_prev_list = [], []
                # for x, _ in loader_curr:
                #     x = x.to(DEVICE)
                #     logits_curr_list.append(model_head_curr(x).detach().cpu())
                #     logits_prev_list.append(model_head_prev(x).detach().cpu())

                # logits_curr = torch.cat(logits_curr_list, dim=0)
                # logits_prev = torch.cat(logits_prev_list, dim=0)
                # cos_sim = torch.nn.functional.cosine_similarity(logits_curr, logits_prev, dim=1)
                # row = {}
                # acc_curr_on_curr = test_metrics(model_head_curr, loader_curr,
                #                         class_order=client_class_order,
                #                         task_index=task)
                # acc_curr_on_old  = test_metrics(model_head_curr, loader_old,
                #                                 class_order=client_class_order,
                #                                 task_index=task - 1) if has_old else float('nan')
                # row[task] = acc_curr_on_curr

                # if has_old:
                #     for old_task, old_loader in old_loaders.items():
                #         row[old_task] = test_metrics(model_head_curr, old_loader,
                #                                      class_order=client_class_order,
                #                                      task_index=old_task)

                # acc_matrix.append(row)
                
                # FM, per_task_forgetting, per_task_max_acc = compute_forgetting(acc_matrix=acc_matrix, current_task=task)
                # bwt = compute_bwt(accuracy_matrix=acc_matrix,task=task)
                # preds = []
                # for x, _ in loader_curr:
                #     x = x.to(DEVICE)
                #     pred = model_head_curr(x).argmax(1)
                #     preds.append(pred.cpu())

                # preds = torch.cat(preds)
                # #print(f"Range predict curr : {preds.min()}, {preds.max()}")
                # preds = []
                # for x, _ in loader_curr:
                #     x = x.to(DEVICE)
                #     pred = model_head_prev(x).argmax(1)
                #     preds.append(pred.cpu())

                # preds = torch.cat(preds)
                #print(f"Range predict old : {preds.min()}, {preds.max()}")
                # pred_old = model_head_prev(loader_old).argmax(1)
                # print(f"Range predict old : {pred_old.min()} , {pred_old.max()}") 
                #fwt = compute_fwt(accuracy_matrix=acc_matrix,task=task,random_baseline=None)
                                # Grad-CAM
                target_layer_curr = [model_curr.layer4[-1]]
                # model_with_grad_cam_curr = BaseCAM(model_curr, target_layer_curr)
                # neuron_important_curr = model_with_grad_cam_curr.get_importance(loader_old,target_layer_curr) if has_old else model_with_grad_cam_curr.get_importance(loader_curr,target_layer_curr)

                # if has_old:
                #     target_layer_prev = [model_prev_task.layer4[-1]]
                #     model_with_grad_cam_prev = BaseCAM(model_prev_task, target_layer_prev)
                #     neuron_important_prev = model_with_grad_cam_prev.get_importance(loader_old,target_layer_prev)
                   
                #     # ===== Drift =====
                #     drift_neuron = torch.norm(neuron_important_curr - neuron_important_prev)

                #     # ===== Range =====
                #     curr_min, curr_max = neuron_important_curr.min(), neuron_important_curr.max()
                #     prev_min, prev_max = neuron_important_prev.min(), neuron_important_prev.max()

                #     # ===== Top-k =====
                #     k_top = 50
                #     top_curr = torch.topk(neuron_important_curr, k_top).indices
                #     top_prev = torch.topk(neuron_important_prev, k_top).indices

                #     # ===== Overlap =====
                #     overlap = len(set(top_curr.tolist()) & set(top_prev.tolist())) / k_top

                #     # ===== Cosine similarity (rất nên có) =====
                #     cosine_neuron = torch.nn.functional.cosine_similarity(
                #         neuron_important_curr.unsqueeze(0),
                #         neuron_important_prev.unsqueeze(0)
                #     ).item()

                # else:
                #     neuron_important_prev = torch.tensor(float('nan'))
                #     drift_neuron = torch.tensor(float('nan'))
                #     curr_min, curr_max = neuron_important_curr.min(), neuron_important_curr.max()
                #     prev_min, prev_max = float('nan'), float('nan')
                #     overlap = float('nan')
                #     cosine_neuron = float('nan')

                # # ===== Log =====
                # logger.info(
                #     #f"Neurun curr : {neuron_important_curr}\n Neuron prev : {neuron_important_prev}\n"
                #     f"Neuron drift={drift_neuron:.4f} | "
                #     f"curr_range=({curr_min:.4f},{curr_max:.4f}) | "
                #     f"prev_range=({prev_min:.4f},{prev_max:.4f}) | "
                #     f"overlap@50={overlap:.4f} | "
                #     f"cosine={cosine_neuron:.4f}"
                # )
                round_global += 1

                logger.info(
                    f'  │  task={task} round_idx={round_idx} round_global={round_global} | '
                    # f'acc_curr={acc_curr_on_curr*100:.2f}%  '
                    # f'FM={ f"{FM*100:.2f}%" if FM == FM else "N/A" }'
                )
                scalar_log = {}
                double_log = {}
                # ── Per-block metrics ────────────────────────────────────────
                for block_idx in [4]:
                    target_layer = f'block{block_idx}'
                    # scatter      = scatters[block_idx]

                    try:
                        # ── Features trên current task data (luôn có) ──────────
                        feat_curr_on_curr_data       = compute_feature_resnet18(model_curr, task, test_data_curr, target_layer, args.seed, args)
                        feat_prev_round_on_curr_data = compute_feature_resnet18(model_prev, task, test_data_curr, target_layer, args.seed, args)

                        # width_curr = compute_width(model_curr, block_idx - 1) if block_idx > 0 else float('nan')
                        # width_prev = compute_width(model_prev, block_idx - 1) if block_idx > 0 else float('nan')

                        # eta_min_on_curr_data, eta_max_on_curr_data, eta_min_n, eta_max_n = compute_eta(feat_curr_on_curr_data)
                        # sigma_on_curr_data = compute_sigma(feat_curr_on_curr_data, feat_prev_round_on_curr_data)
                        eps_on_curr_data   = compute_eps(feat_curr_on_curr_data,   feat_prev_round_on_curr_data)
                        _, cka_on_curr_data = compute_cka(feat_curr_on_curr_data,  feat_prev_round_on_curr_data)
                        #ratio_feat = eta_max_on_curr_data / eta_min_on_curr_data if eta_min_on_curr_data > 0 else float('nan')

                        align_score_on_curr_data = {}
                        for k in [10,20]:
                            align_score_on_curr_data[k], _ = compute_alignment_from_arrays(
                                feat_curr_on_curr_data, feat_prev_round_on_curr_data, "mutual_knn", topk=k, precise=True
                            )

                #         # ── Metrics trên old task data (chỉ khi has_old) ───────
                #         NAN = float('nan')
                #         sigma_on_old_data  = NAN
                #         eps_on_old_data    = NAN
                #         cka_on_old_data    = NAN
                #         linear_cka         = NAN
                #         kernel_cka         = NAN
                #         nl_cka             = NAN
                #         align_score_on_old_data = {}
                #         align_old_150      = NAN

                #         if has_old:
                #             feat_curr_on_old_data = compute_feature_resnet18(model_curr, task, test_data_old, target_layer, args.seed, args)
                #             feat_prev_on_old_data = compute_feature_resnet18(model_prev, task, test_data_old, target_layer, args.seed, args)

                #             sigma_on_old_data  = compute_sigma(feat_curr_on_old_data, feat_prev_on_old_data)
                #             eps_on_old_data    = compute_eps(feat_curr_on_old_data,   feat_prev_on_old_data)
                #             _, cka_on_old_data = compute_cka(feat_curr_on_old_data,   feat_prev_on_old_data)

                #             feat_curr_t = torch.from_numpy(feat_curr_on_old_data).float().to(DEVICE)
                #             feat_prev_t = torch.from_numpy(feat_prev_on_old_data).float().to(DEVICE)
                #             cka_obj    = TorchCKA(device=DEVICE)
                #             linear_cka = cka_obj.linear_CKA(feat_curr_t, feat_prev_t)
                #             kernel_cka = cka_obj.kernel_CKA(feat_curr_t, feat_prev_t, sigma=None)
                #             nl_cka     = kernel_cka - linear_cka

                #             for k in [20, 100, 150]:
                #                 align_score_on_old_data[k], _ = compute_alignment_from_arrays(
                #                     feat_curr_on_old_data, feat_prev_on_old_data, "mutual_knn", topk=k, precise=True
                #                 )
                #             align_old_150 = align_score_on_old_data.get(150, NAN)

                        # # ── Log ────────────────────────────────────────────────
                        # def _fmt(v):
                        #     return f'{v:.4f}' if v == v else 'nan'

                        logger.info(
                            # f'  │  [{block_idx+1}/{num_blocks}] {target_layer} | '
                            # f'FM={_fmt(FM*100)}% '
                            # f'bwt={bwt}, '
                            # f'cosine={cos_sim.mean().item():.4f}  '
                            #f'σ_curr={sigma_on_curr_data:.4f}  ε_curr={eps_on_curr_data:.4f}  '
                                f'cka_curr={cka_on_curr_data:.4f}  '
                                f'align@10_curr={align_score_on_curr_data.get(10, float("nan")):.4f}  '
                                f'eps_curr={eps_on_curr_data:.4f}  '
                            # f'σ_old={_fmt(sigma_on_old_data)}  ε_old={_fmt(eps_on_old_data)}  '
                            # f'linCKA={_fmt(linear_cka)}  nlCKA={_fmt(nl_cka)}  kCKA={_fmt(kernel_cka)}  '
                            # f'dim={_fmt(ratio_feat)}  '
                            # f'align@150_old={_fmt(align_old_150)}  '
                            # f'ACC_curr={acc_curr_on_curr*100:.2f}%  '
                            # f'ACC_old={_fmt(acc_curr_on_old*100) if has_old else "N/A"}  '
                        )

                        # ── CSV ─────────────────────────────────────────────────
                        with open(output_file, 'a') as f:
                            def to_val(x):
                                import torch
                                return x.item() if isinstance(x, torch.Tensor) else x

                            csv_row = [
                                client_id, block_idx, task, round_idx,
                                # sigma_on_curr_data, eps_on_curr_data,
                                # sigma_on_old_data, eps_on_old_data, cos_sim.mean().item(),
                                # to_val(linear_cka), to_val(nl_cka), to_val(kernel_cka), bwt,
                                # acc_curr_on_curr,
                                to_val(eps_on_curr_data), to_val(cka_on_curr_data), to_val(align_score_on_curr_data.get(10, float('nan')))
                            ]
                            f.write(','.join(map(str, csv_row)) + '\n')
                # with open(output_file_v2, 'a') as f:
                #     if task == 0:
                #         # Task 0: chỉ có acc_curr_on_curr, không có old task nào
                #         f.write(
                #             f'{client_id},'
                #             f'nan,'                         # task_old
                #             f'{task},'                      # task_current
                #             f'{round_idx},'
                #             # f'{acc_curr_on_curr:.6f},'
                #             f'nan,'                         # acc_curr_on_old
                #             f'nan,'                         # forgetting_per_task
                #             f'nan,'                         # FM
                #             #f'{bwt},'
                #             f'nan\n'                        # max_past_acc
                #         )
                    # else:
                    #     for old_task in range(task):
                    #         f_per_task   = per_task_forgetting.get(old_task, float('nan'))
                    #         max_past     = per_task_max_acc.get(old_task, float('nan'))

                    #         # acc_curr_on_old: lấy từ row hiện tại của acc_matrix
                    #         acc_on_old = acc_matrix[-1].get(old_task, float('nan'))

                    #         f.write(
                    #             f'{client_id},'
                    #             f'{old_task},'              # task_old
                    #             f'{task},'                  # task_current
                    #             f'{round_idx},'
                    #             f'{acc_curr_on_curr:.6f},'
                    #             f'{acc_on_old:.6f},'        # acc_curr_on_old
                    #             f'{f_per_task:.6f},'        # forgetting per old_task
                    #             f'{FM:.6f},'               # FM (mean across all old tasks)
                    #             #f'{bwt},'
                    #             f'{max_past:.6f}\n'         # max_past_acc per old_task
                    #         )
                    except Exception as e:
                        logger.error(
                            # f'  │  [SKIP] client={client_id} {target_layer} '
                            f'task={task} round={round_idx} | {e}'
                        )
                        # import traceback
                        # logger.debug(traceback.format_exc())   # thêm dòng này để debug dễ hơn
                        # continue

            #         # ── Scalar logs (mỗi round) ──────────────────────────────
            #         if args.use_wandb:
            #             forgetting_pct = forgetting * 100 if forgetting == forgetting else float('nan')


            #             # ① Đặt ở đầu hàm measure_follow_training, NGOÀI vòng lặp
            #             records_3d = []

            #             # ② Trong vòng lặp, thay vì plot ngay, chỉ append:
            #             records_3d.append({
            #                 "global_round": round_global,      # int
            #                 "old_task": eps_on_old_data,       # int (task id cũ)
            #                 "acc_old_curr": acc_curr_on_old,   # float
            #             })




            #             scalar_log.update({
            #                 f"block{block_idx}/task{task}/cosine_similarity":  cos_sim.mean().item(),
            #                 f"block{block_idx}/task{task}/sigma_on_curr_data": sigma_on_curr_data,
            #                 f"block{block_idx}/task{task}/eps_on_curr_data":   eps_on_curr_data,
            #                 f"block{block_idx}/task{task}/cka_on_curr_data":   cka_on_curr_data,
            #                 f"block{block_idx}/task{task}/sigma_on_old_data":  sigma_on_old_data,
            #                 f"block{block_idx}/task{task}/eps_on_old_data":    eps_on_old_data,
            #                 f"block{block_idx}/task{task}/cka_on_old_data":    cka_on_old_data,
            #                 f"block{block_idx}/task{task}/linear_cka":         linear_cka,
            #                 f"block{block_idx}/task{task}/nl_cka":             nl_cka,
            #                 f"block{block_idx}/task{task}/kernel_cka":         kernel_cka,
            #                 f"block{block_idx}/task{task}/align20":            align_score_on_curr_data[20],
            #                 f"block{block_idx}/task{task}/align100":           align_score_on_curr_data[100],
            #                 f"block{block_idx}/task{task}/align150":           align_score_on_curr_data[150],
            #                 f"block{block_idx}/task{task}/ratio_feature":      ratio_feat,
            #                 f"block{block_idx}/task{task}/acc_curr":           acc_curr_on_curr * 100,
            #                 f"block{block_idx}/task{task}/acc_old":            acc_curr_on_old * 100 if has_old else float('nan'),
            #                 f"block{block_idx}/task{task}/forgetting":         forgetting_pct,
            #                 f"block{block_idx}/task{task}/neuron_curr_min":    curr_min,
            #                 f"block{block_idx}/task{task}/neuron_curr_max":    curr_max,
            #                 f"block{block_idx}/task{task}/neuron_prev_min":     prev_min,
            #                 f"block{block_idx}/task{task}/neuron_prev_max":     prev_max,
            #                 f"block{block_idx}/task{task}/overlap@20":    overlap,
            #                 f"block{block_idx}/task{task}/cosine_neuron":      cosine_neuron,     
            #             }, step=round_global)

            #             double_log.update({
            #                 f"double/block{block_idx}/task{task}/sigma/curr": sigma_on_curr_data,
            #                 f"double/block{block_idx}/task{task}/sigma/old":  sigma_on_old_data if has_old else None,
            #                 f"double/block{block_idx}/task{task}/eps/curr":   eps_on_curr_data,
            #                 f"double/block{block_idx}/task{task}/eps/old":    eps_on_old_data   if has_old else None,
            #                 f"double/block{block_idx}/task{task}/cka/curr":   cka_on_curr_data,
            #                 f"double/block{block_idx}/task{task}/cka/old":    cka_on_old_data   if has_old else None,
            #                 f"block{block_idx}/task{task}/curr/neuron_curr_min": curr_min,
            #                 f"block{block_idx}/task{task}/curr/neuron_curr_max": curr_max,
            #                 f"block{block_idx}/task{task}/prev/neuron_prev_min": prev_min if has_old else None,
            #                 f"block{block_idx}/task{task}/prev/neuron_prev_max": prev_max if has_old else None,
            #             }, step=round_global)

            #             # ── Tích lũy data vào bảng (chưa log scatter) ───────
            #             acc_curr_pct = acc_curr_on_curr * 100

            #             if args.use_wandb:
            #                 bt = block_tables[block_idx]   # lấy bảng của đúng block
            #                 acc_curr_pct   = acc_curr_on_curr * 100
            #                 forgetting_pct = forgetting * 100 if forgetting == forgetting else float('nan')

            #                 bt["table_eps"].add_data(              eps_on_curr_data,              acc_curr_pct,   f"task_{task}")
            #                 bt["table_sigma"].add_data(            sigma_on_curr_data,            acc_curr_pct,   f"task_{task}")
            #                 bt["table_cka"].add_data(              cka_on_curr_data,              acc_curr_pct,   f"task_{task}")
            #                 bt["table_align"].add_data(            align_score_on_curr_data[150], acc_curr_pct,   f"task_{task}")
            #                 bt["table_cosine"].add_data(           cos_sim.mean().item(),         acc_curr_pct,   f"task_{task}")
            #                 bt["table_eps_forgetting"].add_data(   eps_on_curr_data,              forgetting_pct, f"task_{task}")
            #                 bt["table_sigma_forgetting"].add_data( sigma_on_curr_data,            forgetting_pct, f"task_{task}")
            #                 bt["table_cka_forgetting"].add_data(   cka_on_curr_data,              forgetting_pct, f"task_{task}")
            #                 bt["table_align_forgetting"].add_data( align_score_on_curr_data[150], forgetting_pct, f"task_{task}")
            #                 bt["table_cosine_forgetting"].add_data(cos_sim.mean().item(),         forgetting_pct, f"task_{task}")

            #                 if has_old:
            #                     acc_old_pct = acc_curr_on_old * 100
            #                     bt["table_eps_old"].add_data(           eps_on_old_data,   acc_old_pct,    f"task_{task}")
            #                     bt["table_sigma_old"].add_data(         sigma_on_old_data, acc_old_pct,    f"task_{task}")
            #                     bt["table_cka_old"].add_data(           cka_on_old_data,   acc_old_pct,    f"task_{task}")
            #                     bt["table_eps_forgetting_old"].add_data(  eps_on_old_data,   forgetting_pct, f"task_{task}")
            #                     bt["table_sigma_forgetting_old"].add_data(sigma_on_old_data, forgetting_pct, f"task_{task}")
            #                     bt["table_cka_forgetting_old"].add_data(  cka_on_old_data,   forgetting_pct, f"task_{task}")
            #                     bt["table_old_curr_eps_acc"].add_data(eps_on_old_data,  acc_old_pct,    "old",  round_global)
            #                     bt["table_old_curr_eps_acc"].add_data(eps_on_curr_data, acc_curr_pct,   "curr", round_global)
            #                     bt["table_old_curr_eps_fgt"].add_data(eps_on_old_data,  forgetting_pct, "old",  round_global)
            #                     bt["table_old_curr_eps_fgt"].add_data(eps_on_curr_data, forgetting_pct, "curr", round_global)
            #                     bt["table_kernel_old"].add_data(kernel_cka, acc_old_pct, f"task_{task}")
            #                     bt["table_nl_cka_old"].add_data(nl_cka, acc_old_pct, f"task_{task}")
            #         if args.use_wandb and scalar_log:
            #             #print(f"DEbug Round{round_global} Round_idex = {round_idx} scalar_log_size{scalar_log.__sizeof__} double_log{double_log.__sizeof__}")
            #             wandb.log({**scalar_log,**double_log,"round_global": round_global})

            # # ================================================================
            # # SAU KHI XONG 25 ROUND → log scatter 1 lần cho cả task
            # # ================================================================
            # # ── SAU vòng for task — log scatter tất cả block ────────────────────
            # if args.use_wandb:
            #     for block_idx in range(num_blocks):
            #         bt = block_tables[block_idx]
            #         wandb.log({
            #             "round_global": round_global,
            #             # ===== CURR DATA =====
            #             f"Custom/Block{block_idx}/Tasks{task}/Epsilon_vs_Accuracy_curr": wandb.plot.scatter(
            #                 bt["table_eps"],   "epsilon",  "accuracy",
            #                 title=f"[Curr] Epsilon vs Accuracy | Block{block_idx} — Task{task}"
            #             ),
            #             f"Custom/Block{block_idx}/Tasks{task}/Sigma_vs_Accuracy_curr": wandb.plot.scatter(
            #                 bt["table_sigma"], "sigma",    "accuracy",
            #                 title=f"[Curr] Sigma vs Accuracy | Block{block_idx} — Task{task}"
            #             ),
            #             f"Custom/Block{block_idx}/Tasks{task}/CKA_vs_Accuracy_curr": wandb.plot.scatter(
            #                 bt["table_cka"],   "cka",      "accuracy",
            #                 title=f"[Curr] CKA vs Accuracy | Block{block_idx} — Task{task}"
            #             ),
            #             f"Custom/Block{block_idx}/Tasks{task}/Align150_vs_Accuracy_curr": wandb.plot.scatter(
            #                 bt["table_align"], "align150", "accuracy",
            #                 title=f"[Curr] Align150 vs Accuracy | Block{block_idx} — Task{task}"
            #             ),
            #             f"Custom/Block{block_idx}/Tasks{task}/Cosine_vs_Accuracy_curr": wandb.plot.scatter(
            #                 bt["table_cosine"],"cosine",   "accuracy",
            #                 title=f"[Curr] Cosine vs Accuracy | Block{block_idx} — Task{task}"
            #             ),
            #             f"Custom/Block{block_idx}/Tasks{task}/Epsilon_vs_Forgetting_curr": wandb.plot.scatter(
            #                 bt["table_eps_forgetting"],    "epsilon",  "forgetting",
            #                 title=f"[Curr] Epsilon vs Forgetting | Block{block_idx} — Task{task}"
            #             ),
            #             f"Custom/Block{block_idx}/Tasks{task}/Sigma_vs_Forgetting_curr": wandb.plot.scatter(
            #                 bt["table_sigma_forgetting"],  "sigma",    "forgetting",
            #                 title=f"[Curr] Sigma vs Forgetting | Block{block_idx} — Task{task}"
            #             ),
            #             f"Custom/Block{block_idx}/Tasks{task}/CKA_vs_Forgetting_curr": wandb.plot.scatter(
            #                 bt["table_cka_forgetting"],    "cka",      "forgetting",
            #                 title=f"[Curr] CKA vs Forgetting | Block{block_idx} — Task{task}"
            #             ),
            #             f"Custom/Block{block_idx}/Tasks{task}/Align150_vs_Forgetting_curr": wandb.plot.scatter(
            #                 bt["table_align_forgetting"],  "align150", "forgetting",
            #                 title=f"[Curr] Align150 vs Forgetting | Block{block_idx} — Task{task}"
            #             ),
            #             f"Custom/Block{block_idx}/Tasks{task}/Cosine_vs_Forgetting_curr": wandb.plot.scatter(
            #                 bt["table_cosine_forgetting"], "cosine",   "forgetting",
            #                 title=f"[Curr] Cosine vs Forgetting | Block{block_idx} — Task{task}"
            #             ),

            #             # ===== OLD DATA =====
            #             f"Custom/Block{block_idx}/Tasks{task}/Epsilon_vs_Accuracy_old": wandb.plot.scatter(
            #                 bt["table_eps_old"],              "epsilon", "accuracy",
            #                 title=f"[Old] Epsilon vs Accuracy | Block{block_idx} — Task{task}"
            #             ),
            #             f"Custom/Block{block_idx}/Tasks{task}/Sigma_vs_Accuracy_old": wandb.plot.scatter(
            #                 bt["table_sigma_old"],            "sigma",   "accuracy",
            #                 title=f"[Old] Sigma vs Accuracy | Block{block_idx} — Task{task}"
            #             ),
            #             f"Custom/Block{block_idx}/Tasks{task}/CKA_vs_Accuracy_old": wandb.plot.scatter(
            #                 bt["table_cka_old"],              "cka",     "accuracy",
            #                 title=f"[Old] CKA vs Accuracy | Block{block_idx} — Task{task}"
            #             ),
            #             f"Custom/Block{block_idx}/Tasks{task}/Kernel_vs_Accuracy_old": wandb.plot.scatter(
            #                 bt["table_kernel_old"],           "kernel",  "accuracy",
            #                 title=f"[Old] Kernel vs Accuracy | Block{block_idx} — Task{task}"
            #             ),
            #             f"Custom/Block{block_idx}/Tasks{task}/NL_CKA_vs_Accuracy_old": wandb.plot.scatter(
            #                 bt["table_nl_cka_old"],           "nl_cka",  "accuracy",
            #                 title=f"[Old] NL_CKA vs Accuracy | Block{block_idx} — Task{task}"
            #             ),
            #             f"Custom/Block{block_idx}/Tasks{task}/Epsilon_vs_Forgetting_old": wandb.plot.scatter(
            #                 bt["table_eps_forgetting_old"],   "epsilon", "forgetting",
            #                 title=f"[Old] Epsilon vs Forgetting | Block{block_idx} — Task{task}"
            #             ),
            #             f"Custom/Block{block_idx}/Tasks{task}/Sigma_vs_Forgetting_old": wandb.plot.scatter(
            #                 bt["table_sigma_forgetting_old"], "sigma",   "forgetting",
            #                 title=f"[Old] Sigma vs Forgetting | Block{block_idx} — Task{task}"
            #             ),
            #             f"Custom/Block{block_idx}/Tasks{task}/CKA_vs_Forgetting_old": wandb.plot.scatter(
            #                 bt["table_cka_forgetting_old"],   "cka",     "forgetting",
            #                 title=f"[Old] CKA vs Forgetting | Block{block_idx} — Task{task}"
            #             ),
            #         })
                # ── W&B Logging ─────────────────────────────────────────────────────────
            if args.use_wandb:
                step = round_global  # dùng round_global làm x-axis xuyên suốt

                # # ── 1. Scalar tổng quan ──────────────────────────────────────────────
                wandb_log = {
                    # f"overview/acc_curr_task{task}":        acc_curr_on_curr * 100,
                    # f"overview/FM":                          FM * 100 if FM == FM else None,
                    # f"overview/BWT":                         bwt if bwt == bwt else None,
                    # f"overview/round_idx":                    round_idx,
                    f"overview/global_round":                  round_global,
                    f"overview/eps_curr":                    eps_on_curr_data,
                    f"overview/cka_curr":                    cka_on_curr_data,
                    f"overview/align10_curr":              align_score_on_curr_data[10],
                    f"overview/align20_curr":              align_score_on_curr_data[20],
                }

                # # acc trên từng old task
                # for old_task in range(task):
                #     acc_on_old = acc_matrix[-1].get(old_task, float('nan'))
                #     wandb_log[f"acc_per_task/task{old_task}_acc_at_task{task}"] = acc_on_old * 100

                # # forgetting per task
                # for old_task, f_val in per_task_forgetting.items():
                #     wandb_log[f"forgetting_per_task/forget_task{old_task}_at_task{task}"] = f_val * 100

                # # max past acc per task (baseline để so sánh)
                # for old_task, max_val in per_task_max_acc.items():
                #     wandb_log[f"max_past_acc/task{old_task}"] = max_val * 100

                wandb_log = {k: v for k, v in wandb_log.items() if v is not None}
                wandb.log(wandb_log, step=step)

            # for s in scatters.values():
            #     s.close()
            logger.info(f'  └── Task {task} done')

    logger.info(f'\n✅  Hoàn thành! CSV → {output_file}')


# ─────────────────────────────────────────────────────────────────────────────
# Helper: freeze fc, fine-tune chỉ backbone trên train data task tprime,
#         rồi đánh giá trên test data task t
# ─────────────────────────────────────────────────────────────────────────────
def fine_tune_backbone_frozen_head_and_eval(
    ckpt_path: str,
    client_id: int,
    train_task: int,       # task dùng để train (tprime)
    eval_task: int,        # task dùng để eval  (t)
    class_order: np.ndarray,
    args,
    lr: float = 1e-3,
    epochs: int = 10,
    patience: int = 3,     # early stopping: dừng nếu loss không giảm sau n epoch
) -> tuple:                # (model_finetuned, acc_after)
    """
    1. Load model từ ckpt_path (backbone + head gốc của task tprime).
    2. Freeze hoàn toàn fc (head) — chỉ backbone được update.
    3. Fine-tune backbone trên train data task tprime cho tới khi hội tụ
       (early stopping theo train loss).
    4. Đánh giá trên test data task t với mask_classes(task_index=eval_task).
    Trả về (model đã fine-tune, acc_after).
    """
    # ── 1. Load model giữ nguyên head ────────────────────────────────────
    model = load_model_with_head(ckpt_path, num_classes=args.classes)
    model.to(DEVICE)

    # ── 2. Freeze fc, chỉ backbone tham gia grad ─────────────────────────
    for param in model.fc.parameters():
        param.requires_grad = False

    backbone_params = [p for n, p in model.named_parameters()
                       if not n.startswith('fc.') and p.requires_grad]
    logger.info(
        f'      [freeze-head] trainable params: '
        f'{sum(p.numel() for p in backbone_params):,} '
        f'(fc frozen: {sum(p.numel() for p in model.fc.parameters()):,})'
    )

    # ── 3. Data ───────────────────────────────────────────────────────────
    train_data = read_client_data_FCL_cifar10(
        client_id, task=train_task,
        classes_per_task=args.cpt,
        count_labels=False, train=True,
    )
    test_data_eval = read_client_data_FCL_cifar10(
        client_id, task=eval_task,
        classes_per_task=args.cpt,
        count_labels=False, train=False,
    )
    train_loader = _make_loader(train_data, batch_size=128)
    test_loader  = _make_loader(test_data_eval, batch_size=256)

    # ── 4. Optimizer (chỉ backbone params) ───────────────────────────────
    optimizer = torch.optim.SGD(
        backbone_params, lr=lr, momentum=0.9, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    criterion = nn.CrossEntropyLoss()

    # ── 5. Fine-tune với early stopping theo train loss ───────────────────
    best_loss   = float('inf')
    no_improve  = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        avg_loss = running_loss / len(train_loader)

        logger.info(
            f'      [fine-tune backbone] epoch {epoch+1}/{epochs} '
            f'loss={avg_loss:.4f}  best={best_loss:.4f}  no_improve={no_improve}'
        )

        # early stopping
        if avg_loss < best_loss - 1e-4:
            best_loss  = avg_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(
                    f'      [early stop] epoch {epoch+1} — '
                    f'loss không giảm sau {patience} epoch liên tiếp'
                )
                break

    # ── 6. Eval trên test data task t (eval_task) ─────────────────────────
    model.eval()
    acc_after = test_metrics(
        model, test_loader,
        class_order=class_order,
        task_index=eval_task,
    )
    return model, acc_after


# ─────────────────────────────────────────────────────────────────────────────
# Hàm đo chính: cross-task + retrain backbone (frozen head)
# ─────────────────────────────────────────────────────────────────────────────
def measure_cross_task_retrain_backbone(args):
    """
    Logic giống measure_all_representation_drift:
      - Duyệt qua tất cả task-pairs (t, tprime), tất cả round_idx 0..24
      - Với mỗi cặp:
          * Load model_tprime tại round_idx (giữ head gốc)
          * Đo acc_before = acc của model_tprime gốc trên test data task t
          * Freeze head, fine-tune backbone trên train data task tprime
            tới khi hội tụ (early stopping)
          * Đo acc_after trên test data task t
          * acc_gap = acc_after - acc_before
      - Ghi CSV + wandb
    """
    root = '/kaggle/working' if args.kaggle else './outputs'
    output_file = (
        f'{root}/cross_task_retrain_backbone'
        f'-{args.partition_options}-{args.backbone}.csv'
    )

    header = (
        'client,t,tprime,round,'
        'acc_before,acc_after,acc_gap\n'
    )
    if not os.path.isfile(output_file) or os.path.getsize(output_file) == 0:
        with open(output_file, 'w') as f:
            f.write(header)

    task_pairs = list(itertools.combinations(range(args.num_tasks), 2))

    for client_id in range(args.num_clients):
        logger.info('=' * 60)
        logger.info(f'  CLIENT {client_id:>2} / {args.num_clients - 1}'
                    f'   ({len(task_pairs)} task-pairs × 25 rounds)')
        logger.info('=' * 60)
        client_class_order = all_class_orders[client_id][:10]

        for (t, tprime) in task_pairs:
            logger.info(f'  ┌── Task pair (t={t}, tprime={tprime})')

            # load test data task t một lần cho cả 25 round
            test_data_t = read_client_data_FCL_cifar10(
                client_id, task=t,
                classes_per_task=args.cpt,
                count_labels=False, train=False,
            )
            loader_t = _make_loader(test_data_t)

            for round_idx in range(25):
                ckpt_tp = get_model_path(args.saving_dir, client_id, tprime, round_idx)
                if not os.path.isfile(ckpt_tp):
                    logger.error(f'  │  [MISSING] {ckpt_tp}')
                    continue

                try:
                    # ── acc_before: model_tprime gốc đánh giá trên data task t ──
                    model_tp_orig = load_model_with_head(ckpt_tp, num_classes=args.classes)
                    acc_before = test_metrics(
                        model_tp_orig, loader_t,
                        class_order=client_class_order,
                        task_index=t,
                    )
                    del model_tp_orig   # giải phóng VRAM trước khi fine-tune
                    torch.cuda.empty_cache()

                    logger.info(
                        f'  │  round={round_idx:>2} | '
                        f'acc_before(task {t}) = {acc_before*100:.2f}%'
                    )

                    # ── fine-tune backbone (freeze head), eval trên task t ──────
                    _, acc_after = fine_tune_backbone_frozen_head_and_eval(
                        ckpt_path=ckpt_tp,
                        client_id=client_id,
                        train_task=tprime,
                        eval_task=t,
                        class_order=client_class_order,
                        args=args,
                        lr=1e-3,
                        epochs=10,
                        patience=3,
                    )

                    acc_gap = acc_after - acc_before
                    logger.info(
                        f'  │  round={round_idx:>2} | '
                        f'acc_after={acc_after*100:.2f}%  '
                        f'gap={acc_gap*100:+.2f}%  '
                        f'({"↑ recover" if acc_gap > 0 else "↓ worse"})'
                    )

                    # ── CSV ────────────────────────────────────────────────────
                    line = (
                        f'{client_id},{t},{tprime},{round_idx},'
                        f'{acc_before:.6f},{acc_after:.6f},{acc_gap:.6f}\n'
                    )
                    with open(output_file, 'a') as f:
                        f.write(line)

                    # ── wandb ──────────────────────────────────────────────────
                    if args.use_wandb:
                        prefix = f'retrain/pair_{t}_{tprime}'
                        wandb.log({
                            'round':                    round_idx,
                            'client':                   client_id,
                            f'{prefix}/acc_before':     acc_before * 100,
                            f'{prefix}/acc_after':      acc_after  * 100,
                            f'{prefix}/acc_gap':        acc_gap    * 100,
                        })

                except Exception as e:
                    logger.error(
                        f'  │  [SKIP] client={client_id} pair=({t},{tprime}) '
                        f'round={round_idx} | {e}'
                    )
                    continue

            logger.info(f'  └── pair (t={t}, tprime={tprime}) done')

    logger.info(f'\n✅  Hoàn thành! CSV → {output_file}')
# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main(args):
    torch.manual_seed(args.seed)

    logger.info('-' * 60)
    logger.info(f'  Device             : {DEVICE}')
    logger.info(f'  partition_options  : {args.partition_options}')
    logger.info(f'  Backbone           : {args.backbone}')
    logger.info(f'  Clients            : {args.num_clients}')
    logger.info(f'  Tasks              : {args.num_tasks}')
    logger.info(f'  CPT                : {args.cpt}')
    logger.info(f'  Saving dir         : {args.saving_dir}')
    logger.info('-' * 60)

    if args.use_wandb:
        wandb.login(key="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")
        wandb.init(
            project="Representation Drift Measurement",
            entity="ducthu2003",
            config=vars(args),
            name=f"{args.model}_{args.backbone}_{args.method}_c{args.num_clients}_t{args.num_tasks}_cpt{args.cpt}_seed{args.seed}",
            group=f"{args.backbone}_{args.method}",
        )
        wandb.define_metric("round")
        wandb.define_metric("block*", step_metric="round")
        wandb.define_metric("scatter*", step_metric="round")
        
    if args.backbone == 'ResNet18':        
        #measure_all_representation_drift(args)
        
        if args.method == 'dynamic':
            measure_follow_training(args)
        elif args.method == 'cross_client':
            measure_all_drift_follow_task_client_pair(args)
        elif args.method == 'cross_task':
            measure_all_representation_drift(args)
        elif args.method == 'retrain_backbone_cross_task':
            measure_cross_task_retrain_backbone(args)
    else:
        raise ValueError(f'Backbone chưa hỗ trợ: {args.backbone}')

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Representation Drift Measurement')

    parser.add_argument('--saving_dir',        type=str,  default=r'D:\FCL\checkpoints_client_task')
    parser.add_argument('--cp_probe',          type=str,  default=r'C:\Thu\FCL\probes_torchvision')
    parser.add_argument('--partition_options', type=str,  default='hetero')
    parser.add_argument('--backbone',          type=str,  default='ResNet18')
    parser.add_argument('--num_clients',       type=int,  default=10)
    parser.add_argument('--num_tasks',         type=int,  default=5)
    parser.add_argument('--cpt',               type=int,  default=2)
    parser.add_argument('--seed',              type=int,  default=42)
    parser.add_argument('--classes',           type=int,  default=10)
    parser.add_argument('--use_wandb',         type=bool, default=False)
    parser.add_argument('--method',             type=str,  default='dynamic')
    parser.add_argument('--kaggle',             type=bool, default=False)
    parser.add_argument('--retrain_epochs',  type=int,   default=10)
    parser.add_argument('--retrain_lr',      type=float, default=1e-3)
    parser.add_argument('--retrain_patience',type=int,   default=3)
    parser.add_argument('--model',             type=str,  default='ALA')

    args = parser.parse_args()
    main(args)