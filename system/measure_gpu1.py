import os
import itertools
import logging
import argparse
import pickle

import torch.nn as nn
import numpy as np
import torch
import wandb
from tqdm import tqdm
from torchvision.models import resnet18
from torchvision.models.resnet import BasicBlock
from sklearn.linear_model import LinearRegression
from system.measure_alignment import compute_alignment, compute_alignment_from_arrays
from system.measure_alignment import compute_alignment
from system.utils.CKA import TorchCKA, hsic
from system.utils.data_utils import *
from torch.utils.data import DataLoader
import sys

# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
# ─────────────────────────────────────────────────────────────────────────────
# Các hàm đo
# ─────────────────────────────────────────────────────────────────────────────
def get_resnet18_blocks(_model):
    return {
        'block0': torch.nn.Sequential(_model.conv1, _model.bn1, _model.relu, _model.maxpool),
        'block1': _model.layer1,
        'block2': _model.layer2,
        'block3': _model.layer3,
        'block4': _model.layer4,
    }

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
    raw_sd = torch.load(ckpt_path, map_location='cpu')

    head_keys = [k for k in raw_sd.keys() if k.startswith('head.')]
    print(f'Head keys: {head_keys}')

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

def test_metrics(model, testloader):
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
            accuracy_matrix = (torch.argmax(output, dim=1) == y).cpu().numpy()
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            test_num += y.shape[0]

    return test_acc / test_num if test_num > 0 else 0.0

def compute_forgetting(model, accuracy_matrix, current_task: int, target_task: int) -> float:
    if target_task >= current_task:
        raise ValueError(f"target_task ({target_task}) phải < current_task ({current_task})")
    acc_values = [accuracy_matrix[t][target_task] for t in range(target_task, current_task)]
    max_acc = max(acc_values)
    current_acc = accuracy_matrix[current_task][target_task]
    return max_acc - current_acc

# ── Các hàm đo — chạy trên GPU tensor, fallback numpy cho LinearRegression ──

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
# ─────────────────────────────────────────────────────────────────────────────
# Hàm đo chính
# ─────────────────────────────────────────────────────────────────────────────
def measure_all_representation_drift(args):
    output_file = f'./outputs/representation_drift-{args.partition_options}-{args.backbone}.csv'

    if not os.path.isfile(output_file):
        with open(output_file, 'w') as f:
            f.write('client,block,t,tprime,'
                    'eta_min,eta_max,eta_min_norm,eta_max_norm,'
                    'sigma,eps,width_t,width_tprime\n')

    #task_pairs = [(0, 3), (0, 4)]
    task_pairs = list(itertools.combinations(range(args.num_tasks), 2))
    num_blocks = 5

    total = args.num_clients * len(task_pairs) * num_blocks
    done  = 0

    for client_id in range(args.num_clients):
        logger.info('=' * 60)
        logger.info(f'  CLIENT {client_id:>2} / {args.num_clients - 1}'
                    f'   ({len(task_pairs)} task-pairs × {num_blocks} blocks)')
        logger.info('=' * 60)
        args.client = client_id

        for (t, tprime) in task_pairs:
            scatters = {
                block_idx: ScatterLogger(f"final_results/block{block_idx}/taskpair_{t}_{tprime}")
                for block_idx in range(num_blocks)
            }

            # ── Tích lũy accuracy_matrix qua tất cả các round ──────────────
            # accuracy_matrix[round][task] = acc của model trained đến `tprime`
            # được test trên task `task` ở round đó
            # Dùng dict để lưu: {round: {task: acc}}
            acc_history = {}  # acc_history[round][task] = float

            for round_idx in range(25):
                logger.info(f'  ┌── Task pair ({t}, {tprime}), round {round_idx}')

                ckpt_t  = get_model_path(args.saving_dir, client_id, t,      round_idx)
                ckpt_tp = get_model_path(args.saving_dir, client_id, tprime, round_idx)

                skip = False
                for ckpt in [ckpt_t, ckpt_tp]:
                    if not os.path.isfile(ckpt):
                        logger.error(f'  [MISSING] {ckpt}')
                        skip = True
                if skip:
                    continue

                model_t      = load_resnet18_from_checkpoint(ckpt_t,  load_head=False)
                model_tprime = load_resnet18_from_checkpoint(ckpt_tp, load_head=False)
                logger.info(f'  │  model_t  ← {ckpt_t}')
                logger.info(f'  │  model_t\' ← {ckpt_tp}')

                test_data_t      = read_client_data_FCL_cifar10(
                    client_id, task=t,      classes_per_task=args.cpt, count_labels=False, train=False)
                test_data_tprime = read_client_data_FCL_cifar10(
                    client_id, task=tprime, classes_per_task=args.cpt, count_labels=False, train=False)

                model_head_tp = load_model_with_head(ckpt_tp, num_classes=args.classes)
                model_head_t = load_model_with_head(ckpt_t, num_classes=args.classes)
                loader_tprime = _make_loader(test_data_tprime)
                loader_t      = _make_loader(test_data_t)
                logits_tprime_list = []
                logits_t_list = []

                for x_tp, _ in loader_tprime:  # chỉ lấy input
                    x_tp = x_tp.to(DEVICE)
                    logits_tprime_list.append(model_head_tp(x_tp).detach().cpu())

                for x_t, _ in loader_t:  # chỉ lấy input
                    x_t = x_t.to(DEVICE)
                    logits_t_list.append(model_head_t(x_t).detach().cpu())

                logits_tprime = torch.cat(logits_tprime_list, dim=0)
                logits_t      = torch.cat(logits_t_list, dim=0)

                cos_sin = torch.nn.functional.cosine_similarity(logits_tprime, logits_t, dim=1)
                
                # BUG FIX: test_metrics trả về 1 giá trị, không phải tuple
                acc_t_on_head = test_metrics(model_head_t,loader_t)
                current_test_acc = test_metrics(model_head_tp, loader_tprime)  # acc trên task tprime
                old_test_acc     = test_metrics(model_head_tp, loader_t)       # acc trên task t (cũ hơn)
                acc_on_t_now = old_test_acc
                # Ghi lại accuracy của round này
                acc_history[round_idx] = {
                    t:      acc_t_on_head,  # acc của model_t trên task t (để tính forgetting)
                    tprime: current_test_acc,
                }

                # ── Tính forgetting ───────────────────────────────────────────
                # Forgetting của task t tại round này = max acc(t) ở các round trước - acc(t) hiện tại
                # Chỉ tính được khi đã có ít nhất 1 round trước đó
                if round_idx > 0:
                    past_accs_on_t = [
                        acc_history[r][t]
                        for r in range(round_idx)   # tất cả round trước round hiện tại
                        if r in acc_history and t in acc_history[r]
                    ]
                    if past_accs_on_t:
                        max_past_acc = max(past_accs_on_t)

                        forgetting = max_past_acc - old_test_acc
                    else:
                        forgetting = float('nan')
                else:
                    forgetting = float('nan')  # round 0: chưa có gì để quên

                for block_idx in range(num_blocks):  # BUG FIX: đổi tên k → block_idx tránh shadow
                    target_layer = f'block{block_idx}'
                    scatter = scatters[block_idx]
                    try:
                        feat_t  = compute_feature_resnet18(
                            model_t,      t,      test_data_t,      target_layer, args.seed, args)
                        feat_tp = compute_feature_resnet18(
                            model_tprime, tprime, test_data_tprime, target_layer, args.seed, args)

                        if block_idx == 0:
                            width_t = width_tp = float('nan')
                        else:
                            width_t  = compute_width(model_t,      block_idx - 1)
                            width_tp = compute_width(model_tprime, block_idx - 1)

                        eta_min, eta_max, eta_min_n, eta_max_n = compute_eta(feat_t)
                        sigma         = compute_sigma(feat_t, feat_tp)
                        eps           = compute_eps(feat_t, feat_tp)
                        hsic_val, cka = compute_cka(feat_t, feat_tp)
                        linear_cka = TorchCKA(device=DEVICE).linear_CKA(torch.from_numpy(feat_t).float().to(DEVICE), torch.from_numpy(feat_tp).float().to(DEVICE))
                        kernel_cka = TorchCKA(device=DEVICE).kernel_CKA(torch.from_numpy(feat_t).float().to(DEVICE), torch.from_numpy(feat_tp).float().to(DEVICE), sigma=None)
                        topk_list = [5, 10, 20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
                        align_score = {}
                        for k in topk_list:
                            align_score[k], _ = compute_alignment_from_arrays(
                                feat_t, feat_tp, "mutual_knn", topk=k, precise=True)

                        done += 1
                        progress = f'[{done}/{total}]'

                        logger.info(
                            f'  │  {progress} {target_layer} | '
                            f'cosine_similarity={cos_sin.mean().item():.4f}  '
                            f'σ={sigma:.4f}  ε={eps:.4f}  CKA={cka:.4f} '
                            f'non-linear_CKA={kernel_cka - linear_cka:.4f} kernel_CKA={kernel_cka:.4f} '
                            f'dim_featuremax/featuremin = {eta_max/eta_min:.4f} '
                            f'align@20={align_score[20]:.4f} align@100={align_score[100]:.4f} '
                            f'ACC({tprime})={current_test_acc*100:.2f}%  '
                            f'ACC({t})={old_test_acc*100:.2f}%  '
                            f'ACCold_real = {acc_t_on_head*100:.2f}%  '
                            f'Forgetting={forgetting*100:.2f}%'
                            if not (forgetting != forgetting) else  # isnan check
                            f'  │  {progress} {target_layer} | '
                            f'cosine_similarity={cos_sin.mean().item():.4f}'
                            f'σ={sigma:.4f}  ε={eps:.4f}  CKA={cka:.4f} '
                            f'non-linear_CKA={kernel_cka - linear_cka:.4f} kernel_CKA={kernel_cka:.4f} '
                            f'dim_featuremax/featuremin = {eta_max/eta_min:.4f} '
                            f'align@20={align_score[20]:.4f} align@100={align_score[100]:.4f} '
                            f'ACC({tprime})={current_test_acc*100:.2f}%  '
                            f'ACC({t})={old_test_acc*100:.2f}%  '
                            f'ACCold_real = {acc_t_on_head*100:.2f}%  '
                            f'Forgetting=N/A (round 0)'
                        )
                        scatter.log_pair("round_vs_cosine_similarity", round_idx, cos_sin)
                        scatter.log_pair("sigma_vs_eps",       sigma,              eps)
                        scatter.log_pair("round_vs_eps",       round_idx,          eps)
                        scatter.log_pair("round_vs_sigma",     round_idx,          sigma)
                        scatter.log_pair("round_vs_cka",       round_idx,          cka)
                        scatter.log_pair("round_vs_linear_cka",  round_idx,          linear_cka)
                        scatter.log_pair("round_vs_non_linear_cka", round_idx,          kernel_cka - linear_cka)
                        scatter.log_pair("round_vs_kernel_cka", round_idx,          kernel_cka)
                        scatter.log_pair("round_vs_align20",   round_idx,          align_score[20])
                        scatter.log_pair("round_vs_align100",   round_idx,          align_score[100])
                        scatter.log_pair("round_vs_align150",   round_idx,          align_score[150])
                        scatter.log_pair("round_vs_ratio_feature", round_idx,          eta_max/eta_min if eta_min > 0 else float('nan'))
                        scatter.log_pair("round_vs_acc_tprime", round_idx,          current_test_acc*100)
                        scatter.log_pair("round_vs_accold",      round_idx,          old_test_acc*100)
                        scatter.log_pair("round_vs_weightnorm_t", round_idx,          width_t)
                        scatter.log_pair("round_vs_weightnorm_tprime", round_idx,          width_tp)
                        scatter.log_pair("cka_vs_eps",         cka,                eps)
                        scatter.log_pair("cka_vs_accold",      cka,                old_test_acc*100)
                        scatter.log_pair("cka_vs_align20",     cka,                align_score[20])
                        scatter.log_pair("cka_vs_align100",    cka,                align_score[100])
                        scatter.log_pair("cka_vs_align150",    cka,                align_score[150])
                        scatter.log_pair("cka_vs_sigma",       cka,                sigma)

                        scatter.log_pair("round_vs_accold",      round_idx,          old_test_acc*100)
                        scatter.log_pair("eps_vs_accold",      eps,                old_test_acc*100)
                        scatter.log_pair("cka_vs_accold",      cka,                old_test_acc*100)
                        scatter.log_pair("kernel_cka_vs_accold", kernel_cka,         old_test_acc*100)
                        scatter.log_pair("cosine_similarity_vs_accold", cos_sin.mean().item(), old_test_acc*100)
                        scatter.log_pair("align100_vs_accold", align_score[100],   old_test_acc*100)
                        scatter.log_pair("sigma_vs_accold",    sigma,              old_test_acc*100)
                        scatter.log_pair("weightnorm_t_vs_accold", width_t,          old_test_acc*100)
                        scatter.log_pair("weightnorm_tprime_vs_accold", width_tp,          old_test_acc*100)
                        scatter.log_pair("ratio_feature_vs_accold", eta_max/eta_min if eta_min > 0 else float('nan'), old_test_acc*100)
                        scatter.log_pair("align150_vs_accold", align_score[150],   old_test_acc*100)
                        scatter.log_pair("non_linear_cka_vs_accold", kernel_cka - linear_cka, old_test_acc*100)

                        # Thêm các scatter mới liên quan forgetting
                        scatter.log_pair("round_vs_forgetting", round_idx,          forgetting * 100 if forgetting == forgetting else float('nan'))
                        scatter.log_pair("eps_vs_forgetting",   eps,                forgetting * 100 if forgetting == forgetting else float('nan'))
                        scatter.log_pair("cka_vs_forgetting",   cka,                forgetting * 100 if forgetting == forgetting else float('nan'))
                        scatter.log_pair("kernel_cka_vs_forgetting", kernel_cka,         forgetting * 100 if forgetting == forgetting else float('nan'))
                        scatter.log_pair("cosine_similarity_vs_forgetting", cos_sin.mean().item(), forgetting * 100 if forgetting == forgetting else float('nan'))
                        scatter.log_pair("align100_vs_forgetting", align_score[100],   forgetting * 100 if forgetting == forgetting else float('nan'))
                        scatter.log_pair("sigma_vs_forgetting",    sigma,              forgetting * 100 if forgetting == forgetting else float('nan'))
                        scatter.log_pair("weightnorm_t_vs_forgetting", width_t,          forgetting * 100 if forgetting == forgetting else float('nan'))
                        scatter.log_pair("weightnorm_tprime_vs_forgetting", width_tp,          forgetting * 100 if forgetting == forgetting else float('nan'))
                        scatter.log_pair("ratio_feature_vs_forgetting", eta_max/eta_min if eta_min > 0 else float('nan'), forgetting * 100 if forgetting == forgetting else float('nan'))       
                        scatter.log_pair("align150_vs_forgetting", align_score[150],   forgetting * 100 if forgetting == forgetting else float('nan'))
                        scatter.log_pair("non_linear_cka_vs_forgetting", kernel_cka - linear_cka, forgetting * 100 if forgetting == forgetting else float('nan')) 

                        if args.use_wandb:
                            wandb.log({
                                'client':               client_id,
                                'block':                block_idx,
                                't':                    t,
                                'tprime':               tprime,
                                'round':                round_idx,
                                'sigma':                sigma,
                                'eps':                  eps,
                                'cka':                  float(cka),
                                f'accuracy_{tprime}':   current_test_acc * 100,
                                f'accuracy_{t}':        old_test_acc * 100,
                                'forgetting':           forgetting * 100 if forgetting == forgetting else None,
                                'eta_min_norm':         eta_min_n,
                                'eta_max_norm':         eta_max_n,
                                'width_t':              width_t,
                                'width_tprime':         width_tp,
                                'pair':                 f'({t},{tprime})',
                                'client_block':         f'c{client_id}_b{block_idx}',
                            })

                        with open(output_file, 'a') as f:
                            row = [client_id, block_idx, t, tprime,
                                   eta_min, eta_max, eta_min_n, eta_max_n,
                                   sigma, eps, width_t, width_tp]
                            f.write(','.join(map(str, row)) + '\n')

                    except Exception as e:
                        logger.error(
                            f'  │  [SKIP] client={client_id} {target_layer} '
                            f't={t} t\'={tprime} round={round_idx} | {e}'
                        )
                        continue

            scatter.close()
            logger.info(f'  └── Task pair ({t}, {tprime}) done')

    logger.info(f'\n✅  Hoàn thành! CSV → {output_file}')


def measure_all_drift_follow_task(args):
    output_file = f'./outputs/representation_drift-{args.partition_options}-{args.backbone}.csv'

    if not os.path.isfile(output_file):
        with open(output_file, 'w') as f:
            f.write('client,block,t,tprime,'
                    'eta_min,eta_max,eta_min_norm,eta_max_norm,'
                    'sigma,eps,width_t,width_tprime\n')

    client_pairs = list(itertools.combinations(range(10), 2))
    num_blocks = 5

    total = args.num_tasks * len(client_pairs) * num_blocks
    done  = 0
    #root = "/kaggle/working/final_results"   # Kaggle chuẩn
    root = "result_client_pair"  # local
    os.makedirs(root, exist_ok=True)
    print(f"Root directory for results: {root}")

    for task_id in range(args.num_tasks):
        logger.info('=' * 60)
        logger.info(f'  TASK {task_id:>2} / {args.num_tasks - 1}'
                    f'   ({len(client_pairs)} client-pairs × {num_blocks} blocks)')
        logger.info('=' * 60)
        args.task = task_id
        pair_scatters = {
            (client, client_prime): {
                block_idx: ScatterLogger(f"{root}/block{block_idx}/clientpair_{client}_{client_prime}")
                for block_idx in range(num_blocks)
            }
            for (client, client_prime) in client_pairs
        }
        for (client, client_prime) in client_pairs:
            scatters = pair_scatters[(client, client_prime)]
            logger.info(f'  ┌── Client pair ({client}, {client_prime})')
            ckpt_client       = get_model_path_no_round(args.saving_dir, client,       task_id)
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
            logger.info(f'  │  model_c  ← {ckpt_client}')
            logger.info(f'  │  model_c\' ← {ckpt_client_prime}')

            test_data_c = read_client_data_FCL_cifar10(
                client, task=task_id,
                classes_per_task=args.cpt,
                count_labels=False, train=False
            )
            test_data_cprime = read_client_data_FCL_cifar10(
                client_prime, task=task_id,
                classes_per_task=args.cpt,
                count_labels=False, train=False
            )
            model_head_cp = load_model_with_head(ckpt_client_prime, num_classes=args.classes)
            model_head_c = load_model_with_head(ckpt_client, num_classes=args.classes)
            loader_tprime = _make_loader(test_data_cprime)
            loader_t      = _make_loader(test_data_c)
            logits_cprime_list = []
            logits_c_list = []

            for x_tp, _ in loader_tprime:  # chỉ lấy input
                x_tp = x_tp.to(DEVICE)
                logits_cprime_list.append(model_head_cp(x_tp).detach().cpu())

            for x_t, _ in loader_t:  # chỉ lấy input
                x_t = x_t.to(DEVICE)
                logits_c_list.append(model_head_c(x_t).detach().cpu())

            logits_cprime = torch.cat(logits_cprime_list, dim=0)
            logits_c      = torch.cat(logits_c_list, dim=0)

            cos_sin = torch.nn.functional.cosine_similarity(logits_cprime, logits_c, dim=1)
            for num_block in range(num_blocks):
                target_layer = f'block{num_block}'
                scatter = scatters[num_block]
                try:
                    feat_c  = compute_feature_resnet18(
                        model_c,      task_id, test_data_c,      target_layer, args.seed, args)
                    feat_cp = compute_feature_resnet18(
                        model_cprime, task_id, test_data_cprime, target_layer, args.seed, args)

                    if num_block == 0:
                        width_c = width_cp = float('nan')
                    else:
                        width_c  = compute_width(model_c,      num_block - 1)
                        width_cp = compute_width(model_cprime, num_block - 1)

                    eta_min, eta_max, eta_min_n, eta_max_n = compute_eta(feat_c)
                    sigma  = compute_sigma(feat_c, feat_cp)
                    eps    = compute_eps(feat_c, feat_cp)
                    hsic, cka = compute_cka(feat_c, feat_cp)
                    
                    linear_cka = TorchCKA(device=DEVICE).linear_CKA(torch.from_numpy(feat_c).float().to(DEVICE), torch.from_numpy(feat_cp).float().to(DEVICE))
                    kernel_cka = TorchCKA(device=DEVICE).kernel_CKA(torch.from_numpy(feat_c).float().to(DEVICE), torch.from_numpy(feat_cp).float().to(DEVICE), sigma=None)
                    topk_list = [5, 10, 20, 50, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
                    align_score = {}
                    for list_ in topk_list:
                        align_score[list_], _ = compute_alignment_from_arrays(
                            feat_c, feat_cp, "mutual_knn", topk=list_, precise=True)
                    done += 1
                    progress = f'[{done}/{total}]'
                    logger.info(
                        f'  │  {progress} {target_layer} | '
                        f'cosine_similarity={cos_sin.mean().item():.4f}  '
                        f'σ={sigma:.4f}  ε={eps:.4f}  CKA={cka:.4f} '
                        f'non-linear_CKA={kernel_cka - linear_cka:.4f} kernel_CKA={kernel_cka:.4f} '
                        f'dim_featuremax/featuremin = {eta_max/eta_min:.4f} '
                        f'align@20={align_score[20]:.4f} align@100={align_score[100]:.4f} '
                        
                    )
                    scatter.log_pair("task_vs_cosine_similarity", task_id, cos_sin.mean().item())
                    scatter.log_pair("task_vs_sigma", task_id, sigma)
                    scatter.log_pair("task_vs_eps", task_id, eps)
                    scatter.log_pair("task_vs_cka", task_id, cka)
                    scatter.log_pair("cosine_similarity_vs_align100", cos_sin.mean().item(), align_score[100])
                    scatter.log_pair("non_linear_cka_vs_cosine_similarity", kernel_cka - linear_cka, cos_sin.mean().item())
                    scatter.log_pair("sigma_vs_eps",       sigma,              eps)

                    scatter.log_pair("cka_vs_eps",         cka,                eps)                        
                    scatter.log_pair("cka_vs_align20",     cka,                align_score[20])
                    scatter.log_pair("cka_vs_align100",    cka,                align_score[100])
                    scatter.log_pair("cka_vs_align150",    cka,                align_score[150])
                    scatter.log_pair("cka_vs_sigma",       cka,                sigma)




                    if args.use_wandb:
                        wandb.log({
                            'client':        client,
                            'block':         num_block,
                            't':             task_id,
                            'tprime':        client_prime,
                            'sigma':         sigma,
                            'eps':           eps,
                            'eta_min_norm':  eta_min_n,
                            'eta_max_norm':  eta_max_n,
                            'width_t':       width_c,
                            'width_tprime':  width_cp,
                            'pair':          f'({client},{client_prime})',
                            'client_block':  f'c{client}_b{k}',
                        })

                except Exception as e:
                    logger.error(
                        f'  │  [SKIP] client={client} {target_layer} '
                        f't={task_id} | {e}'
                    )
                    continue

            logger.info(f'  └── Client pair ({client}, {client_prime}) done')

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
            name=os.name,
        )

    if args.backbone == 'ResNet18':        
        #measure_all_representation_drift(args)
        measure_all_drift_follow_task(args)
    else:
        raise ValueError(f'Backbone chưa hỗ trợ: {args.backbone}')

    if args.use_wandb:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Representation Drift Measurement')

    parser.add_argument('--saving_dir',        type=str,  default=r'D:\FCL\checkpoints_client_task')
    parser.add_argument('--partition_options', type=str,  default='hetero')
    parser.add_argument('--backbone',          type=str,  default='ResNet18')
    parser.add_argument('--num_clients',       type=int,  default=10)
    parser.add_argument('--num_tasks',         type=int,  default=5)
    parser.add_argument('--cpt',               type=int,  default=2)
    parser.add_argument('--seed',              type=int,  default=42)
    parser.add_argument('--classes',           type=int,  default=10)
    parser.add_argument('--use_wandb',         type=bool, default=False)

    args = parser.parse_args()
    main(args)