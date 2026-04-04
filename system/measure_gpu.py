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
from system.utils.CKA import TorchCKA
from system.utils.data_utils import *
from torch.utils.data import DataLoader
import sys
from system.measure_alignment import *
from system.utils.metric import *
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
#logger.info(f'🖥️  Running on: {DEVICE}' + (f' ({torch.cuda.get_device_name(0)})' if DEVICE.type == 'cuda' else ''))


# ─────────────────────────────────────────────────────────────────────────────
# Đường dẫn checkpoint
# ─────────────────────────────────────────────────────────────────────────────
def get_model_path(saving_dir: str, client_id: int, task: int, round: int) -> str:
    return os.path.join(saving_dir, f'client_{client_id}_task_{task}_round_{round}.pt')


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

def compute_feature_resnet18(_model, _model_task_index, _dataset, _target_layer_index: str, seed, args):
    """
    Trích xuất features trên GPU, trả về numpy array.
    """
    blocks = get_resnet18_blocks(_model)

    _model.eval()
    outputs = []

    # Dùng DataLoader để batch inference thay vì từng sample → nhanh hơn nhiều
    loader = DataLoader(_dataset, batch_size=256, shuffle=False,
                        num_workers=4, pin_memory=(DEVICE.type == 'cuda'))

    with torch.no_grad():
        for features, targets in tqdm(loader,
                                      desc=f'Feature M_{_model_task_index}^{_target_layer_index}',
                                      disable=True):
            features = features.to(DEVICE, non_blocking=True)  # (B, C, H, W)

            for block_name, operations in blocks.items():
                features = operations(features)
                if block_name == _target_layer_index:
                    break

            # Flatten → (B, D), đưa về CPU ngay để giải phóng VRAM
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
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            test_num += y.shape[0]

    return test_acc / test_num if test_num > 0 else 0.0, test_num

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

    #task_pairs = list(itertools.combinations(range(args.num_tasks), 2))
    task_pairs = [(0,3),(0,4)]
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
            for round in range(25):
                logger.info(f'  ┌── Task pair ({t}, {tprime}), round {round}')

                ckpt_t  = get_model_path(args.saving_dir, client_id, t, round)
                ckpt_tp = get_model_path(args.saving_dir, client_id, tprime, round)
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

                test_data_t = read_client_data_FCL_cifar10(
                    client_id, task=t,
                    classes_per_task=args.cpt,
                    count_labels=False, train=False
                )
                test_data_tprime = read_client_data_FCL_cifar10(
                    client_id, task=tprime,
                    classes_per_task=args.cpt,
                    count_labels=False, train=False
                )

                model_head_tp = load_model_with_head(ckpt_tp, num_classes=10)
                loader_tprime = DataLoader(test_data_tprime, batch_size=256, shuffle=False,
                                           num_workers=4, pin_memory=(DEVICE.type == 'cuda'))
                loader_t      = DataLoader(test_data_t,      batch_size=256, shuffle=False,
                                           num_workers=4, pin_memory=(DEVICE.type == 'cuda'))

                current_test_acc, _ = test_metrics(model_head_tp, loader_tprime)
                old_test_acc,     _ = test_metrics(model_head_tp, loader_t)

                for k in range(num_blocks):
                    target_layer = f'block{k}'
                    try:
                        feat_t  = compute_feature_resnet18(
                            model_t,      t,      test_data_t,      target_layer, args.seed, args)
                        feat_tp = compute_feature_resnet18(
                            model_tprime, tprime, test_data_tprime, target_layer, args.seed, args)

                        if k == 0:
                            width_t = width_tp = float('nan')
                        else:
                            width_t  = compute_width(model_t,      k - 1)
                            width_tp = compute_width(model_tprime, k - 1)

                        eta_min, eta_max, eta_min_n, eta_max_n = compute_eta(feat_t)
                        sigma  = compute_sigma(feat_t, feat_tp)
                        eps    = compute_eps(feat_t, feat_tp)
                        hsic, cka = compute_cka(feat_t, feat_tp)
                        for k in [5,10,20,50]:
                            align_score, _ = compute_alignment(feat_t, feat_tp,"mutual_knn", topk=k, precise=True)

                        done += 1
                        progress = f'[{done}/{total}]'

                        logger.info(
                            f'  │  {progress} {target_layer} | '
                            f'σ={sigma:.4f}  ε={eps:.4f}  CKA={cka:.4f} align={align_score:.4f} '
                            f'  Accuracy {tprime}: {current_test_acc*100:.4f}'
                            f'  Accuracy {t}: {old_test_acc*100:.4f} '
                            f'η_min={eta_min_n:.4f}  η_max={eta_max_n:.4f}  '
                            f'W_t={width_t:.4f}  W_t\'={width_tp:.4f}'
                            f'  HSIC={hsic:.4f}'
                        )

                        if args.use_wandb:
                            wandb.log({
                                'client':               client_id,
                                'block':                k,
                                't':                    t,
                                'tprime':               tprime,
                                'sigma':                sigma,
                                'eps':                  eps,
                                f'accuracy_{tprime}':   current_test_acc * 100,
                                f'accuracy_{t}':        old_test_acc * 100,
                                'eta_min_norm':         eta_min_n,
                                'eta_max_norm':         eta_max_n,
                                'width_t':              width_t,
                                'width_tprime':         width_tp,
                                'pair':                 f'({t},{tprime})',
                                'client_block':         f'c{client_id}_b{k}',
                            })

                        with open(output_file, 'a') as f:
                            row = [client_id, k, t, tprime,
                                   eta_min, eta_max, eta_min_n, eta_max_n,
                                   sigma, eps, width_t, width_tp]
                            f.write(','.join(map(str, row)) + '\n')

                    except Exception as e:
                        logger.error(
                            f'  │  [SKIP] client={client_id} {target_layer} '
                            f't={t} t\'={tprime} | {e}'
                        )
                        continue

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

    for task_id in range(args.num_tasks):
        logger.info('=' * 60)
        logger.info(f'  TASK {task_id:>2} / {args.num_tasks - 1}'
                    f'   ({len(client_pairs)} client-pairs × {num_blocks} blocks)')
        logger.info('=' * 60)
        args.task = task_id

        for (client, client_prime) in client_pairs:
            logger.info(f'  ┌── Client pair ({client}, {client_prime})')

            ckpt_client       = get_model_path(args.saving_dir, client,       task_id, round=0)
            ckpt_client_prime = get_model_path(args.saving_dir, client_prime, task_id, round=0)
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

            for k in range(num_blocks):
                target_layer = f'block{k}'
                try:
                    feat_c  = compute_feature_resnet18(
                        model_c,      task_id, test_data_c,      target_layer, args.seed, args)
                    feat_cp = compute_feature_resnet18(
                        model_cprime, task_id, test_data_cprime, target_layer, args.seed, args)

                    if k == 0:
                        width_c = width_cp = float('nan')
                    else:
                        width_c  = compute_width(model_c,      k - 1)
                        width_cp = compute_width(model_cprime, k - 1)

                    eta_min, eta_max, eta_min_n, eta_max_n = compute_eta(feat_c)
                    sigma  = compute_sigma(feat_c, feat_cp)
                    eps    = compute_eps(feat_c, feat_cp)
                    hsic, cka = compute_cka(feat_c, feat_cp)

                    done += 1
                    progress = f'[{done}/{total}]'
                    logger.info(
                        f' | {progress} {target_layer} | '
                        f'σ={sigma:.4f}  ε={eps:.4f}  CKA={cka:.4f} '
                        f'η_min={eta_min_n:.4f}  η_max={eta_max_n:.4f}  '
                        f'W_c={width_c:.4f}  W_c\'={width_cp:.4f}  HSIC={hsic:.4f}'
                    )

                    if args.use_wandb:
                        wandb.log({
                            'client':        client,
                            'block':         k,
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
        for round in range(25):
            measure_all_representation_drift(args)
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