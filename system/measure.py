"""
eval_old_task.py
Đánh giá model train tại task=1 (mọi round) trên data task=0, client=0.

Usage:
    python eval_old_task.py --saving_dir ./checkpoints --cpt 2 --num_rounds 25
"""

import argparse
import os

import torch
import torch.nn as nn

from system.utils.data_utils import *
from system.measure_gpu1 import _make_loader
from system.utils.resnet import CIFARResNet18  # ← đổi nếu path khác

# ─────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLIENT_ID   = 0
EVAL_TASK   = 1    # model được train ở task này
OLD_TASK    = 1    # đánh giá trên data của task này
NUM_CLASSES = 10
# ─────────────────────────────────────────────

def get_all_ckpts_for_task(saving_dir: str, client_id: int, task: int):
    """Tự scan folder, lấy tất cả file của client+task, sort theo round."""
    prefix = f"client_{client_id}_task_{task}_round_"
    files = []
    for fname in os.listdir(saving_dir):
        if fname.startswith(prefix) and fname.endswith(".pt"):
            round_idx = int(fname.replace(prefix, "").replace(".pt", ""))
            files.append((round_idx, os.path.join(saving_dir, fname)))
    return sorted(files, key=lambda x: x[0])  # sort theo round
def load_model(ckpt_path: str) -> nn.Module:
    model = CIFARResNet18(num_classes=NUM_CLASSES).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)

    # Hỗ trợ cả raw state-dict lẫn dict có key 'model' / 'state_dict'
    state = ckpt
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in ckpt:
                state = ckpt[key]
                break

    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader) -> float:
    correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total   += y.size(0)
    return correct / total * 100.0 if total > 0 else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saving_dir", type=str, required=True)
    parser.add_argument("--cpt",        type=int, default=2)
    parser.add_argument("--eval_task",  type=int, default=1,
                        help="Task của model cần eval (default=1)")
    parser.add_argument("--old_task",   type=int, default=1,
                        help="Task data để test forgetting (default=0)")
    args = parser.parse_args()

    # ── Load data old task ────────────────────────────────────────────────
    test_data_old = read_client_data_FCL_cifar10(
        CLIENT_ID,
        task=args.old_task,
        classes_per_task=args.cpt,
        count_labels=False,
        train=False,
    )
    loader_old = _make_loader(test_data_old)
    # ── DEBUG: kiểm tra model predict gì ─────────────────────────────────
    print("\n[DEBUG] Kiểm tra 1 batch đầu tiên:")
    model_debug = load_model(ckpt_list[0][1])
    for x, y in loader_old:
        x = x.to(DEVICE)
        logits = model_debug(x)
        preds  = logits.argmax(dim=1).cpu()
        print(f"  True labels : {y.unique().tolist()}")
        print(f"  Pred labels : {preds.unique().tolist()}")
        print(f"  Logits range: min={logits.min():.3f}  max={logits.max():.3f}")
        print(f"  Logits mean per class: {logits.mean(0).detach().cpu().tolist()}")
        break
    # ── Scan checkpoint ───────────────────────────────────────────────────
    ckpt_list = get_all_ckpts_for_task(args.saving_dir, CLIENT_ID, args.eval_task)

    if not ckpt_list:
        print(f"[ERROR] Không tìm thấy checkpoint nào cho task={args.eval_task} trong {args.saving_dir}")
        return

    print(f"\nTìm thấy {len(ckpt_list)} checkpoints cho task={args.eval_task}")
    print(f"Round range: {ckpt_list[0][0]} → {ckpt_list[-1][0]}")
    print(f"\n{'Round':>6}  {'Acc on task-{} (%)'.format(args.old_task):>20}  Ckpt")
    print("-" * 80)

    results = []
    for round_idx, ckpt_path in ckpt_list:
        model   = load_model(ckpt_path)
        acc_old = evaluate(model, loader_old)
        results.append((round_idx, acc_old))
        print(f"{round_idx:>6}  {acc_old:>19.2f}%  {ckpt_path}")

    # ── Tóm tắt ──────────────────────────────────────────────────────────
    if results:
        rounds, accs = zip(*results)
        best_round, best_acc = max(results, key=lambda x: x[1])
        print("-" * 80)
        print(f"  Best  → round {best_round:>3}  acc = {best_acc:.2f}%")
        print(f"  Mean  → {sum(accs)/len(accs):.2f}%")
        print(f"  Final → round {rounds[-1]:>3}  acc = {accs[-1]:.2f}%")

if __name__ == "__main__":
    main()