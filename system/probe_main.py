import os
import itertools
import logging
from pprint import pprint
import argparse

import torch
from torchvision.models import resnet18
from wandb.util import np
from system.utilities_probe.utils import EarlyStoppingConfig
from system.utilities_probe.metrics import Accuracy, Loss
from system.utilities_probe.configs import TrainingConfig
from system.utilities_probe.evaluation import PredictionBasedEvaluator
from system.utilities_probe.trainer import ProbeEvaluator
from system.utilities_probe.utils import gpu_information_summary, set_seed
from system.task_data_loader.scenarios import TaskConfig, SimpleScenario
import wandb
from torch.utils.data import Dataset
from system.utils.data_utils import read_client_data_FCL_cifar10

logger = logging.getLogger(__name__)
BLOCK_NAMES = ["block0", "block1", "block2", "block3", "block4"]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

class RemappedDataset(Dataset):
    def __init__(self, dataset, client_id: int, task_id: int, classes_per_task: int = 2):
        self.dataset = dataset
        all_class_orders = np.load(
            './dataset/class_order/class_order_cifar10.npy',
            allow_pickle=True
        )
        client_order = all_class_orders[client_id][:10]
        start = task_id * classes_per_task
        task_classes = client_order[start: start + classes_per_task]
        self.label_map = {int(orig): new for new, orig in enumerate(task_classes)}
        print(f"[RemappedDataset] client={client_id} task={task_id} "
              f"classes={task_classes.tolist()} map={self.label_map}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, self.label_map[int(y)]


def load_model(path, device):
    model = resnet18(pretrained=False, num_classes=2)
    model.fc = torch.nn.Identity()
    state = torch.load(path, map_location=device)
    new_state = {}
    for k, v in state.items():
        if k.startswith("base."):
            new_state[k[len("base."):]] = v
        elif k.startswith("head."):
            pass
        else:
            new_state[k] = v
    model.load_state_dict(new_state, strict=False)
    return model.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Core: 1 probe = 1 task = 1 model  (follow design gốc)
# ─────────────────────────────────────────────────────────────────────────────

def probe_single_task(
    model,
    client_id: int,
    target_task_id: int,
    block_name: str,
    probe_caller: str,
    experiment_name: str,
    args,
) -> float:
    """
    Follow đúng design gốc của paper:
      - Scenario chỉ chứa đúng 1 task (target_task_id)
        → giống cl_task.choose_task(target_task) trong main_probe.py gốc
      - ProbeEvaluator + TrainingConfig + optimizer tạo mới hoàn toàn
      → Không có shared state, không có multi-head, không có optimizer leak
    """
    # Scenario 1 task duy nhất — đúng như choose_task() gốc
    train_ds = read_client_data_FCL_cifar10(
        client_id, task=target_task_id, classes_per_task=2, train=True)
    test_ds = read_client_data_FCL_cifar10(
        client_id, task=target_task_id, classes_per_task=2, train=False)
    train_ds = RemappedDataset(
        train_ds, client_id=client_id, task_id=target_task_id, classes_per_task=2)
    test_ds = RemappedDataset(
        test_ds, client_id=client_id, task_id=target_task_id, classes_per_task=2)

    scenario = SimpleScenario(tasks=[
        TaskConfig(
            train=train_ds,
            test=test_ds,
            id=str(target_task_id),
            nb_classes=2,
        )
    ])

    # Config + optimizer mới hoàn toàn mỗi lần gọi
    early_stop_cfg = EarlyStoppingConfig(
        model_name=experiment_name.replace("/", "_"),
        patience=args.patience,
        verbose=True,
        delta=0.001,
        directory="/kaggle/working/early_stopping_checkpoints",
    )
    cfg = TrainingConfig(
        prediction_evaluator=PredictionBasedEvaluator(
            metrics=[Accuracy(), Loss()],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        ),
        seed_value=args.seed_value,
        nb_epochs=args.epochs,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        logging_step=4000,
        early_stopping_config=early_stop_cfg,
        is_probe=True,
        save_progress=False,
        saving_dir=args.saving_dir,
        experiment_name=experiment_name,
        optimizer=None,  # → ProbeEvaluator tự tạo AdamW mới
        # optimizer=None (default) → ModelCoach tự tạo AdamW mới
    )

    probe_evaluator = ProbeEvaluator(
        blocks_to_prob=[block_name],
        data_stream=scenario,
        half_precision=True,
        training_configs=cfg,
    )

    set_seed(args.seed_value, n_gpu=1)

    results = probe_evaluator.probe(
        model=model,
        target_id_task=str(target_task_id),
        probe_caller=probe_caller,
    )

    return results[block_name][f"task_{target_task_id}"]


# ─────────────────────────────────────────────────────────────────────────────
# Main measurement loop
# ─────────────────────────────────────────────────────────────────────────────

def measure_probe_forgetting(args):

    def ckpt_path(client_id, task_id, round_idx):
        return os.path.join(
            args.saving_dir,
            f"client_{client_id}_task_{task_id}_round_{round_idx}.pt"
        )

    task_pairs = list(itertools.combinations(range(args.num_tasks), 2))
    CACHE_DIR = args.dir_probe_cache
    os.makedirs(CACHE_DIR, exist_ok=True)

    csv_rows = []

    for client_id in range(args.num_clients):
        logger.info("=" * 65)
        logger.info(f"  CLIENT {client_id}")
        logger.info("=" * 65)

        if args.use_wandb:
            wandb.init(
                project="Representation Drift Measurement",
                entity="ducthu2003",
                name=f"client{client_id}_linear_probe",
                group=f"linear_probe_client{client_id}",
                config={"client_id": client_id, "epochs": args.epochs},
                reinit=True,
            )

        # ── BASELINE ─────────────────────────────────────────────────────────
        # Probe task t trên model_t_round{num_rounds}
        # → đo "upper bound" accuracy khi backbone vừa học xong task t
        baseline_acc = {}   # [t][block_name] = float

        for t in range(args.num_tasks):
            path = ckpt_path(client_id, t, args.num_rounds)
            if not os.path.isfile(path):
                logger.error(f"  [MISSING baseline] {path}")
                continue

            model_t = load_model(path, args.device)  # model mới cho mỗi t
            baseline_acc[t] = {}

            for block_name in BLOCK_NAMES:
                cache_file = os.path.join(
                    CACHE_DIR,
                    f"baseline_c{client_id}_t{t}_{block_name}.pt"
                )

                if os.path.exists(cache_file):
                    acc = torch.load(cache_file)["acc"]
                    logger.info(
                        f"  [baseline CACHE] t={t} {block_name} acc={acc:.4f}")
                else:
                    acc = probe_single_task(
                        model=model_t,
                        client_id=client_id,
                        target_task_id=t,
                        block_name=block_name,
                        probe_caller=f"c{client_id}_t{t}_{block_name}_baseline",
                        experiment_name=f"baseline_c{client_id}_t{t}_{block_name}",
                        args=args,
                    )
                    torch.save({"acc": acc}, cache_file)
                    logger.info(
                        f"  [baseline TRAINED] t={t} {block_name} acc={acc:.4f}")

                baseline_acc[t][block_name] = acc

        # ── FORGETTING ───────────────────────────────────────────────────────
        # Probe task t trên model_tprime_round{r}
        # → đo "degraded" accuracy sau khi backbone đã học task tprime
        for (t, tprime) in task_pairs:
            if t not in baseline_acc:
                logger.error(f"  [SKIP] No baseline t={t}")
                continue

            logger.info(f"  Pair t={t} → tprime={tprime}")

            for round_idx in [5,10,11,12,13,14,15,20]:
                path = ckpt_path(client_id, tprime, round_idx)
                if not os.path.isfile(path):
                    logger.warning(f"  [MISSING] {path}")
                    continue

                # Model mới cho mỗi (tprime, round_idx)
                model_tprime = load_model(path, args.device)

                for block_name in BLOCK_NAMES:
                    acc_after = probe_single_task(
                        model=model_tprime,
                        client_id=client_id,
                        target_task_id=t,
                        block_name=block_name,
                        probe_caller=(
                            f"c{client_id}_t{t}_{block_name}"
                            f"_tp{tprime}_r{round_idx}"
                        ),
                        experiment_name=(
                            f"pair_c{client_id}_t{t}_tp{tprime}"
                            f"_r{round_idx}_{block_name}"
                        ),
                        args=args,
                    )

                    baseline  = baseline_acc[t][block_name]
                    forgetting = baseline - acc_after

                    logger.info(
                        f"    round={round_idx:02d} {block_name}"
                        f"  base={baseline:.4f}"
                        f"  after={acc_after:.4f}"
                        f"  forget={forgetting:.4f}"
                    )

                    csv_rows.append({
                        "client":             client_id,
                        "task_t":             t,
                        "task_tprime":        tprime,
                        "round":              round_idx,
                        "block_name":         block_name,
                        "block_acc_baseline": baseline,
                        "block_acc_tprime":   acc_after,
                        "forgetting":         forgetting,
                    })

                    if args.use_wandb:
                        wandb.log({
                            f"{block_name}/forgetting/pair{t}_{tprime}": forgetting,
                            f"{block_name}/acc_tprime/pair{t}_{tprime}": acc_after,
                            f"{block_name}/baseline/pair{t}_{tprime}":   baseline,
                            "round": round_idx,
                        })

        if args.use_wandb:
            wandb.finish()

    return csv_rows


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    n_gpu, device = gpu_information_summary()
    set_seed(args.seed_value, n_gpu=n_gpu)
    args.device = device

    rows = measure_probe_forgetting(args)

    import csv
    out_path = os.path.join(args.saving_dir, "probe_forgetting_results.csv")
    if rows:
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--saving_dir",      type=str,
                        default="C:/Thu/FCL/checkpoint/weightAVGClient0")
    parser.add_argument("--num_clients",     type=int, default=1)
    parser.add_argument("--num_tasks",       type=int, default=5)
    parser.add_argument("--num_rounds",      type=int, default=15)
    parser.add_argument("--epochs",          type=int, default=100)
    parser.add_argument("--batch_size",      type=int, default=128)
    parser.add_argument("--num_workers",     type=int, default=0)
    parser.add_argument("--lr",              type=float, default=0.001)
    parser.add_argument("--seed_value",      type=int, default=42)
    parser.add_argument("--use_wandb",       action="store_true")
    parser.add_argument("--dir_probe_cache", type=str, default="./probe_cache")
    parser.add_argument("--patience",        type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    pprint(vars(args))
    main(args)