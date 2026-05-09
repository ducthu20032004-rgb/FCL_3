import os
import itertools
import logging
from pprint import pprint
import argparse

import torch
from torchvision.models import resnet18

from system.main_probe import BLOCK_NAMES
from system.utilities_probe.metrics import Accuracy, Loss
from system.utilities_probe.configs import TrainingConfig
from system.utilities_probe.evaluation import PredictionBasedEvaluator
from system.utilities_probe.trainer import ProbeEvaluator
from system.utilities_probe.utils import gpu_information_summary, set_seed
from system.task_data_loader.scenarios import Scenario, TaskConfig ,SimpleScenario
import wandb
from torch.utils.data import DataLoader
from system.utils.data_utils import read_client_data_FCL_cifar10,read_client_data_FCL_cifar100
from torchvision.models import ResNet18_Weights
logger = logging.getLogger(__name__)
def load_model(path, device):
    # random_seed = 1609
    # torch.manual_seed(random_seed)
    model = resnet18(pretrained=False, num_classes=10).to(device)
    state = torch.load(path, map_location=device)

    # Remap keys: "base.xxx" -> "xxx", "head.xxx" -> "fc.xxx"
    new_state = {}
    for k, v in state.items():
        if k.startswith("base."):
            new_state[k[len("base."):]] = v
        elif k.startswith("head."):
            new_state["fc." + k[len("head."):]] = v
        else:
            new_state[k] = v

    model.load_state_dict(new_state)
    return model.to(device)
def measure_probe_forgetting(args):

    def ckpt(client_id, task_id, round_idx):
        return os.path.join(
            args.saving_dir,
            f"client_{client_id}_task_{task_id}_round_{round_idx}.pt"
        )

    def make_training_config(args, experiment_name):
        """Tạo TrainingConfig — fix bug seed_value khai báo 2 lần"""
        return TrainingConfig(
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
            early_stopping_config=None,
            is_probe=True,
            save_progress=True,
            saving_dir=args.saving_dir,
            experiment_name=experiment_name,
        )

    task_pairs = list(itertools.combinations(range(args.num_tasks), 2))
    csv_rows   = []

    for client_id in range(args.num_clients):
        logger.info("=" * 65)
        logger.info(f"  CLIENT {client_id} / {args.num_clients - 1}")
        logger.info("=" * 65)

        if args.use_wandb:
            wandb.init(
                project = "Representation Drift Measurement",
                entity  = "ducthu2003",
                name    = f"client{client_id}_linear_probe",
                group   = f"linear_probe_client{client_id}",
                config  = {
                    "client_id":        client_id,
                    "epochs_probe":     args.epochs,
                    "lr_probe":         args.lr,
                    "classes_per_task": 5,
                },
                reinit=True,
            )

        # Build Scenario chứa tất cả task của client
        tasks = []
        for task_id in range(args.num_tasks):
            train_ds = read_client_data_FCL_cifar10(client_id, task=task_id, classes_per_task=5, train=True)
            test_ds  = read_client_data_FCL_cifar10(client_id, task=task_id, classes_per_task=5, train=False)
            tasks.append(TaskConfig(train=train_ds, test=test_ds, id=str(task_id), nb_classes=5))
        cl_task = SimpleScenario(tasks=tasks)

        for (t, tprime) in task_pairs:
            logger.info(f"  Task pair  t={t} (cũ)  tprime={tprime} (mới)")

            # ── Bước 1: tính baseline acc dùng model_t ──────────────────────
            ckpt_t_path = ckpt(client_id, t, round_idx=24)
            if not os.path.isfile(ckpt_t_path):
                logger.error(f"  [MISSING baseline] {ckpt_t_path}")
                continue

            model_t = load_model(ckpt_t_path, args.device)

            baseline_acc_per_block = {}   # block_name -> float
            for block_idx, block_name in enumerate(BLOCK_NAMES):
                cfg = make_training_config(
                    args,
                    experiment_name=f"client{client_id}_t{t}_block{block_idx}_baseline",
                )
                probe_evaluator = ProbeEvaluator(
                    blocks_to_prob=[block_name],
                    data_stream=cl_task,
                    half_precision=True,
                    training_configs=cfg,
                    
                )
                results = probe_evaluator.probe(
                    model=model_t,
                    target_id_task=str(t),  # Chỉ probe trên task t để lấy baseline
                    probe_caller=f"client{client_id}_task{t}_block{block_idx}_baseline",
                )
                block_results = results[block_name]
                print("DEBUG block_results:", block_results) 
                baseline_acc_per_block[block_name] = block_results[f"task_{t}"]
                logger.info(f"    [baseline] block={block_name}  acc={baseline_acc_per_block[block_name]:.4f}")

            # ── Bước 2: qua từng round, probe model_tprime trên data task t ─
            for round_idx in range(args.num_rounds):
                ckpt_tprime_path = ckpt(client_id, tprime, round_idx)
                if not os.path.isfile(ckpt_tprime_path):
                    logger.warning(f"  [MISSING] {ckpt_tprime_path}, skip round {round_idx}")
                    continue

                model_tprime = load_model(ckpt_tprime_path, args.device)
                
                for block_idx, block_name in enumerate(BLOCK_NAMES):
                    cfg = make_training_config(
                        args,
                        experiment_name=f"client{client_id}_t{t}_tprime{tprime}_block{block_idx}_round{round_idx}",
                    )
                    probe_evaluator = ProbeEvaluator(
                        blocks_to_prob=[block_name],
                        data_stream=cl_task,
                        half_precision=True,
                        training_configs=cfg,
                          # Chỉ probe trên task t để lấy baseline
                    )
                    results = probe_evaluator.probe(
                        model=model_tprime,
                        target_id_task=str(t),  # Chỉ probe trên task t để so sánh với baseline
                        probe_caller=f"client{client_id}_task{t}_block{block_idx}_round{round_idx}",
                    )

                    block_results = results[block_name]
                    print("DEBUG block_results:", block_results) 
                    acc_tprime = block_results[f"task_{t}"]
                    forgetting = baseline_acc_per_block[block_name] - acc_tprime
                    
                    logger.info(
                        f"    round={round_idx:02d}  block={block_name}"
                        f"  block_acc_baseline={baseline_acc_per_block[block_name]:.4f}"
                        f"  block_acc_{tprime}={acc_tprime:.4f}"
                        f"  forgetting={forgetting:.4f}"
                    )

                    row = {
                        "client":         client_id,
                        "task_t":         t,
                        "task_tprime":    tprime,
                        "round":          round_idx,
                        "block_idx":      block_idx,
                        "block_name":     block_name,
                        "block_acc_baseline":   baseline_acc_per_block[block_name],
                        "block_acc_tprime":     acc_tprime,
                        "forgetting":     forgetting,
                    }
                    csv_rows.append(row)

                    if args.use_wandb:
                        wandb.log({
                            f"{block_name}/forgetting/t{t}_tprime{tprime}": forgetting,
                            f"{block_name}/acc_tprime/t{t}_tprime{tprime}": acc_tprime,
                            "round": round_idx,
                        })

        if args.use_wandb:
            wandb.finish()

    return csv_rows


def main(args):
    
    n_gpu, device = gpu_information_summary()
    set_seed(args.seed_value,  n_gpu=n_gpu)
    args.device = device
    logger.info(f"Using device: {device} with {n_gpu} GPU(s)")
    if args.use_wandb:
        wandb.init(project="FCL-probe", config=vars(args))
    rows = measure_probe_forgetting(args)

    # Lưu CSV kết quả
    import csv
    out_path = os.path.join(args.saving_dir, "probe_forgetting_results.csv")
    if rows:
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Saved results to {out_path}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--saving_dir", type=str,
                        default="C:/Thu/FCL/checkpoint/weightAVGClient0")

    parser.add_argument("--num_clients", type=int, default=1)
    parser.add_argument("--num_tasks", type=int, default=5)
    parser.add_argument("--num_rounds", type=int, default=25)
    parser.add_argument("--nb_classes", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--lr", type=float, default=0.001)

    parser.add_argument("--seed_value", type=int, default=42)

    parser.add_argument("--use_wandb", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    pprint(vars(args))

    main(args)