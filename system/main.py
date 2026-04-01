import os
import sys
import copy
import torch
import argparse
import time
import warnings
import numpy as np
import torchvision
import json
import wandb
from argparse import Namespace
from types import SimpleNamespace

from flcore.servers.serveravg import FedAvg
from flcore.servers.serverala import FedALA
from flcore.servers.serverdbe import FedDBE
from flcore.servers.serveras import FedAS
from flcore.servers.serverweit import FedWeIT
from flcore.servers.serveraffcl import FedAFFCL
from flcore.servers.servertarget import FedTARGET
from flcore.servers.serverl2p import FedL2P
from flcore.servers.serverPILORA import PILORA

from flcore.servers.serverLANDER import LANDERServer
from flcore.servers.serverGLFC import GLFCServer

from flcore.trainmodel.models import *

from flcore.trainmodel.AFFCL_models import AFFCLModel
from flcore.servers.serverstgm import FedSTGM
from flcore.servers.serverfcil import FedFCIL

from flcore.trainmodel.bilstm import *
from flcore.trainmodel.alexnet import *
from flcore.trainmodel.mobilenet_v2 import *
from flcore.trainmodel.transformer import *
from flcore.trainmodel.vit_prompt_l2p import *
from flcore.trainmodel.PILORA.VLT import *
from flcore.trainmodel.PILORA.VITLORA import vitlora

warnings.simplefilter("ignore")
torch.manual_seed(0)
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def run(args):

    if args.partition_options == "tuan":
        partition_name = "thu_partition"

        name=f"{args.dataset}_{args.model}_{args.algorithm}_{args.optimizer}_lr{args.local_learning_rate}_{partition_name}_{args.note}" if args.note\
                  else f"{args.dataset}_{args.model}_{args.algorithm}_{args.optimizer}_lr{args.local_learning_rate}_{partition_name}"

    elif args.partition_options == "hetero":
        partition_name = "hetero_partition"
        name = f"{args.dataset}_{args.model}_{args.algorithm}_{args.optimizer}_lr{args.local_learning_rate}_classpertask{args.cpt}_numtasks{args.num_tasks}_numclient{args.num_clients}_alpha{args.alpha}_taskdisoder{args.task_disorder}"
    
    if args.wandb:
        wandb.login(key="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")
        wandb.init(
            project="my-awesome-project",
            entity="ducthu2003",
            config=vars(args),
            name=name, 
        )
    print("Wandb run:", wandb.run)
    time_list = []
    model_str = args.model
    args.model_str = model_str

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.model
        if model_str == "CNN": # non-convex
            if "CIFAR100" in args.dataset:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            elif "CIFAR10" in args.dataset:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            elif "IMAGENET1k" in args.dataset:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            elif "IMAGENET1k224" in args.dataset:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=179776).to(args.device)
            else:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)

        elif model_str == "ResNet50":
            args.model = torchvision.models.resnet50(pretrained=False, num_classes=args.num_classes).to(args.device)
        elif model_str == "ResNet50-pretrained":
            weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
            args.model = torchvision.models.resnet50(weights=weights, num_classes=args.num_classes).to(args.device)
        elif model_str == "ResNet34":
            args.model = torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes).to(args.device)
        elif model_str == "ResNet18":
            args.model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes).to(args.device)
        elif model_str == "Swin_t":
            args.model = torchvision.models.swin_t(weights=None, num_classes=args.num_classes).to(args.device)
        elif model_str == "AFFCLModel":    
            args.model = AFFCLModel(args).to(args.device)
        elif model_str == "VitL2P":
            args.model = VitL2P(
                num_classes=args.num_classes,
                n_prompts=args.n_prompts,
                prompt_length=args.prompt_length,
                prompt_pool=args.prompt_pool,
                pool_size=args.pool_size).to(args.device)
        elif model_str == "VLT":
            args.model = VLT(modelname='vit_base_patch16_224_dino',
                num_classes=args.num_classes,
                pretrained=False,
                r = 4,
                lora_layer = [0]).to(args.device)
        else:
            raise NotImplementedError

        # select algorithm
        if args.algorithm == "FedAvg":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedAvg(args, i)

        elif args.algorithm == "FedALA":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedALA(args, i)

        elif args.algorithm == "FedDBE":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedDBE(args, i)

        elif args.algorithm == "FedWeIT":
            # args.model = None
            server = FedWeIT(args, i)

        elif args.algorithm == "PreciseFCL":
            # args.head = copy.deepcopy(args.model.classifier.fc_classifier)
            # args.model.classifier.fc_classifier = nn.Identity()
            # args.model.classifier = BaseHeadSplit_AFFCL(args.model.classifier, args.head)
            server = FedAFFCL(args, i)

        elif args.algorithm == 'FedAS':
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedAS(args, i)

        elif args.algorithm == "FedFCIL":
            server = FedFCIL(args, i)
            
        elif args.algorithm == "FedSTGM":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedSTGM(args, i)

        elif args.algorithm == "FedTARGET":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedTARGET(args, i)

        elif args.algorithm == "FedL2P":
            # args.head = copy.deepcopy(args.model.fc)
            # args.model.fc = nn.Identity()
            # args.model = BaseHeadSplit(args.model, args.head)
            server = FedL2P(args, i)

        elif args.algorithm == "GLFC":
            server = GLFCServer(args, i)
        elif args.algorithm == "LANDER":
            server = LANDERServer(args, i)

        elif args.algorithm == "PILORA":
            server = PILORA(args, i)

        else:
            raise NotImplementedError

        # ====== LOAD CKPT & EVAL-ONLY ======
        if getattr(args, "load_checkpoint", False):
            print(f"Loading checkpoint from {args.ckpt_path} for evaluation only...")
            start_round = server.load_checkpoint(args.ckpt_path)

            server.eval(task=server.current_task, glob_iter=start_round, flag="global")
            print("[DONE] Eval from checkpoint.")
            return
        # ====== LOAD CKPT & training ======
        if args.resume and args.ckpt_path:
            print(f"[CKPT] Resume training from: {args.ckpt_path}")
            start_round = server.load_checkpoint(args.ckpt_path)
            args.start_round = start_round    # cực quan trọng
            print(f"[RESUME] Continue from round {args.start_round}")
        else:
            args.start_round = 0
        try:
            from utils.partition_viz import visualize_and_print_partition
            visualize_and_print_partition(server, args, fig_dir="figures")
        except Exception as e:
            print(f"[WARN] Partition visualization failed: {e}")

        server.train()

        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    
    # Global average
    print("All done!")

    # server = FedAvg(args, args.times) đang comment lại vì khởi tạo server lần 2 khi retrain sẽ bị lỗi do thiếu args.start_round, trong khi server.train() đã được gọi ở trên và đã sử dụng args.start_round

    # ===== LOAD CKPT & EVAL ONLY =====
    if getattr(args, "load_checkpoint", False):
        start_round = server.load_checkpoint(args.ckpt_path)

        server.eval(task=server.current_task, glob_iter=start_round, flag="global")
        print("[DONE] Eval from checkpoint.")
        exit(0)
if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfp', type=str, default="./hparams/FedAvg.json", help='Configuration path for training')
    parser.add_argument('--note', type=str, default=None, help='Optional note to add to save name')
    parser.add_argument('--wandb', type=bool, default=False, help='Log on wandb')
    parser.add_argument('--offlog', type=bool, default=False, help='Save wandb logger')
    parser.add_argument('--log', type=bool, default=False, help='Print logger')
    parser.add_argument('--debug', type=bool, default=False, help='When use Debug, turn off forgetting')
    parser.add_argument('--cpt', type=int, default=2, help='Class per task')
    parser.add_argument('--nt', type=int, default=None, help='Num tasks')
    parser.add_argument('--seval', action='store_true', help='Log Spatio Gradient')
    parser.add_argument('--teval', action='store_true', help='Log Temporal Gradient')
    parser.add_argument('--pca_eval', action='store_true', help='Log PCA Gradient')
    parser.add_argument('--num_clients', type=int, default=10, help='Number of clients')
    parser.add_argument('--save_checkpoint', action='store_true', help='Save global model checkpoints')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--ckpt_path', type=str, default='', help='Checkpoint path')
    parser.add_argument('--out_folder', type=str, default='./results/', help='Output folder')

    parser.add_argument('--device_id', type=str, default='0', help='cuda device id')

    parser.add_argument('--partition_options',
        type=str,
        choices=["tuan", "hetero"],
        default="hetero",
        help="Data partition scheme: 'tuan' uses the repo's class-order slicing; 'hetero' are heterogeneous data partitioning."
    )

    # --- data partition knobs (task-level permutation) ---
    parser.add_argument('--alpha', type=float, default=0.3,
                        help='Dirichlet alpha for per-class split across clients (smaller = more skew).')
    parser.add_argument('--task_disorder', type=float, default=0.5,
                        help='Task-order disorder in [0,1]: 0 -> same as master; 1 -> random permutation.')
    parser.add_argument('--client_disorder', type=str, default=None,
                        help='Optional per-client disorder override as CSV or [list], e.g. "0,0.2,0.8". '
                            'Client 0 is forced to 0.0 (master) regardless.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for the partitioner.')


    # GLFC
    parser.add_argument("--glfc_T", type=float, default=2.0)
    parser.add_argument("--glfc_alpha", type=float, default=0.5)             # class-imbalance compensation exponent
    parser.add_argument("--glfc_distill", type=float, default=1.0)           # KD coeff
    parser.add_argument("--glfc_relation", type=float, default=1.0)          # relation KD coeff
    parser.add_argument("--glfc_mem_per_class", type=int, default=20)
    parser.add_argument("--use_memory", action="store_true", default=True)

    # LANDER
    parser.add_argument("--lander_T", type=float, default=2.0)
    parser.add_argument("--lander_kd", type=float, default=0.5)
    parser.add_argument("--lander_lambda_bound", type=float, default=1.0)
    parser.add_argument("--lander_radius", type=float, default=0.5)
    parser.add_argument("--lander_text_encoder", type=str, default="clip-ViT-B-32")
    parser.add_argument("--lander_text_template", type=str, default="a photo of a {}")

    # Shared / model side
    parser.add_argument("--feature_dim", type=int, default=512)

    args = parser.parse_args()
    # LƯU FLAGS TỪ CLI
    cli_resume = args.resume
    cli_ckpt_path = args.ckpt_path
    cli_save_ckpt = args.save_checkpoint
    with open(args.cfp, 'r') as f:
        cfdct = json.load(f)
    if args.note is not None:
        cfdct['note'] = args.note
    if args.nt is not None:
        cfdct['num_tasks'] = args.nt


    cfdct['nt'] = args.nt
    cfdct['wandb'] = args.wandb
    cfdct['offlog'] = args.offlog
    cfdct['log'] = args.log
    cfdct['debug'] = args.debug
    cfdct['cpt'] = args.cpt
    cfdct['seval'] = args.seval
    cfdct['teval'] = args.teval
    cfdct['pca_eval'] = args.pca_eval
    cfdct['partition_options'] = args.partition_options
    cfdct['device_id'] = args.device_id
    cfdct["num_clients"] = args.num_clients

    cfdct['out_folder'] = args.out_folder

    cfdct['alpha'] = args.alpha
    cfdct['task_disorder'] = args.task_disorder
    cfdct['client_disorder'] = args.client_disorder
    cfdct['seed'] = args.seed


    # print(args.seval)
    # print(args.teval)
    # print(args.pca_eval)

    if "tgm" not in cfdct:
        cfdct['tgm'] = True

    if "coreset" not in cfdct:
        cfdct['coreset'] = False

    args = Namespace(**cfdct)
    # GẮN LẠI FLAGS
    args.resume = cli_resume
    args.ckpt_path = cli_ckpt_path
    args.save_checkpoint = cli_save_ckpt
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"


    run(args)

