# system/flcore/servers/serverGLFC.py
import copy
from xmlrpc import client
import torch
import os
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
                # === Ghi CSV (thay thế / bổ sung wandb) ===
import csv, os
from system.flcore.servers.serverbase import Server
from system.flcore.clients.clientGLFC import clientGLFC
from torch.utils.data import DataLoader
from system.measure_gpu1 import get_resnet18_blocks, compute_eps, compute_cka, compute_alignment_from_arrays, DEVICE
from system.utils.data_utils import read_client_data_FCL_cifar100, read_client_data_FCL_imagenet1k
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
def compute_feature_resnet18_wrap(_model, _model_task_index, _dataset, _target_layer_index: str, seed, args):
    """
    Trích xuất features trên GPU, trả về numpy array (N, D).
    Chấp nhận Dataset chuẩn hoặc list of (x, y) tuples.
    """

    # ===== unwrap BaseHeadSplit =====
    backbone = _model.base if hasattr(_model, "base") else _model

    blocks = get_resnet18_blocks(backbone)

    backbone.eval()

    outputs = []

    loader = _make_loader(_dataset, batch_size=256)

    with torch.no_grad():
        for features, targets in tqdm(
            loader,
            desc=f'Feature M_{_model_task_index}^{_target_layer_index}',
            disable=True
        ):
            features = features.to(DEVICE, non_blocking=True)

            for block_name, operations in blocks.items():
                features = operations(features)

                if block_name == _target_layer_index:
                    break

            outputs.append(torch.flatten(features, 1).cpu())

    return torch.cat(outputs, dim=0).numpy()
class GLFCServer(Server):
    """
    GLFC server:
    - Keeps a pool of previous global models (proxy teacher); here we use the last best global as teacher.
    - Aggregates client prototypes (mean of normalized features) to broadcast if needed.
    """
    def __init__(self, args, times):
        super().__init__(args, times)
        # Select slow clients if your base provides this API
        try:
            self.set_slow_clients()
        except Exception:
            pass

        # Create clients
        self.set_clients(clientGLFC)

        self.best_global = None
        self.global_prototypes = None  # [C, D]

    def train(self):
        for round in range(self.global_rounds):
            # sample a subset
            self.selected_clients = self.select_clients()
            # broadcast current global model; also share teacher (prev best)
            teacher = self.best_global
            for c in self.selected_clients:
                try:
                    c.set_teacher(teacher)
                except Exception:
                    pass
                self.send_models(c)
            model_global_before = copy.deepcopy(self.global_model)
            # local updates
            self.receive_models()
            try:
                client._model_after_local = copy.deepcopy(client.model)
            except Exception:
                pass
            if i == 24:  
                try:
                    save_dir =  "/home/ghostm211/Thu/FCL_3/weight_feddbe/weight_client_round"  # Thay đổi đường dẫn theo nhu cầu
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = f"{save_dir}/client_{client.id}_task_{task}_round_{i}.pt"
                    
                    model_to_save = client.model
                    torch.save(model_to_save.state_dict(), save_path)
                    print(f"[SAVED] client={client.id} task={task} round={i} → {save_path}")
                except Exception as save_err:
                    print(f"[ERROR] Failed to save client {client.id} task {task} round {i}: {save_err}")
                    import traceback; traceback.print_exc()            
            # aggregate as FedAvg (Server base usually has this)
            self.aggregate_parameters()
            model_global_after = copy.deepcopy(self.global_model)
            if getattr(self.args, "measure_drift", True):
                print("Drift thanh cong")
                try:
                    # dict lưu kết quả per client per block
                    drift_results = {}  # client.id -> {block_idx -> {metric: value}}

                    for j, client in enumerate(self.selected_clients):

                        if not hasattr(client, "_model_after_local"):
                            continue
                        if client.id in [5,6,7,8,9]:
                            continue
                        test_data = read_client_data_FCL_cifar10(
                            client.id, task=task,
                            classes_per_task=self.args.cpt,
                            count_labels=False, train=False
                        )

                        drift_results[client.id] = {}

                        for block_idx in [4]:
                            target_layer = f'block{block_idx}'
                            try:
                                feat_global = compute_feature_resnet18_wrap(
                                    model_global_before, task, test_data, target_layer,
                                    self.args.seed, self.args)
                                feat_local = compute_feature_resnet18_wrap(
                                    client._model_after_local, task, test_data, target_layer,
                                    self.args.seed, self.args)
                                feat_aggre = compute_feature_resnet18_wrap(
                                    model_global_after, task, test_data, target_layer,
                                    self.args.seed, self.args)

                                drift_trained  = compute_eps(feat_global, feat_local)
                                drift_aggre    = compute_eps(feat_local,  feat_aggre)
                                drift_global = compute_eps(feat_global,feat_aggre)
                                # _,cka_trained    = compute_cka(feat_global, feat_local)
                                # _,cka_aggre      = compute_cka(feat_local,  feat_aggre)
                                #_,cka_global = compute_cka(feat_global,feat_aggre)
                                cknna_trained,_  = compute_alignment_from_arrays(feat_global, feat_local, "mutual_knn", topk=10, precise=True)
                                cknna_aggre ,_   = compute_alignment_from_arrays(feat_local,  feat_aggre, "mutual_knn", topk=10, precise=True)
                                cknna_global,_ = compute_alignment_from_arrays(feat_global,feat_aggre, "mutual_knn", topk=10, precise=True)
                                drift_results[client.id][block_idx] = {
                                    "drift_trained": drift_trained,
                                    "drift_aggre":   drift_aggre,
                                    "drift_global":drift_global,
                                    # "cka_trained":   cka_trained,
                                    # "cka_aggre":     cka_aggre,
                                    #"cka_global":cka_global,
                                    "cknna_global":cknna_global,
                                    "cknna_trained": cknna_trained,
                                    "cknna_aggre":   cknna_aggre,

                                }

                                # print per block per client
                                print(
                                    f"[Drift] round={disp_round} task={task} client={client.id} {target_layer} | "
                                    # f"drift_trained={drift_trained:.4f} drift_aggre={drift_aggre:.4f} | drift_global = {drift_global} |"
                                    # f"cka_trained={cka_trained:.4f} cka_aggre={cka_aggre:.4f} | cka_global = {cka_global} |"
                                    f"cknna_trained={cknna_trained:.4f} cknna_aggre={cknna_aggre:.4f} cknna_global = {cknna_global}"
                                )

                            except Exception as e:
                                print(f"[Drift] client={client.id} block{block_idx} error: {e}")
                                drift_results[client.id][block_idx] = None

                        # append vào client_summaries đúng client
                        block_summary = {}
                        for block_idx in [4]:
                            r = drift_results[client.id].get(block_idx)
                            if r is None:
                                continue
                            for metric_name, val in r.items():
                                block_summary[f"block{block_idx}/{metric_name}"] = val
                        client_summaries[j].update(block_summary)

                    # wandb log: per-client per block per metric
                    if getattr(self.args, "wandb", True):
                        # Determine whether to log this round
                        log_every_round = getattr(self.args, "wandb_drift_every_round", True)  # test mode
                        final_round_of_task = 24  # production: only log at round 24
                        
                        should_log = log_every_round or (i == final_round_of_task)
                        
                        if should_log and i == 0:  # keep your existing round==0 guard
                            for cid in drift_results:
                                for block_idx in [4]:
                                    if drift_results[cid].get(block_idx) is None:
                                        continue
                                    wb_drift = {"round": disp_round, "task": task, "client": cid}
                                    for metric_name in ["drift_trained", "drift_aggre", "cka_trained", "cka_aggre"]:
                                        if metric_name in drift_results[cid][block_idx]:
                                            wb_drift[f"drift/client{cid}/block{block_idx}/{metric_name}"] = float(drift_results[cid][block_idx][metric_name])
                                    wandb.log(wb_drift)

                except Exception as e:
                    print(f"[Drift measure] warning: {e}")

                    import traceback; traceback.print_exc()   


                csv_path = "/home/ghostm211/Thu/FCL_3/weight_feddbe/feddbe_drift_results.csv"
                fieldnames = ["round", "task", "client", "block",
                            "drift_trained", "drift_aggre", "drift_global",
                            "cknna_trained", "cknna_aggre", "cknna_global"]

                write_header = not os.path.exists(csv_path)

                with open(csv_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    if write_header:
                        writer.writeheader()

                    for cid in drift_results:
                        for block_idx in [4]:
                            r = drift_results[cid].get(block_idx)
                            if r is None:
                                continue
                            writer.writerow({
                                "round":           round,
                                "task":          task,
                                "client":        cid,
                                "block":         block_idx,
                                "drift_trained": round(float(r.get("drift_trained", float("nan"))), 6),
                                "drift_aggre":   round(float(r.get("drift_aggre",   float("nan"))), 6),
                                "drift_global":  round(float(r.get("drift_global",  float("nan"))), 6),
                                # "cka_trained":   round(float(r.get("cka_trained",   float("nan"))), 6),
                                # "cka_aggre":     round(float(r.get("cka_aggre",     float("nan"))), 6),
                                # "cka_global":    round(float(r.get("cka_global",    float("nan"))), 6),
                                "cknna_trained": round(float(r.get("cknna_trained", float("nan"))), 6),
                                "cknna_aggre":   round(float(r.get("cknna_aggre",   float("nan"))), 6),
                                "cknna_global":  round(float(r.get("cknna_global",  float("nan"))), 6)
                            })
            # track best global by validation if base supports it
            if hasattr(self, "global_test"):
                acc = self.global_test()
            else:
                acc = None
            if acc is None or (self.rs_test_acc and self.rs_test_acc[-1] >= max(self.rs_test_acc)):
                self.best_global = copy.deepcopy(self.global_model)

            # collect and average client prototypes
            self._aggregate_prototypes()

            # record / display as usual
            self.print_result(round)

        self.save_results()

    def _aggregate_prototypes(self):
        # Gather available client prototypes and average
        protos = []
        for c in self.clients:
            p = getattr(c, "upload_prototypes", None)
            if p is not None:
                protos.append(p)
        if protos:
            P = torch.stack(protos, dim=0).float()
            self.global_prototypes = torch.nanmean(P, dim=0)  # [C, D]
