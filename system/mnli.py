# # # import pandas as pd
# # # import numpy as np
# # # import matplotlib.pyplot as plt

# # # df = pd.read_csv(r"C:\Thu\FCL\outputs\client_representation_drift-hetero-ResNet18.csv")

# # # blocks = sorted(df['block_idx'].unique())
# # # tasks  = sorted(df['t'].unique())

# # # TASK_COLORS = ['#1a237e', '#7b1fa2', '#e91e63', '#ff6f00', '#f9d71c']

# # # def make_fig(raw_col, scale, y_label, title, fname):
# # #     agg = df.groupby(['t', 'block_idx'])[raw_col].agg(['mean', 'std']).reset_index()

# # #     fig, ax = plt.subplots(figsize=(11, 6))
# # #     ax.set_facecolor('#f8f9ff')
# # #     fig.patch.set_facecolor('white')

# # #     for t_idx, t_val in enumerate(tasks):
# # #         color = TASK_COLORS[t_idx]
# # #         sub   = agg[agg.t == t_val].sort_values('block_idx')
# # #         mean  = sub['mean'].values * scale
# # #         std   = sub['std'].values  * scale
# # #         x     = sub['block_idx'].values
# # #         ax.fill_between(x, mean - std, mean + std,
# # #                         color=color, alpha=0.18, linewidth=0, zorder=2)
# # #         ax.plot(x, mean, color=color, linewidth=2.2, marker='o',
# # #                 ms=6, zorder=3, label=f"task t={t_val}")

# # #     x_min = min(blocks)
# # #     x_max = max(blocks)
# # #     ax.set_xlim(x_min - 0.05, x_max + 0.3)

# # #     ax.set_xlabel("Block (Layer)", fontsize=11)
# # #     ax.set_ylabel(y_label,         fontsize=11)
# # #     ax.set_title(title,            fontsize=12, fontweight='bold', pad=10)
# # #     ax.set_xticks(blocks)
# # #     ax.grid(True, linestyle='--', alpha=0.5, color='#ccccdd')

# # #     ax.legend(fontsize=9.5, framealpha=0.9, loc='upper left')

# # #     plt.tight_layout()
# # #     plt.savefig(fname, dpi=180, bbox_inches='tight')
# # #     print(f"Saved → {fname}")
# # #     plt.show()


# # # make_fig('eps', 1.0,
# # #          "Magnitude",
# # #          "Cross-Client Representation Drift",
# # #          r"C:\Thu\FCL\outputs\final3_eps.png")

# # # make_fig('cka', 100.0,
# # #          "Similarity %",
# # #          "Cross-Client Representation Similarity · CKA×100",
# # #          r"C:\Thu\FCL\outputs\final3_cka.png")

# # # print("Done!")

# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt

# # df = pd.read_csv(r"C:\Thu\FCL\outputs\client_representation_drift-hetero-ResNet18.csv")

# # blocks = sorted(df['block_idx'].unique())
# # tasks  = sorted(df['t'].unique())

# # # Màu theo block — dùng colormap để tự sinh đủ màu dù nhiều block
# # cmap = plt.cm.get_cmap('tab10', len(blocks))
# # BLOCK_COLORS = [cmap(i) for i in range(len(blocks))]

# # def make_fig(raw_col, scale, y_label, title, fname):
# #     agg = df.groupby(['t', 'block_idx'])[raw_col].agg(['mean', 'std']).reset_index()

# #     fig, ax = plt.subplots(figsize=(11, 6))
# #     ax.set_facecolor('#f8f9ff')
# #     fig.patch.set_facecolor('white')

# #     for b_idx, b_val in enumerate(blocks):
# #         color = BLOCK_COLORS[b_idx]
# #         sub   = agg[agg.block_idx == b_val].sort_values('t')   # ← sort theo task
# #         mean  = sub['mean'].values * scale
# #         std   = sub['std'].values  * scale
# #         x     = sub['t'].values                                # ← x là task

# #         ax.fill_between(x, mean - std, mean + std,
# #                         color=color, alpha=0.18, linewidth=0, zorder=2)
# #         ax.plot(x, mean, color=color, linewidth=2.2, marker='o',
# #                 ms=6, zorder=3, label=f"block {b_val}")

# #     x_min = min(tasks)
# #     x_max = max(tasks)
# #     ax.set_xlim(x_min - 0.05, x_max + 0.3)

# #     ax.set_xlabel("Task (t)",  fontsize=11)
# #     ax.set_ylabel(y_label,     fontsize=11)
# #     ax.set_title(title,        fontsize=12, fontweight='bold', pad=10)
# #     ax.set_xticks(tasks)
# #     ax.grid(True, linestyle='--', alpha=0.5, color='#ccccdd')

# #     ax.legend(fontsize=9.5, framealpha=0.9, loc='upper left',
# #               ncol=2 if len(blocks) > 6 else 1)   # tự wrap legend nếu nhiều block

# #     plt.tight_layout()
# #     plt.savefig(fname, dpi=180, bbox_inches='tight')
# #     print(f"Saved → {fname}")
# #     plt.show()


# # # make_fig('eps', 1.0,
# # #          "Magnitude",
# # #          "Cross-Client Representation Drift (by block)",
# # #          r"C:\Thu\FCL\outputs\final3_eps_by_task.png")

# # # make_fig('cka', 100.0,
# # #          "Similarity %",
# # #          "Cross-Client Representation Similarity · CKA×100 (by block)",
# # #          r"C:\Thu\FCL\outputs\final3_cka_by_task.png")
# # # make_fig('sigma',1.0,
# # #          "Distance Feature Space",
# # #          "Cross-Client Representation Distance ",
# # #             r"C:\Thu\FCL\outputs\final3_sigma_by_task.png")
# # make_fig('align@10',100.0,
# #          "Similarity %",
# #          "Cross-Client Representation Similarity · CKNNA×100 (by block)",
# #          r"C:\Thu\FCL\outputs\final3_cknna_by_task.png")
# # print("Done!")
# from torch.utils.data import DataLoader
# from system.measure_gpu1 import *
# import os
# def get_model_path(saving_dir: str, client_id: int, task: int, round: int) -> str:
#     return os.path.join(saving_dir, f'client_{client_id}_task_{task}_round_{round}.pt')
# path_model = get_model_path("/home/ghostm211/Thu/FCL_3/weight_fedweit/weight_client_round", 0, 0, 0)
# test_data_t      = read_client_data_FCL_cifar10(
#                     0, task=0,      classes_per_task=2,
#                     count_labels=False, train=False)
# def _make_loader(dataset, batch_size: int = 256):
#     """
#     Tao DataLoader an toan, xu ly moi kieu tra ve cua read_client_data_FCL_cifar10:

# <<<<<<< HEAD
# import pandas as pd
# import numpy as np

# CSV_PATH = r"C:\Thu\FCL\outputs\client_representation_drift-hetero-ResNet18.csv"

# df = pd.read_csv(CSV_PATH)

# # Rename align columns for convenience
# df = df.rename(columns={"align@10": "cknna10", "align@20": "cknna20"})

# METRICS = ["eps", "cka", "cknna10", "sigma"]

# # ─────────────────────────────────────────────────────────────────────────────
# # 1. Average over ALL client pairs  →  mean per (t, block_idx)
# # ─────────────────────────────────────────────────────────────────────────────
# by_t_block = (
#     df.groupby(["t", "block_idx"])[METRICS]
#     .mean()
#     .reset_index()
# )

# # ─────────────────────────────────────────────────────────────────────────────
# # 2. Average over ALL client pairs  →  mean per t  (averaged across blocks too)
# # ─────────────────────────────────────────────────────────────────────────────
# by_t = (
#     df.groupby("t")[METRICS]
#     .mean()
#     .reset_index()
# )

# # ═════════════════════════════════════════════════════════════════════════════
# # PRINT SECTION 1 – Per (task, block) averaged over all client pairs
# # ═════════════════════════════════════════════════════════════════════════════
# tasks = sorted(by_t_block["t"].unique())
# blocks = sorted(by_t_block["block_idx"].unique())

# col_w = 12
# hdr_w = 8

# print("=" * 80)
# print("  AVERAGE METRICS PER (TASK, BLOCK)  –  averaged over all client pairs")
# print("=" * 80)

# for t in tasks:
#     subset = by_t_block[by_t_block["t"] == t].set_index("block_idx")

#     # Header
#     print(f"\n  Task t = {t}")
#     print(f"  {'Block':<{hdr_w}}", end="")
#     for m in METRICS:
#         print(f"  {m:>{col_w}}", end="")
#     print()
#     print("  " + "-" * (hdr_w + len(METRICS) * (col_w + 2)))

#     for b in blocks:
#         if b not in subset.index:
#             continue
#         row = subset.loc[b]
#         print(f"  {b:<{hdr_w}}", end="")
#         for m in METRICS:
#             v = row[m]
#             # eps is tiny → scientific; others → fixed
#             if m == "eps":
#                 print(f"  {v:>{col_w}.4e}", end="")
#             elif m == "cknna10":
#                 print(f"  {v:>{col_w}.4f}", end="")
#             else:
#                 print(f"  {v:>{col_w}.6f}", end="")
#         print()

# # ═════════════════════════════════════════════════════════════════════════════
# # PRINT SECTION 2 – Per task, averaged over all blocks AND all client pairs
# # ═════════════════════════════════════════════════════════════════════════════
# print("\n")
# print("=" * 80)
# print("  AVERAGE METRICS PER TASK  –  averaged over all blocks & client pairs")
# print("=" * 80)
# print(f"\n  {'Task':<{hdr_w}}", end="")
# for m in METRICS:
#     print(f"  {m:>{col_w}}", end="")
# print()
# print("  " + "-" * (hdr_w + len(METRICS) * (col_w + 2)))

# for _, row in by_t.iterrows():
#     t = int(row["t"])
#     print(f"  {t:<{hdr_w}}", end="")
#     for m in METRICS:
#         v = row[m]
#         if m == "eps":
#             print(f"  {v:>{col_w}.4e}", end="")
#         elif m == "cknna10":
#             print(f"  {v:>{col_w}.4f}", end="")
#         else:
#             print(f"  {v:>{col_w}.6f}", end="")
#     print()

# print("\nDone.")
# =======
#       Case 1 - torch.utils.data.Dataset chuan  -> dung truc tiep
#       Case 2 - tuple/list 2 phan tu (X, Y) voi X,Y la array/tensor (N,...) -> TensorDataset
#       Case 3 - list of (x_i, y_i) sample tuples -> stack roi TensorDataset
#     num_workers=0 de tranh loi pickle / seek khi data da duoc load san vao RAM.
#     """
#     from torch.utils.data import TensorDataset, Dataset

#     # Case 1: torch.utils.data.Dataset chuan
#     if isinstance(dataset, Dataset):
#         return DataLoader(dataset, batch_size=batch_size, shuffle=False,
#                           num_workers=0, pin_memory=(DEVICE.type == 'cuda'))

#     # Case 2: (X, Y) - moi phan tu la array/tensor ca batch
#     # Nhan dien: co dung 2 phan tu va phan tu dau co >= 2 chieu (batch dim + feature dims)
#     if (isinstance(dataset, (tuple, list))
#             and len(dataset) == 2
#             and hasattr(dataset[0], 'shape')
#             and len(np.shape(dataset[0])) >= 2):
#         X, Y = dataset
#         xs = torch.as_tensor(np.array(X, dtype=np.float32))
#         ys = torch.as_tensor(np.array(Y)).long()
#         return DataLoader(TensorDataset(xs, ys), batch_size=batch_size, shuffle=False,
#                           num_workers=0, pin_memory=(DEVICE.type == 'cuda'))

#     # Case 3: list of (x_i, y_i) sample tuples
#     xs, ys = [], []
#     for x, y in dataset:
#         xs.append(torch.as_tensor(np.array(x, dtype=np.float32)))
#         ys.append(torch.as_tensor(np.array(y)).long())
#     xs = torch.stack(xs)
#     ys = torch.stack(ys)
#     return DataLoader(TensorDataset(xs, ys), batch_size=batch_size, shuffle=False,
#                       num_workers=0, pin_memory=(DEVICE.type == 'cuda'))



# loader_t = _make_loader(test_data_t, batch_size=128)
# model = load_resnet18_from_checkpoint(path_model, load_head=False)
# feat_t = compute_feature_resnet18(path_model,loader_t,0,seed =42 ,args=args)
# print(path_model)
# print(feat_t.shape)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Representation Drift Measurement')

#     parser.add_argument('--saving_dir',        type=str,  default=r'D:\FCL\checkpoints_client_task')
#     parser.add_argument('--cp_probe',          type=str,  default=r'C:\Thu\FCL\probes_torchvision')
#     parser.add_argument('--partition_options', type=str,  default='hetero')
#     parser.add_argument('--backbone',          type=str,  default='ResNet18')
#     parser.add_argument('--num_clients',       type=int,  default=10)
#     parser.add_argument('--num_tasks',         type=int,  default=5)
#     parser.add_argument('--cpt',               type=int,  default=2)
#     parser.add_argument('--seed',              type=int,  default=42)
#     parser.add_argument('--classes',           type=int,  default=10)
#     parser.add_argument('--use_wandb',         type=bool, default=False)
#     parser.add_argument('--method',             type=str,  default='dynamic')
#     parser.add_argument('--kaggle',             type=bool, default=False)
#     parser.add_argument('--retrain_epochs',  type=int,   default=10)
#     parser.add_argument('--retrain_lr',      type=float, default=1e-3)
#     parser.add_argument('--retrain_patience',type=int,   default=3)
#     parser.add_argument('--model',             type=str,  default='ALA')
#     args = parser.parse_args()
#     main(args)
# >>>>>>> 61ea87e49662544c3e97fcb8e1f61c9ddfcd266c

import numpy as np
data = np.load('./dataset/cifar10-classes/0.npy')
print(data.shape, data.dtype)
print(data.min(), data.max())