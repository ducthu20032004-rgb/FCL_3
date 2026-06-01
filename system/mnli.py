# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt

# # df = pd.read_csv(r"C:\Thu\FCL\outputs\client_representation_drift-hetero-ResNet18.csv")

# # blocks = sorted(df['block_idx'].unique())
# # tasks  = sorted(df['t'].unique())

# # TASK_COLORS = ['#1a237e', '#7b1fa2', '#e91e63', '#ff6f00', '#f9d71c']

# # def make_fig(raw_col, scale, y_label, title, fname):
# #     agg = df.groupby(['t', 'block_idx'])[raw_col].agg(['mean', 'std']).reset_index()

# #     fig, ax = plt.subplots(figsize=(11, 6))
# #     ax.set_facecolor('#f8f9ff')
# #     fig.patch.set_facecolor('white')

# #     for t_idx, t_val in enumerate(tasks):
# #         color = TASK_COLORS[t_idx]
# #         sub   = agg[agg.t == t_val].sort_values('block_idx')
# #         mean  = sub['mean'].values * scale
# #         std   = sub['std'].values  * scale
# #         x     = sub['block_idx'].values
# #         ax.fill_between(x, mean - std, mean + std,
# #                         color=color, alpha=0.18, linewidth=0, zorder=2)
# #         ax.plot(x, mean, color=color, linewidth=2.2, marker='o',
# #                 ms=6, zorder=3, label=f"task t={t_val}")

# #     x_min = min(blocks)
# #     x_max = max(blocks)
# #     ax.set_xlim(x_min - 0.05, x_max + 0.3)

# #     ax.set_xlabel("Block (Layer)", fontsize=11)
# #     ax.set_ylabel(y_label,         fontsize=11)
# #     ax.set_title(title,            fontsize=12, fontweight='bold', pad=10)
# #     ax.set_xticks(blocks)
# #     ax.grid(True, linestyle='--', alpha=0.5, color='#ccccdd')

# #     ax.legend(fontsize=9.5, framealpha=0.9, loc='upper left')

# #     plt.tight_layout()
# #     plt.savefig(fname, dpi=180, bbox_inches='tight')
# #     print(f"Saved → {fname}")
# #     plt.show()


# # make_fig('eps', 1.0,
# #          "Magnitude",
# #          "Cross-Client Representation Drift",
# #          r"C:\Thu\FCL\outputs\final3_eps.png")

# # make_fig('cka', 100.0,
# #          "Similarity %",
# #          "Cross-Client Representation Similarity · CKA×100",
# #          r"C:\Thu\FCL\outputs\final3_cka.png")

# # print("Done!")

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# df = pd.read_csv(r"C:\Thu\FCL\outputs\client_representation_drift-hetero-ResNet18.csv")

# blocks = sorted(df['block_idx'].unique())
# tasks  = sorted(df['t'].unique())

# # Màu theo block — dùng colormap để tự sinh đủ màu dù nhiều block
# cmap = plt.cm.get_cmap('tab10', len(blocks))
# BLOCK_COLORS = [cmap(i) for i in range(len(blocks))]

# def make_fig(raw_col, scale, y_label, title, fname):
#     agg = df.groupby(['t', 'block_idx'])[raw_col].agg(['mean', 'std']).reset_index()

#     fig, ax = plt.subplots(figsize=(11, 6))
#     ax.set_facecolor('#f8f9ff')
#     fig.patch.set_facecolor('white')

#     for b_idx, b_val in enumerate(blocks):
#         color = BLOCK_COLORS[b_idx]
#         sub   = agg[agg.block_idx == b_val].sort_values('t')   # ← sort theo task
#         mean  = sub['mean'].values * scale
#         std   = sub['std'].values  * scale
#         x     = sub['t'].values                                # ← x là task

#         ax.fill_between(x, mean - std, mean + std,
#                         color=color, alpha=0.18, linewidth=0, zorder=2)
#         ax.plot(x, mean, color=color, linewidth=2.2, marker='o',
#                 ms=6, zorder=3, label=f"block {b_val}")

#     x_min = min(tasks)
#     x_max = max(tasks)
#     ax.set_xlim(x_min - 0.05, x_max + 0.3)

#     ax.set_xlabel("Task (t)",  fontsize=11)
#     ax.set_ylabel(y_label,     fontsize=11)
#     ax.set_title(title,        fontsize=12, fontweight='bold', pad=10)
#     ax.set_xticks(tasks)
#     ax.grid(True, linestyle='--', alpha=0.5, color='#ccccdd')

#     ax.legend(fontsize=9.5, framealpha=0.9, loc='upper left',
#               ncol=2 if len(blocks) > 6 else 1)   # tự wrap legend nếu nhiều block

#     plt.tight_layout()
#     plt.savefig(fname, dpi=180, bbox_inches='tight')
#     print(f"Saved → {fname}")
#     plt.show()


# # make_fig('eps', 1.0,
# #          "Magnitude",
# #          "Cross-Client Representation Drift (by block)",
# #          r"C:\Thu\FCL\outputs\final3_eps_by_task.png")

# # make_fig('cka', 100.0,
# #          "Similarity %",
# #          "Cross-Client Representation Similarity · CKA×100 (by block)",
# #          r"C:\Thu\FCL\outputs\final3_cka_by_task.png")
# # make_fig('sigma',1.0,
# #          "Distance Feature Space",
# #          "Cross-Client Representation Distance ",
# #             r"C:\Thu\FCL\outputs\final3_sigma_by_task.png")
# make_fig('align@10',100.0,
#          "Similarity %",
#          "Cross-Client Representation Similarity · CKNNA×100 (by block)",
#          r"C:\Thu\FCL\outputs\final3_cknna_by_task.png")
# print("Done!")

import numpy as np
arr = np.load("dataset/cifar10-classes/0.npy")
print(arr.shape)  # (N, H, W, 3)