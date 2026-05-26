# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import matplotlib.cm as cm
# # import matplotlib.lines as mlines
# # from scipy import stats
# # import sys, os

# # # ══════════════════════════════════════════════════════
# # #  CONFIG
# # # ══════════════════════════════════════════════════════
# # CSV_FILE   = './outputs/client_representation_drift-hetero-ResNet18.csv'
# # BLOCK      = 3                    # block muốn vẽ
# # X_METRICS  = ['cka', 'cosine_similarity', 'sigma','align150']   # 4 metrics trục X
# # Y_METRIC   = 'old_test_acc'       # target: acc@task_t
# # OUTPUT_DIR = './outputs/scatter'
# # ANNOTATE_CLIENT = True            # hiện số client cạnh điểm
# # # ══════════════════════════════════════════════════════

# # METRIC_LABELS = {
# #     'sigma':            'σ (sigma)',
# #     'eps':              'ε (epsilon)',
# #     'cka':              'CKA',
# #     'linear_cka':       'Linear CKA',
# #     'kernel_cka':       'Kernel CKA',
# #     'cosine_similarity':'Cosine Similarity',
# #     'align100':         'Align@100',
# #     'align150':         'Align@150',
# #     'old_test_acc':     'ACC @ task t',
# #     'current_test_acc': "ACC @ task t'",
# # }

# # os.makedirs(OUTPUT_DIR, exist_ok=True)


# # def load(path):
# #     df = pd.read_csv(path)
# #     df.columns = df.columns.str.strip()
# #     df['pair'] = df['t'].astype(int).astype(str) + '→' + df['tprime'].astype(int).astype(str)
# #     return df


# # def plot_metrics_vs_acc(df, block, x_metrics, y_metric, annotate=True):
# #     df_b = df[df['block'] == block].copy()
# #     if df_b.empty:
# #         print(f'[SKIP] No data for block={block}')
# #         return

# #     # ── Palette theo task pair ───────────────────────
# #     pairs   = sorted(df_b['pair'].unique())
# #     cmap    = cm.get_cmap('tab10', len(pairs))
# #     pair_color = {p: cmap(i) for i, p in enumerate(pairs)}

# #     # ── Lọc metrics có trong data ────────────────────
# #     x_metrics = [m for m in x_metrics if m in df_b.columns]
# #     n = len(x_metrics)

# #     fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5.5), sharey=True)
# #     if n == 1:
# #         axes = [axes]

# #     yl = METRIC_LABELS.get(y_metric, y_metric)

# #     for ax, xm in zip(axes, x_metrics):
# #         xl = METRIC_LABELS.get(xm, xm)
# #         sub = df_b.dropna(subset=[xm, y_metric])

# #         # ── Scatter từng pair, client là điểm ────────
# #         for pair in pairs:
# #             grp = sub[sub['pair'] == pair].sort_values('client')
# #             c   = pair_color[pair]
# #             ax.scatter(
# #                 grp[xm], grp[y_metric],
# #                 color=c, s=70, alpha=0.85,
# #                 edgecolors='white', linewidths=0.5,
# #                 zorder=4, label=pair
# #             )
# #             # # Nối các client theo thứ tự để thấy trajectory
# #             # ax.plot(
# #             #     grp[xm], grp[y_metric],
# #             #     color=c, linewidth=0.8, alpha=0.4,
# #             #     linestyle='-', zorder=3
# #             # )
# #             # Annotate số client
# #             if annotate:
# #                 for _, row in grp.iterrows():
# #                     ax.annotate(
# #                         f"C{int(row['client'])}",
# #                         (row[xm], row[y_metric]),
# #                         textcoords='offset points',
# #                         xytext=(4, 3),
# #                         fontsize=7,
# #                         color=c,
# #                         alpha=0.9
# #                     )

# #         # ── Trend line tổng thể ───────────────────────
# #         x_all = sub[xm].values
# #         y_all = sub[y_metric].values
# #         if len(x_all) > 2:
# #             slope, intercept, r, p, _ = stats.linregress(x_all, y_all)
# #             x_line = np.linspace(x_all.min(), x_all.max(), 300)
# #             ax.plot(x_line, slope * x_line + intercept,
# #                     color='crimson', linewidth=2.0,
# #                     linestyle='--', zorder=5,
# #                     label=f'Trend r={r:.3f}')
# #             ax.set_title(f'{xl}\nr={r:.3f}  p={p:.3f}', fontsize=11)
# #         else:
# #             ax.set_title(xl, fontsize=11)

# #         ax.set_xlabel(xl, fontsize=11)
# #         if ax == axes[0]:
# #             ax.set_ylabel(yl, fontsize=11)
# #         ax.grid(True, linestyle=':', alpha=0.35)

# #     # ── Legend chung: task pair + trend ──────────────
# #     pair_handles = [
# #         mlines.Line2D([], [], color=pair_color[p], marker='o',
# #                       markersize=7, linewidth=1, label=p)
# #         for p in pairs
# #     ]
# #     trend_handle = mlines.Line2D([], [], color='crimson', linewidth=2,
# #                                   linestyle='--', label='Trend line')
# #     fig.legend(
# #         handles=pair_handles + [trend_handle],
# #         title='Task pair',
# #         loc='upper right',
# #         bbox_to_anchor=(1.0, 1.0),
# #         fontsize=9,
# #         title_fontsize=10,
# #         framealpha=0.85
# #     )

# #     fig.suptitle(
# #         f'Block {block}  —  Metrics vs {yl}  (colored by task pair, labeled by client)',
# #         fontsize=13, fontweight='bold', y=1.02
# #     )
# #     plt.tight_layout()

# #     # fname = os.path.join(OUTPUT_DIR, f'scatter_block{block}_multi_vs_{y_metric}.png')
# #     # plt.savefig(fname, dpi=150, bbox_inches='tight')
# #     plt.show()
# #     plt.close()
# #     # print(f'Saved: {fname}')
# #     print(f'Metrics plotted: {x_metrics}')


# # # ══════════════════════════════════════════════════════
# # if __name__ == '__main__':
# #     csv  = sys.argv[1] if len(sys.argv) > 1 else CSV_FILE
# #     blk  = int(sys.argv[2]) if len(sys.argv) > 2 else BLOCK

# #     df = load(csv)
# #     print(f'Loaded {len(df)} rows | blocks={sorted(df.block.unique())}')
# #     print(f'Pairs: {sorted(df.pair.unique())}')
# #     print(f'Clients: {sorted(df.client.unique())}\n')

# #     plot_metrics_vs_acc(df, blk, X_METRICS, Y_METRIC, ANNOTATE_CLIENT)

# """
# plot_divergence.py — Visualize client divergence (1 - CKA) across tasks & blocks

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# USAGE EXAMPLES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#   # Layout v1: X=client1, Y=client2, Z=divergence, grouped by task
#   python plot_divergence.py --block 2 --mode bar3d    --layout v1
#   python plot_divergence.py --block 2 --mode surface  --layout v1
#   python plot_divergence.py --block 2 --mode heatmap  --layout v1
#   python plot_divergence.py --block 2 --mode grouped  --layout v1

#   # Layout v2: X=client pair label, Y=task, Z=divergence
#   python plot_divergence.py --block 2 --mode bar3d    --layout v2
#   python plot_divergence.py --block 2 --mode surface  --layout v2
#   python plot_divergence.py --block 2 --mode heatmap  --layout v2
#   python plot_divergence.py --block 2 --mode grouped  --layout v2

#   # Multiple blocks side by side
#   python plot_divergence.py --block all --mode heatmap --layout v2

#   # Save to file
#   python plot_divergence.py --block 4 --mode bar3d --layout v1 --output out.png

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ARGUMENTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#   --block   0-4 or "all"                          (default: 2)
#   --mode    bar3d | surface | heatmap | grouped   (default: heatmap)
#   --layout  v1 | v2                               (default: v1)
#             v1 → X=client1, Y=client2  (matrix view per task)
#             v2 → X=pair label,  Y=task (timeline view)
#   --metric  cka | align150                        (default: cka)
#   --data    path to .csv or .xlsx                 (default: client_pair.xlsx)
#   --output  save path e.g. out.png                (default: show window)
#   --dpi     output DPI                            (default: 150)
# """

# import argparse
# import sys
# from typing import List, Optional

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# # ══════════════════════════════════════════════════════════════
# # CLI
# # ══════════════════════════════════════════════════════════════

# def parse_args():
#     p = argparse.ArgumentParser(description="Client divergence visualizer")
#     p.add_argument("--block",  default="2",
#                    help="Block index 0-4 or 'all'")
#     p.add_argument("--mode",   default="heatmap",
#                    choices=["bar3d", "surface", "heatmap", "grouped"],
#                    help="Plot type")
#     p.add_argument("--layout", default="v1", choices=["v1", "v2"],
#                    help="v1=client1 x client2 matrix | v2=pair x task")
#     p.add_argument("--metric", default="cka", choices=["cka", "align150"],
#                    help="Metric to compute divergence from")
#     p.add_argument("--data",   default="client_pair.xlsx",
#                    help="Path to .csv or .xlsx data file")
#     p.add_argument("--output", default=None,
#                    help="Save figure to file (png/pdf). Omit = show window")
#     p.add_argument("--dpi",    default=150, type=int)
#     return p.parse_args()


# # ══════════════════════════════════════════════════════════════
# # DATA
# # ══════════════════════════════════════════════════════════════

# def load_data(path, metric):
#     try:
#         df = pd.read_csv(path) if path.endswith(".csv") else \
#              pd.read_excel(path, sheet_name=0)
#     except FileNotFoundError:
#         sys.exit("[ERROR] File not found: {}".format(path))

#     col = "cka" if metric == "cka" else "align@150"
#     required = {"block_idx", "t", "client1", "client2", col}
#     missing = required - set(df.columns)
#     if missing:
#         sys.exit("[ERROR] Missing columns: {}".format(missing))

#     df["divergence"] = 1.0 - df[col].clip(0, 1)
#     df["pair"] = df.apply(
#         lambda r: "{}-{}".format(int(r.client1), int(r.client2)), axis=1)
#     return df


# def resolve_blocks(arg):
#     if arg.lower() == "all":
#         return [0, 1, 2, 3, 4]
#     try:
#         b = int(arg)
#         if b not in range(5):
#             sys.exit("[ERROR] --block must be 0-4 or 'all'")
#         return [b]
#     except ValueError:
#         sys.exit("[ERROR] --block must be an integer 0-4 or 'all'")


# # ══════════════════════════════════════════════════════════════
# # PIVOT HELPERS
# # ══════════════════════════════════════════════════════════════

# def pivot_v1(sub, task):
#     """rows=client1, cols=client2, value=divergence for given task."""
#     d = sub[sub["t"] == task].copy()
#     clients = sorted(set(d["client1"].unique()) | set(d["client2"].unique()))
#     mat = pd.DataFrame(np.nan, index=clients, columns=clients)
#     for _, row in d.iterrows():
#         mat.loc[int(row.client1), int(row.client2)] = row.divergence
#         mat.loc[int(row.client2), int(row.client1)] = row.divergence
#     return mat


# def pivot_v2(sub):
#     """rows=pair, cols=task, value=divergence."""
#     return sub.pivot_table(index="pair", columns="t",
#                            values="divergence", aggfunc="mean")


# # ══════════════════════════════════════════════════════════════
# # SHARED
# # ══════════════════════════════════════════════════════════════

# CMAP = "RdYlGn_r"
# VMIN, VMAX = 0.0, 1.0


# def _fig_title(block, mode, layout, metric):
#     m = "1-CKA" if metric == "cka" else "1-align@150"
#     lbl = "X=C1,Y=C2" if layout == "v1" else "X=pair,Y=task"
#     return "Block {}  |  Divergence ({})  mode={}  layout={}\ngreen=converge  ->  red=diverge".format(
#         block, m, mode, lbl)


# def _save_or_show(fig, output, dpi):
#     fig.tight_layout()
#     if output:
#         fig.savefig(output, dpi=dpi, bbox_inches="tight")
#         print("[OK] Saved -> {}".format(output))
#     else:
#         plt.show()
#     plt.close(fig)


# # ══════════════════════════════════════════════════════════════
# # MODE 1 — bar3d
# # ══════════════════════════════════════════════════════════════

# def _bar3d_v1(ax, sub, tasks, clients):
#     n_tasks = len(tasks)
#     c_idx = {c: i for i, c in enumerate(clients)}
#     colors = cm.plasma(np.linspace(0.15, 0.85, n_tasks))
#     bar_w = 0.15
#     offsets = np.linspace(-bar_w * n_tasks / 2, bar_w * n_tasks / 2, n_tasks)

#     for ti, (task, color) in enumerate(zip(tasks, colors)):
#         d = sub[sub["t"] == task]
#         for _, row in d.iterrows():
#             xi = c_idx[int(row.client1)] + offsets[ti]
#             yi = c_idx[int(row.client2)] + offsets[ti]
#             ax.bar3d(xi, yi, 0, bar_w * 0.85, bar_w * 0.85,
#                      row.divergence, color=color, alpha=0.75, zsort="average")

#     ax.set_xticks(range(len(clients)))
#     ax.set_xticklabels(["C{}".format(c) for c in clients], fontsize=6)
#     ax.set_yticks(range(len(clients)))
#     ax.set_yticklabels(["C{}".format(c) for c in clients], fontsize=6)
#     ax.set_xlabel("Client 1", labelpad=8)
#     ax.set_ylabel("Client 2", labelpad=8)
#     ax.set_zlabel("Divergence")
#     ax.set_zlim(0, 1)
#     handles = [plt.Rectangle((0, 0), 1, 1,
#                color=cm.plasma(np.linspace(0.15, 0.85, n_tasks))[i])
#                for i in range(n_tasks)]
#     ax.legend(handles, ["Task {}".format(t) for t in tasks],
#               loc="upper left", fontsize=7)


# def _bar3d_v2(ax, sub, tasks, pairs):
#     p_idx = {p: i for i, p in enumerate(pairs)}
#     bar_w, bar_d = 0.6, 0.6

#     for _, row in sub.iterrows():
#         xi = p_idx[row.pair]
#         yi = tasks.index(row.t)
#         dz = float(row.divergence)
#         c = cm.RdYlGn_r(dz)
#         ax.bar3d(xi, yi, 0, bar_w, bar_d, dz,
#                  color=c, alpha=0.8, zsort="average")

#     ax.set_xticks(range(len(pairs)))
#     ax.set_xticklabels(pairs, fontsize=5, rotation=45, ha="right")
#     ax.set_yticks(range(len(tasks)))
#     ax.set_yticklabels(["Task {}".format(t) for t in tasks], fontsize=7)
#     ax.set_xlabel("Client Pair", labelpad=10)
#     ax.set_ylabel("Task", labelpad=8)
#     ax.set_zlabel("Divergence")
#     ax.set_zlim(0, 1)


# def plot_bar3d(df, blocks, layout, metric, output, dpi):
#     n = len(blocks)
#     fig = plt.figure(figsize=(9 * n, 7))

#     for idx, block in enumerate(blocks):
#         ax = fig.add_subplot(1, n, idx + 1, projection="3d")
#         sub = df[df["block_idx"] == block].copy()
#         tasks = sorted(sub["t"].unique())

#         if layout == "v1":
#             clients = sorted(set(sub["client1"].unique()) |
#                              set(sub["client2"].unique()))
#             _bar3d_v1(ax, sub, tasks, clients)
#         else:
#             pairs = sorted(sub["pair"].unique())
#             _bar3d_v2(ax, sub, tasks, pairs)

#         ax.set_title(_fig_title(block, "bar3d", layout, metric), fontsize=8, pad=12)
#         ax.view_init(elev=28, azim=-50)

#     _save_or_show(fig, output, dpi)


# # ══════════════════════════════════════════════════════════════
# # MODE 2 — surface
# # ══════════════════════════════════════════════════════════════

# def _surface_v1(ax, sub, tasks):
#     clients = sorted(set(sub["client1"].unique()) |
#                      set(sub["client2"].unique()))
#     n = len(clients)
#     colors = cm.plasma(np.linspace(0.1, 0.9, len(tasks)))

#     for task, color in zip(tasks, colors):
#         mat = pivot_v1(sub, task).values.astype(float)
#         np.fill_diagonal(mat, 0.0)
#         mat = np.nan_to_num(mat, nan=0.5)
#         X, Y = np.meshgrid(range(n), range(n))
#         ax.plot_surface(X, Y, mat, alpha=0.45, cmap="RdYlGn_r",
#                         vmin=0, vmax=1, linewidth=0)

#     ax.set_xticks(range(n))
#     ax.set_xticklabels(["C{}".format(c) for c in clients], fontsize=7)
#     ax.set_yticks(range(n))
#     ax.set_yticklabels(["C{}".format(c) for c in clients], fontsize=7)
#     ax.set_xlabel("Client 1", labelpad=8)
#     ax.set_ylabel("Client 2", labelpad=8)
#     ax.set_zlabel("Divergence")
#     ax.set_zlim(0, 1)
#     handles = [plt.Line2D([0], [0],
#                color=cm.plasma(np.linspace(0.1, 0.9, len(tasks)))[i], lw=3)
#                for i in range(len(tasks))]
#     ax.legend(handles, ["Task {}".format(t) for t in tasks], fontsize=7)


# def _surface_v2(ax, sub):
#     mat = pivot_v2(sub)
#     pairs = list(mat.index)
#     tasks = list(mat.columns)
#     Z = np.nan_to_num(mat.values.astype(float), nan=0.5)
#     X, Y = np.meshgrid(range(len(tasks)), range(len(pairs)))
#     surf = ax.plot_surface(X, Y, Z, cmap="RdYlGn_r",
#                            vmin=0, vmax=1, alpha=0.85, linewidth=0)
#     ax.set_xticks(range(len(tasks)))
#     ax.set_xticklabels(["T{}".format(t) for t in tasks], fontsize=7)
#     ax.set_yticks(range(len(pairs)))
#     ax.set_yticklabels(pairs, fontsize=5)
#     ax.set_xlabel("Task", labelpad=8)
#     ax.set_ylabel("Client Pair", labelpad=10)
#     ax.set_zlabel("Divergence")
#     ax.set_zlim(0, 1)
#     plt.colorbar(surf, ax=ax, shrink=0.4, pad=0.1, label="Divergence")


# def plot_surface(df, blocks, layout, metric, output, dpi):
#     n = len(blocks)
#     fig = plt.figure(figsize=(9 * n, 7))

#     for idx, block in enumerate(blocks):
#         ax = fig.add_subplot(1, n, idx + 1, projection="3d")
#         sub = df[df["block_idx"] == block].copy()
#         tasks = sorted(sub["t"].unique())

#         if layout == "v1":
#             _surface_v1(ax, sub, tasks)
#         else:
#             _surface_v2(ax, sub)

#         ax.set_title(_fig_title(block, "surface", layout, metric), fontsize=8, pad=12)
#         ax.view_init(elev=30, azim=-55)

#     _save_or_show(fig, output, dpi)


# # ══════════════════════════════════════════════════════════════
# # MODE 3 — heatmap
# # ══════════════════════════════════════════════════════════════

# def _heatmap_v1(axes_row, sub, tasks, block):
#     clients = sorted(set(sub["client1"].unique()) |
#                      set(sub["client2"].unique()))
#     for ax, task in zip(axes_row, tasks):
#         mat = pivot_v1(sub, task)
#         arr = mat.values.copy(); np.fill_diagonal(arr, np.nan); mat = pd.DataFrame(arr, index=mat.index, columns=mat.columns)
#         im = ax.imshow(mat.values, vmin=VMIN, vmax=VMAX,
#                        cmap=CMAP, aspect="auto", interpolation="nearest")
#         ax.set_xticks(range(len(clients)))
#         ax.set_xticklabels(["C{}".format(c) for c in clients], fontsize=7)
#         ax.set_yticks(range(len(clients)))
#         ax.set_yticklabels(["C{}".format(c) for c in clients], fontsize=7)
#         ax.set_title("Block {} — Task {}".format(block, task), fontsize=9, fontweight="bold")
#         # ax.set_xlabel("Client 2", fontsize=8)
#         # ax.set_ylabel("Client 1", fontsize=8)
#         plt.colorbar(im, ax=ax, label="Divergence", fraction=0.046, pad=0.04)
#         # annotate
#         for i in range(mat.shape[0]):
#             for j in range(mat.shape[1]):
#                 v = mat.values[i, j]
#                 if not np.isnan(v):
#                     ax.text(j, i, "{:.2f}".format(v), ha="center", va="center",
#                             fontsize=6, color="black" if v < 0.65 else "white")


# def _heatmap_v2(ax, sub, block):
#     mat = pivot_v2(sub)
#     im = ax.imshow(mat.values, vmin=VMIN, vmax=VMAX,
#                    cmap=CMAP, aspect="auto", interpolation="nearest")
#     ax.set_xticks(range(len(mat.columns)))
#     ax.set_xticklabels(["Task {}".format(t) for t in mat.columns], fontsize=8)
#     ax.set_yticks(range(len(mat.index)))
#     ax.set_yticklabels(mat.index, fontsize=7)
#     ax.set_title("Block {}  |  Divergence (pair x task)".format(block),
#                  fontsize=10, fontweight="bold")
#     ax.set_xlabel("Task", fontsize=9)
#     ax.set_ylabel("Client Pair", fontsize=9)
#     plt.colorbar(im, ax=ax, label="Divergence", fraction=0.03, pad=0.04)
#     for i in range(mat.shape[0]):
#         for j in range(mat.shape[1]):
#             v = mat.values[i, j]
#             if not np.isnan(v):
#                 ax.text(j, i, "{:.2f}".format(v), ha="center", va="center",
#                         fontsize=6, color="black" if v < 0.65 else "white")


# def plot_heatmap(df, blocks, layout, metric, output, dpi):
#     tasks = sorted(df["t"].unique())

#     if layout == "v1":
#         n_tasks = len(tasks)
#         n_blocks = len(blocks)
#         fig, axes = plt.subplots(n_blocks, n_tasks,
#                                  figsize=(4 * n_tasks, 4 * n_blocks),
#                                  squeeze=False)
#         for row_idx, block in enumerate(blocks):
#             sub = df[df["block_idx"] == block].copy()
#             _heatmap_v1(axes[row_idx], sub, tasks, block)
#     else:
#         n = len(blocks)
#         fig, axes = plt.subplots(1, n, figsize=(10 * n, 7), squeeze=False)
#         for col_idx, block in enumerate(blocks):
#             sub = df[df["block_idx"] == block].copy()
#             _heatmap_v2(axes[0][col_idx], sub, block)

#     m = "1-CKA" if metric == "cka" else "1-align@150"
#     fig.suptitle("Client Divergence Heatmap ({})  layout={}   green=converge -> red=diverge".format(m, layout),
#                  fontsize=11, y=1.01)
#     _save_or_show(fig, output, dpi)


# # ══════════════════════════════════════════════════════════════
# # MODE 4 — grouped bar
# # ══════════════════════════════════════════════════════════════

# def _grouped_v1(ax, sub, tasks, clients):
#     n_tasks = len(tasks)
#     c_idx = {c: i for i, c in enumerate(clients)}
#     bar_w = 0.8 / n_tasks
#     colors = cm.plasma(np.linspace(0.1, 0.9, n_tasks))

#     for ti, (task, color) in enumerate(zip(tasks, colors)):
#         d = sub[sub["t"] == task]
#         xs, ys = [], []
#         for _, row in d.iterrows():
#             xi = c_idx.get(int(row.client2))
#             if xi is not None:
#                 offset = (ti - n_tasks / 2 + 0.5) * bar_w
#                 xs.append(xi + offset)
#                 ys.append(row.divergence)
#         ax.bar(xs, ys, width=bar_w * 0.85, color=color,
#                alpha=0.8, label="Task {}".format(task))

#     ax.set_xticks(range(len(clients)))
#     ax.set_xticklabels(["C{}".format(c) for c in clients], fontsize=8)
#     ax.set_ylim(0, 1)
#     ax.axhline(0.5, color="red", lw=1, ls="--", alpha=0.5)
#     ax.set_xlabel("Client 2")
#     ax.set_ylabel("Divergence")
#     ax.legend(fontsize=7, loc="upper right")
#     ax.grid(axis="y", alpha=0.3)


# def _grouped_v2(ax, sub, tasks, pairs):
#     n_tasks = len(tasks)
#     p_idx = {p: i for i, p in enumerate(pairs)}
#     bar_w = 0.8 / n_tasks
#     colors = cm.plasma(np.linspace(0.1, 0.9, n_tasks))

#     for ti, (task, color) in enumerate(zip(tasks, colors)):
#         d = sub[sub["t"] == task]
#         xs, ys = [], []
#         for _, row in d.iterrows():
#             xi = p_idx.get(row.pair)
#             if xi is not None:
#                 offset = (ti - n_tasks / 2 + 0.5) * bar_w
#                 xs.append(xi + offset)
#                 ys.append(row.divergence)
#         ax.bar(xs, ys, width=bar_w * 0.85, color=color,
#                alpha=0.8, label="Task {}".format(task))

#     ax.set_xticks(range(len(pairs)))
#     ax.set_xticklabels(pairs, fontsize=6, rotation=45, ha="right")
#     ax.set_ylim(0, 1)
#     ax.axhline(0.5, color="red", lw=1, ls="--", alpha=0.5, label="0.5 threshold")
#     ax.set_xlabel("Client Pair")
#     ax.set_ylabel("Divergence")
#     ax.legend(fontsize=7, loc="upper right")
#     ax.grid(axis="y", alpha=0.3)


# def plot_grouped(df, blocks, layout, metric, output, dpi):
#     n = len(blocks)
#     fig, axes = plt.subplots(1, n, figsize=(12 * n, 5), squeeze=False)

#     for col_idx, block in enumerate(blocks):
#         ax = axes[0][col_idx]
#         sub = df[df["block_idx"] == block].copy()
#         tasks = sorted(sub["t"].unique())

#         if layout == "v1":
#             clients = sorted(set(sub["client1"].unique()) |
#                              set(sub["client2"].unique()))
#             _grouped_v1(ax, sub, tasks, clients)
#         else:
#             pairs = sorted(sub["pair"].unique())
#             _grouped_v2(ax, sub, tasks, pairs)

#         ax.set_title(_fig_title(block, "grouped", layout, metric), fontsize=9, fontweight="bold")

#     m = "1-CKA" if metric == "cka" else "1-align@150"
#     fig.suptitle("Client Divergence Grouped Bar ({})  layout={}".format(m, layout),
#                  fontsize=11, y=1.01)
#     _save_or_show(fig, output, dpi)


# # ══════════════════════════════════════════════════════════════
# # MAIN
# # ══════════════════════════════════════════════════════════════

# DISPATCHERS = {
#     "bar3d":   plot_bar3d,
#     "surface": plot_surface,
#     "heatmap": plot_heatmap,
#     "grouped": plot_grouped,
# }


# def main():
#     args = parse_args()
#     df = load_data(args.data, args.metric)
#     blocks = resolve_blocks(args.block)

#     print("[INFO] block(s)={}  mode={}  layout={}  metric={}  rows={}".format(
#         blocks, args.mode, args.layout, args.metric, len(df)))

#     DISPATCHERS[args.mode](df, blocks, args.layout, args.metric,
#                            args.output, args.dpi)


# if __name__ == "__main__":
#     main()


# # Phien bản vẽ acc_curr theo round_global, chia theo task, có mean ± std band, và chấm giá trị thật ở mỗi round.
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# from pathlib import Path

# # Đọc dữ liệu từ file CSV
# csv_path = r'C:\Thu\FCL\material_experiment\dynamic\acc_curr_block4.csv'

# try:
#     df = pd.read_csv(csv_path)
#     print("Columns:", df.columns.tolist())
#     print("\nFirst few rows:")
#     print(df.head(20))
# except FileNotFoundError:
#     print(f"Error: File not found at {csv_path}")
#     print("Please check the file path and try again.")
#     exit()

# # Giả sử dữ liệu có cấu trúc:
# # - Cột 'round_global': round number (1-125 cho 5 tasks, mỗi task 25 rounds)
# # - Cột 'acc_curr': accuracy value
# # - Cột 'task': task ID (0-4 hoặc 1-5)

# # Xử lý NaN: thay thế bằng 0.0
# df = df.fillna(0.0)
# print("\nNaN values replaced with 0.0")

# # Nếu không có cột 'task', tạo nó dựa trên round_global
# if 'task' not in df.columns:
#     df['task'] = ((df['round_global'] - 1) // 25).astype(int)

# print("\nData shape:", df.shape)
# print("Unique tasks:", sorted(df['task'].unique()))

# # Tính mean và std cho mỗi task
# fig, axes = plt.subplots(2, 3, figsize=(16, 10))
# fig.suptitle('Accuracy vs Round Global by Task (with Mean ± Std)', fontsize=16, fontweight='bold')

# axes = axes.flatten()

# # Màu sắc cho các task
# colors = plt.cm.Set1(np.linspace(0, 1, 5))

# # Vẽ cho mỗi task
# for task_id in sorted(df['task'].unique()):
#     task_data = df[df['task'] == task_id].copy()
    
#     # Reset round_global cho mỗi task (từ 1 đến 25)
#     task_data = task_data.reset_index(drop=True)
#     task_data['round_in_task'] = range(1, len(task_data) + 1)
    
#     # Tính mean và std
#     mean_acc = task_data['acc_curr'].mean()
#     std_acc = task_data['acc_curr'].std()
    
#     ax = axes[task_id]
    
#     # Vẽ std band (vùng mờ)
#     ax.fill_between(task_data['round_in_task'], 
#                      mean_acc - std_acc, mean_acc + std_acc,
#                      alpha=0.15, color=colors[task_id], 
#                      label=f'Std: ±{std_acc:.2f}')
    
#     # Vẽ mean line (nét rõ)
#     ax.plot(task_data['round_in_task'], [mean_acc]*len(task_data), 
#             linewidth=3, color=colors[task_id], 
#             label=f'Mean: {mean_acc:.2f}', zorder=3)
    
#     # Vẽ chấm ở giá trị thật
#     ax.scatter(task_data['round_in_task'], task_data['acc_curr'], 
#                s=60, color=colors[task_id], alpha=0.8, 
#                edgecolors='black', linewidth=0.5, zorder=4, label='Actual')
    
#     # Set labels và titles
#     ax.set_xlabel('Round in Task', fontsize=11, fontweight='bold')
#     ax.set_ylabel('Accuracy (acc_curr)', fontsize=11, fontweight='bold')
#     ax.set_title(f'Task {task_id}', fontsize=12, fontweight='bold')
#     ax.legend(loc='best', fontsize=9)
#     ax.grid(True, alpha=0.3)
#     ax.set_ylim([min(task_data['acc_curr']) - 1, max(task_data['acc_curr']) + 1])

# # Ẩn subplot thứ 6 (nếu có)
# if len(df['task'].unique()) < 6:
#     axes[-1].set_visible(False)

# plt.tight_layout()
# plt.savefig('acc_curr_block4_by_task.png', dpi=300, bbox_inches='tight')
# print("\n✓ Plot saved as 'acc_curr_block4_by_task.png'")
# plt.show()

# # Vẽ biểu đồ tổng hợp (tất cả các task trên một đồ thị)
# fig2, ax2 = plt.subplots(figsize=(14, 8))

# for task_id in sorted(df['task'].unique()):
#     task_data = df[df['task'] == task_id].copy()
#     task_data = task_data.reset_index(drop=True)
#     task_data['round_in_task'] = range(1, len(task_data) + 1)
    
#     mean_acc = task_data['acc_curr'].mean()
#     std_acc = task_data['acc_curr'].std()
    
#     # Offset x-axis cho từng task để dễ nhìn
#     x_offset = np.array(task_data['round_in_task']) + (task_id * 25)
    
#     # Vẽ std band
#     ax2.fill_between(x_offset, 
#                       mean_acc - std_acc, mean_acc + std_acc,
#                       alpha=0.15, color=colors[task_id])
    
#     # Vẽ mean line (nét rõ)
#     ax2.plot(x_offset, [mean_acc]*len(task_data), 
#              linewidth=2.5, color=colors[task_id], 
#              label=f'Task {task_id} (mean: {mean_acc:.2f})', zorder=3)
    
#     # Vẽ chấm ở giá trị thật
#     ax2.scatter(x_offset, task_data['acc_curr'], 
#                 s=40, color=colors[task_id], alpha=0.7, 
#                 edgecolors='black', linewidth=0.3, zorder=4)

# ax2.set_xlabel('Round Global', fontsize=12, fontweight='bold')
# ax2.set_ylabel('Accuracy (acc_curr)', fontsize=12, fontweight='bold')
# ax2.set_title('All Tasks: Accuracy vs Round Global', fontsize=14, fontweight='bold')
# ax2.legend(loc='best', fontsize=10, ncol=5)
# ax2.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.savefig('acc_curr_block4_all_tasks.png', dpi=300, bbox_inches='tight')
# print("✓ Plot saved as 'acc_curr_block4_all_tasks.png'")
# plt.show()

# # In statistics
# print("\n" + "="*60)
# print("STATISTICS BY TASK")
# print("="*60)
# for task_id in sorted(df['task'].unique()):
#     task_data = df[df['task'] == task_id]
#     print(f"\nTask {task_id}:")
#     print(f"  Mean Accuracy: {task_data['acc_curr'].mean():.4f}")
#     print(f"  Std Accuracy:  {task_data['acc_curr'].std():.4f}")
#     print(f"  Min Accuracy:  {task_data['acc_curr'].min():.4f}")
#     print(f"  Max Accuracy:  {task_data['acc_curr'].max():.4f}")
"""
plot_pairs_combined.py — Vẽ tất cả pairs nối tiếp nhau trên cùng 1 biểu đồ
Cách dùng:
    python plot_pairs_combined.py --file C:\Thu\FCL\block4_gap_eps.csv
    python plot_pairs_combined.py --file data.csv --out ./charts --dpi 150 --sigma 1.5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from scipy.ndimage import gaussian_filter1d

# ─── Palette ──────────────────────────────────────────────────────────────────
DARK_BG  = "#0d1117"
PANEL_BG = "#161b22"
GRID_COL = "#21262d"
TEXT_COL = "#e6edf3"
MUTED    = "#8b949e"

# Màu cho từng pair — tự động cycle nếu > 10 pairs
PAIR_COLORS = [
    "#58a6ff",  # blue
    "#f78166",  # coral
    "#56d364",  # green
    "#e3b341",  # gold
    "#bc8cff",  # purple
    "#39d3f0",  # cyan
    "#ff7b72",  # red-orange
    "#ffa657",  # orange
    "#79c0ff",  # light blue
    "#d2a8ff",  # lavender
    "#7ee787",  # mint
    "#ff9a9a",  # pink
]

FONT_MONO = "monospace"


def smooth(y, sigma=1.5):
    if len(y) < 4:
        return y.astype(float)
    return gaussian_filter1d(y.astype(float), sigma=sigma)


def pair_sort_key(p):
    parts = p.replace("pair_", "").split("_")
    a, b = int(parts[0]), int(parts[1])
    # Nhóm theo task mới học (b), trong nhóm sort theo task cũ (a)
    # → 0-1 | 0-2,1-2 | 0-3,1-3,2-3 | 0-4,1-4,2-4,3-4 | ...
    return (b, a)


def main():
    parser = argparse.ArgumentParser(
        description="Vẽ tất cả pairs nối tiếp trên cùng 1 biểu đồ")
    parser.add_argument("--file", required=True,
                        help="Đường dẫn tới file CSV")
    parser.add_argument("--out", default=None,
                        help="Thư mục lưu ảnh (mặc định: cùng thư mục file)")
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument("--sigma", type=float, default=1.5,
                        help="Độ mượt Gaussian (mặc định: 1.5)")
    parser.add_argument("--height", type=float, default=6.5,
                        help="Chiều cao figure (mặc định: 6.5)")
    args = parser.parse_args()

    fpath = Path(args.file)
    if not fpath.exists():
        print(f"[ERROR] Không tìm thấy file: {fpath}")
        sys.exit(1)

    df = pd.read_csv(fpath)
    required = {"pair", "round", "value"}
    if not required.issubset(df.columns):
        print(f"[ERROR] Cần các cột: {required}  —  Hiện có: {list(df.columns)}")
        sys.exit(1)

    pairs = sorted(df["pair"].unique(), key=pair_sort_key)
    n_pairs = len(pairs)
    n_rounds = df["round"].nunique()

    print(f"[INFO] File   : {fpath.name}")
    print(f"[INFO] Pairs  : {n_pairs}  |  Rounds/pair: {n_rounds}")

    # ── Tính x-offset cho mỗi pair ────────────────────────────────────────────
    # Mỗi pair chiếm n_rounds ô, cách nhau 1 ô (separator)
    GAP = 1  # số ô trống giữa các pair

    pair_info = {}   # pair_name -> {color, x_start, rounds, values, mean, std}
    x_cursor = 0

    for i, pname in enumerate(pairs):
        sub = df[df["pair"] == pname].sort_values("round")
        rounds = sub["round"].values
        values = sub["value"].values.astype(float)
        n = len(rounds)

        pair_info[pname] = {
            "color":   PAIR_COLORS[i % len(PAIR_COLORS)],
            "x_start": x_cursor,
            "x_global": x_cursor + rounds,   # vị trí x tuyệt đối
            "rounds":  rounds,
            "values":  values,
            "mean":    np.mean(values),
            "std":     np.std(values, ddof=1),
            "n":       n,
        }
        x_cursor += n + GAP

    total_x = x_cursor - GAP  # bỏ gap cuối

    # ── Figure ────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "figure.facecolor": DARK_BG,
        "axes.facecolor":   PANEL_BG,
        "text.color":       TEXT_COL,
        "xtick.color":      MUTED,
        "ytick.color":      MUTED,
        "axes.edgecolor":   GRID_COL,
        "axes.labelcolor":  MUTED,
        "grid.color":       GRID_COL,
        "font.family":      FONT_MONO,
    })

    fig_w = min(max(16, n_pairs * n_rounds * 0.22), 40)
    fig, ax = plt.subplots(figsize=(fig_w, args.height), facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)

    # ── Vẽ từng pair ──────────────────────────────────────────────────────────
    for pname, info in pair_info.items():
        c       = info["color"]
        xg      = info["x_global"]
        vals    = info["values"]
        mn      = info["mean"]
        sd      = info["std"]
        x_start = info["x_start"]
        n       = info["n"]

        # --- Vùng nền phân tách pair (alternating shade)
        pair_idx = pairs.index(pname)
        shade_alpha = 0.04 if pair_idx % 2 == 0 else 0.09
        ax.axvspan(x_start - 0.5, x_start + n - 0.5,
                   color=c, alpha=shade_alpha, linewidth=0, zorder=1)

        # --- Đường kẻ dọc phân cách (trừ pair đầu)
        if pair_idx > 0:
            ax.axvline(x_start - 0.5, color=GRID_COL,
                       linewidth=1.2, linestyle="-", alpha=0.8, zorder=2)

        # --- Băng ±std theo mean
        ax.fill_between(xg, mn - sd, mn + sd,
                        color=c, alpha=0.12, linewidth=0, zorder=3)

        # --- Đường mean nằm ngang trong phạm vi pair
        ax.hlines(mn, x_start - 0.3, x_start + n - 0.7,
                  colors=c, linewidth=1.1, linestyle="--",
                  alpha=0.55, zorder=4)

        # --- Đường smooth
        vals_sm = smooth(vals, args.sigma)
        ax.plot(xg, vals_sm, color=c, linewidth=2.0, alpha=0.95, zorder=5)

        # --- Fill smooth ±std
        ax.fill_between(xg, vals_sm - sd, vals_sm + sd,
                        color=c, alpha=0.15, linewidth=0, zorder=4)

        # --- Điểm thật
        ax.scatter(xg, vals, color=c, s=18, zorder=6,
                   edgecolors=DARK_BG, linewidths=0.4, alpha=0.9)

        # (stat box vẽ sau khi ylim đã ổn định)

    # ── X-ticks: hiện round number trong mỗi pair ─────────────────────────────
    tick_positions = []
    tick_labels    = []
    for pname, info in pair_info.items():
        for r, xp in zip(info["rounds"], info["x_global"]):
            tick_positions.append(xp)
            tick_labels.append(str(r))

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=5.5, rotation=70,
                       color=MUTED, fontfamily=FONT_MONO)
    ax.set_xlim(-1, total_x + 1)

    # ── Nhãn pair phía trên (dùng secondary x-axis trick) ─────────────────────
    ax2 = ax.twiny()
    ax2.set_facecolor("none")
    ax2.spines[:].set_visible(False)
    ax2.set_xlim(ax.get_xlim())

    pair_centers = []
    pair_labels  = []
    for pname, info in pair_info.items():
        xc = info["x_start"] + (info["n"] - 1) / 2
        pair_centers.append(xc)
        pair_labels.append(pname.replace("pair_", "").replace("_", " → "))

    ax2.set_xticks(pair_centers)
    ax2.set_xticklabels(pair_labels, fontsize=8.5, fontweight="bold",
                        fontfamily=FONT_MONO)
    for tick, pname in zip(ax2.get_xticklabels(), pairs):
        tick.set_color(pair_info[pname]["color"])

    # ── Grid & styling ─────────────────────────────────────────────────────────
    ax.grid(axis="y", color=GRID_COL, linewidth=0.5, linestyle="-", alpha=0.7)
    ax.grid(axis="x", visible=False)
    ax.spines[:].set_color(GRID_COL)
    ax.set_ylabel("value", fontsize=9, color=MUTED, fontfamily=FONT_MONO)
    ax.set_xlabel("round  (per pair)", fontsize=8, color=MUTED,
                  fontfamily=FONT_MONO, labelpad=4)

    # ── Stat box trong từng pair — góc trên phải, tọa độ data thật ────────────
    ylo, yhi = ax.get_ylim()
    yrange   = yhi - ylo

    for pname, info in pair_info.items():
        c      = info["color"]
        xs     = info["x_start"]
        n      = info["n"]
        mn     = info["mean"]
        sd     = info["std"]

        x_box = xs + n - 0.6                    # sát cạnh phải của pair
        y_box = yhi - yrange * 0.01             # sát mép trên

        stat_txt = (
            f"μ = {mn:+.3f}\n"
            f"σ =  {sd:.3f}"
        )
        ax.text(
            x_box, y_box, stat_txt,
            va="top", ha="right",
            fontsize=6.2, fontfamily=FONT_MONO,
            color=TEXT_COL,
            bbox=dict(boxstyle="round,pad=0.38", facecolor=DARK_BG,
                      edgecolor=c, alpha=0.92, linewidth=0.9),
            zorder=10,
        )

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.suptitle(
        f"  {fpath.stem}  —  all pairs combined",
        fontsize=12, fontweight="bold",
        color=TEXT_COL, fontfamily=FONT_MONO,
        x=0.01, ha="left", y=0.995,
    )

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.subplots_adjust(left=0.055, right=0.985, top=0.91, bottom=0.13)

    # ── Lưu ──────────────────────────────────────────────────────────────────
    out_dir = Path(args.out) if args.out else fpath.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{fpath.stem}_combined.png"
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight", facecolor=DARK_BG)
    print(f"\n[DONE] Đã lưu → {out_path}")

    # In bảng tóm tắt
    print(f"\n{'pair':<14} {'mean':>9} {'std':>8} {'min':>9} {'max':>9}")
    print("─" * 52)
    for pname, info in pair_info.items():
        v = info["values"]
        print(f"{pname:<14} {info['mean']:>+9.4f} {info['std']:>8.4f} "
              f"{v.min():>+9.4f} {v.max():>+9.4f}")

    plt.show()


if __name__ == "__main__":
    main()