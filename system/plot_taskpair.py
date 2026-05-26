# """
# Vẽ biểu đồ eps vs acc theo taskpair và block chỉ định.

# Cấu trúc thư mục:
#   <base_dir>\block<N>\taskpair_X_Y\<filename>.csv

# Mỗi file CSV có định dạng 2 cột: x, y (ví dụ: eps, acc)
# Hoặc 3 cột: round, eps, acc

# Cách chạy:
#   python ploteps.py --block 4 --file eps_vs_accold.csv --taskpairs taskpair_0_1 taskpair_0_2 --plot 1
#   python ploteps.py --block 4 --file eps_vs_accold.csv --taskpairs all --plot 6
#   python ploteps.py --block 4 --file eps_vs_accold.csv --taskpairs taskpair_0_1 --plot all
#   python ploteps.py --block 4 --file eps_vs_accold.csv --taskpairs all --plot 1 --save

# Plot options:
#   1 – Scatter eps vs acc, màu theo round (colorbar)
#   2 – Scatter round vs eps, màu theo acc
#   3 – Line: round → eps & acc (2 đường)
#   4 – 3D scatter: eps / round / acc  (interactive)
#   5 – Heatmap 1 dòng theo round
#   6 – Scatter eps vs acc + regression line
#   all – xuất tất cả (trừ 4) thành file ảnh
# """

# import argparse
# import os
# import glob
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# # ─────────────────────────────────────────────
# # Config
# # ─────────────────────────────────────────────
# DEFAULT_BASE_DIR = r"C:\Thu\representation_drift_temporal_25_4-hetero-ResNet18.csv"

# COLORS = [
#     "#534AB7", "#0F6E56", "#993C1D", "#BA7517",
#     "#185FA5", "#C2185B", "#00838F", "#6D4C41",
#     "#558B2F", "#4527A0"
# ]
# MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h"]


# # ─────────────────────────────────────────────
# # Load data
# # ─────────────────────────────────────────────
# def load_taskpair_data(base_dir: str, block: int, taskpairs: list, filename: str):
#     """
#     Load dữ liệu từ nhiều taskpair.
#     Hỗ trợ:
#       - 2 cột: (eps, acc)         => round tự sinh 0,1,2,...
#       - 3 cột: (round, eps, acc)  => dùng trực tiếp
#     Trả về dict: { "taskpair_X_Y": pd.DataFrame(round, eps, acc) }
#     """
#     block_dir = os.path.join(base_dir, f"block{block}")

#     if not os.path.isdir(block_dir):
#         raise FileNotFoundError(f"Không tìm thấy thư mục: {block_dir}")

#     if taskpairs == ["all"]:
#         found = sorted(glob.glob(os.path.join(block_dir, "taskpair_*")))
#         taskpairs = [os.path.basename(p) for p in found if os.path.isdir(p)]
#         if not taskpairs:
#             raise FileNotFoundError(f"Không có taskpair nào trong: {block_dir}")
#         print(f"[AUTO] {len(taskpairs)} taskpair: {taskpairs}")

#     data = {}
#     for tp in taskpairs:
#         csv_path = os.path.join(block_dir, tp, filename)
#         if not os.path.isfile(csv_path):
#             print(f"[WARN] Không tìm thấy: {csv_path} — bỏ qua")
#             continue
#         try:
#             df = pd.read_csv(csv_path)
#             df.columns = df.columns.str.strip()
#             cols = df.columns.tolist()

#             if len(cols) == 2:
#                 df = df.rename(columns={cols[0]: "eps", cols[1]: "acc"})
#                 df["round"] = range(len(df))
#             elif len(cols) >= 3:
#                 df = df.rename(columns={cols[0]: "round", cols[1]: "eps", cols[2]: "acc"})
#             else:
#                 print(f"[WARN] {csv_path} có ít hơn 2 cột — bỏ qua")
#                 continue

#             df["round"] = pd.to_numeric(df["round"], errors="coerce")
#             df["eps"]   = pd.to_numeric(df["eps"],   errors="coerce")
#             df["acc"]   = pd.to_numeric(df["acc"],   errors="coerce")
#             df = df.dropna(subset=["eps", "acc"]).sort_values("round").reset_index(drop=True)

#             data[tp] = df
#             print(f"[OK] {tp}: {len(df)} điểm")

#         except Exception as e:
#             print(f"[ERR] {csv_path}: {e}")

#     if not data:
#         raise ValueError("Không load được dữ liệu nào.")

#     return data


# # ─────────────────────────────────────────────
# # Helpers
# # ─────────────────────────────────────────────
# def _label(tp: str) -> str:
#     return tp.replace("taskpair_", "Pair ").replace("_", "→")

# def _title(block: int, tp: str, suffix: str = "") -> str:
#     return f"Block {block} – {_label(tp)}" + (f"  |  {suffix}" if suffix else "")


# # ─────────────────────────────────────────────
# # Plot 1 – Scatter eps vs acc, màu theo round
# # ─────────────────────────────────────────────
# def plot1(data: dict, block: int):
#     for tp, df in data.items():
#         fig, ax = plt.subplots(figsize=(7, 5))

#         sc = ax.scatter(df["eps"], df["acc"],
#                         c=df["round"], cmap="plasma",
#                         s=60, alpha=0.85)
#         plt.colorbar(sc, ax=ax, label="Round")

#         ax.set_xlabel("eps")
#         ax.set_ylabel("acc")
#         ax.set_title(_title(block, tp, "eps vs acc"))
#         ax.grid(True, alpha=0.3)
#         plt.tight_layout()


# # ─────────────────────────────────────────────
# # Plot 2 – Scatter round vs eps, màu theo acc
# # ─────────────────────────────────────────────
# def plot2(data: dict, block: int):
#     for tp, df in data.items():
#         fig, ax = plt.subplots(figsize=(7, 5))

#         sc = ax.scatter(df["round"], df["eps"],
#                         c=df["acc"], cmap="viridis",
#                         s=60)
#         plt.colorbar(sc, ax=ax, label="acc")

#         ax.set_xlabel("Round")
#         ax.set_ylabel("eps")
#         ax.set_title(_title(block, tp, "round vs eps"))
#         ax.grid(True, alpha=0.3)
#         plt.tight_layout()


# # ─────────────────────────────────────────────
# # Plot 3 – Line: round → eps & acc
# # ─────────────────────────────────────────────
# def plot3(data: dict, block: int):
#     for i, (tp, df) in enumerate(data.items()):
#         fig, ax = plt.subplots(figsize=(8, 5))

#         ax.plot(df["round"], df["eps"],
#                 marker="o", color=COLORS[i % len(COLORS)],
#                 label="eps", linewidth=1.8)
#         ax.plot(df["round"], df["acc"],
#                 marker="s", color=COLORS[(i + 1) % len(COLORS)],
#                 label="acc", linewidth=1.8, linestyle="--")

#         ax.set_xlabel("Round")
#         ax.set_title(_title(block, tp, "eps & acc over rounds"))
#         ax.legend()
#         ax.grid(True, alpha=0.3)
#         plt.tight_layout()


# # ─────────────────────────────────────────────
# # Plot 4 – 3D scatter (interactive)
# # ─────────────────────────────────────────────
# def plot4(data: dict, block: int):
#     for tp, df in data.items():
#         fig = plt.figure(figsize=(9, 7))
#         ax  = fig.add_subplot(111, projection="3d")

#         ax.scatter(df["eps"], df["round"], df["acc"],
#                    c=df["round"], cmap="plasma", s=60)

#         ax.set_xlabel("eps")
#         ax.set_ylabel("Round")
#         ax.set_zlabel("acc")
#         ax.set_title(_title(block, tp, "3D"))
#         plt.tight_layout()

#     print("[TIP] Kéo chuột để xoay 3D")


# # ─────────────────────────────────────────────
# # Plot 5 – Heatmap 1 dòng theo round
# # ─────────────────────────────────────────────
# def plot5(data: dict, block: int, hue: str = "eps"):
#     for tp, df in data.items():
#         fig, ax = plt.subplots(figsize=(10, 2))

#         values = df[hue].values.reshape(1, -1)
#         im = ax.imshow(values, aspect="auto", cmap="YlOrRd")
#         plt.colorbar(im, ax=ax, label=hue)

#         ax.set_xticks(range(len(df)))
#         ax.set_xticklabels(df["round"].astype(int), fontsize=8)
#         ax.set_yticks([])
#         ax.set_xlabel("Round")
#         ax.set_title(_title(block, tp, f"{hue} heatmap"))
#         plt.tight_layout()


# # ─────────────────────────────────────────────
# # Plot 6 – Scatter eps vs acc + regression
# # ─────────────────────────────────────────────
# def plot6(data: dict, block: int):
#     for tp, df in data.items():
#         fig, ax = plt.subplots(figsize=(7, 5))

#         sc = ax.scatter(df["eps"], df["acc"],
#                         c=df["round"], cmap="plasma",
#                         s=60, alpha=0.85)
#         plt.colorbar(sc, ax=ax, label="Round")

#         if len(df) > 1:
#             z  = np.polyfit(df["eps"], df["acc"], 1)
#             p  = np.poly1d(z)
#             xs = np.linspace(df["eps"].min(), df["eps"].max(), 100)
#             ax.plot(xs, p(xs), color="red", linewidth=2,
#                     label=f"slope={z[0]:.4f}")
#             ax.legend()
#             print(f"[{tp}] slope = {z[0]:.4f}")

#         ax.set_xlabel("eps")
#         ax.set_ylabel("acc")
#         ax.set_title(_title(block, tp, "eps vs acc + trend"))
#         ax.grid(True, alpha=0.3)
#         plt.tight_layout()


# # ─────────────────────────────────────────────
# # Main
# # ─────────────────────────────────────────────
# def main():
#     parser = argparse.ArgumentParser(
#         description="Vẽ eps vs acc theo taskpair",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog=__doc__
#     )
#     parser.add_argument("--block", type=int, required=True,
#                         help="Block cần vẽ, ví dụ: 4")
#     parser.add_argument("--file", required=True,
#                         help="Tên file CSV, ví dụ: eps_vs_accold.csv")
#     parser.add_argument("--taskpairs", nargs="+", default=["all"],
#                         help="Taskpair: taskpair_0_1 taskpair_1_2 ... hoặc all")
#     parser.add_argument("--plot", default="1",
#                         help="Plot option: 1/2/3/4/5/6/all (mặc định: 1)")
#     parser.add_argument("--hue", default="eps", choices=["eps", "acc"],
#                         help="Biến heatmap plot 5 (mặc định: eps)")
#     parser.add_argument("--save", action="store_true",
#                         help="Lưu ảnh PNG thay vì hiện interactive")
#     parser.add_argument("--base", default=DEFAULT_BASE_DIR,
#                         help=f"Thư mục gốc (mặc định: {DEFAULT_BASE_DIR})")
#     args = parser.parse_args()

#     data = load_taskpair_data(
#         base_dir=args.base,
#         block=args.block,
#         taskpairs=args.taskpairs,
#         filename=args.file
#     )
#     print(f"\n[INFO] block={args.block} | file='{args.file}' | "
#           f"{len(data)} taskpair | plot={args.plot}\n")

#     choice = args.plot.lower()

#     def _save(suffix):
#         if args.save:
#             fname = f"block{args.block}_plot{suffix}.png"
#             plt.savefig(fname, dpi=150, bbox_inches="tight")
#             print(f"[SAVE] {fname}")
#             plt.close("all")

#     if choice in ("1", "all"):
#         plot1(data, args.block)
#         _save("1_scatter_eps_acc")

#     if choice in ("2", "all"):
#         plot2(data, args.block)
#         _save("2_scatter_round_eps")

#     if choice in ("3", "all"):
#         plot3(data, args.block)
#         _save("3_line_round")

#     if choice == "4":
#         plot4(data, args.block)

#     if choice in ("5", "all"):
#         plot5(data, args.block, hue=args.hue)
#         _save(f"5_heatmap_{args.hue}")

#     if choice in ("6", "all"):
#         plot6(data, args.block)
#         _save("6_regression")

#     if not args.save:
#         plt.show()


# if __name__ == "__main__":
#     main()

# #!/usr/bin/env python3
# """
# Representation Drift Visualization Script
# Usage:
#     python plot_drift.py --input PATH --method eps_old --blocks 0 1 2 3 4
#     python plot_drift.py --input PATH --method cka_curr --blocks 0 2 4
#     python plot_drift.py --input PATH --method cosine_similarity
# """

# import argparse
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# from itertools import combinations
# import sys
# import os

# # ── Friendly display names for each metric column ──────────────────────────
# METRIC_LABELS = {
#     "sigma_old":          "σ (old)",
#     "eps_old":            "ε (old)",
#     "cka_old":            "CKA (old)",
#     "linear_cka_old":     "Linear CKA (old)",
#     "kernel_cka_old":     "Kernel CKA (old)",
#     "cka_curr":           "CKA (curr)",
#     "linear_cka_curr":    "Linear CKA (curr)",
#     "kernel_cka_curr":    "Kernel CKA (curr)",
#     "old_test_acc":       "Old Test Acc",
#     "current_test_acc":   "Current Test Acc",
#     "acc_t_on_head":      "Acc(t) on Head",
#     "forgetting_drop":    "Forgetting Drop",
#     "cosine_similarity":  "Cosine Similarity",
#     "align150":           "Align@150",
#     "drift_neuron":       "Drift (neuron)",
#     "cosine_neuron":      "Cosine (neuron)",
#     "overlap_at50":       "Overlap@50",
#     "drift_per_acc_unit": "Drift per Acc Unit",
# }


# def parse_args():
#     parser = argparse.ArgumentParser(description="Plot representation drift metrics.")
#     parser.add_argument(
#         "--input", "-i",
#         default=r"C:\Thu\representation_drift_temporal_25_4-hetero-ResNet18.csv",
#         help="Path to the CSV file"
#     )
#     parser.add_argument(
#         "--method", "-m",
#         default="cosine_similarity",
#         choices=list(METRIC_LABELS.keys()),
#         help="Which metric column to plot"
#     )
#     parser.add_argument(
#         "--blocks", "-b",
#         nargs="+",
#         type=int,
#         default=None,
#         help="Block indices to include (default: all blocks in file)"
#     )
#     parser.add_argument(
#         "--clients", "-c",
#         nargs="+",
#         type=int,
#         default=None,
#         help="Client indices to include (default: all clients)"
#     )
#     parser.add_argument(
#         "--output", "-o",
#         default=None,
#         help="Output file path (e.g. plot.png). If omitted, shows interactive window."
#     )
#     parser.add_argument(
#         "--figsize",
#         nargs=2,
#         type=float,
#         default=[14, 7],
#         metavar=("WIDTH", "HEIGHT"),
#         help="Figure size in inches (default: 14 7)"
#     )
#     return parser.parse_args()


# def load_data(path):
#     """Load CSV with flexible separator detection."""
#     # Try tab, then comma
#     for sep in ["\t", ","]:
#         try:
#             df = pd.read_csv(path, sep=sep)
#             if len(df.columns) > 3:
#                 return df
#         except Exception:
#             continue
#     raise ValueError(f"Cannot parse file: {path}")


# def build_task_pairs(blocks):
#     """Return ordered list of (t, tprime) pairs in temporal order."""
#     # t < tprime, ordered by tprime then t
#     pairs = []
#     for t in blocks:
#         for tp in blocks:
#             if tp > t:
#                 pairs.append((t, tp))
#     # Sort: first by gap (tprime - t), then by t → chronological reading
#     pairs.sort(key=lambda x: (x[1], x[0]))
#     return pairs


# def get_pair_data(df, t, tprime, method, clients):
#     """Return (rounds, values_per_client) for a given task pair across clients."""
#     mask = (df["t"] == t) & (df["tprime"] == tprime)
#     if clients is not None:
#         mask &= df["client"].isin(clients)
#     sub = df[mask].copy()

#     # Group by block (round index) and aggregate over clients
#     grouped = sub.groupby("block")[method]
#     rounds = sorted(sub["block"].unique())
#     means = [grouped.get_group(r).mean() for r in rounds]
#     stds  = [grouped.get_group(r).std(ddof=0) for r in rounds]
#     raws  = [grouped.get_group(r).values      for r in rounds]
#     return rounds, np.array(means), np.array(stds), raws


# def main():
#     args = parse_args()

#     # ── Load ──────────────────────────────────────────────────────────────
#     if not os.path.exists(args.input):
#         sys.exit(f"[ERROR] File not found: {args.input}")

#     df = load_data(args.input)
#     df.columns = df.columns.str.strip()

#     # Validate method
#     if args.method not in df.columns:
#         sys.exit(f"[ERROR] Column '{args.method}' not found. Available: {list(df.columns)}")

#     # Filter blocks / clients
#     all_blocks  = sorted(df["block"].unique())
#     all_clients = sorted(df["client"].unique())
#     blocks  = args.blocks  if args.blocks  is not None else all_blocks
#     clients = args.clients if args.clients is not None else None  # None = all

#     # ── Build task pairs ──────────────────────────────────────────────────
#     pairs = build_task_pairs(blocks)
#     if not pairs:
#         sys.exit("[ERROR] Need at least 2 distinct block indices to form task pairs.")

#     # ── Color palette ─────────────────────────────────────────────────────
#     cmap   = cm.get_cmap("tab10", len(pairs))
#     colors = [cmap(i) for i in range(len(pairs))]

#     # ── X-axis: sequential positions with pair separators ────────────────
#     # Each pair occupies N_rounds x-slots; we add a small gap between pairs.
#     GAP = 0.8  # fractional gap between pairs

#     fig, ax = plt.subplots(figsize=tuple(args.figsize))
#     ax.set_facecolor("#0f1117")
#     fig.patch.set_facecolor("#0f1117")

#     x_cursor = 0
#     pair_xtick_centers = []   # label tick positions
#     pair_xtick_labels  = []

#     for idx, (t, tp) in enumerate(pairs):
#         rounds, means, stds, raws = get_pair_data(df, t, tp, args.method, clients)
#         if len(rounds) == 0:
#             continue

#         color = colors[idx]
#         xs = [x_cursor + r_i for r_i, _ in enumerate(rounds)]

#         # ── Scatter raw data points ───────────────────────────────────────
#         for xi, raw_vals in zip(xs, raws):
#             ax.scatter(
#                 [xi] * len(raw_vals), raw_vals,
#                 color=color, alpha=0.35, s=18, zorder=3, linewidths=0
#             )

#         # ── Mean line ────────────────────────────────────────────────────
#         ax.plot(
#             xs, means,
#             color=color, linewidth=2.2, zorder=5,
#             label=f"({t}→{tp})"
#         )

#         # ── Std band ─────────────────────────────────────────────────────
#         ax.fill_between(
#             xs,
#             means - stds,
#             means + stds,
#             color=color, alpha=0.18, zorder=2
#         )

#         # ── Tick: center of this pair's segment ──────────────────────────
#         pair_xtick_centers.append(np.mean(xs))
#         pair_xtick_labels.append(f"t={t}\nt'={tp}")

#         # Vertical separator before next pair
#         if idx < len(pairs) - 1:
#             x_cursor += len(rounds) + GAP
#         else:
#             x_cursor += len(rounds)

#     # ── Axes styling ──────────────────────────────────────────────────────
#     metric_name = METRIC_LABELS.get(args.method, args.method)
#     ax.set_title(
#         f"Representation Drift  ·  {metric_name}",
#         color="white", fontsize=15, fontweight="bold", pad=14
#     )
#     ax.set_ylabel(metric_name, color="#c8d0e0", fontsize=11)
#     ax.set_xlabel("Task pairs  (each segment = rounds within that pair)", color="#c8d0e0", fontsize=10)

#     ax.set_xticks(pair_xtick_centers)
#     ax.set_xticklabels(pair_xtick_labels, color="#a0a8b8", fontsize=8.5)
#     ax.tick_params(axis="y", colors="#a0a8b8")

#     for spine in ax.spines.values():
#         spine.set_edgecolor("#2a2f3f")
#     ax.yaxis.grid(True, color="#1e2335", linewidth=0.8, zorder=0)
#     ax.set_axisbelow(True)

#     # Pair boundary lines
#     x_cursor2 = 0
#     for idx, (t, tp) in enumerate(pairs):
#         rounds, *_ = get_pair_data(df, t, tp, args.method, clients)
#         if len(rounds) == 0:
#             continue
#         x_cursor2 += len(rounds)
#         if idx < len(pairs) - 1:
#             ax.axvline(x_cursor2 + GAP / 2, color="#2a2f4a", linewidth=1.2, linestyle="--", zorder=1)
#             x_cursor2 += GAP

#     legend = ax.legend(
#         title="Task pair (t→t')",
#         title_fontsize=9,
#         fontsize=8.5,
#         loc="upper right",
#         framealpha=0.25,
#         edgecolor="#2a2f4a",
#         labelcolor="white",
#     )
#     legend.get_title().set_color("#c8d0e0")

#     plt.tight_layout()

#     # ── Save or show ──────────────────────────────────────────────────────
#     if args.output:
#         plt.savefig(args.output, dpi=150, bbox_inches="tight",
#                     facecolor=fig.get_facecolor())
#         print(f"[✓] Saved → {args.output}")
#     else:
#         plt.show()


# if __name__ == "__main__":
#     main()
"""
3D Scatter Plot: eps_curr / eps_old / forgetting_drop
- eps: chỉ vẽ 5 round đầu + EMA smoothing + Quiver arrows
- forgetting: vẽ đủ tất cả round (để thấy giá trị cuối)
- Giả thuyết: 5 round đầu của eps dự đoán forgetting cuối

Dùng lệnh:
  python plot_3d_perpair_interactive.py --curr true --old true
  python plot_3d_perpair_interactive.py --curr true
  python plot_3d_perpair_interactive.py --old true
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.lines import Line2D

# ── Argparse ──────────────────────────────────────────────────────────────────
def str2bool(v):
    return v.lower() in ("true", "1", "yes")

parser = argparse.ArgumentParser()
parser.add_argument("--curr", type=str2bool, default=False, metavar="true/false")
parser.add_argument("--old",  type=str2bool, default=False, metavar="true/false")
parser.add_argument("--early_rounds", type=int, default=25,
                    help="Số round đầu để vẽ eps (default: 25)")
parser.add_argument("--ema_alpha", type=float, default=0.4,
                    help="EMA smoothing factor 0-1, nhỏ=mượt hơn (default: 0.4)")
args = parser.parse_args()

if not args.curr and not args.old:
    parser.error("Phải chỉ định ít nhất --curr true hoặc --old true")

N_EARLY   = args.early_rounds
EMA_ALPHA = args.ema_alpha
print(f"[CFG] eps_curr={args.curr}  eps_old={args.old}  early_rounds={N_EARLY}  ema_alpha={EMA_ALPHA}")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE  = r"C:\Thu\FCL"
BLOCK = "block4"

PATH_CURR = os.path.join(BASE, f"{BLOCK}_eps_curr.csv")
PATH_OLD  = os.path.join(BASE, f"{BLOCK}_eps_old.csv")
PATH_FORG = os.path.join(BASE, f"{BLOCK}_forgetting_drop.csv")

# ── Style ─────────────────────────────────────────────────────────────────────
ROUND_CMAP   = cm.get_cmap("plasma")
TASK_COLORS  = [cm.get_cmap("tab10")(i) for i in range(10)]
S_MIN, S_MAX = 60, 220

# Marker phân biệt rõ ràng giữa curr và old
MARKER_CURR = "^"   # tam giác lên — dễ nhận
MARKER_OLD  = "s"   # hình vuông — khác hẳn

# ── Load ──────────────────────────────────────────────────────────────────────
def load(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    return df

curr_df = load(PATH_CURR)
old_df  = load(PATH_OLD)
forg_df = load(PATH_FORG)

pairs = sorted(curr_df["pair"].unique())
print(f"Tìm thấy {len(pairs)} task pair: {pairs}")

forg_min = forg_df["value"].min()
forg_max = forg_df["value"].max()

def forg_to_size(f_vals):
    if forg_max == forg_min:
        return np.full(len(f_vals), (S_MIN + S_MAX) / 2)
    return S_MIN + (f_vals - forg_min) / (forg_max - forg_min) * (S_MAX - S_MIN)

def ema(series, alpha=EMA_ALPHA):
    """Exponential Moving Average."""
    result = np.zeros(len(series))
    result[0] = series.iloc[0]
    for i in range(1, len(series)):
        result[i] = alpha * series.iloc[i] + (1 - alpha) * result[i - 1]
    return result

def get_pair_data(pair):
    c = curr_df[curr_df["pair"] == pair][["round","value"]].rename(columns={"value":"eps_curr"})
    o = old_df [old_df ["pair"] == pair][["round","value"]].rename(columns={"value":"eps_old"})
    f = forg_df[forg_df["pair"] == pair][["round","value"]].rename(columns={"value":"forgetting"})
    return c.merge(o, on="round").merge(f, on="round").sort_values("round").reset_index(drop=True)

def round_colors(rounds):
    r = np.array(rounds, dtype=float)
    rmin, rmax = r.min(), r.max()
    norm = (r - rmin) / (rmax - rmin) if rmax > rmin else np.full(len(r), 0.5)
    return [ROUND_CMAP(v) for v in norm]

# ── Vẽ từng pair ──────────────────────────────────────────────────────────────
for idx, pair in enumerate(pairs):
    data = get_pair_data(pair)
    if data.empty:
        print(f"[SKIP] {pair}: không có dữ liệu")
        continue

    # ── Tách dữ liệu ─────────────────────────────────────────────────────────
    early = data[data["round"] < N_EARLY].copy().reset_index(drop=True)
    full  = data.copy()

    if len(early) == 0:
        print(f"[SKIP] {pair}: không có early rounds")
        continue

    early["eps_curr_ema"] = ema(early["eps_curr"])
    early["eps_old_ema"]  = ema(early["eps_old"])

    sizes_early = forg_to_size(early["forgetting"].values)
    sizes_full  = forg_to_size(full["forgetting"].values)
    colors_early = round_colors(early["round"].values)
    colors_full  = round_colors(full["round"].values)

    fig = plt.figure(figsize=(12, 8))
    fig.canvas.manager.set_window_title(f"{BLOCK} | {pair}")
    ax = fig.add_subplot(111, projection="3d")

    legend_elems = []

    # ════════════════════════════════════════════════════════════════════════
    # 1. FORGETTING — tất cả round (cloud điểm mờ, marker tròn nhỏ)
    #    Chiếu tại Y = mean(eps) của pair, chỉ vẽ 1 lần duy nhất mỗi round
    # ════════════════════════════════════════════════════════════════════════
    eps_ref = (
        (data["eps_curr"].mean() if args.curr else 0) +
        (data["eps_old"].mean()  if args.old  else 0)
    ) / max(int(args.curr) + int(args.old), 1)

    ax.scatter(
        full["round"].values,
        np.full(len(full), eps_ref),
        full["forgetting"].values,
        c=colors_full,
        marker="o", s=sizes_full * 0.5,
        alpha=0.30, linewidths=0,
        zorder=1
    )
    legend_elems.append(
        Line2D([0],[0], marker="o", color="gray", markersize=7,
               alpha=0.4, linestyle="None",
               label="forgetting (all rounds, Y=ε̄)"))

    # ════════════════════════════════════════════════════════════════════════
    # 2. EPS — chỉ EMA points + quiver, KHÔNG vẽ raw scatter riêng
    # ════════════════════════════════════════════════════════════════════════
    def draw_eps_ema(eps_ema_col, marker, edge_col, edge_lw,
                     point_alpha, quiver_color_override, label_ema):
        """
        Chỉ vẽ EMA points (1 điểm / round) + quiver arrows.
        Không vẽ thêm raw scatter để tránh điểm thừa.
        """
        # ── EMA scatter (1 điểm / round, rõ nét) ────────────────────────
        ax.scatter(
            early["round"].values,
            early[eps_ema_col].values,
            early["forgetting"].values,
            c=colors_early,
            marker=marker,
            s=sizes_early,
            alpha=point_alpha,
            edgecolors=edge_col,
            linewidths=edge_lw,
            zorder=5
        )

        # ── Quiver arrows: t → t+1 ───────────────────────────────────────
        for i in range(len(early) - 1):
            dx = early["round"].iloc[i+1]      - early["round"].iloc[i]
            dy = early[eps_ema_col].iloc[i+1]  - early[eps_ema_col].iloc[i]
            dz = early["forgetting"].iloc[i+1] - early["forgetting"].iloc[i]
            col = quiver_color_override if quiver_color_override else colors_early[i]
            ax.quiver(
                early["round"].iloc[i],
                early[eps_ema_col].iloc[i],
                early["forgetting"].iloc[i],
                dx, dy, dz,
                color=col, alpha=0.75,
                arrow_length_ratio=0.3, linewidth=1.4
            )

        # ── START marker (star lớn) ───────────────────────────────────────
        ax.scatter(
            early["round"].iloc[0],
            early[eps_ema_col].iloc[0],
            early["forgetting"].iloc[0],
            color=colors_early[0], marker="*", s=380,
            edgecolors="black", linewidths=1.2, zorder=10
        )

        legend_elems.append(
            Line2D([0],[0], marker=marker, color="w",
                   markerfacecolor=ROUND_CMAP(0.5),
                   markeredgecolor=edge_col,
                   markeredgewidth=edge_lw,
                   markersize=10, linestyle="None",
                   label=label_ema))

    if args.curr:
        draw_eps_ema(
            eps_ema_col="eps_curr_ema",
            marker=MARKER_CURR,          # ▲ tam giác lên
            edge_col="white",
            edge_lw=0.8,
            point_alpha=0.92,
            quiver_color_override=None,  # dùng màu theo round
            label_ema="eps_curr EMA  [▲ white-edge]"
        )

    if args.old:
        draw_eps_ema(
            eps_ema_col="eps_old_ema",
            marker=MARKER_OLD,           # ■ hình vuông
            edge_col="black",
            edge_lw=1.2,
            point_alpha=0.75,
            quiver_color_override=None,
            label_ema="eps_old  EMA  [■ black-edge]"
        )

    # ── Nối eps_curr ↔ eps_old tại cùng round (chỉ khi cả 2 bật) ────────────
    if args.curr and args.old:
        for _, row in early.iterrows():
            ax.plot(
                [row["round"],         row["round"]],
                [row["eps_old_ema"],   row["eps_curr_ema"]],
                [row["forgetting"],    row["forgetting"]],
                color="black", alpha=0.18, linewidth=0.7, linestyle=":"
            )
        legend_elems.append(
            Line2D([0],[0], color="black", alpha=0.3, linewidth=1,
                   linestyle=":", label="curr ↔ old (same round)"))

    # ── START legend entry ────────────────────────────────────────────────────
    legend_elems += [
        Line2D([0],[0], marker="*", color="w",
               markerfacecolor=ROUND_CMAP(0.0), markeredgecolor="black",
               markersize=14,
               label=f"START (round {early['round'].min()})"),
        Line2D([0],[0], marker=">", color="gray", markersize=8,
               linestyle="None",
               label="Quiver: direction t→t+1 (EMA)"),
    ]

    # ── Colorbar ──────────────────────────────────────────────────────────────
    sm = cm.ScalarMappable(cmap=ROUND_CMAP,
                           norm=mcolors.Normalize(vmin=data["round"].min(),
                                                  vmax=data["round"].max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.12, shrink=0.5, aspect=18)
    cbar.set_label("Round", fontsize=9)

    # ── Labels & Title ────────────────────────────────────────────────────────
    ax.set_xlabel("Round",           fontsize=11, labelpad=10)
    ax.set_ylabel("Epsilon (ε)",     fontsize=11, labelpad=10)
    ax.set_zlabel("Forgetting Drop", fontsize=11, labelpad=10)

    shown = " & ".join(filter(None,[
        "eps_curr" if args.curr else "",
        "eps_old"  if args.old  else ""
    ]))
    ax.set_title(
        f"Block {BLOCK}  —  {pair}\n"
        f"eps ({shown}): {N_EARLY} round đầu + EMA (α={EMA_ALPHA})  |  forgetting: all rounds\n"
        f"Color: tím (round đầu) → vàng (round cuối)",
        fontsize=10, fontweight="bold", pad=14
    )

    ax.legend(handles=legend_elems, loc="upper left", fontsize=8, framealpha=0.75)
    ax.text2D(0.01, 0.01,
              f"Giả thuyết: {N_EARLY} round đầu eps → dự đoán forgetting cuối\n"
              f"Point size ∝ forgetting ({forg_min:.3f}–{forg_max:.3f})",
              transform=ax.transAxes, fontsize=7, color="gray")

    plt.tight_layout()

print("[TIP] Kéo chuột để xoay. Mũi tên = hướng di chuyển eps theo thời gian (EMA).")
plt.show()