"""
plot_resnet_eps.py  —  v3
==========================
Vẽ 4 biểu đồ (2×2) cho 4 anchor task (0,1,2,3).
Mỗi ô: 5 đường rolling mean ± std band cho 5 ResNet block.

CÁCH DÙNG:
  Chỉ cần đặt đường dẫn file block 0 vào BLOCK0_PATH.
  Script tự suy ra block1 → block4.

  Ví dụ:
    BLOCK0_PATH = r"C:\Thu\FCL\Client0_block0_eps_old.csv"
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ─────────────────────────────────────────────
#  ĐẶT ĐƯỜNG DẪN FILE BLOCK 0 Ở ĐÂY
# ─────────────────────────────────────────────
BLOCK0_PATH = r"C:\Thu\FCL\Client0_block0_linear_cka.csv"

Y_LABEL         = "similarity"  # nhãn trục y (thay đổi nếu cần)
ROUNDS_PER_TASK = 25
NUM_TASKS       = 5
WINDOW          = 5   # rolling window (tăng để mượt hơn, giảm để giữ chi tiết)

# ─────────────────────────────────────────────
COLORS       = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#6a4c93"]
BLOCK_LABELS = [f"Block {i}" for i in range(NUM_TASKS)]


def get_block_paths(block0_path: str) -> list:
    return [re.sub(r"block\d+", f"block{i}", block0_path) for i in range(NUM_TASKS)]


def load_block(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[["anchor", "target"]] = (
        df["pair"].str.extract(r"pair_(\d+)_(\d+)").astype(int)
    )
    return df[["anchor", "target", "round", "value"]]


def load_all_blocks(paths: list) -> list:
    dfs = []
    for i, p in enumerate(paths):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Không tìm thấy file block {i}:\n  {p}\nKiểm tra lại BLOCK0_PATH."
            )
        dfs.append(load_block(p))
    return dfs


def plot_anchor(ax, anchor: int, dfs: list):
    target_tasks = list(range(anchor + 1, NUM_TASKS))
    n_seg = len(target_tasks)

    for block_idx, df in enumerate(dfs):
        df_anchor = df[df["anchor"] == anchor]
        x_all, y_all = [], []

        for seg_i, t in enumerate(target_tasks):
            df_pair = df_anchor[df_anchor["target"] == t].sort_values("round")
            if df_pair.empty:
                continue
            x_vals = seg_i * ROUNDS_PER_TASK + df_pair["round"].values
            y_vals = df_pair["value"].values
            x_all.extend(x_vals)
            y_all.extend(y_vals)

        if not x_all:
            continue

        order    = np.argsort(x_all)
        x_sorted = np.array(x_all)[order]
        y_sorted = np.array(y_all)[order]

        s      = pd.Series(y_sorted)
        y_mean = s.rolling(WINDOW, center=True, min_periods=1).mean().values
        y_std  = s.rolling(WINDOW, center=True, min_periods=1).std().fillna(0).values

        ax.plot(x_sorted, y_mean,
                color=COLORS[block_idx], linewidth=1.8,
                label=BLOCK_LABELS[block_idx], alpha=0.95, zorder=3)
        ax.fill_between(x_sorted,
                        y_mean - y_std,
                        y_mean + y_std,
                        color=COLORS[block_idx], alpha=0.20, zorder=2)

    # Đường kẻ dọc phân tách task
    for seg_i in range(1, n_seg):
        ax.axvline(x=seg_i * ROUNDS_PER_TASK, color="gray",
                   linewidth=0.8, linestyle="--", alpha=0.5)

    tick_pos = [seg_i * ROUNDS_PER_TASK for seg_i in range(n_seg)]
    tick_lbl = [f"T{t}" for t in target_tasks]
    # end_x    = (n_seg - 1) * ROUNDS_PER_TASK + (ROUNDS_PER_TASK - 1)
    # tick_pos.append(end_x)
    # tick_lbl.append(f"R{ROUNDS_PER_TASK - 1}")
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbl, fontsize=9)
    ax.set_xlim(-1, n_seg * ROUNDS_PER_TASK)

    ax.set_title(f"Anchor Task {anchor}", fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel("Task", fontsize=9)
    ax.set_ylabel(Y_LABEL, fontsize=9)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.7)


def main():
    paths = get_block_paths(BLOCK0_PATH)
    print("Đọc các file:")
    for i, p in enumerate(paths):
        print(f"  Block {i}: {p}")

    dfs = load_all_blocks(paths)
    print("Đọc xong.\n")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "Similarity Across Tasks by CKA",
        fontsize=13, fontweight="bold"
    )

    for idx, anchor in enumerate([0, 1, 2, 3]):
        ax = axes[idx // 2][idx % 2]
        plot_anchor(ax, anchor, dfs)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # giữ chỗ cho suptitle

    out_path = os.path.join(os.path.dirname(BLOCK0_PATH), "resnet_eps_plot.png")
    #plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Đã lưu: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()