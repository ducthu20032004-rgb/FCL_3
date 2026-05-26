# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import sys
# import os

# # ─── CONFIG ──────────────────────────────────────────────────────────────────
# CSV_FILE   = r"C:\Thu\FCL\material_experiment\dynamic\cka_on_old_data_block3.csv"   # <-- đổi đường dẫn file ở đây
# X_COL      = "round_global"           # <-- tên cột trục X
# Y_COL      = "block4/cka_on_old_data"      # <-- tên cột trục Y
# WINDOW     = 7                        # rolling window để tính mean ± std
# SHOW_RAW   = True                     # True = vẽ scatter raw points
# TITLE      = "block3 / cka_on_old_data theo round ở block 3"
# X_LABEL    = "Round"
# Y_LABEL    = "CKA so sánh 2 model hiện tại và trước ở cùng dataset cũ (%)"
# # ─────────────────────────────────────────────────────────────────────────────

# if len(sys.argv) > 1:
#     CSV_FILE = sys.argv[1]

# if not os.path.exists(CSV_FILE):
#     print(f"[ERR] Không tìm thấy file: {CSV_FILE}")
#     sys.exit(1)

# df = pd.read_csv(CSV_FILE)
# print(f"[OK] Loaded {CSV_FILE}  shape={df.shape}")
# print(f"     Columns: {df.columns.tolist()}")

# if X_COL not in df.columns or Y_COL not in df.columns:
#     print(f"[ERR] Cần cột '{X_COL}' và '{Y_COL}'. Có: {df.columns.tolist()}")
#     sys.exit(1)

# df = df[[X_COL, Y_COL]].dropna(subset=[Y_COL]).sort_values(X_COL).reset_index(drop=True)

# x   = df[X_COL].values
# y   = df[Y_COL].values

# roll   = pd.Series(y).rolling(window=WINDOW, center=True, min_periods=1)
# y_mean = roll.mean().values
# y_std  = roll.std(ddof=0).fillna(0).values

# fig, ax = plt.subplots(figsize=(12, 5))

# ax.fill_between(x, y_mean - y_std, y_mean + y_std,
#                 alpha=0.18, color="#378ADD", label="Mean ± Std")

# ax.plot(x, y_mean, color="#378ADD", linewidth=2, label="Mean")

# if SHOW_RAW:
#     ax.scatter(x, y, s=18, color="#378ADD", alpha=0.45, zorder=3, label="Raw")

# ax.set_title(TITLE, fontsize=14, fontweight="normal", pad=12)
# ax.set_xlabel(X_LABEL, fontsize=12)
# ax.set_ylabel(Y_LABEL, fontsize=12)

# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)
# ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

# overall_mean = np.mean(y)
# overall_std  = np.std(y, ddof=0)
# info = f"n={len(y)}   mean={overall_mean:.1f}   std={overall_std:.1f}   min={y.min():.1f}   max={y.max():.1f}"
# ax.set_title(f"{TITLE}\n{info}", fontsize=13, fontweight="normal", pad=10)

# band_patch = mpatches.Patch(color="#378ADD", alpha=0.18, label=f"Mean ± Std  (window={WINDOW})")
# line_patch  = plt.Line2D([0],[0], color="#378ADD", linewidth=2, label="Rolling mean")
# handles = [band_patch, line_patch]
# if SHOW_RAW:
#     raw_patch = plt.Line2D([0],[0], marker='o', color='w',
#                            markerfacecolor="#378ADD", markersize=6, alpha=0.6, label="Raw points")
#     handles.append(raw_patch)
# ax.legend(handles=handles, fontsize=10, framealpha=0.7)

# plt.tight_layout()

# out_png = CSV_FILE.replace(".csv", "_chart.png")
# plt.savefig(out_png, dpi=150, bbox_inches="tight")
# print(f"[OK] Saved → {out_png}")
# plt.show()
"""
Vẽ biểu đồ 3D: round_global vs eps vs accuracy
Chỉ định tham số qua command line.

Cách dùng:
  python plot_3d_drift.py --csv <đường_dẫn> --eps <tên_cột> --acc <tên_cột> [--block <số>]

Ví dụ:
  python plot_3d_drift.py --csv data.csv --eps eps_current --acc accuracy_current
  python plot_3d_drift.py --csv data.csv --eps eps_old     --acc accuracy_old --block 2
  python plot_3d_drift.py --csv data.csv --eps eps_current --acc accuracy_old --block 0 --out my_plot.png

Các cột eps hợp lệ  : eps_current, eps_old
Các cột acc hợp lệ  : accuracy_current, accuracy_old
Block               : 0, 1, 2, 3, 4  (bỏ qua --block để vẽ tất cả)
"""

import argparse
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps


# ──────────────────────────────────────────────────────────────
# Parse command-line
# ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Vẽ 3D: round_global x eps x accuracy, màu theo task"
)
parser.add_argument("--csv",   required=True,  help="Đường dẫn tới file CSV")
parser.add_argument("--eps",   required=True,  help="Tên cột epsilon trục Y  (eps_current | eps_old)")
parser.add_argument("--acc",   required=True,  help="Tên cột accuracy trục Z  (accuracy_current | accuracy_old)")
parser.add_argument("--block", type=int, default=None,
                    help="Block muốn lọc: 0-4  (bỏ qua = vẽ tất cả)")
parser.add_argument("--out",   default=None,
                    help="Tên file ảnh đầu ra  (mặc định tự sinh)")
parser.add_argument("--elev",  type=float, default=25,  help="Góc elevation (mặc định 25)")
parser.add_argument("--azim",  type=float, default=-55, help="Góc azimuth   (mặc định -55)")

args = parser.parse_args()

CSV_PATH     = args.csv
EPS_COL      = args.eps
ACC_COL      = args.acc
BLOCK_FILTER = args.block


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def parse_tensor(val):
    """Chuyển 'tensor(0.0025, ...)' hoặc float thô -> float."""
    if isinstance(val, (int, float)):
        return float(val) if not (isinstance(val, float) and np.isnan(val)) else np.nan
    if isinstance(val, str):
        m = re.search(r"tensor\(\s*([+-]?[\d.e+\-]+)", val)
        if m:
            return float(m.group(1))
    return np.nan


# ──────────────────────────────────────────────────────────────
# Đọc & chuẩn bị dữ liệu
# ──────────────────────────────────────────────────────────────
print(f"Đọc: {CSV_PATH}")
df = pd.read_csv(CSV_PATH, index_col=[0, 1, 2, 3],
                 engine="python", on_bad_lines="skip")
df = df.reset_index()
df.columns = ["client_id", "block_id", "task_id", "round_global"] + list(df.columns[4:])

print(f"Các cột có sẵn: {list(df.columns)}\n")

# Kiểm tra cột eps được chỉ định có tồn tại không
if EPS_COL not in df.columns:
    print(f"❌ ERROR: Cột EPS '{EPS_COL}' không tồn tại.")
    print(f"Các cột hiện có: {list(df.columns)}")
    sys.exit(1)

# Kiểm tra cột acc được chỉ định có tồn tại không
if ACC_COL not in df.columns:
    print(f"❌ ERROR: Cột ACC '{ACC_COL}' không tồn tại.")
    print(f"Các cột hiện có: {list(df.columns)}")
    sys.exit(1)

print(f"✓ Sử dụng cột EPS: {EPS_COL}")
print(f"✓ Sử dụng cột ACC: {ACC_COL}\n")

# Parse tensor strings - chỉ parse cột eps và acc được chỉ định
df[EPS_COL] = df[EPS_COL].apply(parse_tensor)
df[ACC_COL] = df[ACC_COL].apply(parse_tensor)

for col in ["round_global", "block_id", "task_id"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ──────────────────────────────────────────────────────────────
# Lọc block
# ──────────────────────────────────────────────────────────────
plot_df = df.copy()
if BLOCK_FILTER is not None:
    available_blocks = sorted(df["block_id"].dropna().unique().astype(int).tolist())
    if BLOCK_FILTER not in available_blocks:
        print(f"❌ ERROR: Block {BLOCK_FILTER} không tồn tại.")
        print(f"Các block có sẵn: {available_blocks}")
        sys.exit(1)
    plot_df = plot_df[plot_df["block_id"] == BLOCK_FILTER]

plot_df = plot_df.dropna(subset=["round_global", EPS_COL, ACC_COL])

if plot_df.empty:
    print("❌ ERROR: Không có dữ liệu sau khi lọc. Kiểm tra lại tham số.")
    sys.exit(1)

print(f"✓ Dữ liệu sau lọc: {len(plot_df)} rows\n")

# ──────────────────────────────────────────────────────────────
# Màu theo task
# ──────────────────────────────────────────────────────────────
tasks   = sorted(plot_df["task_id"].dropna().unique())
n_tasks = len(tasks)
cmap    = colormaps.get_cmap("tab10")
colors  = {task: cmap(i / max(n_tasks - 1, 1)) for i, task in enumerate(tasks)}

# ──────────────────────────────────────────────────────────────
# Vẽ 3D
# ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 8))
ax  = fig.add_subplot(111, projection="3d")

for task in tasks:
    subset  = plot_df[plot_df["task_id"] == task]
    grouped = (subset
               .groupby("round_global", as_index=False)[[EPS_COL, ACC_COL]]
               .mean()
               .sort_values("round_global"))

    x = grouped["round_global"].values
    y = grouped[EPS_COL].values
    z = grouped[ACC_COL].values

    ax.plot(x, y, z,
            color=colors[task], linewidth=2,
            label=f"Task {int(task)}")
    ax.scatter(x, y, z,
               color=colors[task], s=30,
               alpha=0.8, depthshade=True)

# ──────────────────────────────────────────────────────────────
# Nhãn & tiêu đề
# ──────────────────────────────────────────────────────────────
block_label = f"Block {BLOCK_FILTER}" if BLOCK_FILTER is not None else "All Blocks"

ax.set_xlabel("round_global", fontsize=11, labelpad=10)
ax.set_ylabel(EPS_COL,        fontsize=11, labelpad=10)
ax.set_zlabel(ACC_COL,        fontsize=11, labelpad=10)
ax.set_title(
    f"3D Representation Drift  [{block_label}]\n"
    f"X: round_global   Y: {EPS_COL}   Z: {ACC_COL}",
    fontsize=13, fontweight="bold", pad=18
)
ax.legend(title="Task ID", bbox_to_anchor=(1.1, 0.5),
          loc="center left", fontsize=9)
ax.view_init(elev=args.elev, azim=args.azim)

plt.tight_layout()

# ──────────────────────────────────────────────────────────────
# Lưu ảnh
# ──────────────────────────────────────────────────────────────
if args.out:
    out_name = args.out
else:
    block_str = f"block{BLOCK_FILTER}" if BLOCK_FILTER is not None else "blockAll"
    out_name  = f"3d_{EPS_COL}_{ACC_COL}_{block_str}.png"

plt.savefig(out_name, dpi=150, bbox_inches="tight")
print(f"✓ Saved: {out_name}")
plt.show()