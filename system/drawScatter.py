import argparse
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import mplcursors
import numpy as np


# ================= FIX =================
matplotlib.use('TkAgg')
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False
# =======================================

# ================= CONFIG =================
def parse_axis_from_filename(filename):
    name = os.path.splitext(filename)[0]  # bỏ .csv
    parts = name.split("_vs_")

    if len(parts) == 2:
        x_label = parts[0]
        y_label = parts[1]
    else:
        x_label = "X"
        y_label = "Y"

    return x_label, y_label
import re

def smart_convert(x):
    # 🔥 nếu đã là số thì giữ nguyên
    if isinstance(x, (int, float)):
        return x
    # 🔥 case tensor vector
    if "tensor([" in x:
        nums = re.findall(r"[-+]?\d*\.?\d+", x)
        if len(nums) > 0:
            nums = [float(n) for n in nums]
            return np.mean(nums)  # 🔥 LẤY MEAN Ở ĐÂY
    # 🔥 nếu là string
    if isinstance(x, str):
        x = x.strip()

        # case: tensor(0.123, device='cuda:0')
        match = re.search(r"tensor\(([-+]?\d*\.?\d+)", x)
        if match:
            return float(match.group(1))

        # case: "0.123"
        try:
            return float(x)
        except:
            return None

    return None
ROOT_DIR = r"C:\Thu\FCL\final_results\block4"

   
# 👉 đổi tên file ở đây là xong

# hoặc:
# SELECTED_PAIRS = ["taskpair_0_1", "taskpair_0_2"]
parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    type=str,
    default="line",
    choices=["scatter", "line"],
    help="Plot mode: scatter hoặc line"
)
parser.add_argument(
    "--filename",
    type=str,
    default="task_vs_cosine_similarity.csv",
    help="Tên file CSV chứa cột X,Y để vẽ (phần còn lại của path sẽ tự động tìm trong các folder con)"
)
parser.add_argument(
    "--sel",
    action="store_true",
    help="Nếu có flag này thì chỉ vẽ taskpair 0_1→0_4; nếu không có flag sẽ vẽ tất cả"
)

args = parser.parse_args()

if args.sel:
    SELECTED_PAIRS = ["taskpair_0_1", "taskpair_0_2", "taskpair_0_3", "taskpair_0_4"]  # hoặc None để lấy tất cả
else:
    SELECTED_PAIRS = SELECTED_PAIRS = ["taskpair_0_1","taskpair_0_2","taskpair_0_3","taskpair_0_4","taskpair_1_2","taskpair_1_3","taskpair_1_4","taskpair_2_3","taskpair_2_4","taskpair_3_4"]  # hoặc ["taskpair_0_1", "taskpair_0_2"] để chỉ lấy 2 cặp
PLOT_MODE = args.mode
TARGET_FILE_NAME = args.filename + ".csv" 
# =========================================

plt.figure(figsize=(8, 6))
import numpy as np
colors = plt.cm.tab10(np.linspace(0, 1, len(SELECTED_PAIRS) if SELECTED_PAIRS else 10))
color_idx = 0
scatters = []

# 🔥 duyệt từng taskpair (folder con)
for pair_name in os.listdir(ROOT_DIR):
    pair_path = os.path.join(ROOT_DIR, pair_name)

    if not os.path.isdir(pair_path):
        continue

    # 🔥 filter taskpair
    if SELECTED_PAIRS is not None and pair_name not in SELECTED_PAIRS:
        continue

    file_path = os.path.join(pair_path, TARGET_FILE_NAME)

    if not os.path.exists(file_path):
        print(f"⚠️ Không có file: {file_path}")
        continue

    try:
        # 🔥 đọc file (2 cột x,y)
        df = pd.read_csv(file_path)

        
        # nếu không có header thì fallback
        if df.shape[1] >= 2:
            df = df.iloc[:, :2]
            df.columns = ["X", "Y"]
        else:
            continue

        # df["X"] = pd.to_numeric(df["X"], errors="coerce")
        # df["Y"] = pd.to_numeric(df["Y"], errors="coerce")
        df["X"] = df["X"].apply(smart_convert)
        df["Y"] = df["Y"].apply(smart_convert)
        df = df.dropna()

        if len(df) == 0:
            continue
        if PLOT_MODE == "scatter":        
            sc = plt.scatter(
                df["X"],
                df["Y"],
                alpha=1,
                s=20,
                label=pair_name
            )
        elif PLOT_MODE == "line":
            sc = plt.plot(
                df["X"],
                df["Y"],
                linewidth=2,
                marker='o',
                markersize=4,
                color=colors[color_idx],  # 🔥 màu riêng
                label=pair_name
            )[0]

        color_idx += 1

        # 🔥 attach metadata
        sc._pair_name = pair_name
        sc._data = list(zip(df["X"], df["Y"]))

        scatters.append(sc)

    except Exception as e:
        print(f"❌ Error: {file_path} | {e}")

# ================= HOVER =================
cursor = mplcursors.cursor(scatters, hover=True)

@cursor.connect("add")
def on_add(sel):
    sc = sel.artist
    index = sel.index

    pair_name = sc._pair_name
    x, y = sc._data[index]

    sel.annotation.set_text(
        f"{pair_name}\nX={x:.4f}\nY={y:.4f}"
    )

    sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)

# =========================================
x_label, y_label = parse_axis_from_filename(TARGET_FILE_NAME)

plt.xlabel(x_label)
plt.ylabel(y_label)
plt.title(f"{PLOT_MODE.upper()} plot: {x_label} vs {y_label}")

plt.legend(fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.show()