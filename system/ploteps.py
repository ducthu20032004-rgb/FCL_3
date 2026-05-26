
# import matplotlib
# print(matplotlib.get_backend())

# import argparse
# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# # ─────────────────────────────────────────────
# # Config
# # ─────────────────────────────────────────────
# BASE_DIR = r"C:\Thu\FCL\results_14_4"
# N_ROUNDS = 25

# TASKPAIR = "0_4","0_1","0_2"   # 🔥 chỉ cần sửa dòng này


# # ─────────────────────────────────────────────
# # Load data
# # ─────────────────────────────────────────────
# def load_block(block: int):
#     path = os.path.join(
#         BASE_DIR,
#         f"block{block}",
#         f"taskpair_{TASKPAIR}",
#         "eps_vs_accold.csv"
#     )

#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Không tìm thấy file: {path}")

#     df = pd.read_csv(path, header=None, names=["eps", "acc"])

#     df = df.apply(pd.to_numeric, errors="coerce")
#     df = df.dropna().astype(float)
#     df = df.head(N_ROUNDS).copy()
#     df["round"] = range(1, len(df) + 1)

#     return df


# # ─────────────────────────────────────────────
# # Plot 1 – Scatter eps vs acc
# # ─────────────────────────────────────────────
# def plot1(df, block):
#     fig, ax = plt.subplots(figsize=(7, 5))

#     sc = ax.scatter(df["eps"], df["acc"],
#                     c=df["round"], cmap="plasma",
#                     s=60, alpha=0.85)

#     plt.colorbar(sc, ax=ax, label="Round")

#     ax.set_xlabel("eps_old")
#     ax.set_ylabel("accuracy_old")
#     ax.set_title(f"Block {block} – Taskpair {TASKPAIR}")
#     ax.grid(True, alpha=0.3)

#     plt.tight_layout()


# # ─────────────────────────────────────────────
# # Plot 2 – eps vs round (color = acc)
# # ─────────────────────────────────────────────
# def plot2(df, block):
#     fig, ax = plt.subplots(figsize=(7, 5))

#     sc = ax.scatter(df["round"], df["eps"],
#                     c=df["acc"], cmap="viridis",
#                     s=60)

#     plt.colorbar(sc, ax=ax, label="accuracy_old")

#     ax.set_xlabel("Round")
#     ax.set_ylabel("eps_old")
#     ax.set_title(f"Block {block} – Taskpair {TASKPAIR}")
#     ax.grid(True, alpha=0.3)

#     plt.tight_layout()


# # ─────────────────────────────────────────────
# # Plot 3 – Line: round → eps & acc
# # ─────────────────────────────────────────────
# def plot3(df, block):
#     fig, ax = plt.subplots(figsize=(8, 5))

#     ax.plot(df["round"], df["eps"], marker="o", label="eps")
#     ax.plot(df["round"], df["acc"], marker="s", label="accuracy_old")

#     ax.set_xlabel("Round")
#     ax.set_title(f"Block {block} – Taskpair {TASKPAIR}")
#     ax.legend()
#     ax.grid(True, alpha=0.3)

#     plt.tight_layout()


# # ─────────────────────────────────────────────
# # Plot 4 – 3D
# # ─────────────────────────────────────────────
# def plot4(df, block):
#     fig = plt.figure(figsize=(9, 7))
#     ax = fig.add_subplot(111, projection="3d")

#     ax.scatter(df["eps"], df["round"], df["acc"],
#                c=df["round"], cmap="plasma", s=60)

#     ax.set_xlabel("eps_old")
#     ax.set_ylabel("Round")
#     ax.set_zlabel("accuracy_old")
#     ax.set_title(f"Block {block} – Taskpair {TASKPAIR}")

#     print("[TIP] Kéo chuột để xoay 3D")
#     plt.tight_layout()


# # ─────────────────────────────────────────────
# # Plot 5 – Heatmap (1 dòng)
# # ─────────────────────────────────────────────
# def plot5(df, block, hue="eps"):
#     fig, ax = plt.subplots(figsize=(10, 2))

#     data = df[hue].values.reshape(1, -1)

#     im = ax.imshow(data, aspect="auto", cmap="YlOrRd")

#     plt.colorbar(im, ax=ax, label=hue)

#     ax.set_xticks(range(len(df)))
#     ax.set_xticklabels(df["round"])
#     ax.set_yticks([])

#     ax.set_xlabel("Round")
#     ax.set_title(f"{hue} evolution – Block {block} – Taskpair {TASKPAIR}")

#     plt.tight_layout()


# # ─────────────────────────────────────────────
# # Plot 6 – Regression
# # ─────────────────────────────────────────────
# def plot6(df, block):
#     fig, ax = plt.subplots(figsize=(7, 5))

#     ax.scatter(df["eps"], df["acc"],
#                c=df["round"], cmap="plasma",
#                s=60, alpha=0.85)

#     if len(df) > 1:
#         z = np.polyfit(df["eps"], df["acc"], 1)
#         p = np.poly1d(z)

#         xs = np.linspace(df["eps"].min(), df["eps"].max(), 100)
#         ax.plot(xs, p(xs), color="red", linewidth=2)

#         print(f"[INFO] slope = {z[0]:.4f}")

#     ax.set_xlabel("eps_old")
#     ax.set_ylabel("accuracy_old")
#     ax.set_title(f"Block {block} – Taskpair {TASKPAIR}")
#     ax.grid(True, alpha=0.3)

#     plt.tight_layout()


# # ─────────────────────────────────────────────
# # Main
# # ─────────────────────────────────────────────
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--block", type=int, required=True)
#     parser.add_argument("--plot", default="1")
#     parser.add_argument("--save", action="store_true")
#     parser.add_argument("--hue", default="eps", choices=["eps", "acc"])

#     args = parser.parse_args()

#     df = load_block(args.block)

#     print(f"[OK] Block {args.block} | Taskpair {TASKPAIR} | {len(df)} points")

#     def _save(name):
#         if args.save:
#             fname = f"block{args.block}_tp{TASKPAIR}_{name}.png"
#             plt.savefig(fname, dpi=150, bbox_inches="tight")
#             print(f"[SAVE] {fname}")
#             plt.close()

#     choice = args.plot.lower()

#     if choice in ("1", "all"):
#         plot1(df, args.block)
#         _save("scatter")

#     if choice in ("2", "all"):
#         plot2(df, args.block)
#         _save("eps_vs_round")

#     if choice in ("3", "all"):
#         plot3(df, args.block)
#         _save("line")

#     if choice == "4":
#         plot4(df, args.block)

#     if choice in ("5", "all"):
#         plot5(df, args.block, hue=args.hue)
#         _save(f"heatmap_{args.hue}")

#     if choice in ("6", "all"):
#         plot6(df, args.block)
#         _save("regression")

#     if not args.save:
#         plt.show()


# if __name__ == "__main__":
#     main()
# # Phien ban load từ file 
# """
# Vẽ mối quan hệ eps vs accuracy_old cho một block chỉ định.
# Dữ liệu load từ file CSV duy nhất:
#   C:/Thu/FCL/outputs/representation_drift_temporal_14_4-hetero-ResNet18.csv

# Cách chạy:
#   python plot_eps_acc_v2.py --block 4 --plot 1       # scatter, màu theo task
#   python plot_eps_acc_v2.py --block 4 --plot 2       # scatter, màu theo round (colorbar)
#   python plot_eps_acc_v2.py --block 4 --plot 3       # facet 5 subplot (1 task / 1 ô)
#   python plot_eps_acc_v2.py --block 4 --plot 4       # 3D scatter (eps, acc, round) interactive
#   python plot_eps_acc_v2.py --block 4 --plot 5       # heatmap task x round; --hue acc de doi
#   python plot_eps_acc_v2.py --block 4 --plot 6       # scatter + regression line moi task
#   python plot_eps_acc_v2.py --block 4 --plot all     # xuat tat ca (tru 4) thanh file anh
#   python plot_eps_acc_v2.py --block 4 --plot 1 --save  # luu anh thay vi hien
# """
import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
DATA_FILE = r"C:\Thu\FCL\outputs\representation_drift_temporal_25_4-hetero-ResNet18.csv"
N_TASKS   = 5  # Client 0-4 (client 0 = no task data)
N_ROUNDS  = 25
TASK_COLORS  = ["#534AB7", "#0F6E56", "#993C1D", "#BA7517", "#185FA5"]
TASK_MARKERS = ["o", "s", "^", "D", "P"]


# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
def load_block(block: int, data_file: str = DATA_FILE):
    """
    Doc file CSV, loc theo client va block chi dinh.
    
    Note: 'task' column chứa float values, không phải 0-4
    Thay vào đó, dùng 'client' (1-4) làm task/federated client
    Client 0 là aggregate data
    """
    # Buoc 1: doc raw va clean tensor strings
    with open(data_file, encoding="utf-8") as fh:
        raw = fh.read()

    # Xoa ", device='cuda:0'" (hoac cpu) ben trong tensor(...)
    raw = re.sub(r",\s*device='[^']*'", "", raw)
    # Xoa wrapper tensor(...) => giu gia tri so ben trong
    raw = re.sub(r"tensor\(([^)]*)\)", r"\1", raw)

    # Buoc 2: parse CSV tu string da clean
    df_raw = pd.read_csv(StringIO(raw))
    df_raw.columns = df_raw.columns.str.strip()

    # Debug: in blocks tim thay
    blocks_found = sorted(
        pd.to_numeric(df_raw["block"], errors="coerce")
        .dropna().unique().astype(int).tolist()
    )
    print(f"[DEBUG] Loaded {df_raw.shape[0]} rows | blocks available: {blocks_found}")

    # Buoc 3: loc theo block (không lọc client - lấy tất cả clients)
    df_raw["block"] = pd.to_numeric(df_raw["block"], errors="coerce")
    df_raw = df_raw[df_raw["block"] == block].copy()

    if df_raw.empty:
        print(f"[WARN] Khong co du lieu cho block={block}")
        return pd.DataFrame(columns=["round", "eps", "acc", "client"])

    print(f"[DEBUG] Block {block}: {len(df_raw)} rows")

    # Buoc 4: xac dinh cot eps va accuracy
    eps_col = None
    if "eps_current" in df_raw.columns:
        non_null = df_raw["eps_current"].notna().sum()
        if non_null > 0:
            eps_col = "eps_current"
            print(f"[OK] Dùng eps_current ({non_null}/{len(df_raw)} non-null)")
    
    if eps_col is None and "eps_old" in df_raw.columns:
        non_null = df_raw["eps_old"].notna().sum()
        if non_null > 0:
            eps_col = "eps_old"
            print(f"[WARN] Dùng eps_old ({non_null}/{len(df_raw)} non-null)")
    
    if eps_col is None:
        print(f"[WARN] Không có cột eps với dữ liệu")
        eps_col = "eps_current"

    acc_col = None
    if "accuracy_current" in df_raw.columns:
        non_null = df_raw["accuracy_current"].notna().sum()
        if non_null > 0:
            acc_col = "accuracy_current"
            print(f"[OK] Dùng accuracy_current ({non_null}/{len(df_raw)} non-null)")
    
    if acc_col is None and "accuracy_old" in df_raw.columns:
        non_null = df_raw["accuracy_old"].notna().sum()
        if non_null > 0:
            acc_col = "accuracy_old"
            print(f"[WARN] Dùng accuracy_old ({non_null}/{len(df_raw)} non-null)")
    
    if acc_col is None:
        print(f"[WARN] Không có cột accuracy với dữ liệu")
        acc_col = "accuracy_current"

    # Buoc 5: tong hop theo CLIENT (không phải task)
    # Client 0 = aggregate, Client 1-4 = federated clients
    rows = []
    df_raw["client"] = pd.to_numeric(df_raw["client"], errors="coerce")
    df_raw["round"] = pd.to_numeric(df_raw["round"], errors="coerce")
    
    for client_id in sorted(df_raw["client"].dropna().unique().astype(int)):
        d = df_raw[df_raw["client"] == client_id].copy()
        
        if d.empty:
            print(f"[SKIP] Khong co du lieu client={client_id}, block={block}")
            continue

        # Vì 'round' column có thể là float hoặc NaN, cần xử lý
        d["round_num"] = d["round"]
        
        d["eps"] = pd.to_numeric(d[eps_col], errors="coerce")
        d["acc"] = pd.to_numeric(d[acc_col], errors="coerce")

        out = d[["round_num", "eps", "acc"]].rename(columns={"round_num": "round"})
        out = out.dropna()
        
        # Bỏ qua client nếu không có dữ liệu hợp lệ
        if out.empty:
            print(f"[SKIP] Client {client_id} không có dữ liệu hợp lệ")
            continue
        
        out["client"] = client_id
        rows.append(out)

    if not rows:
        print(f"[WARN] Block {block} không có dữ liệu hợp lệ cho bất kỳ client nào")
        return pd.DataFrame(columns=["round", "eps", "acc", "client"])

    result = pd.concat(rows, ignore_index=True)
    print(f"[OK] Block {block}: {len(result)} data points "
          f"(eps='{eps_col}', acc='{acc_col}')")
    return result


# ─────────────────────────────────────────────
# Option 1 – Scatter, mau theo client
# ─────────────────────────────────────────────
def plot1(df, block, ax=None, standalone=True):
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 5))
    
    if df.empty:
        ax.text(0.5, 0.5, f"Block {block}: No data",
                transform=ax.transAxes, ha='center', va='center', fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        for c in sorted(df["client"].unique()):
            d = df[df["client"] == c]
            if not d.empty:
                ax.scatter(d["eps"], d["acc"],
                           color=TASK_COLORS[c % len(TASK_COLORS)], 
                           marker=TASK_MARKERS[c % len(TASK_MARKERS)],
                           s=50, alpha=0.8, label=f"Client {c}")
    
    ax.set_xlabel("eps")
    ax.set_ylabel("accuracy")
    ax.set_title(f"Block {block} - eps vs accuracy (color by client)")
    ax.legend(title="Client", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    if standalone:
        plt.tight_layout()


# ─────────────────────────────────────────────
# Option 2 – Scatter, mau theo round
# ─────────────────────────────────────────────
def plot2(df, block, ax=None, standalone=True):
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 5))
    
    if df.empty:
        ax.text(0.5, 0.5, f"Block {block}: No data",
                transform=ax.transAxes, ha='center', va='center', fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        sc = ax.scatter(df["eps"], df["acc"],
                        c=df["round"], cmap="plasma",
                        s=50, alpha=0.8)
        if standalone:
            plt.colorbar(sc, ax=ax, label="Round")
        else:
            plt.colorbar(sc, ax=ax, label="Round", shrink=0.8)
    
    ax.set_xlabel("eps")
    ax.set_ylabel("accuracy")
    ax.set_title(f"Block {block} - eps vs accuracy (color by round)")
    ax.grid(True, alpha=0.3)
    if standalone:
        plt.tight_layout()


# ─────────────────────────────────────────────
# Option 3 – Facet 1x5, moi o 1 client
# ─────────────────────────────────────────────
def plot3(df, block, standalone=True):
    fig, axes = plt.subplots(1, 5, figsize=(15, 4), sharey=True)
    fig.suptitle(f"Block {block} - eps vs accuracy | each subplot = 1 client", y=1.01)
    sc = None
    
    for c in range(5):
        ax = axes[c]
        d = df[df["client"] == c]
        if d.empty:
            ax.text(0.5, 0.5, f"Client {c}: No data", 
                   transform=ax.transAxes, ha='center', va='center')
        else:
            sc = ax.scatter(d["eps"], d["acc"],
                            c=d["round"], cmap="viridis",
                            s=40, alpha=0.85)
            
            if len(d) > 1:
                z = np.polyfit(d["eps"], d["acc"], 1)
                p = np.poly1d(z)
                xs = np.linspace(d["eps"].min(), d["eps"].max(), 50)
                ax.plot(xs, p(xs), "--", color=TASK_COLORS[c], linewidth=1.5, alpha=0.7)
        
        ax.set_title(f"Client {c}", color=TASK_COLORS[c], fontweight="bold")
        ax.set_xlabel("eps")
        if c == 0:
            ax.set_ylabel("accuracy")
        ax.grid(True, alpha=0.3)

    if sc is not None:
        plt.colorbar(sc, ax=axes[-1], label="Round", shrink=0.9)
    plt.tight_layout()


# ─────────────────────────────────────────────
# Option 4 – 3D scatter
# ─────────────────────────────────────────────
def plot4(df, block):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    
    if df.empty:
        ax.text(0, 0, 0, f"Block {block}: No data")
    else:
        for c in sorted(df["client"].unique()):
            d = df[df["client"] == c]
            if not d.empty:
                ax.scatter(d["eps"], d["round"], d["acc"],
                           color=TASK_COLORS[c % len(TASK_COLORS)], 
                           marker=TASK_MARKERS[c % len(TASK_MARKERS)],
                           s=45, alpha=0.8, label=f"Client {c}")
    
    ax.set_xlabel("eps")
    ax.set_ylabel("round")
    ax.set_zlabel("accuracy")
    ax.set_title(f"Block {block} - 3D: eps / round / accuracy")
    ax.legend(title="Client", loc="upper left")
    plt.tight_layout()
    print("[TIP] Drag mouse to rotate 3D plot.")


# ─────────────────────────────────────────────
# Option 5 – Heatmap client x round
# ─────────────────────────────────────────────
def plot5(df, block, hue="eps", standalone=True):
    if df.empty:
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.text(0.5, 0.5, f"Block {block}: No data",
                transform=ax.transAxes, ha='center', va='center')
        plt.tight_layout()
        return
    
    pivot = df.pivot_table(index="client", columns="round", values=hue, aggfunc="first")
    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    plt.colorbar(im, ax=ax, label=hue)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{x:.3f}" for x in pivot.columns], fontsize=8, rotation=45)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"Client {int(c)}" for c in pivot.index])
    ax.set_xlabel("Round")
    ax.set_title(f"Block {block} - Heatmap {hue} (client x round)")
    plt.tight_layout()


# ─────────────────────────────────────────────
# Option 6 – Scatter + regression line per client
# ─────────────────────────────────────────────
def plot6(df, block, standalone=True):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    if df.empty:
        ax.text(0.5, 0.5, f"Block {block}: No data",
                transform=ax.transAxes, ha='center', va='center', fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        for c in sorted(df["client"].unique()):
            d = df[df["client"] == c]
            if not d.empty:
                ax.scatter(d["eps"], d["acc"],
                           color=TASK_COLORS[c % len(TASK_COLORS)], 
                           marker=TASK_MARKERS[c % len(TASK_MARKERS)],
                           s=45, alpha=0.7, label=f"Client {c}")
                if len(d) > 1:
                    z = np.polyfit(d["eps"], d["acc"], 1)
                    p = np.poly1d(z)
                    xs = np.linspace(d["eps"].min(), d["eps"].max(), 80)
                    ax.plot(xs, p(xs), color=TASK_COLORS[c % len(TASK_COLORS)], linewidth=2)
    
    ax.set_xlabel("eps")
    ax.set_ylabel("accuracy")
    ax.set_title(f"Block {block} - eps vs accuracy + trendline per client")
    ax.legend(title="Client", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Plot eps vs accuracy from CSV")
    parser.add_argument("--block", type=int, required=True,
                        help="Block to plot (0-24)")
    parser.add_argument("--plot", default="1",
                        help="Plot option: 1/2/3/4/5/6/all (default: 1)")
    parser.add_argument("--save", action="store_true",
                        help="Save images instead of showing interactive")
    parser.add_argument("--hue", default="eps", choices=["eps", "acc"],
                        help="Variable for heatmap (option 5, default: eps)")
    parser.add_argument("--file", default=DATA_FILE,
                        help="Path to CSV file")
    args = parser.parse_args()

    df = load_block(args.block, data_file=args.file)

    choice = args.plot.lower()
    save   = args.save

    def _save(name):
        if save:
            fname = f"block{args.block}_{name}.png"
            plt.savefig(fname, dpi=150, bbox_inches="tight")
            print(f"[SAVE] {fname}")
            plt.close()

    if choice in ("1", "all"):
        plot1(df, args.block)
        _save("opt1_scatter_client")

    if choice in ("2", "all"):
        plot2(df, args.block)
        _save("opt2_scatter_round")

    if choice in ("3", "all"):
        plot3(df, args.block)
        _save("opt3_facet_client")

    if choice == "4":
        plot4(df, args.block)

    if choice in ("5", "all"):
        plot5(df, args.block, hue=args.hue)
        _save(f"opt5_heatmap_{args.hue}")

    if choice in ("6", "all"):
        plot6(df, args.block)
        _save("opt6_regression")

    if not save:
        plt.show()


if __name__ == "__main__":
    main()
# """
# Vẽ biểu đồ round vs <y_column> theo taskpair và block chỉ định.

# Cấu trúc thư mục:
#   <BASE_DIR>\\block<N>\\<taskpair>\\<filename>.csv

# Cách chạy:
#   python plot_ratio_feature.py --block 4 --file round_vs_ratio_feature.csv --taskpairs taskpair_2_3
#   python plot_ratio_feature.py --block 4 --file round_vs_ratio_feature.csv --taskpairs taskpair_0_1 taskpair_2_3 --plot line
#   python plot_ratio_feature.py --block 4 --file round_vs_ratio_feature.csv --taskpairs all --plot scatter
#   python plot_ratio_feature.py --block 4 --file round_vs_ratio_feature.csv --taskpairs all --plot both --save

# Plot options:
#   scatter  – Scatter round vs y, màu theo giá trị y (colorbar)
#   line     – Line round vs y, mỗi taskpair 1 màu
#   both     – Line + scatter trên cùng 1 axes (mặc định)
# """

# import argparse
# import os
# import glob
# import pandas as pd
# import matplotlib.pyplot as plt

# # ─────────────────────────────────────────────
# # Config mặc định
# # ─────────────────────────────────────────────
# DEFAULT_BASE_DIR = r"C:\Thu\FCL\outputs\representation_drift_temporal_13_4-hetero-ResNet18.csv"

# COLORS  = [
#     "#534AB7", "#0F6E56", "#993C1D", "#BA7517",
#     "#185FA5", "#C2185B", "#00838F", "#6D4C41",
#     "#558B2F", "#4527A0",
# ]
# MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h"]


# # ─────────────────────────────────────────────
# # Load data  (kế thừa từ ploteps.py)
# # ─────────────────────────────────────────────
# def load_data(base_dir, block, taskpairs, filename):
#     """
#     Load CSV từ nhiều taskpair.
#       - 1 cột  : dùng làm y, round tự sinh 0,1,2,...
#       - 2+ cột : cột 0 = round, cột 1 = y
#     Trả về: (dict { tp: DataFrame(round, y) }, tên_cột_y)
#     """
#     block_dir = os.path.join(base_dir, f"block{block}")
#     if not os.path.isdir(block_dir):
#         raise FileNotFoundError(f"Không tìm thấy thư mục: {block_dir}")

#     tp_list = list(taskpairs)
#     if "all" in tp_list:
#         found   = sorted(glob.glob(os.path.join(block_dir, "taskpair_*")))
#         tp_list = [os.path.basename(p) for p in found if os.path.isdir(p)]
#         if not tp_list:
#             raise FileNotFoundError(f"Không có taskpair nào trong: {block_dir}")
#         print(f"[AUTO] {len(tp_list)} taskpair: {tp_list}")

#     data  = {}
#     y_col = None

#     for tp in tp_list:
#         csv_path = os.path.join(block_dir, tp, filename)
#         if not os.path.isfile(csv_path):
#             print(f"[WARN] Không tìm thấy: {csv_path} — bỏ qua")
#             continue
#         try:
#             df   = pd.read_csv(csv_path)
#             df.columns = df.columns.str.strip()
#             cols = df.columns.tolist()

#             if len(cols) >= 2:
#                 y_col = cols[1]
#                 df = df[[cols[0], cols[1]]].rename(
#                     columns={cols[0]: "round", cols[1]: "y"})
#             elif len(cols) == 1:
#                 y_col = cols[0]
#                 df = df.rename(columns={cols[0]: "y"})
#                 df["round"] = range(len(df))
#             else:
#                 print(f"[WARN] {csv_path} rỗng — bỏ qua")
#                 continue

#             df["round"] = pd.to_numeric(df["round"], errors="coerce")
#             df["y"]     = pd.to_numeric(df["y"],     errors="coerce")
#             df = (df.dropna(subset=["round", "y"])
#                     .sort_values("round")
#                     .reset_index(drop=True))

#             data[tp] = df
#             print(f"[OK] {tp}: {len(df)} điểm  |  "
#                   f"{y_col} ∈ [{df['y'].min():.4f}, {df['y'].max():.4f}]")

#         except Exception as e:
#             print(f"[ERR] {csv_path}: {e}")

#     if not data:
#         raise ValueError("Không load được dữ liệu nào.")

#     return data, (y_col or "y")


# # ─────────────────────────────────────────────
# # Draw helpers
# # ─────────────────────────────────────────────
# def _label(tp):
#     return tp.replace("taskpair_", "Pair ").replace("_", "→")

# def draw_scatter(ax, df, color, marker):
#     return ax.scatter(df["round"], df["y"],
#                       c=df["y"], cmap="plasma",
#                       s=65, zorder=4, alpha=0.9,
#                       edgecolors="white", linewidths=0.4)

# def draw_line(ax, df, color, marker, label):
#     ax.plot(df["round"], df["y"],
#             color=color, marker=marker,
#             linewidth=1.8, markersize=5,
#             label=label, alpha=0.85, zorder=3)

# def draw_both(ax, df, color, marker, label):
#     ax.plot(df["round"], df["y"],
#             color=color, linewidth=1.5, alpha=0.5, zorder=2)
#     ax.scatter(df["round"], df["y"],
#                color=color, marker=marker,
#                s=60, zorder=4, alpha=0.9, label=label)


# # ─────────────────────────────────────────────
# # Figure 1 – tất cả taskpair trên 1 axes
# # ─────────────────────────────────────────────
# def plot_combined(data, block, filename, y_col, plot_type, save, save_dir):
#     fig, ax = plt.subplots(figsize=(10, 5))
#     sc_last = None

#     for i, (tp, df) in enumerate(data.items()):
#         color  = COLORS[i % len(COLORS)]
#         marker = MARKERS[i % len(MARKERS)]
#         label  = _label(tp)

#         if plot_type == "scatter":
#             sc_last = draw_scatter(ax, df, color, marker)
#         elif plot_type == "line":
#             draw_line(ax, df, color, marker, label)
#         else:
#             draw_both(ax, df, color, marker, label)

#     if plot_type == "scatter" and sc_last is not None:
#         plt.colorbar(sc_last, ax=ax, label=y_col)
#         for i, (tp, df) in enumerate(data.items()):
#             mid = len(df) // 2
#             ax.annotate(_label(tp),
#                         xy=(df["round"].iloc[mid], df["y"].iloc[mid]),
#                         fontsize=7, alpha=0.75,
#                         color=COLORS[i % len(COLORS)])
#     else:
#         ax.legend(loc="best", fontsize=9)

#     ax.set_xlabel("Round", fontsize=11)
#     ax.set_ylabel(y_col, fontsize=11)
#     ax.set_title(
#         f"Block {block}  |  {filename}  |  round vs {y_col}  [{plot_type}]",
#         fontsize=12, fontweight="bold")
#     ax.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.show()
#     if save:
#         _save_fig(fig, f"block{block}_{plot_type}_{y_col}", save_dir)


# # ─────────────────────────────────────────────
# # Figure 2 – subplot riêng từng taskpair
# # ─────────────────────────────────────────────
# def plot_subplots(data, block, filename, y_col, plot_type, save, save_dir):
#     n     = len(data)
#     ncols = min(n, 3)
#     nrows = (n + ncols - 1) // ncols

#     fig, axes = plt.subplots(nrows, ncols,
#                               figsize=(6 * ncols, 4.5 * nrows),
#                               squeeze=False)
#     axes_flat = axes.flatten()
#     last_i    = 0

#     for i, (tp, df) in enumerate(data.items()):
#         ax     = axes_flat[i]
#         color  = COLORS[i % len(COLORS)]
#         marker = MARKERS[i % len(MARKERS)]
#         label  = _label(tp)
#         last_i = i

#         if plot_type == "scatter":
#             sc = draw_scatter(ax, df, color, marker)
#             plt.colorbar(sc, ax=ax, label=y_col)
#         elif plot_type == "line":
#             draw_line(ax, df, color, marker, label)
#             ax.legend(fontsize=8)
#         else:
#             draw_both(ax, df, color, marker, label)
#             ax.legend(fontsize=8)

#         ax.set_xlabel("Round")
#         ax.set_ylabel(y_col)
#         ax.set_title(f"Block {block} – {label}", fontweight="bold")
#         ax.grid(True, alpha=0.3)

#     for j in range(last_i + 1, len(axes_flat)):
#         axes_flat[j].set_visible(False)

#     plt.suptitle(
#         f"Block {block}  |  {filename}  |  round vs {y_col}  [{plot_type}]",
#         fontsize=13, fontweight="bold", y=1.01)
#     plt.tight_layout()
#     plt.show()
#     if save:
#         _save_fig(fig, f"block{block}_{plot_type}_{y_col}_subplots", save_dir)


# # ─────────────────────────────────────────────
# # Save helper
# # ─────────────────────────────────────────────
# def _save_fig(fig, stem, save_dir):
#     os.makedirs(save_dir, exist_ok=True)
#     path = os.path.join(save_dir, f"{stem}.png")
#     fig.savefig(path, dpi=150, bbox_inches="tight")
#     print(f"[SAVE] {path}")
#     plt.close(fig)


# # ─────────────────────────────────────────────
# # Entry point
# # ─────────────────────────────────────────────
# def main():
#     parser = argparse.ArgumentParser(
#         description="Vẽ round vs <y> theo taskpair",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog=__doc__,
#     )
#     parser.add_argument("--block", type=int, required=True,
#                         help="Block cần vẽ, ví dụ: 4")
#     parser.add_argument("--file", required=True,
#                         help="Tên file CSV, ví dụ: round_vs_ratio_feature.csv")
#     parser.add_argument("--taskpairs", nargs="+", default=["all"],
#                         help="Taskpair: taskpair_0_1 taskpair_2_3 ... hoặc all")
#     parser.add_argument("--plot", default="both",
#                         choices=["scatter", "line", "both"],
#                         help="Loại biểu đồ: scatter / line / both  (mặc định: both)")
#     parser.add_argument("--save", action="store_true",
#                         help="Lưu ảnh PNG thay vì hiện cửa sổ")
#     parser.add_argument("--savedir", default=".",
#                         help="Thư mục lưu ảnh (mặc định: thư mục hiện tại)")
#     parser.add_argument("--base", default=DEFAULT_BASE_DIR,
#                         help=f"Thư mục gốc (mặc định: {DEFAULT_BASE_DIR})")
#     args = parser.parse_args()

#     print(f"\n{'='*55}")
#     print(f"  block     : {args.block}")
#     print(f"  file      : {args.file}")
#     print(f"  taskpairs : {args.taskpairs}")
#     print(f"  plot      : {args.plot}")
#     print(f"  save      : {args.save}")
#     print(f"{'='*55}\n")

#     data, y_col = load_data(
#         base_dir  = args.base,
#         block     = args.block,
#         taskpairs = args.taskpairs,
#         filename  = args.file,
#     )

#     print(f"\n[INFO] {len(data)} taskpair | y_col='{y_col}' | plot={args.plot}\n")

#     # Hình tổng hợp (luôn vẽ)
#     plot_combined(data, args.block, args.file, y_col,
#                   args.plot, args.save, args.savedir)

#     # Subplot riêng nếu có nhiều hơn 1 taskpair
#     if len(data) > 1:
#         plot_subplots(data, args.block, args.file, y_col,
#                       args.plot, args.save, args.savedir)

# \

# if __name__ == "__main__":
#     main()