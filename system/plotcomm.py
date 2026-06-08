"""
plot_drift.py
=============
Vẽ biểu đồ drift / CKA theo round từ file drift_results.csv.
Tất cả tuỳ chỉnh nằm trong phần CONFIG bên dưới.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ═══════════════════════════════════════════════════════════════════════
#  CONFIG  ── chỉnh tại đây, KHÔNG cần sửa phần code bên dưới
# ═══════════════════════════════════════════════════════════════════════

# --- Đường dẫn file đầu vào ------------------------------------------
CSV_PATH = r"C:\Thu\FCL\Book1.csv"

# --- Chọn nhóm metric muốn vẽ (chọn 1 trong 2 nhóm) -----------------
#   "drift"  → drift_trained, drift_aggre, drift_global
#   "cka"    → cka_trained,   cka_aggre,   cka_global
#   "cknna"  → cknna_trained, cknna_aggre, cknna_global
METRIC_GROUP = "drift"          # "drift", "cka" hoặc "cknna"

# --- Chọn đường nào được hiển thị (True = vẽ, False = ẩn) -----------
SHOW_TRAINED = True
SHOW_AGGRE   = True
SHOW_GLOBAL  = True
SHOW_CLIENT  = True

# --- Nhãn trên biểu đồ (chuẩn hoá cho paper) ------------------------
#   Thay đổi tuỳ theo ký hiệu trong bài viết của bạn
LABEL_TRAINED = r"$\mathcal{D}_\mathrm{trained}$"      # drift/CKA của model local
LABEL_AGGRE   = r"$\mathcal{D}_\mathrm{aggr}$"        # drift/CKA sau aggregation
LABEL_GLOBAL  = r"$\mathcal{D}_\mathrm{global}$"      # drift/CKA global
LABEL_CLIENT  = r"$\mathcal{D}_\mathrm{client}$"      # drift/CKA của client (nếu có)

# --- Tiêu đề và nhãn trục -------------------------------------------
TITLE    = r"Similarity Inner Round"
XLABEL   = "Communication Round"
YLABEL_DRIFT = r"Drift"
YLABEL_CKA   = r"CKA Similarity"
YLABEL_CKRNA = r"CKNNA Similarity"

# --- Nhãn task trên trục X ------------------------------------------
#   Mỗi phần tử: (round_bắt_đầu_task, "Tên task")
#   Script tự tính nếu để rỗng [], nhưng bạn có thể ghi đè:
TASK_LABELS = []    # [] = tự động; hoặc ví dụ: [(1,"T1"),(26,"T2"),...]

# --- Màu sắc các đường -----------------------------------------------
COLOR_TRAINED = "#2196F3"   # xanh dương
COLOR_AGGRE   = "#FF5722"   # cam
COLOR_GLOBAL  = "#4CAF50"   # xanh lá

# --- Kiểu đường (linestyle) ------------------------------------------
LS_TRAINED = "-"
LS_AGGRE   = "--"
LS_GLOBAL  = "-."

# --- Độ dày đường ----------------------------------------------------
LW = 1.8

# --- Font size -------------------------------------------------------
FONTSIZE_TITLE  = 25
FONTSIZE_AXIS   = 15
FONTSIZE_TICK   = 15
FONTSIZE_LEGEND = 15
FONTSIZE_TASK   = 12     # chú thích tên task trên trục X

# --- Kích thước figure (inch) ----------------------------------------
FIG_W = 10
FIG_H = 4.5

# --- Lưu file --------------------------------------------------------
SAVE_PATH = "drift_plot.pdf"    # None để không lưu; hoặc "drift_plot.png"
DPI       = 300

# ═══════════════════════════════════════════════════════════════════════
#  CODE  ── không cần chỉnh bên dưới
# ═══════════════════════════════════════════════════════════════════════

def load_and_average(csv_path: str) -> pd.DataFrame:
    """Đọc CSV, lấy trung bình 5 client theo từng round."""
    df = pd.read_csv(csv_path)
    cols = ["round", "task",
            "drift_trained", "drift_aggre", "drift_global",
            "cka_trained",   "cka_aggre",   "cka_global",
            "cknna_trained","cknna_aggre", "cknna_global","cknna_client","Drift_client"]
    df = df[cols]
    avg = df.groupby(["round", "task"]).mean().reset_index()
    avg = avg.sort_values("round").reset_index(drop=True)
    return avg


def get_task_boundaries(df: pd.DataFrame):
    """Trả về danh sách (round_đầu_task, task_id) khi task thay đổi."""
    boundaries = []
    prev_task = None
    for _, row in df.iterrows():
        t = int(row["task"])
        r = int(row["round"])
        if t != prev_task:
            boundaries.append((r, t))
            prev_task = t
    return boundaries


def pick_columns(group: str):
    if group == "drift":
        return ("drift_trained", "drift_aggre", "drift_global", "Drift_client")
    elif group == "cka":
        return ("cka_trained", "cka_aggre", "cka_global")
    elif group == "cknna":
        return ("cknna_trained", "cknna_aggre", "cknna_global", "cknna_client")
    else:
        raise ValueError(f"METRIC_GROUP phải là 'drift' hoặc 'cka' hoặc 'cknna', nhận được: '{group}'")


def main():
    # --- Đọc dữ liệu -------------------------------------------------
    avg = load_and_average(CSV_PATH)

    col_trained, col_aggre, col_global, col_client = pick_columns(METRIC_GROUP)
    ylabel = YLABEL_DRIFT if METRIC_GROUP == "drift" else YLABEL_CKA if METRIC_GROUP == "cka" else YLABEL_CKRNA

    rounds = avg["round"].values

    # =====================================================
    # Mean ± Std của toàn bộ rounds
    # =====================================================
    stats_lines = []

    # if SHOW_TRAINED:
    #     mean_val = avg[col_trained].mean()
    #     std_val  = avg[col_trained].std()
    #     stats_lines.append(f"Trained : {mean_val:.4f} ± {std_val:.4f}")

    # if SHOW_AGGRE:
    #     mean_val = avg[col_aggre].mean()
    #     std_val  = avg[col_aggre].std()
    #     stats_lines.append(f"Aggr    : {mean_val:.4f} ± {std_val:.4f}")

    # if SHOW_GLOBAL:
    #     mean_val = avg[col_global].mean()
    #     std_val  = avg[col_global].std()
    #     stats_lines.append(f"Global  : {mean_val:.4f} ± {std_val:.4f}")

    # if SHOW_CLIENT:
    #     mean_val = avg[col_client].mean()
    #     std_val  = avg[col_client].std()
    #     stats_lines.append(f"Client  : {mean_val:.4f} ± {std_val:.4f}")

    stats_text = "\n".join(stats_lines)

    print("\n===== Mean ± Std =====")
    print(stats_text)

    # --- Task boundaries ---------------------------------------------
    boundaries = get_task_boundaries(avg)
    if TASK_LABELS:
        task_label_map = {r: lbl for r, lbl in TASK_LABELS}
    else:
        task_label_map = {r: f"Task {t}" for r, t in boundaries}

    # --- Figure ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor("white")   # ← nền trắng cho toàn figure
    ax.set_facecolor("white")          # ← nền trắng cho vùng vẽ

    # Vùng tô nền cho từng task
    task_colors = ["white", "white"]
    for i, (r_start, _) in enumerate(boundaries):
        r_end = boundaries[i + 1][0] if i + 1 < len(boundaries) else rounds[-1] + 1
        ax.axvspan(r_start - 0.5, r_end - 0.5,
                   color=task_colors[i % 2], alpha=0.5, zorder=0)

    # Đường dọc phân cách task
    for i, (r_start, _) in enumerate(boundaries):
        if i == 0:
            continue
        ax.axvline(x=r_start - 0.5, color="gray", linewidth=0.8,
                   linestyle=":", zorder=1)

    # Vẽ các đường metric
    lines = []
    if SHOW_TRAINED:
        l, = ax.plot(rounds, avg[col_trained],
                     label=LABEL_TRAINED,
                     color=COLOR_TRAINED, linestyle=LS_TRAINED, linewidth=LW)
        lines.append(l)

    if SHOW_AGGRE:
        l, = ax.plot(rounds, avg[col_aggre],
                     label=LABEL_AGGRE,
                     color=COLOR_AGGRE, linestyle=LS_AGGRE, linewidth=LW)
        lines.append(l)

    if SHOW_GLOBAL:
        l, = ax.plot(rounds, avg[col_global],
                     label=LABEL_GLOBAL,
                     color=COLOR_GLOBAL, linestyle=LS_GLOBAL, linewidth=LW)
        lines.append(l)

    # if SHOW_CLIENT:
    #     l, = ax.plot(rounds, avg[col_client],
    #                  label=LABEL_CLIENT,
    #                  color="#9C27B0", linestyle=":", linewidth=LW)
    #     lines.append(l)

    # --- Nhãn task trên top ------------------------------------------
    ax2 = ax.twiny()
    ax2.set_facecolor("white")         # ← nền trắng cho trục task
    ax2.set_xlim(ax.get_xlim())

    # Đặt tick ở giữa mỗi vùng task
    mid_positions = []
    mid_labels    = []
    for i, (r_start, _) in enumerate(boundaries):
        r_end = boundaries[i + 1][0] if i + 1 < len(boundaries) else rounds[-1] + 1
        mid_positions.append((r_start + r_end - 1) / 2)
        mid_labels.append(task_label_map.get(r_start, ""))

    ax2.set_xticks(mid_positions)
    ax2.set_xticklabels(mid_labels, fontsize=FONTSIZE_TASK)
    ax2.tick_params(axis="x", length=0)

    # --- Nhãn, tiêu đề, legend ---------------------------------------
    #ax.set_title(TITLE, fontsize=FONTSIZE_TITLE, pad=28)
    ax.set_xlabel(XLABEL, fontsize=FONTSIZE_AXIS)
    ax.set_ylabel(ylabel,  fontsize=FONTSIZE_AXIS)
    ax.tick_params(axis="both", labelsize=FONTSIZE_TICK)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(25))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax.set_xlim(rounds[0] - 1, rounds[-1] + 1)

    ax.legend(handles=lines, fontsize=FONTSIZE_LEGEND,
              loc="upper right", framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # =====================================================
    # Hiển thị Mean ± Std trên hình
    # =====================================================
    ax.text(
        0.02, 0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray")
    )

    fig.tight_layout()

    # --- Lưu / hiển thị ----------------------------------------------
    if SAVE_PATH:
        fig.savefig(SAVE_PATH, dpi=DPI, bbox_inches="tight", facecolor="white")  # ← facecolor="white"
        print(f"[✓] Đã lưu biểu đồ: {SAVE_PATH}")

    plt.show()


if __name__ == "__main__":
    main()