# import argparse
# import re
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from io import StringIO
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# # ─────────────────────────────────────────────
# # Config
# # ─────────────────────────────────────────────
# DATA_FILE = r"C:\Thu\FCL\outputs\representation_drift_temporal_25_4-hetero-ResNet18.csv"
# N_TASKS   = 5
# N_ROUNDS  = 25
# TASK_COLORS  = ["#534AB7", "#0F6E56", "#993C1D", "#BA7517", "#185FA5"]
# TASK_MARKERS = ["o", "s", "^", "D", "P"]


# # ─────────────────────────────────────────────
# # Load data
# # ─────────────────────────────────────────────
# def load_block(block: int, data_file: str = DATA_FILE):
#     """
#     Doc file CSV lon, loc theo client=0 va block chi dinh.

#     Van de goc: tensor(0.99, device='cuda:0') chua dau phay ben trong ngoac
#     => pandas dem nham cot => toan bo du lieu bi lech.

#     Giai phap: doc raw text => clean bang regex => moi parse CSV.
    
#     🔧 FIX: Bỏ qua task/block không có dữ liệu thay vì raise error
#     """
#     # Buoc 1: doc raw va clean tensor strings
#     with open(data_file, encoding="utf-8") as fh:
#         raw = fh.read()

#     # Xoa ", device='cuda:0'" (hoac cpu) ben trong tensor(...)
#     raw = re.sub(r",\s*device='[^']*'", "", raw)
#     # Xoa wrapper tensor(...) => giu gia tri so ben trong
#     # vi du: tensor(0.9951) => 0.9951 | tensor(nan) => nan
#     raw = re.sub(r"tensor\(([^)]*)\)", r"\1", raw)

#     # Buoc 2: parse CSV tu string da clean
#     df_raw = pd.read_csv(StringIO(raw))
#     df_raw.columns = df_raw.columns.str.strip()

#     # Debug: in blocks tim thay
#     blocks_found = sorted(
#         pd.to_numeric(df_raw["block"], errors="coerce")
#         .dropna().unique().astype(int).tolist()
#     )
#     print(f"[DEBUG] Loaded {df_raw.shape[0]} rows | blocks available: {blocks_found}")

#     # Buoc 3: loc client=0 va block chi dinh
#     if "client" in df_raw.columns:
#         df_raw = df_raw[pd.to_numeric(df_raw["client"], errors="coerce") == 0]
#     df_raw["block"] = pd.to_numeric(df_raw["block"], errors="coerce")
#     df_raw = df_raw[df_raw["block"] == block].copy()

#     if df_raw.empty:
#         print(f"[WARN] Khong co du lieu cho block={block}, tra ve dataframe trong")
#         return pd.DataFrame(columns=["round", "eps", "acc", "task"])

#     # 🔧 BUOC 4: xac dinh cot eps va accuracy
#     # Ưu tiên _current (có dữ liệu thực), fallback sang _old nếu cần
    
#     eps_col = None
#     if "eps_current" in df_raw.columns:
#         non_null = df_raw["eps_current"].notna().sum()
#         if non_null > 0:
#             eps_col = "eps_current"
#             print(f"[OK] Dùng eps_current ({non_null}/{len(df_raw)} non-null)")
    
#     # Fallback sang eps_old
#     if eps_col is None and "eps_old" in df_raw.columns:
#         non_null = df_raw["eps_old"].notna().sum()
#         if non_null > 0:
#             eps_col = "eps_old"
#             print(f"[WARN] eps_current không có data, dùng eps_old ({non_null}/{len(df_raw)} non-null)")
    
#     # Nếu không tìm thấy cột eps nào có dữ liệu, để trống
#     if eps_col is None:
#         print(f"[WARN] Không tìm thấy cột eps với dữ liệu, dùng NaN")
#         eps_col = "eps_current"  # Giữ mặc định, sẽ là NaN

#     # Kiểm tra accuracy_current trước
#     acc_col = None
#     if "accuracy_current" in df_raw.columns:
#         non_null = df_raw["accuracy_current"].notna().sum()
#         if non_null > 0:
#             acc_col = "accuracy_current"
#             print(f"[OK] Dùng accuracy_current ({non_null}/{len(df_raw)} non-null)")
    
#     # Fallback sang accuracy_old
#     if acc_col is None and "accuracy_old" in df_raw.columns:
#         non_null = df_raw["accuracy_old"].notna().sum()
#         if non_null > 0:
#             acc_col = "accuracy_old"
#             print(f"[WARN] accuracy_current không có data, dùng accuracy_old ({non_null}/{len(df_raw)} non-null)")
    
#     if acc_col is None:
#         print(f"[WARN] Không tìm thấy cột accuracy với dữ liệu, dùng NaN")
#         acc_col = "accuracy_current"  # Giữ mặc định, sẽ là NaN

#     # Buoc 5: tong hop theo task
#     rows = []
#     for t in range(N_TASKS):
#         d = df_raw[pd.to_numeric(df_raw["task"], errors="coerce") == t].copy()
#         if d.empty:
#             print(f"[SKIP] Khong co du lieu task={t}, block={block}")
#             continue

#         d["round_num"] = pd.to_numeric(d["round"], errors="coerce")
#         d = d.sort_values("round_num").head(N_ROUNDS).copy()

#         d["eps"] = pd.to_numeric(d[eps_col], errors="coerce")
#         d["acc"] = pd.to_numeric(d[acc_col], errors="coerce")

#         out = d[["round_num", "eps", "acc"]].rename(columns={"round_num": "round"})
#         out = out.dropna()
        
#         # 🔧 Bỏ qua task nếu không có dữ liệu hợp lệ
#         if out.empty:
#             print(f"[SKIP] Task {t} không có dữ liệu hợp lệ sau khi xoá NaN")
#             continue
        
#         out["task"] = t
#         rows.append(out)

#     if not rows:
#         print(f"[WARN] Block {block} không có dữ liệu hợp lệ cho bất kỳ task nào")
#         return pd.DataFrame(columns=["round", "eps", "acc", "task"])

#     result = pd.concat(rows, ignore_index=True)
#     print(f"[OK] Block {block}: {len(result)} diem du lieu "
#           f"(eps='{eps_col}', acc='{acc_col}')")
#     return result


# # ─────────────────────────────────────────────
# # Option 1 – Scatter, mau theo task
# # ─────────────────────────────────────────────
# def plot1(df, block, ax=None, standalone=True):
#     if standalone:
#         fig, ax = plt.subplots(figsize=(7, 5))
    
#     if df.empty:
#         ax.text(0.5, 0.5, f"Block {block}: Không có dữ liệu",
#                 transform=ax.transAxes, ha='center', va='center', fontsize=12)
#         ax.set_xlim(0, 1)
#         ax.set_ylim(0, 1)
#     else:
#         for t in range(N_TASKS):
#             d = df[df["task"] == t]
#             if not d.empty:
#                 ax.scatter(d["eps"], d["acc"],
#                            color=TASK_COLORS[t], marker=TASK_MARKERS[t],
#                            s=50, alpha=0.8, label=f"Task {t}")
    
#     ax.set_xlabel("eps")
#     ax.set_ylabel("accuracy")
#     ax.set_title(f"Block {block} - eps vs acc (mau theo task)")
#     ax.legend(title="Task", bbox_to_anchor=(1.01, 1), loc="upper left")
#     ax.grid(True, alpha=0.3)
#     if standalone:
#         plt.tight_layout()


# # ─────────────────────────────────────────────
# # Option 2 – Scatter, mau theo round (colorbar)
# # ─────────────────────────────────────────────
# def plot2(df, block, ax=None, standalone=True):
#     if standalone:
#         fig, ax = plt.subplots(figsize=(7, 5))
    
#     if df.empty:
#         ax.text(0.5, 0.5, f"Block {block}: Không có dữ liệu",
#                 transform=ax.transAxes, ha='center', va='center', fontsize=12)
#         ax.set_xlim(0, 1)
#         ax.set_ylim(0, 1)
#     else:
#         sc = ax.scatter(df["eps"], df["acc"],
#                         c=df["round"], cmap="plasma",
#                         s=50, alpha=0.8)
#         if standalone:
#             plt.colorbar(sc, ax=ax, label="Round")
#         else:
#             plt.colorbar(sc, ax=ax, label="Round", shrink=0.8)
    
#     ax.set_xlabel("eps")
#     ax.set_ylabel("accuracy")
#     ax.set_title(f"Block {block} - eps vs acc (mau theo round)")
#     ax.grid(True, alpha=0.3)
#     if standalone:
#         plt.tight_layout()


# # ─────────────────────────────────────────────
# # Option 3 – Facet 1x5, moi o 1 task
# # ─────────────────────────────────────────────
# def plot3(df, block, standalone=True):
#     fig, axes = plt.subplots(1, N_TASKS, figsize=(15, 4), sharey=True)
#     fig.suptitle(f"Block {block} - eps vs acc | moi subplot = 1 task (mau = round)", y=1.01)
#     sc = None
    
#     for t, ax in enumerate(axes):
#         d = df[df["task"] == t]
#         if d.empty:
#             ax.text(0.5, 0.5, f"Task {t}: No data", 
#                    transform=ax.transAxes, ha='center', va='center')
#         else:
#             sc = ax.scatter(d["eps"], d["acc"],
#                             c=d["round"], cmap="viridis",
#                             s=40, alpha=0.85)
            
#             if len(d) > 1:
#                 z = np.polyfit(d["eps"], d["acc"], 1)
#                 p = np.poly1d(z)
#                 xs = np.linspace(d["eps"].min(), d["eps"].max(), 50)
#                 ax.plot(xs, p(xs), "--", color=TASK_COLORS[t], linewidth=1.5, alpha=0.7)
        
#         ax.set_title(f"Task {t}", color=TASK_COLORS[t], fontweight="bold")
#         ax.set_xlabel("eps")
#         if t == 0:
#             ax.set_ylabel("accuracy")
#         ax.grid(True, alpha=0.3)

#     if sc is not None:
#         plt.colorbar(sc, ax=axes[-1], label="Round", shrink=0.9)
#     plt.tight_layout()


# # ─────────────────────────────────────────────
# # Option 4 – 3D scatter (eps, acc, round) interactive
# # ─────────────────────────────────────────────
# def plot4(df, block):
#     fig = plt.figure(figsize=(9, 7))
#     ax = fig.add_subplot(111, projection="3d")
    
#     if df.empty:
#         ax.text(0, 0, 0, f"Block {block}: Không có dữ liệu")
#     else:
#         for t in range(N_TASKS):
#             d = df[df["task"] == t]
#             if not d.empty:
#                 ax.scatter(d["eps"], d["round"], d["acc"]*100,
#                            color=TASK_COLORS[t], marker=TASK_MARKERS[t],
#                            s=45, alpha=0.8, label=f"Task {t}")
    
#     ax.set_xlabel("eps")
#     ax.set_ylabel("Round")
#     ax.set_zlabel("accuracy")
#     ax.set_title(f"Block {block} - 3D: eps / round / acc")
#     ax.legend(title="Task", loc="upper left")
#     plt.tight_layout()
#     print("[TIP] Keo chuot de xoay bieu do 3D.")


# # ─────────────────────────────────────────────
# # Option 5 – Heatmap task x round
# # ─────────────────────────────────────────────
# def plot5(df, block, hue="eps", standalone=True):
#     if df.empty:
#         fig, ax = plt.subplots(figsize=(14, 4))
#         ax.text(0.5, 0.5, f"Block {block}: Không có dữ liệu",
#                 transform=ax.transAxes, ha='center', va='center')
#         plt.tight_layout()
#         return
    
#     pivot = df.pivot_table(index="task", columns="round", values=hue, aggfunc="first")
#     fig, ax = plt.subplots(figsize=(14, 4))
#     im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", interpolation="nearest")
#     plt.colorbar(im, ax=ax, label=hue)
#     ax.set_xticks(range(len(pivot.columns)))
#     ax.set_xticklabels(pivot.columns.astype(int), fontsize=8)
#     ax.set_yticks(range(N_TASKS))
#     ax.set_yticklabels([f"Task {t}" for t in range(N_TASKS)])
#     ax.set_xlabel("Round")
#     ax.set_title(f"Block {block} - Heatmap {hue} (task x round)")
#     plt.tight_layout()


# # ─────────────────────────────────────────────
# # Option 6 – Scatter + regression line moi task
# # ─────────────────────────────────────────────
# def plot6(df, block, standalone=True):
#     fig, ax = plt.subplots(figsize=(8, 5))
    
#     if df.empty:
#         ax.text(0.5, 0.5, f"Block {block}: Không có dữ liệu",
#                 transform=ax.transAxes, ha='center', va='center', fontsize=12)
#         ax.set_xlim(0, 1)
#         ax.set_ylim(0, 1)
#     else:
#         for t in range(N_TASKS):
#             d = df[df["task"] == t]
#             if not d.empty:
#                 ax.scatter(d["eps"], d["acc"],
#                            color=TASK_COLORS[t], marker=TASK_MARKERS[t],
#                            s=45, alpha=0.7, label=f"Task {t}")
#                 if len(d) > 1:
#                     z = np.polyfit(d["eps"], d["acc"], 1)
#                     p = np.poly1d(z)
#                     xs = np.linspace(d["eps"].min(), d["eps"].max(), 80)
#                     ax.plot(xs, p(xs), color=TASK_COLORS[t], linewidth=2)
    
#     ax.set_xlabel("eps")
#     ax.set_ylabel("accuracy")
#     ax.set_title(f"Block {block} - eps vs acc + trendline moi task")
#     ax.legend(title="Task", bbox_to_anchor=(1.01, 1), loc="upper left")
#     ax.grid(True, alpha=0.3)
#     plt.tight_layout()


# # ─────────────────────────────────────────────
# # Main
# # ─────────────────────────────────────────────
# def main():
#     parser = argparse.ArgumentParser(description="Ve eps vs accuracy tu file CSV duy nhat")
#     parser.add_argument("--block", type=int, required=True,
#                         help="Block can ve (0-4)")
#     parser.add_argument("--plot", default="1",
#                         help="Phuong an: 1/2/3/4/5/6/all  (mac dinh: 1)")
#     parser.add_argument("--save", action="store_true",
#                         help="Luu anh thay vi hien interactive")
#     parser.add_argument("--hue", default="eps", choices=["eps", "acc"],
#                         help="Bien hien thi trong heatmap (option 5, mac dinh: eps)")
#     parser.add_argument("--file", default=DATA_FILE,
#                         help="Duong dan toi file CSV")
#     args = parser.parse_args()

#     df = load_block(args.block, data_file=args.file)

#     choice = args.plot.lower()
#     save   = args.save

#     def _save(name):
#         if save:
#             fname = f"block{args.block}_{name}.png"
#             plt.savefig(fname, dpi=150, bbox_inches="tight")
#             print(f"[SAVE] {fname}")
#             plt.close()

#     if choice in ("1", "all"):
#         plot1(df, args.block)
#         _save("opt1_scatter_task")

#     if choice in ("2", "all"):
#         plot2(df, args.block)
#         _save("opt2_scatter_round")

#     if choice in ("3", "all"):
#         plot3(df, args.block)
#         _save("opt3_facet_task")

#     if choice == "4":
#         plot4(df, args.block)

#     if choice in ("5", "all"):
#         plot5(df, args.block, hue=args.hue)
#         _save(f"opt5_heatmap_{args.hue}")

#     if choice in ("6", "all"):
#         plot6(df, args.block)
#         _save("opt6_regression")

#     if not save:
#         plt.show()


# if __name__ == "__main__":
#     main()

# """
# plot_pairs.py — Vẽ biểu đồ từng pair từ file CSV wandb
# Cách dùng:
#     python plot_pairs.py --file C:\Thu\FCL\block4_gap_eps.csv
#     python plot_pairs.py --file data.csv --out ./charts --dpi 150
# """

# import argparse
# import sys
# import os
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec
# from matplotlib.patches import FancyBboxPatch
# from scipy.ndimage import gaussian_filter1d

# # ─── Palette ──────────────────────────────────────────────────────────────────
# DARK_BG   = "#0d1117"
# PANEL_BG  = "#161b22"
# ACCENT    = "#58a6ff"        # xanh dương sáng
# ACCENT2   = "#f78166"        # cam san hô cho raw dots
# GRID_COL  = "#21262d"
# TEXT_COL  = "#e6edf3"
# MUTED     = "#8b949e"
# FILL_COL  = "#1f6feb"        # fill std band

# FONT_TITLE = dict(fontsize=13, fontweight="bold", color=TEXT_COL, fontfamily="monospace")
# FONT_STAT  = dict(fontsize=9,  color=TEXT_COL,   fontfamily="monospace")
# FONT_AXIS  = dict(fontsize=8,  color=MUTED,       fontfamily="monospace")


# def smooth(y, sigma=2.0):
#     """Gaussian smoothing."""
#     if len(y) < 4:
#         return y
#     return gaussian_filter1d(y.astype(float), sigma=sigma)


# def plot_pair(ax, rounds, values, pair_name, sigma=2.0):
#     """Vẽ một pair lên axes đã cho."""
#     r  = np.array(rounds)
#     v  = np.array(values, dtype=float)
#     mn = np.mean(v)
#     sd = np.std(v, ddof=1)

#     # Smooth mean line — sử dụng tất cả điểm theo thứ tự round
#     sort_idx = np.argsort(r)
#     rs = r[sort_idx]
#     vs = v[sort_idx]
#     vs_sm = smooth(vs, sigma)

#     # --- Vùng ± std nằm ngang (constant band)
#     ax.axhspan(mn - sd, mn + sd,
#                color=FILL_COL, alpha=0.10, linewidth=0, zorder=1)

#     # --- Đường mean nằm ngang
#     ax.axhline(mn, color=ACCENT, linewidth=1.2, linestyle="--",
#                alpha=0.55, zorder=2, label="mean")

#     # --- Fill smooth ± std
#     ax.fill_between(rs, vs_sm - sd, vs_sm + sd,
#                     color=FILL_COL, alpha=0.18, zorder=3)

#     # --- Đường smooth
#     ax.plot(rs, vs_sm, color=ACCENT, linewidth=2.0, zorder=4, label="smooth")

#     # --- Các điểm thật
#     ax.scatter(rs, vs, color=ACCENT2, s=22, zorder=5,
#                edgecolors="none", alpha=0.85, label="data")

#     # --- Styling
#     ax.set_facecolor(PANEL_BG)
#     ax.set_title(f"pair  {pair_name.replace('pair_', '').replace('_', ' → ')}",
#                  **FONT_TITLE, pad=6)
#     ax.tick_params(colors=MUTED, labelsize=7)
#     ax.spines[:].set_color(GRID_COL)
#     ax.grid(color=GRID_COL, linewidth=0.5, linestyle="-", alpha=0.6)
#     ax.set_xlabel("round", **FONT_AXIS)
#     ax.set_ylabel("value", **FONT_AXIS)
#     ax.set_xticks(rs)
#     ax.xaxis.set_tick_params(labelsize=6, rotation=45)

#     # --- Stat box góc trên phải
#     stat_text = (
#         f"μ = {mn:+.3f}\n"
#         f"σ = {sd:.3f}\n"
#         f"n = {len(v)}"
#     )
#     ax.text(
#         0.98, 0.97, stat_text,
#         transform=ax.transAxes,
#         va="top", ha="right",
#         **FONT_STAT,
#         bbox=dict(boxstyle="round,pad=0.4", facecolor="#0d1117",
#                   edgecolor=ACCENT, alpha=0.85, linewidth=0.8),
#         zorder=6,
#     )

#     return mn, sd




# Phien bản vẽ từng biểu đồ riêng pair từ file CSV chuẩn: cột "pair", "round", "value"
# def main():
#     parser = argparse.ArgumentParser(
#         description="Vẽ biểu đồ từng pair từ file CSV wandb")
#     parser.add_argument("--file", required=True,
#                         help="Đường dẫn tới file CSV (vd: C:\\Thu\\FCL\\data.csv)")
#     parser.add_argument("--out", default=None,
#                         help="Thư mục lưu ảnh (mặc định: cùng thư mục với file)")
#     parser.add_argument("--dpi", type=int, default=130,
#                         help="Độ phân giải ảnh (mặc định: 130)")
#     parser.add_argument("--sigma", type=float, default=1.5,
#                         help="Độ mượt Gaussian (mặc định: 1.5)")
#     parser.add_argument("--cols", type=int, default=3,
#                         help="Số cột trong figure (mặc định: 3)")
#     args = parser.parse_args()

#     # ── Đọc file ──────────────────────────────────────────────────────────────
#     fpath = Path(args.file)
#     if not fpath.exists():
#         print(f"[ERROR] Không tìm thấy file: {fpath}")
#         sys.exit(1)

#     df = pd.read_csv(fpath)
#     required = {"pair", "round", "value"}
#     if not required.issubset(df.columns):
#         print(f"[ERROR] File CSV cần có các cột: {required}")
#         print(f"        Cột hiện có: {list(df.columns)}")
#         sys.exit(1)

#     pairs = df["pair"].unique()
#     # Sắp xếp pairs theo thứ tự số
#     def pair_sort_key(p):
#         parts = p.replace("pair_", "").split("_")
#         return tuple(int(x) for x in parts)
#     pairs = sorted(pairs, key=pair_sort_key)

#     n_pairs = len(pairs)
#     n_cols  = args.cols
#     n_rows  = (n_pairs + n_cols - 1) // n_cols

#     print(f"[INFO] File  : {fpath.name}")
#     print(f"[INFO] Pairs : {n_pairs}")
#     print(f"[INFO] Layout: {n_rows} hàng × {n_cols} cột")

#     # ── Vẽ ────────────────────────────────────────────────────────────────────
#     plt.rcParams.update({
#         "figure.facecolor":  DARK_BG,
#         "axes.facecolor":    PANEL_BG,
#         "text.color":        TEXT_COL,
#         "xtick.color":       MUTED,
#         "ytick.color":       MUTED,
#         "axes.edgecolor":    GRID_COL,
#         "axes.labelcolor":   MUTED,
#         "grid.color":        GRID_COL,
#         "font.family":       "monospace",
#     })

#     fig_w = n_cols * 4.8
#     fig_h = n_rows * 3.8 + 1.2          # +1.2 cho title chính

#     fig = plt.figure(figsize=(fig_w, fig_h), facecolor=DARK_BG)
#     fig.suptitle(
#         f"  {fpath.stem}  —  pair-wise rounds",
#         fontsize=15, fontweight="bold",
#         color=TEXT_COL, fontfamily="monospace",
#         x=0.02, ha="left", y=0.995,
#     )

#     gs = gridspec.GridSpec(n_rows, n_cols,
#                            figure=fig,
#                            hspace=0.62, wspace=0.35,
#                            left=0.05, right=0.97,
#                            top=0.96, bottom=0.04)

#     summary_rows = []

#     for idx, pair_name in enumerate(pairs):
#         row, col = divmod(idx, n_cols)
#         ax = fig.add_subplot(gs[row, col])

#         sub = df[df["pair"] == pair_name].sort_values("round")
#         rounds = sub["round"].values
#         values = sub["value"].values

#         mn, sd = plot_pair(ax, rounds, values, pair_name, sigma=args.sigma)
#         summary_rows.append({"pair": pair_name, "mean": mn, "std": sd,
#                               "n_rounds": len(values)})
#         print(f"  {pair_name:<12}  μ={mn:+.4f}  σ={sd:.4f}  n={len(values)}")

#     # Ẩn các subplot trống
#     for idx in range(n_pairs, n_rows * n_cols):
#         row, col = divmod(idx, n_cols)
#         fig.add_subplot(gs[row, col]).set_visible(False)

#     # ── Lưu file ──────────────────────────────────────────────────────────────
#     out_dir = Path(args.out) if args.out else fpath.parent
#     out_dir.mkdir(parents=True, exist_ok=True)
#     out_path = out_dir / f"{fpath.stem}_pairs.png"

#     fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight",
#                 facecolor=DARK_BG)
#     print(f"\n[DONE] Đã lưu ảnh → {out_path}")
#     plt.show()


# if __name__ == "__main__":
#     main()

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
    return y.astype(float)


def pair_sort_key(p):
    parts = p.replace("pair_", "").split("_")
    a, b = int(parts[0]), int(parts[1])
    # Nhóm theo task mới học (b), trong nhóm sort theo task cũ (a)
    # → 0-1 | 0-2,1-2 | 0-3,1-3,2-3 | 0-4,1-4,2-4,3-4 | ...
    return (a,b)


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


# Phiên bản vẽ tất cả pairs trên CÙNG 1 biểu đồ (overlay) — có thể dùng để so sánh trực quan các pair với nhau, nhưng sẽ hơi rối nếu có nhiều pair hoặc nhiều round.
# """
# plot_pairs_overlay.py — Vẽ tất cả pairs trên CÙNG 1 biểu đồ (overlay)
# Cách dùng:
#     python plot_pairs_overlay.py --file C:\Thu\FCL\block4_gap_eps.csv
#     python plot_pairs_overlay.py --file data.csv --out ./charts --dpi 150
# """

# import argparse
# import sys
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import matplotlib.lines as mlines
# from scipy.ndimage import gaussian_filter1d

# DARK_BG  = "#0d1117"
# PANEL_BG = "#161b22"
# GRID_COL = "#21262d"
# TEXT_COL = "#e6edf3"
# MUTED    = "#8b949e"

# PAIR_COLORS = [
#     "#58a6ff", "#f83205", "#56d364", "#941a66",
#     "#000000", "#E8D90D", "#762E2E", "#ffa657",
#     "#ff3064", "#d2a8ff", "#7e85e7", "#9a1e1e",
# ]
# FONT_MONO = "monospace"


# def smooth(y, sigma=1.5):
#     if len(y) < 4:
#         return y.astype(float)
#     return gaussian_filter1d(y.astype(float), sigma=sigma)


# def pair_sort_key(p):
#     parts = p.replace("pair_", "").split("_")
#     a, b = int(parts[0]), int(parts[1])
#     return (b, a)


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--file",   required=True)
#     parser.add_argument("--out",    default=None)
#     parser.add_argument("--dpi",    type=int,   default=140)
#     parser.add_argument("--sigma",  type=float, default=1.5)
#     parser.add_argument("--height", type=float, default=7.0)
#     parser.add_argument("--width",  type=float, default=13.0)
#     args = parser.parse_args()

#     fpath = Path(args.file)
#     if not fpath.exists():
#         print(f"[ERROR] Không tìm thấy file: {fpath}"); sys.exit(1)

#     df = pd.read_csv(fpath)
#     required = {"pair", "round", "value"}
#     if not required.issubset(df.columns):
#         print(f"[ERROR] Cần cột: {required}"); sys.exit(1)
#     # ── Chuyển scalar → radian qua arccos ─────────────────────────────────────
#     raw = df["value"].values.astype(float)
#     df["value"] = np.arccos(np.clip(raw, -1.0, 1.0))   # ∈ [0, π]
#     pairs = sorted(df["pair"].unique(), key=pair_sort_key)

#     n_pairs  = len(pairs)
#     n_rounds = df["round"].nunique()
#     print(f"[INFO] File  : {fpath.name}")
#     print(f"[INFO] Pairs : {n_pairs}  |  Rounds/pair: {n_rounds}")

#     plt.rcParams.update({
#         "figure.facecolor": DARK_BG, "axes.facecolor": PANEL_BG,
#         "text.color": TEXT_COL, "xtick.color": MUTED, "ytick.color": MUTED,
#         "axes.edgecolor": GRID_COL, "axes.labelcolor": MUTED,
#         "grid.color": GRID_COL, "font.family": FONT_MONO,
#     })

#     fig, ax = plt.subplots(figsize=(args.width, args.height), facecolor=DARK_BG)
#     ax.set_facecolor(PANEL_BG)

#     # ── Vẽ từng pair ──────────────────────────────────────────────────────────
#     legend_items = []

#     for i, pname in enumerate(pairs):
#         sub    = df[df["pair"] == pname].sort_values("round")
#         rounds = sub["round"].values
#         vals   = sub["value"].values.astype(float)
#         c      = PAIR_COLORS[i % len(PAIR_COLORS)]
#         label  = pname.replace("pair_", "").replace("_", "→")

#         mn = np.mean(vals)
#         sd = np.std(vals, ddof=1)
#         vs = smooth(vals, args.sigma)

#         # Fill ±std quanh smooth
#         ax.fill_between(rounds, vs - sd, vs + sd,
#                         color=c, alpha=0.10, linewidth=0, zorder=2)

#         # Đường mean nằm ngang
#         ax.hlines(mn, rounds[0], rounds[-1],
#                   colors=c, linewidth=0.9, linestyle="--", alpha=0.45, zorder=3)

#         # Đường smooth
#         ax.plot(rounds, vs, color=c, linewidth=2.0, alpha=0.90, zorder=4)

#         # Điểm thật
#         ax.scatter(rounds, vals, color=c, s=16, zorder=5,
#                    edgecolors=DARK_BG, linewidths=0.4, alpha=0.80)

#         legend_items.append((c, label, mn, sd))

#     # ── X-ticks = số round thật ───────────────────────────────────────────────
#     all_rounds = sorted(df["round"].unique())
#     ax.set_xticks(all_rounds)
#     ax.set_xticklabels([str(r) for r in all_rounds],
#                        fontsize=7, color=MUTED)
#     ax.set_xlim(all_rounds[0] - 0.8, all_rounds[-1] + 0.8)

#     ax.grid(axis="y", color=GRID_COL, linewidth=0.5, alpha=0.7)
#     ax.grid(axis="x", color=GRID_COL, linewidth=0.3, alpha=0.4)
#     ax.spines[:].set_color(GRID_COL)
#     ax.set_ylabel("value", fontsize=9, color=MUTED)
#     ax.set_xlabel("round", fontsize=8, color=MUTED, labelpad=4)

#     # ── Legend góc trên phải — mỗi pair 1 dòng: màu + tên + μ±σ ─────────────
#     handles = []
#     for c, label, mn, sd in legend_items:
#         h = mlines.Line2D([], [], color=c, linewidth=2.0,
#                           marker="o", markersize=4,
#                           markerfacecolor=c, markeredgecolor=DARK_BG,
#                           label=f"{label:<8}  μ={mn:+.3f}  ±{sd:.3f}")
#         handles.append(h)

#     leg = ax.legend(
#         handles=handles,
#         loc="upper right",
#         fontsize=7.2,
#         framealpha=0.92,
#         facecolor=DARK_BG,
#         edgecolor=MUTED,
#         handlelength=1.6,
#         labelspacing=0.38,
#         borderpad=0.7,
#         prop={"family": FONT_MONO, "size": 7.2},
#     )
#     for text in leg.get_texts():
#         text.set_color(TEXT_COL)

#     # ── Title ─────────────────────────────────────────────────────────────────
#     fig.suptitle(
#         f"  {fpath.stem}  —  all pairs overlay",
#         fontsize=12, fontweight="bold", color=TEXT_COL,
#         fontfamily=FONT_MONO, x=0.01, ha="left", y=0.995,
#     )

#     fig.subplots_adjust(left=0.06, right=0.985, top=0.93, bottom=0.09)

#     # ── Lưu ──────────────────────────────────────────────────────────────────
#     out_dir  = Path(args.out) if args.out else fpath.parent
#     out_dir.mkdir(parents=True, exist_ok=True)
#     out_path = out_dir / f"{fpath.stem}_overlay.png"
#     fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight", facecolor=DARK_BG)
#     print(f"\n[DONE] Đã lưu → {out_path}")

#     print(f"\n{'pair':<14} {'mean':>9} {'std':>8} {'min':>9} {'max':>9}")
#     print("─" * 52)
#     for pname in pairs:
#         v = df[df["pair"]==pname]["value"].values.astype(float)
#         print(f"{pname:<14} {np.mean(v):>+9.4f} {np.std(v,ddof=1):>8.4f} "
#               f"{v.min():>+9.4f} {v.max():>+9.4f}")

#     plt.show()

# if __name__ == "__main__":
#     main()


# # """
# # plot_nonlinear_cka.py — Vẽ non-linear CKA overlay từ kernel CKA - linear CKA
# # Cách dùng:
# #     python plot_nonlinear_cka.py --kernel C:\Thu\FCL\block4_kernel_cka.csv --linear C:\Thu\FCL\block4_linear_cka.csv
# #     python plot_nonlinear_cka.py --kernel kernel.csv --linear linear.csv --out ./charts --dpi 150
# # """

# # import argparse
# # import sys
# # from pathlib import Path

# # import numpy as np
# # import pandas as pd
# # import matplotlib.pyplot as plt
# # import matplotlib.patches as mpatches
# # import matplotlib.lines as mlines
# # from scipy.ndimage import gaussian_filter1d

# # DARK_BG  = "#0d1117"
# # PANEL_BG = "#161b22"
# # GRID_COL = "#21262d"
# # TEXT_COL = "#e6edf3"
# # MUTED    = "#8b949e"

# # PAIR_COLORS = [
# #     "#58a6ff", "#f83205", "#56d364", "#941a66",
# #     "#000000", "#E8D90D", "#762E2E", "#ffa657",
# #     "#5e0b95", "#d2a8ff", "#7e85e7", "#9a1e1e",
# # ]
# # FONT_MONO = "monospace"


# # def smooth(y, sigma=1.5):
# #     if len(y) < 4:
# #         return y.astype(float)
# #     return gaussian_filter1d(y.astype(float), sigma=sigma)


# # def pair_sort_key(p):
# #     parts = p.replace("pair_", "").split("_")
# #     a, b = int(parts[0]), int(parts[1])
# #     return (b, a)


# # def main():
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("--kernel", required=True, help="Đường dẫn file kernel CKA")
# #     parser.add_argument("--linear", required=True, help="Đường dẫn file linear CKA")
# #     parser.add_argument("--out",    default=None)
# #     parser.add_argument("--dpi",    type=int,   default=140)
# #     parser.add_argument("--sigma",  type=float, default=1.5)
# #     parser.add_argument("--height", type=float, default=7.0)
# #     parser.add_argument("--width",  type=float, default=13.0)
# #     args = parser.parse_args()

# #     kernel_path = Path(args.kernel)
# #     linear_path = Path(args.linear)
    
# #     if not kernel_path.exists():
# #         print(f"[ERROR] Không tìm thấy file kernel: {kernel_path}"); sys.exit(1)
# #     if not linear_path.exists():
# #         print(f"[ERROR] Không tìm thấy file linear: {linear_path}"); sys.exit(1)

# #     # ── Đọc dữ liệu ──────────────────────────────────────────────────────────
# #     df_kernel = pd.read_csv(kernel_path)
# #     df_linear = pd.read_csv(linear_path)
    
# #     required = {"pair", "round", "value"}
# #     if not required.issubset(df_kernel.columns):
# #         print(f"[ERROR] Kernel CKA cần cột: {required}"); sys.exit(1)
# #     if not required.issubset(df_linear.columns):
# #         print(f"[ERROR] Linear CKA cần cột: {required}"); sys.exit(1)

# #     # ── Tính non-linear CKA = kernel CKA - linear CKA ────────────────────────
# #     # Merge 2 dataframe theo pair và round
# #     df_merged = df_kernel[["pair", "round", "value"]].copy()
# #     df_merged.rename(columns={"value": "kernel_value"}, inplace=True)
    
# #     df_linear_copy = df_linear[["pair", "round", "value"]].copy()
# #     df_linear_copy.rename(columns={"value": "linear_value"}, inplace=True)
    
# #     df_merged = df_merged.merge(df_linear_copy, on=["pair", "round"], how="inner")
    
# #     # Kiểm tra xem có dữ liệu merge được không
# #     if df_merged.empty:
# #         print("[ERROR] Không thể merge kernel CKA và linear CKA (pair/round không khớp)"); 
# #         sys.exit(1)
    
# #     # Tính non-linear CKA
# #     df_merged["value"] = df_merged["kernel_value"]/df_merged["linear_value"]
# #     df = df_merged[["pair", "round", "value"]].copy()
    
# #     pairs = sorted(df["pair"].unique(), key=pair_sort_key)
# #     n_pairs  = len(pairs)
# #     n_rounds = df["round"].nunique()
# #     print(f"[INFO] Kernel file : {kernel_path.name}")
# #     print(f"[INFO] Linear file : {linear_path.name}")
# #     print(f"[INFO] Pairs       : {n_pairs}  |  Rounds/pair: {n_rounds}")
# #     print(f"[INFO] Công thức   : non-linear CKA = kernel CKA / linear CKA")

# #     plt.rcParams.update({
# #         "figure.facecolor": DARK_BG, "axes.facecolor": PANEL_BG,
# #         "text.color": TEXT_COL, "xtick.color": MUTED, "ytick.color": MUTED,
# #         "axes.edgecolor": GRID_COL, "axes.labelcolor": MUTED,
# #         "grid.color": GRID_COL, "font.family": FONT_MONO,
# #     })

# #     fig, ax = plt.subplots(figsize=(args.width, args.height), facecolor=DARK_BG)
# #     ax.set_facecolor(PANEL_BG)

# #     # ── Vẽ từng pair ──────────────────────────────────────────────────────────
# #     legend_items = []

# #     for i, pname in enumerate(pairs):
# #         sub    = df[df["pair"] == pname].sort_values("round")
# #         rounds = sub["round"].values
# #         vals   = sub["value"].values.astype(float)
# #         c      = PAIR_COLORS[i % len(PAIR_COLORS)]
# #         label  = pname.replace("pair_", "").replace("_", "→")

# #         mn = np.mean(vals)
# #         sd = np.std(vals, ddof=1)
# #         vs = smooth(vals, args.sigma)

# #         # # Fill ±std quanh smooth
# #         # ax.fill_between(rounds, vs - sd, vs + sd,
# #         #                 color=c, alpha=0.10, linewidth=0, zorder=2)

# #         # Đường mean nằm ngang
# #         ax.hlines(mn, rounds[0], rounds[-1],
# #                   colors=c, linewidth=0.9, linestyle="--", alpha=0.45, zorder=3)

# #         # Đường smooth
# #         ax.plot(rounds, vs, color=c, linewidth=2.0, alpha=0.90, zorder=4)

# #         # Điểm thật
# #         ax.scatter(rounds, vals, color=c, s=16, zorder=5,
# #                    edgecolors=DARK_BG, linewidths=0.4, alpha=0.80)

# #         legend_items.append((c, label, mn, sd))

# #     # ── X-ticks = số round thật ───────────────────────────────────────────────
# #     all_rounds = sorted(df["round"].unique())
# #     ax.set_xticks(all_rounds)
# #     ax.set_xticklabels([str(r) for r in all_rounds],
# #                        fontsize=7, color=MUTED)
# #     ax.set_xlim(all_rounds[0] - 0.8, all_rounds[-1] + 0.8)

# #     ax.grid(axis="y", color=GRID_COL, linewidth=0.5, alpha=0.7)
# #     ax.grid(axis="x", color=GRID_COL, linewidth=0.3, alpha=0.4)
# #     ax.spines[:].set_color(GRID_COL)
# #     ax.set_ylabel("non-linear CKA", fontsize=9, color=MUTED)
# #     ax.set_xlabel("round", fontsize=8, color=MUTED, labelpad=4)

# #     # ── Legend góc trên phải — mỗi pair 1 dòng: màu + tên + μ±σ ─────────────
# #     handles = []
# #     for c, label, mn, sd in legend_items:
# #         h = mlines.Line2D([], [], color=c, linewidth=2.0,
# #                           marker="o", markersize=4,
# #                           markerfacecolor=c, markeredgecolor=DARK_BG,
# #                           label=f"{label:<8}  μ={mn:+.3f}  ±{sd:.3f}")
# #         handles.append(h)

# #     leg = ax.legend(
# #         handles=handles,
# #         loc="upper right",
# #         fontsize=7.2,
# #         framealpha=0.92,
# #         facecolor=DARK_BG,
# #         edgecolor=MUTED,
# #         handlelength=1.6,
# #         labelspacing=0.38,
# #         borderpad=0.7,
# #         prop={"family": FONT_MONO, "size": 7.2},
# #     )
# #     for text in leg.get_texts():
# #         text.set_color(TEXT_COL)

# #     # ── Title ─────────────────────────────────────────────────────────────────
# #     fig.suptitle(
# #         f"  non-linear CKA (kernel − linear)  —  all pairs overlay",
# #         fontsize=12, fontweight="bold", color=TEXT_COL,
# #         fontfamily=FONT_MONO, x=0.01, ha="left", y=0.995,
# #     )

# #     fig.subplots_adjust(left=0.06, right=0.985, top=0.93, bottom=0.09)

# #     # ── Lưu ──────────────────────────────────────────────────────────────────
# #     out_dir  = Path(args.out) if args.out else kernel_path.parent
# #     out_dir.mkdir(parents=True, exist_ok=True)
# #     out_path = out_dir / "nonlinear_cka_overlay.png"
# #     fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight", facecolor=DARK_BG)
# #     print(f"\n[DONE] Đã lưu → {out_path}")

# #     # ── In bảng thống kê ──────────────────────────────────────────────────────
# #     print(f"\n{'pair':<14} {'mean':>9} {'std':>8} {'min':>9} {'max':>9}")
# #     print("─" * 52)
# #     for pname in pairs:
# #         v = df[df["pair"]==pname]["value"].values.astype(float)
# #         print(f"{pname:<14} {np.mean(v):>+9.4f} {np.std(v,ddof=1):>8.4f} "
# #               f"{v.min():>+9.4f} {v.max():>+9.4f}")

# #     plt.show()

# # if __name__ == "__main__":
# #     main()