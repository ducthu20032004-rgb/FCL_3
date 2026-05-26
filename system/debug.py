
# # # import torch
# # # from torchvision.models import resnet18
# # # from torch.utils.data import DataLoader
# # # from system.measure import load_model_with_head
# # # from system.utils.data_utils import read_client_data_FCL_cifar10

# # # test_data_t = read_client_data_FCL_cifar10(
# # #                 0, task=1,
# # #                 classes_per_task=2,
# # #                 count_labels=False, train=False
# # #             )

# # # raw_sd = torch.load('weight-client-avg-resnet18/client_0_task_1.pt', map_location='cpu')
# # # print('head.weight shape:', raw_sd['head.weight'].shape)  # expect (10, 512)

# # # # Load model task 1 và thử predict trên 1 batch của task 0
# # # model_debug = load_model_with_head('weight-client-avg-resnet18/client_0_task_1.pt', num_classes=10)
# # # loader_debug = DataLoader(test_data_t, batch_size=10, shuffle=False)
# # # x, y = next(iter(loader_debug))
# # # with torch.no_grad():
# # #     out = model_debug(x)
# # # print('True labels (task 0):', y)
# # # print('Predicted labels:', torch.argmax(out, dim=1))
# # # print('Logits:\n', out)

# # # import torch


# # # sd1 = torch.load('weight-client-avg-resnet18/client_0_task_1.pt', map_location='cpu')
# # # sd2 = torch.load('weight-client-avg-resnet18/client_1_task_1.pt', map_location='cpu')

# # # for k in sd1:
# # #     if not torch.allclose(sd1[k], sd2[k]):
# # #         print("DIFFERENT at", k)
# # #         break
# # # else:
# # #     print("IDENTICAL")

# # import matplotlib.pyplot as plt
# # import numpy as np

# # # --- Dữ liệu từ log bạn gửi (block 0 → block 4, round 0 → 9) ---
# # blocks = [0, 1, 2, 3, 4]

# # # epsilon từng round, block 0→4
# # epsilon = [
# #     [0.1612, 0.0896, 10.5388, 42.6045, 46.2603],
# #     [0.0912, 0.0789, 9.7629, 40.9637, 39.4238],
# #     [0.1174, 0.0661, 11.0996, 36.7517, 33.9683],
# #     [0.1048, 0.0880, 8.5887, 35.3171, 33.1160],
# #     [0.0998, 0.0744, 9.3891, 35.6355, 34.2299],
# #     [0.1025, 0.0793, 8.7048, 36.9261, 32.0563],
# #     [0.1179, 0.0778, 9.5368, 32.9956, 30.6864],
# #     [0.0931, 0.0686, 8.3490, 30.8989, 30.9740],
# #     [0.1336, 0.0883, 8.9007, 35.6457, 31.2280],
# #     [0.0924, 0.0715, 9.6322, 32.1583, 30.0]  # block4 ε approx
# # ]

# # # CKA từng round, block 0→4
# # cka = [
# #     [0.0184, 0.0241, 0.0369, 0.0670, 0.1643],
# #     [0.0198, 0.0281, 0.0451, 0.0874, 0.2011],
# #     [0.0205, 0.0313, 0.0478, 0.0901, 0.2117],
# #     [0.0201, 0.0304, 0.0511, 0.0967, 0.2100],
# #     [0.0208, 0.0303, 0.0502, 0.0969, 0.2187],
# #     [0.0210, 0.0309, 0.0498, 0.1019, 0.2328],
# #     [0.0214, 0.0312, 0.0486, 0.0989, 0.2320],
# #     [0.0220, 0.0323, 0.0522, 0.1047, 0.2431],
# #     [0.0217, 0.0309, 0.0487, 0.1023, 0.2442],
# #     [0.0219, 0.0317, 0.0508, 0.1077, 0.245]  # block4 CKA approx
# # ]

# # # Forgetting theo round (chỉ đo block cuối, block4)
# # forgetting = [88.2, 88.2, 90.0, 90.6, 90.5, 90.8, 90.65, 90.6, 91.0, 90.75]

# # # --- Vẽ epsilon và CKA theo block ---
# # plt.figure(figsize=(12,5))
# # for r in range(len(epsilon)):
# #     plt.plot(blocks, epsilon[r], label=f'ε round {r}', linestyle='--', alpha=0.4, color='red')
# #     plt.plot(blocks, cka[r], label=f'CKA round {r}', linestyle='-', alpha=0.4, color='blue')

# # plt.xlabel('Block')
# # plt.ylabel('Value')
# # plt.title('Evolution of ε (red dashed) and CKA (blue) across blocks')
# # plt.legend([],[], frameon=False)  # ẩn legend dài dòng
# # plt.grid(True)
# # plt.show()

# # # --- Vẽ Forgetting theo round ---
# # plt.figure(figsize=(8,4))
# # plt.plot(range(len(forgetting)), forgetting, marker='o', color='purple')
# # plt.xlabel('Round')
# # plt.ylabel('Forgetting (%)')
# # plt.title('Catastrophic Forgetting over Rounds (block4)')
# # plt.grid(True)
# # plt.show()
# """
# plot_eps_gap.py — Vẽ eps_curr vs eps_old đồng thời, theo từng pair nối tiếp
# Cách dùng:
#     python plot_eps_gap.py --curr C:\Thu\FCL\block4_eps_curr.csv \
#                            --old  C:\Thu\FCL\block4_eps_old.csv
#     python plot_eps_gap.py --curr curr.csv --old old.csv --out ./charts --dpi 150
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

# # ─── Palette ──────────────────────────────────────────────────────────────────
# DARK_BG  = "#0d1117"
# PANEL_BG = "#161b22"
# GRID_COL = "#21262d"
# TEXT_COL = "#e6edf3"
# MUTED    = "#8b949e"
# SEP_COL  = "#30363d"

# # Màu cho từng pair
# PAIR_COLORS = [
#     "#58a6ff",  # blue
#     "#f78166",  # coral
#     "#56d364",  # green
#     "#e3b341",  # gold
#     "#bc8cff",  # purple
#     "#39d3f0",  # cyan
#     "#ff7b72",  # red-orange
#     "#ffa657",  # orange
#     "#79c0ff",  # light blue
#     "#d2a8ff",  # lavender
#     "#7ee787",  # mint
#     "#ff9a9a",  # pink
# ]

# # curr = màu pair gốc sáng, old = phiên bản tối/muted hơn
# def darken(hex_color, factor=0.55):
#     """Làm tối màu hex."""
#     h = hex_color.lstrip("#")
#     r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
#     r2 = int(r * factor + 30)
#     g2 = int(g * factor + 30)
#     b2 = int(b * factor + 30)
#     return f"#{min(r2,255):02x}{min(g2,255):02x}{min(b2,255):02x}"

# FONT_MONO = "monospace"


# def smooth(y, sigma=1.5):
#     if len(y) < 4:
#         return y.astype(float)
#     return gaussian_filter1d(y.astype(float), sigma=sigma)


# def pair_sort_key(p):
#     parts = p.replace("pair_", "").split("_")
#     a, b = int(parts[0]), int(parts[1])
#     # Nhom theo task moi hoc (b), trong nhom sort theo task cu (a)
#     # -> 0-1 | 0-2,1-2 | 0-3,1-3,2-3 | 0-4,1-4,...
#     return (b, a)


# def load_csv(path):
#     fpath = Path(path)
#     if not fpath.exists():
#         print(f"[ERROR] Không tìm thấy file: {fpath}")
#         sys.exit(1)
#     df = pd.read_csv(fpath)
#     required = {"pair", "round", "value"}
#     if not required.issubset(df.columns):
#         print(f"[ERROR] File '{fpath.name}' cần cột: {required} — có: {list(df.columns)}")
#         sys.exit(1)
#     return df


# def main():
#     parser = argparse.ArgumentParser(
#         description="Vẽ eps_curr vs eps_old theo pair nối tiếp trên 1 biểu đồ")
#     parser.add_argument("--curr", required=True, help="CSV của eps_curr")
#     parser.add_argument("--old",  required=True, help="CSV của eps_old")
#     parser.add_argument("--out",  default=None,  help="Thư mục lưu ảnh")
#     parser.add_argument("--dpi",  type=int,   default=140)
#     parser.add_argument("--sigma", type=float, default=1.5, help="Độ mượt Gaussian")
#     parser.add_argument("--height", type=float, default=7.0, help="Chiều cao figure")
#     parser.add_argument("--gap-panel", action="store_true",
#                         help="Thêm panel phụ vẽ gap (curr - old) bên dưới")
#     args = parser.parse_args()

#     df_curr = load_csv(args.curr)
#     df_old  = load_csv(args.old)

#     curr_path = Path(args.curr)

#     # Lấy danh sách pair chung, sắp xếp
#     pairs_curr = set(df_curr["pair"].unique())
#     pairs_old  = set(df_old["pair"].unique())
#     pairs_common = sorted(pairs_curr & pairs_old, key=pair_sort_key)
#     pairs_only_curr = pairs_curr - pairs_old
#     pairs_only_old  = pairs_old  - pairs_curr

#     if pairs_only_curr:
#         print(f"[WARN] Chỉ có trong curr: {sorted(pairs_only_curr)}")
#     if pairs_only_old:
#         print(f"[WARN] Chỉ có trong old : {sorted(pairs_only_old)}")

#     n_pairs  = len(pairs_common)
#     n_rounds = df_curr["round"].nunique()
#     GAP      = 1  # khoảng trống giữa các pair

#     print(f"[INFO] curr : {curr_path.name}")
#     print(f"[INFO] old  : {Path(args.old).name}")
#     print(f"[INFO] Pairs: {n_pairs}  |  Rounds/pair: {n_rounds}")

#     # ── Tính offset x cho mỗi pair ───────────────────────────────────────────
#     pair_meta = {}
#     x_cursor  = 0

#     for i, pname in enumerate(pairs_common):
#         sub_c = df_curr[df_curr["pair"] == pname].sort_values("round")
#         sub_o = df_old [df_old ["pair"] == pname].sort_values("round")

#         rounds_c = sub_c["round"].values
#         vals_c   = sub_c["value"].values.astype(float)
#         rounds_o = sub_o["round"].values
#         vals_o   = sub_o["value"].values.astype(float)

#         # Lấy round chung
#         common_rounds = np.intersect1d(rounds_c, rounds_o)
#         mask_c = np.isin(rounds_c, common_rounds)
#         mask_o = np.isin(rounds_o, common_rounds)
#         rc = rounds_c[mask_c]; vc = vals_c[mask_c]
#         ro = rounds_o[mask_o]; vo = vals_o[mask_o]
#         gap_vals = vc - vo   # curr - old

#         color_curr = PAIR_COLORS[i % len(PAIR_COLORS)]
#         color_old  = darken(color_curr, factor=0.5)

#         pair_meta[pname] = {
#             "idx":         i,
#             "color_curr":  color_curr,
#             "color_old":   color_old,
#             "x_start":     x_cursor,
#             "x_global_c":  x_cursor + rc,
#             "x_global_o":  x_cursor + ro,
#             "rounds_c":    rc,
#             "rounds_o":    ro,
#             "vals_c":      vc,
#             "vals_o":      vo,
#             "gap":         gap_vals,
#             "mean_c":      np.mean(vc),
#             "std_c":       np.std(vc,  ddof=1),
#             "mean_o":      np.mean(vo),
#             "std_o":       np.std(vo,  ddof=1),
#             "mean_gap":    np.mean(gap_vals),
#             "std_gap":     np.std(gap_vals, ddof=1),
#             "n":           len(rc),
#         }
#         x_cursor += len(rc) + GAP

#     total_x = x_cursor - GAP

#     # ── Figure layout ─────────────────────────────────────────────────────────
#     plt.rcParams.update({
#         "figure.facecolor": DARK_BG,
#         "axes.facecolor":   PANEL_BG,
#         "text.color":       TEXT_COL,
#         "xtick.color":      MUTED,
#         "ytick.color":      MUTED,
#         "axes.edgecolor":   GRID_COL,
#         "axes.labelcolor":  MUTED,
#         "grid.color":       GRID_COL,
#         "font.family":      FONT_MONO,
#     })

#     # Mỗi round ~0.22 inch, cap 40 để không quá rộng
#     fig_w = min(max(16, n_pairs * n_rounds * 0.22), 40)

#     if args.gap_panel:
#         fig, (ax, ax_gap) = plt.subplots(
#             2, 1, figsize=(fig_w, args.height + 2.5),
#             facecolor=DARK_BG,
#             gridspec_kw={"height_ratios": [3, 1.2], "hspace": 0.12}
#         )
#     else:
#         fig, ax = plt.subplots(figsize=(fig_w, args.height),
#                                facecolor=DARK_BG)
#         ax_gap = None

#     ax.set_facecolor(PANEL_BG)
#     if ax_gap is not None:
#         ax_gap.set_facecolor(PANEL_BG)

#     # ── Vẽ từng pair ──────────────────────────────────────────────────────────
#     for pname, m in pair_meta.items():
#         cc   = m["color_curr"]
#         co   = m["color_old"]
#         xgc  = m["x_global_c"]
#         xgo  = m["x_global_o"]
#         vc   = m["vals_c"]
#         vo   = m["vals_o"]
#         xs   = m["x_start"]
#         n    = m["n"]
#         idx  = m["idx"]

#         # --- Nền pair xen kẽ
#         shade = 0.04 if idx % 2 == 0 else 0.08
#         ax.axvspan(xs - 0.5, xs + n - 0.5,
#                    color=cc, alpha=shade, linewidth=0, zorder=1)

#         # --- Đường kẻ dọc phân cách
#         if idx > 0:
#             ax.axvline(xs - 0.5, color=SEP_COL, linewidth=1.0,
#                        linestyle="-", alpha=1.0, zorder=2)

#         # --- Fill gap giữa smooth curr và smooth old
#         sort_idx = np.argsort(xgc)
#         xgc_s = xgc[sort_idx]
#         vc_sm = smooth(vc, args.sigma)[sort_idx]
#         vo_sm = smooth(vo, args.sigma)[np.argsort(xgo)]

#         ax.fill_between(
#             xgc_s, vc_sm, vo_sm,
#             where=(vc_sm >= vo_sm), interpolate=True,
#             color=cc, alpha=0.18, zorder=3,
#         )
#         ax.fill_between(
#             xgc_s, vc_sm, vo_sm,
#             where=(vc_sm < vo_sm), interpolate=True,
#             color="#f78166", alpha=0.18, zorder=3,
#         )

#         # --- eps_old: vẽ TRƯỚC (dưới)
#         # Glow effect: vẽ đường dày mờ trước rồi đường mảnh sắc nét sau
#         ax.plot(xgo, vo_sm, color=co, linewidth=6,
#                 linestyle="-", alpha=0.10, zorder=4, solid_capstyle="round")
#         ax.plot(xgo, vo_sm, color=co, linewidth=2.0,
#                 linestyle=(0, (4, 2)),   # nét đứt dài
#                 alpha=0.90, zorder=5, solid_capstyle="round")
#         # Marker vuông rỗng cho old
#         ax.scatter(xgo, vo, s=28, zorder=6, alpha=0.85,
#                    facecolors="none", edgecolors=co,
#                    linewidths=1.4, marker="s")

#         # --- eps_curr: vẽ SAU (trên)
#         # Glow effect
#         ax.plot(xgc, vc_sm, color=cc, linewidth=6,
#                 linestyle="-", alpha=0.12, zorder=7, solid_capstyle="round")
#         ax.plot(xgc, vc_sm, color=cc, linewidth=2.2,
#                 linestyle="-", alpha=0.97, zorder=8, solid_capstyle="round")
#         # Marker tròn đặc cho curr
#         ax.scatter(xgc, vc, color=cc, s=22, zorder=9,
#                    edgecolors=DARK_BG, linewidths=0.5,
#                    alpha=0.92, marker="o")

#         # --- Mean lines
#         ax.hlines(m["mean_c"], xs - 0.3, xs + n - 0.7,
#                   colors=cc, linewidth=0.9, linestyle=":",
#                   alpha=0.45, zorder=4)
#         ax.hlines(m["mean_o"], xs - 0.3, xs + n - 0.7,
#                   colors=co, linewidth=0.9, linestyle=":",
#                   alpha=0.45, zorder=4)

#         # --- Gap panel
#         if ax_gap is not None:
#             gv  = m["gap"]
#             gx  = xgc
#             gv_sm = smooth(gv, args.sigma)

#             ax_gap.axvspan(xs - 0.5, xs + n - 0.5,
#                            color=cc, alpha=shade, linewidth=0, zorder=1)
#             if idx > 0:
#                 ax_gap.axvline(xs - 0.5, color=SEP_COL, linewidth=1.0, zorder=2)

#             ax_gap.fill_between(gx, 0, gv_sm,
#                                  where=(gv_sm >= 0),
#                                  color=cc, alpha=0.30, zorder=3)
#             ax_gap.fill_between(gx, 0, gv_sm,
#                                  where=(gv_sm < 0),
#                                  color=co, alpha=0.30, zorder=3)
#             ax_gap.plot(gx, gv_sm, color=cc, linewidth=1.6, zorder=5)
#             ax_gap.scatter(gx, gv, color=cc, s=12, zorder=6,
#                            edgecolors=DARK_BG, linewidths=0.3, alpha=0.8)
#             ax_gap.axhline(0, color=MUTED, linewidth=0.8, linestyle="-", alpha=0.4)

#     # ── X-ticks ───────────────────────────────────────────────────────────────
#     tick_pos, tick_lbl = [], []
#     for pname, m in pair_meta.items():
#         for r, xp in zip(m["rounds_c"], m["x_global_c"]):
#             tick_pos.append(xp); tick_lbl.append(str(r))

#     ax.set_xticks(tick_pos)
#     ax.set_xticklabels(tick_lbl, fontsize=5.5, rotation=70, color=MUTED)
#     ax.set_xlim(-1, total_x + 1)

#     if ax_gap is not None:
#         ax_gap.set_xticks(tick_pos)
#         ax_gap.set_xticklabels(tick_lbl, fontsize=5.5, rotation=70, color=MUTED)
#         ax_gap.set_xlim(-1, total_x + 1)
#         ax_gap.set_ylabel("gap\n(curr−old)", fontsize=8, color=MUTED,
#                           fontfamily=FONT_MONO, labelpad=4)
#         ax_gap.grid(axis="y", color=GRID_COL, linewidth=0.5, alpha=0.7)
#         ax_gap.grid(axis="x", visible=False)
#         ax_gap.spines[:].set_color(SEP_COL)
#     else:
#         ax.set_xlabel("round  (per pair)", fontsize=8, color=MUTED,
#                       fontfamily=FONT_MONO, labelpad=4)

#     # ── Secondary x-axis: nhãn pair ──────────────────────────────────────────
#     ax2 = ax.twiny()
#     ax2.set_facecolor("none")
#     ax2.spines[:].set_visible(False)
#     ax2.set_xlim(ax.get_xlim())

#     ctr_pos, ctr_lbl, ctr_col = [], [], []
#     for pname, m in pair_meta.items():
#         xc = m["x_start"] + (m["n"] - 1) / 2
#         ctr_pos.append(xc)
#         ctr_lbl.append(pname.replace("pair_", "").replace("_", " → "))
#         ctr_col.append(m["color_curr"])

#     ax2.set_xticks(ctr_pos)
#     ax2.set_xticklabels(ctr_lbl, fontsize=8.5, fontweight="bold",
#                         fontfamily=FONT_MONO)
#     for tick, col in zip(ax2.get_xticklabels(), ctr_col):
#         tick.set_color(col)

#     # ── Grid & labels ─────────────────────────────────────────────────────────
#     ax.grid(axis="y", color=GRID_COL, linewidth=0.5, linestyle="-", alpha=0.7)
#     ax.grid(axis="x", visible=False)
#     ax.spines[:].set_color(SEP_COL)
#     ax.set_ylabel("eps value", fontsize=9, color=MUTED, fontfamily=FONT_MONO)

#     # ── Stat box + chú thích đường — VẼ TRONG TỪNG PAIR, góc trên ───────────
#     _, yhi = ax.get_ylim()
#     # Tính khoảng giá trị để offset box xuống dưới top
#     yrange = ax.get_ylim()[1] - ax.get_ylim()[0]

#     for pname, m in pair_meta.items():
#         cc  = m["color_curr"]
#         co  = m["color_old"]
#         xs  = m["x_start"]
#         n   = m["n"]

#         # x: góc phải của pair (tính bằng data coords)
#         x_box = xs + n - 0.6

#         # y: luôn đặt ở top của axes (dùng data coords)
#         y_box = ax.get_ylim()[1] - yrange * 0.01

#         # Màu viền = màu pair, dòng curr đậm, old nhạt
#         stat_txt = (
#             f"━● eps_curr\n"
#             f"   \u03bc={m['mean_c']:+.3f} \u03c3={m['std_c']:.3f}\n"
#             f"╌□ eps_old\n"
#             f"   \u03bc={m['mean_o']:+.3f} \u03c3={m['std_o']:.3f}\n"
#             f"\u0394\u03bc={m['mean_gap']:+.4f}"
#         )
#         ax.text(
#             x_box, y_box, stat_txt,
#             va="top", ha="right",
#             fontsize=5.8, fontfamily=FONT_MONO,
#             color=TEXT_COL,
#             bbox=dict(
#                 boxstyle="round,pad=0.4",
#                 facecolor=DARK_BG,
#                 edgecolor=cc,
#                 alpha=0.92,
#                 linewidth=0.9,
#             ),
#             zorder=10,
#         )

#     # ── Legend kiểu đường — 1 hàng ngang phía dưới figure ───────────────────
#     legend_handles = [
#         mlines.Line2D([], [], color="white", linewidth=2.0,
#                       linestyle="-", marker="o", markersize=4,
#                       markerfacecolor="white", markeredgecolor=DARK_BG,
#                       label="eps_curr  (━ ●  nét liền, chấm tròn đặc)"),
#         mlines.Line2D([], [], color="white", linewidth=1.6,
#                       linestyle=(0, (4, 2)), marker="s", markersize=4,
#                       markerfacecolor="none", markeredgecolor="white",
#                       label="eps_old   (╌ □  nét đứt, vuông rỗng)"),
#         mpatches.Patch(color="white", alpha=0.25,
#                        label="vùng fill = gap (curr \u2212 old)"),
#     ]

#     leg = ax.legend(
#         handles=legend_handles,
#         loc="lower center",
#         bbox_to_anchor=(0.5, -0.13),
#         fontsize=7.5,
#         framealpha=0.88,
#         facecolor=DARK_BG,
#         edgecolor=GRID_COL,
#         ncol=3,
#         handlelength=1.8,
#         labelspacing=0.3,
#         columnspacing=2.0,
#         prop={"family": FONT_MONO, "size": 7.5},
#     )
#     for text in leg.get_texts():
#         text.set_color(TEXT_COL)

#     # ── Title ─────────────────────────────────────────────────────────────────
#     stem_curr = Path(args.curr).stem
#     stem_old  = Path(args.old).stem
#     fig.suptitle(
#         f"{stem_curr}  vs  {stem_old}",
#         fontsize=11, fontweight="bold", color=TEXT_COL,
#         fontfamily=FONT_MONO, x=0.01, ha="left", y=0.995,
#     )

#     if args.gap_panel:
#         fig.subplots_adjust(
#             left=0.055, right=0.985,
#             top=0.91,   bottom=0.13,
#             hspace=0.12,
#         )
#     else:
#         fig.subplots_adjust(
#             left=0.055, right=0.985,
#             top=0.91,   bottom=0.13,
#         )

#     # ── Lưu ──────────────────────────────────────────────────────────────────
#     out_dir = Path(args.out) if args.out else Path(args.curr).parent
#     out_dir.mkdir(parents=True, exist_ok=True)
#     suffix  = "_gap_panel" if args.gap_panel else ""
#     out_path = out_dir / f"eps_curr_vs_old{suffix}_combined.png"
#     fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight", facecolor=DARK_BG)
#     print(f"\n[DONE] Đã lưu → {out_path}")

#     # ── Bảng tóm tắt ─────────────────────────────────────────────────────────
#     print(f"\n{'pair':<14} {'μ_curr':>9} {'μ_old':>9} {'Δμ(curr-old)':>13} {'σ_gap':>8}")
#     print("─" * 58)
#     for pname, m in pair_meta.items():
#         print(f"{pname:<14} {m['mean_c']:>+9.4f} {m['mean_o']:>+9.4f} "
#               f"{m['mean_gap']:>+13.4f} {m['std_gap']:>8.4f}")

#     plt.show()


# if __name__ == "__main__":
#     main()
import numpy as np
all_class_orders = np.load('./dataset/class_order/class_order_cifar10.npy', allow_pickle=True)
print("shape:", all_class_orders.shape)
print("dtype:", all_class_orders.dtype)
print("content:", all_class_orders)
print("content[0]:", all_class_orders[0])