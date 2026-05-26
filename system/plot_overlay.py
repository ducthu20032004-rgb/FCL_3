"""
plot_blocks_heatmap.py — Heatmap (pair × block) + sparkline trend bên phải
  - Trái: heatmap seaborn, mỗi ô = mean(25 rounds), in giá trị + tô màu
  - Phải: line chart overlay 10 pairs, cùng thứ tự trục Y

Cách dùng:
    python plot_blocks_heatmap.py --dir C:\Thu\FCL
    python plot_blocks_heatmap.py --dir C:\Thu\FCL --client Client0 --dpi 160
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# ─── Theme ────────────────────────────────────────────────────────────────────
DARK_BG  = "#0d1117"
PANEL_BG = "#161b22"
CARD_BG  = "#1c2128"
GRID_COL = "#21262d"
TEXT_COL = "#e6edf3"
MUTED    = "#8b949e"

# 10 màu tương phản cho 10 pairs
PAIR_COLORS = [
    "#58a6ff",  # 0→1  blue
    "#f78166",  # 0→2  coral
    "#3fb950",  # 1→2  green
    "#e3b341",  # 0→3  gold
    "#bc8cff",  # 1→3  purple
    "#ff7b72",  # 2→3  red-orange
    "#ffa657",  # 0→4  orange
    "#39d3f0",  # 1→4  cyan
    "#f778ba",  # 2→4  hot pink
    "#7ee787",  # 3→4  mint
]

FONT_MONO = "monospace"


def pair_sort_key(p):
    parts = p.replace("pair_", "").split("_")
    a, b = int(parts[0]), int(parts[1])
    return (b, a)


def load_block(fpath: Path) -> dict:
    df = pd.read_csv(fpath)
    required = {"pair", "round", "value"}
    if not required.issubset(df.columns):
        print(f"[WARN] Thiếu cột trong {fpath.name}")
        return {}
    return {pname: grp["value"].mean() for pname, grp in df.groupby("pair")}


def make_heatmap_cmap():
    """Custom dark→blue colormap: thấp=tối, cao=sáng xanh."""
    colors = [
        "#0d1117",   # 0.0  — tối hoàn toàn (DARK_BG)
        "#0c2d48",   # 0.25
        "#1158a0",   # 0.55
        "#2b7fd4",   # 0.75
        "#58a6ff",   # 0.90
        "#a3d4ff",   # 1.0  — sáng nhất
    ]
    return LinearSegmentedColormap.from_list("dark_blue", colors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",    required=True)
    parser.add_argument("--client", default="Client0")
    parser.add_argument("--out",    default=None)
    parser.add_argument("--dpi",    type=int, default=160)
    parser.add_argument("--method", default = "eps_old", help="Chỉ dùng để chọn file pattern, không ảnh hưởng plot")
    args = parser.parse_args()

    base_dir = Path(args.dir)
    n_blocks = 5
    file_pat = f"{args.client}_block{{b}}_{args.method}.csv"

    # ── Đọc dữ liệu ──────────────────────────────────────────────────────────
    block_data = []
    for b in range(n_blocks):
        fp = base_dir / file_pat.format(b=b)
        if not fp.exists():
            print(f"[ERROR] Không tìm thấy: {fp}"); sys.exit(1)
        block_data.append(load_block(fp))
        print(f"[INFO] Block {b}: {fp.name} — {len(block_data[-1])} pairs")

    all_pairs = sorted(
        {p for bd in block_data for p in bd}, key=pair_sort_key
    )
    n_pairs = len(all_pairs)

    # ── Build DataFrame: rows=pairs, cols=blocks ──────────────────────────────
    col_labels = [f"Block {b}" for b in range(n_blocks)]
    row_labels  = [p.replace("pair_", "").replace("_", " → ") for p in all_pairs]

    data = np.full((n_pairs, n_blocks), np.nan)
    for b_idx, bd in enumerate(block_data):
        for p_idx, pname in enumerate(all_pairs):
            if pname in bd:
                data[p_idx, b_idx] = bd[pname]

    df_heat = pd.DataFrame(data, index=row_labels, columns=col_labels)

    # ── Delta column: B4 - B0 ─────────────────────────────────────────────────
    delta = data[:, -1] - data[:, 0]

    # ── Global value range ────────────────────────────────────────────────────
    vmin = np.nanmin(data)
    vmax = np.nanmax(data)

    # ── Figure layout: heatmap (wide) + sparkline (narrow) ───────────────────
    plt.rcParams.update({
        "figure.facecolor": DARK_BG,
        "axes.facecolor":   CARD_BG,
        "text.color":       TEXT_COL,
        "xtick.color":      MUTED,
        "ytick.color":      MUTED,
        "axes.edgecolor":   GRID_COL,
        "axes.labelcolor":  MUTED,
        "grid.color":       GRID_COL,
        "font.family":      FONT_MONO,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
    })

    fig = plt.figure(figsize=(16, 6.5), facecolor=DARK_BG)

    # GridSpec: [heatmap | delta_bar | sparkline]  widths ratio 5:0.8:3.5
    gs = gridspec.GridSpec(
        1, 3,
        figure=fig,
        width_ratios=[5, 0.7, 3.5],
        wspace=0.06,
    )
    ax_heat  = fig.add_subplot(gs[0])
    ax_delta = fig.add_subplot(gs[1])
    ax_spark = fig.add_subplot(gs[2])

    # ══════════════════════════════════════════════════════════════════════════
    # 1. HEATMAP
    # ══════════════════════════════════════════════════════════════════════════
    cmap = make_heatmap_cmap()

    sns.heatmap(
        df_heat,
        ax=ax_heat,
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        annot=True,
        fmt=".4f",
        annot_kws={"size": 9, "family": FONT_MONO, "weight": "bold"},
        linewidths=1.2,
        linecolor=DARK_BG,
        cbar=False,
        square=False,
    )

    # Style annotation text color dựa trên độ sáng ô
    norm_vals = (data - vmin) / (vmax - vmin + 1e-9)
    for i, row in enumerate(ax_heat.texts):
        # texts đi theo thứ tự row-major
        p_idx = i // n_blocks
        b_idx = i %  n_blocks
        brightness = norm_vals[p_idx, b_idx]
        row.set_color(TEXT_COL if brightness > 0.35 else MUTED)
        row.set_fontsize(8.5)

    # Row labels màu theo pair
    ax_heat.set_yticklabels(
        ax_heat.get_yticklabels(),
        rotation=0, fontsize=10, fontfamily=FONT_MONO,
    )
    for tick, c in zip(ax_heat.get_yticklabels(), PAIR_COLORS[:n_pairs]):
        tick.set_color(c)

    ax_heat.set_xticklabels(
        ax_heat.get_xticklabels(),
        rotation=0, fontsize=10, fontweight="bold",
        color=TEXT_COL, fontfamily=FONT_MONO,
    )
    ax_heat.set_ylabel("")
    ax_heat.set_xlabel("")
    ax_heat.tick_params(left=False, bottom=False)
    ax_heat.set_facecolor(CARD_BG)

    # Title heatmap
    ax_heat.set_title(
        "mean accuracy  (avg of 25 rounds per block)",
        fontsize=10, color=MUTED, fontfamily=FONT_MONO, pad=10, loc="left",
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 2. DELTA BAR (B4 - B0)
    # ══════════════════════════════════════════════════════════════════════════
    y_pos  = np.arange(n_pairs)
    bar_h  = 0.55

    for i, (d, c) in enumerate(zip(delta, PAIR_COLORS[:n_pairs])):
        color = "#f85149" if d < 0 else "#3fb950"
        ax_delta.barh(i, d, height=bar_h, color=color, alpha=0.85,
                      left=0, zorder=3)

    ax_delta.axvline(0, color=MUTED, linewidth=0.7, alpha=0.5, zorder=2)
    ax_delta.set_yticks([])
    ax_delta.set_ylim(-0.5, n_pairs - 0.5)
    ax_delta.invert_yaxis()
    ax_delta.set_facecolor(CARD_BG)
    ax_delta.spines[:].set_color(GRID_COL)
    ax_delta.spines["left"].set_visible(False)
    ax_delta.tick_params(axis="x", labelsize=7, colors=MUTED)
    ax_delta.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax_delta.set_title("Δ B4−B0", fontsize=9, color=MUTED,
                       fontfamily=FONT_MONO, pad=10)
    ax_delta.grid(axis="x", color=GRID_COL, linewidth=0.4,
                  linestyle=":", alpha=0.6)

    # ══════════════════════════════════════════════════════════════════════════
    # 3. SPARKLINE TREND
    # ══════════════════════════════════════════════════════════════════════════
    x_blocks = np.arange(n_blocks)
    x_labels_short = [f"B{i}" for i in range(n_blocks)]

    # Shared y range với padding
    y_pad   = (vmax - vmin) * 0.08
    y_lo    = vmin - y_pad
    y_hi    = vmax + y_pad

    for p_idx in range(n_pairs):
        c    = PAIR_COLORS[p_idx % len(PAIR_COLORS)]
        yv   = data[p_idx]
        mask = ~np.isnan(yv)
        xv   = x_blocks[mask]
        yvals = yv[mask]

        # Glow
        ax_spark.plot(xv, yvals, color=c, linewidth=6,  alpha=0.08, zorder=2)
        # Main line
        ax_spark.plot(xv, yvals, color=c, linewidth=2.0, alpha=0.92, zorder=4,
                      solid_capstyle="round")
        # Dots
        ax_spark.scatter(xv, yvals, color=c, s=45, zorder=5,
                         edgecolors=DARK_BG, linewidths=0.8)

        # Label pair ở đầu (B0) — sát bên trái
        ax_spark.text(
            -0.18, yvals[0],
            row_labels[p_idx],
            ha="right", va="center",
            fontsize=7.5, color=c,
            fontfamily=FONT_MONO,
        )

        # Giá trị tại B0 và B4
        ax_spark.text(
            xv[0] + 0.06, yvals[0] + (y_hi - y_lo) * 0.012,
            f"{yvals[0]:.3f}",
            ha="left", va="bottom",
            fontsize=6.5, color=c, alpha=0.80,
            fontfamily=FONT_MONO,
        )
        ax_spark.text(
            xv[-1] + 0.06, yvals[-1] + (y_hi - y_lo) * 0.012,
            f"{yvals[-1]:.3f}",
            ha="left", va="bottom",
            fontsize=6.5, color=c, alpha=0.80,
            fontfamily=FONT_MONO,
        )

    ax_spark.set_xlim(-1.0, n_blocks - 0.2)
    ax_spark.set_ylim(y_lo, y_hi)
    ax_spark.set_xticks(x_blocks)
    ax_spark.set_xticklabels(x_labels_short, fontsize=9,
                              fontfamily=FONT_MONO, color=TEXT_COL)
    ax_spark.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax_spark.tick_params(axis="y", labelsize=8, colors=MUTED)
    ax_spark.set_facecolor(CARD_BG)
    ax_spark.spines[:].set_color(GRID_COL)
    ax_spark.spines["left"].set_color(GRID_COL)
    ax_spark.grid(axis="y", color=GRID_COL, linewidth=0.5, alpha=0.7)
    ax_spark.grid(axis="x", color=GRID_COL, linewidth=0.3,
                  linestyle=":", alpha=0.4)
    ax_spark.set_title(
        "trend  (overlay all pairs)",
        fontsize=10, color=MUTED, fontfamily=FONT_MONO, pad=10, loc="left",
    )

    # ── Colorbar manual (dọc bên phải heatmap, trong fig) ────────────────────
    cbar_ax = fig.add_axes([0.365, 0.13, 0.008, 0.72])   # [left,bot,w,h]
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.yaxis.set_tick_params(color=MUTED, labelsize=7)
    cbar.outline.set_edgecolor(GRID_COL)
    plt.setp(cbar.ax.yaxis.get_ticklabels(),
             color=MUTED, fontfamily=FONT_MONO, fontsize=7)

    # ── Suptitle ──────────────────────────────────────────────────────────────
    fig.suptitle(
        f"  {args.client}  —  task-pair accuracy across blocks",
        fontsize=13, fontweight="bold",
        color=TEXT_COL, fontfamily=FONT_MONO,
        x=0.01, ha="left", y=1.01,
    )

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.subplots_adjust(left=0.09, right=0.97, top=0.93, bottom=0.09)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir  = Path(args.out) if args.out else base_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.client}_heatmap.png"
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight", facecolor=DARK_BG)
    print(f"\n[DONE] Đã lưu → {out_path}")

    # ── Terminal table ────────────────────────────────────────────────────────
    print(f"\n{'pair':<14}" + "".join(f"  {'Block '+str(b):>8}" for b in range(n_blocks)) + "   Δ(B4-B0)")
    print("─" * (14 + n_blocks * 10 + 12))
    for p_idx, lbl in enumerate(row_labels):
        row = f"{lbl:<14}"
        for v in data[p_idx]:
            row += f"  {v:.5f}" if not np.isnan(v) else "       NaN"
        row += f"  {delta[p_idx]:+.5f}"
        print(row)

    plt.show()


if __name__ == "__main__":
    main()