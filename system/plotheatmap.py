"""
Multi-Heatmap — So sánh CKA drift của nhiều phương pháp song song.

File format (mỗi file):
    pair,round,value
    pair_i_j,0,0.56...     (i < j, i=eval task, j=training task)

Layout:
    Mỗi file input → 1 heatmap (subplot)
    Rows (Y) = eval task           : T0, T1, T2, T3
    Cols (X) = sequential training : After T1, After T2, After T3, After T4
    Cell(i,j) = CKA của task-i đo tại thời điểm sau khi train xong task-j

Usage:
    # Tất cả cùng round (hoặc all)
    python plot_heatmap_multi.py --files er.csv derpp.csv cls_er.csv esmer.csv
    python plot_heatmap_multi.py --files er.csv derpp.csv --round 24

    # Mỗi file một round riêng (phải đúng số lượng file)
    python plot_heatmap_multi.py --files er.csv derpp.csv cls_er.csv esmer.csv \
                                 --rounds 24 all 10 24

    # Thêm options
    python plot_heatmap_multi.py --files er.csv derpp.csv \
        --rounds 24 all \
        --titles "ER" "DER++" \
        --ncols 2 \
        --out compare.png
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path


# ─────────────────────────────────────────────
# Load + aggregate
# ─────────────────────────────────────────────

def parse_round(s: str):
    """'all' → 'all', otherwise → int"""
    s = str(s).strip().lower()
    return "all" if s == "all" else int(s)


def load(filepath: str, round_sel):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    def parse(s):
        s = s.strip().replace("pair", "", 1)
        parts = [x for x in s.split("_") if x.strip() != ""]
        return int(parts[0]), int(parts[1])

    df[["i", "j"]] = pd.DataFrame(df["pair"].apply(parse).tolist(), index=df.index)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["round"] = pd.to_numeric(df["round"], errors="coerce").astype("Int64")
    df = df[df["i"] < df["j"]]

    if round_sel == "all":
        agg = df.groupby(["i", "j"])["value"].mean().reset_index()
    else:
        r = int(round_sel)
        agg = df[df["round"] == r][["i", "j", "value"]].copy()
        if agg.empty:
            avail = sorted(df["round"].dropna().unique().tolist())
            raise ValueError(f"Round {r} not found in '{filepath}'. Available: {avail}")

    return agg


# ─────────────────────────────────────────────
# Build display matrix
# ─────────────────────────────────────────────

def build_matrix(agg: pd.DataFrame, n_tasks: int):
    n = n_tasks - 1
    mat = np.full((n, n), np.nan)

    for _, row in agg.iterrows():
        i, j, v = int(row["i"]), int(row["j"]), float(row["value"])
        # i = eval task  → row (Y-axis)
        # j = training task, "After T{j}" → col (X-axis), 0-based so j-1
        r = i
        c = j - 1
        if 0 <= r < n and 0 <= c < n:
            mat[r, c] = v

    return mat


# ─────────────────────────────────────────────
# Plot multi-heatmap (side by side)
# ─────────────────────────────────────────────

def round_label_str(r) -> str:
    return "all rounds (mean)" if r == "all" else f"round {r}"

# ─────────────────────────────────────────────
# NEW: Tính mean theo block
# ─────────────────────────────────────────────
def compute_block_means(mat: np.ndarray, n_tasks: int, block_size: int = 2):
    """
    Tính mean theo các block:
    - All pairs
    - Diagonal (same task)
    - Off-diagonal (forgetting / interference)
    - Early tasks, Late tasks, Cross blocks
    """
    n = n_tasks - 1
    results = {}

    valid = ~np.isnan(mat)
    results["Overall"] = float(np.mean(mat[valid]))

    # Diagonal
    diag = np.diag(mat)
    results["Diagonal (same task)"] = float(np.nanmean(diag))

    # Off-diagonal
    off_diag = mat.copy()
    np.fill_diagonal(off_diag, np.nan)
    results["Off-diagonal"] = float(np.nanmean(off_diag))

    # Block theo group of tasks
    if block_size > 1:
        for start in range(0, n, block_size):
            end = min(start + block_size, n)
            block = mat[start:end, start:end]
            results[f"Block {start}-{end-1}"] = float(np.nanmean(block))

            # Cross block
            if start + block_size < n:
                cross = mat[start:end, end:]
                results[f"Cross {start}-{end-1} → later"] = float(np.nanmean(cross))

    return results
def plot_multi(matrices, titles, round_sels, n_tasks, y_label,
               ncols=None, global_colorscale=False, suptitle=None, cmap_colors=None):
    """
    matrices   : list of np.ndarray
    titles     : list of str
    round_sels : list of (int | 'all') — one per matrix
    """
    n_plots = len(matrices)
    if ncols is None:
        ncols = n_plots
    nrows = int(np.ceil(n_plots / ncols))

    n = n_tasks - 1

    # Y-axis = eval task : T0, T1, T2, T3
    row_labels = [f"T{i}" for i in range(n)]

    # X-axis = sequential training timeline : After T1, After T2, ...
    col_labels = [f"After T{j+1}" for j in range(n)]

    cell = 1.6
    fig_w = ncols * (n * cell + 1.4) + 0.6
    fig_h = nrows * (n * cell + 2.0) + (0.8 if suptitle else 0.4)

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)

    # Colormap — càng lớn càng đỏ
    if cmap_colors is None:
        cmap_colors = ["#2166ac", "#92c5de", "#f7f7f7", "#f4a582", "#d73027"]
    cmap = mcolors.LinearSegmentedColormap.from_list("bdr", cmap_colors, N=256)
    cmap.set_bad("#e8e8e8")

    # Global color scale
    if global_colorscale:
        all_vals = np.concatenate([m[~np.isnan(m)] for m in matrices])
        g_vmin, g_vmax = float(all_vals.min()), float(all_vals.max())
    else:
        g_vmin = g_vmax = None

    ims = []
    for idx, (mat, title, rsel) in enumerate(zip(matrices, titles, round_sels)):
        r_idx = idx // ncols
        c_idx = idx % ncols
        ax = axes[r_idx][c_idx]

        if global_colorscale:
            vmin, vmax = g_vmin, g_vmax
        else:
            valid = mat[~np.isnan(mat)]
            vmin = float(valid.min()) if len(valid) else 0.0
            vmax = float(valid.max()) if len(valid) else 1.0

        masked = np.ma.masked_invalid(mat)
        im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        ims.append(im)

        # Colorbar riêng cho từng subplot
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.06)
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=16)
        # if idx == 0:
        #     cb.set_label("Forgetting",fontsize=20)
        # elif idx == n_plots - 1:
        #     cb.set_label(y_label, fontsize=20)

        # Cell annotations
        for rr in range(n):
            for cc in range(n):
                v = mat[rr, cc]
                if not np.isnan(v):
                    bright = (v - vmin) / (vmax - vmin + 1e-9)
                    tc = "white" if bright < 0.35 or bright > 0.75 else "#222222"
                    ax.text(cc, rr, f"{v:.3f}",
                            ha="center", va="center",
                            fontsize=20, color=tc, fontweight="bold")

        # X-axis ticks — sequential training timeline
        ax.set_xticks(range(n))
        ax.set_xticklabels(col_labels, fontsize=20)
        ax.xaxis.set_ticks_position("bottom")
        ax.set_xlabel("Sequence Training →", fontsize=20, labelpad=6)

        # Y-axis ticks — eval task
        ax.set_yticks(range(n))
        if idx % ncols == 0:
            # Chỉ subplot cột đầu mới hiện ytick labels
            ax.set_yticklabels(row_labels, fontsize=20)
            ax.set_ylabel("Eval Task", fontsize=25, labelpad=6)
        else:
            ax.set_yticklabels([])

        # Grid lines
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", length=0)

        # Title + round label
        rstr = round_label_str(rsel)
        ax.set_title(f"{title}\n", fontsize=20, fontweight="bold", pad=10)

    # Ẩn axes thừa
    for idx in range(n_plots, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.subplots_adjust(wspace=0.35, hspace=0.55)

    if suptitle:
        fig.suptitle(suptitle, fontsize=20, fontweight="bold", y=1.01)

    plt.tight_layout()
    plt.show()
    return fig


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Plot multiple CKA heatmaps side by side for method comparison."
    )
    p.add_argument("--files",   nargs="+", required=True,
                   help="CSV file paths, one per method")
    p.add_argument("--round",   default="all",
                   help="Default round for ALL files ('all' or int). "
                        "Ignored for files that have a matching --rounds entry.")
    p.add_argument("--rounds",  nargs="*", default=None,
                   help="Per-file round list (must match --files count). "
                        "E.g.: --rounds 24 all 10 24  "
                        "Overrides --round for each file individually.")
    p.add_argument("--titles",  nargs="*", default=None,
                   help="Subplot titles (defaults to filename stems)")
    p.add_argument("--ntasks",  type=int, default=5,
                   help="Total number of tasks (default: 5)")
    p.add_argument("--y",       type=str, default="Magnitude",
                   help="Colorbar label")
    p.add_argument("--ncols",   type=int, default=None,
                   help="Subplot columns (default: all in one row)")
    p.add_argument("--global-scale", action="store_true",
                   help="Use shared global color scale across all plots (default: per-plot scale)")
    p.add_argument("--suptitle", default=None,
                   help="Overall figure title")
    args = p.parse_args()

    n_files = len(args.files)

    # ── Resolve per-file round list ───────────────────────────────────────────
    if args.rounds is not None:
        if len(args.rounds) != n_files:
            p.error(f"--rounds must have exactly {n_files} entries "
                    f"(one per file), got {len(args.rounds)}.")
        round_sels = [parse_round(r) for r in args.rounds]
    else:
        default_round = parse_round(args.round)
        round_sels = [default_round] * n_files

    # ── Titles ────────────────────────────────────────────────────────────────
    if args.titles and len(args.titles) == n_files:
        titles = args.titles
    else:
        titles = [Path(f).stem for f in args.files]
        if args.titles:
            print("[!] --titles count mismatch, using filenames as titles.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"[+] Files  : {args.files}")
    print(f"[+] Rounds : {round_sels}")
    #print(f"[+] Titles : {titles}")
    print(f"[+] Tasks  : {args.ntasks}")

    # ── Load data ─────────────────────────────────────────────────────────────
    matrices = []
    for f, t, rsel in zip(args.files, titles, round_sels):
        print(f"\n── {t} | file: {f} | round: {rsel} ──")
        agg = load(f, rsel)
        mat = build_matrix(agg, args.ntasks)
        
        # In matrix
        df_show = pd.DataFrame(
            np.round(mat, 3),
            index=[f"T{i}" for i in range(args.ntasks-1)],
            columns=[f"After T{j+1}" for j in range(args.ntasks-1)],
        )
        print("Matrix:")
        print(df_show.to_string())
        
        # === TÍNH BLOCK MEANS ===
        block_means = compute_block_means(mat, args.ntasks, 5)
        print("\nBlock Means:")
        for k, v in block_means.items():
            print(f"  {k:30s}: {v:.4f}")
        
        matrices.append(mat)
    
    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_multi(
        matrices=matrices,
        titles=titles,
        round_sels=round_sels,
        n_tasks=args.ntasks,
        y_label=args.y,
        ncols=args.ncols,
        global_colorscale=args.global_scale,
        suptitle=args.suptitle,
    )


if __name__ == "__main__":
    main()