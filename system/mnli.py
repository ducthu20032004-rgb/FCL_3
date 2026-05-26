import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("C:\Thu\FCL\outputs\representation_drift_temporal_14_4-hetero-ResNet18.csv")

blocks = sorted(df['block_idx'].unique())
tasks  = sorted(df['t'].unique())

TASK_COLORS = ['#1a237e', '#7b1fa2', '#e91e63', '#ff6f00', '#f9d71c']

def make_fig(raw_col, scale, y_label, title, fname):
    # tính mean+std trên 45 cặp, group (t, block_idx)
    agg = df.groupby(['t','block_idx'])[raw_col].agg(['mean','std']).reset_index()
    # tính mean+std tổng hợp theo block (trung bình qua task)
    agg_b = agg.groupby('block_idx')[['mean','std']].mean().reset_index()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_facecolor('#f8f9ff')
    fig.patch.set_facecolor('white')

    for t_idx, t_val in enumerate(tasks):
        color = TASK_COLORS[t_idx]
        sub   = agg[agg.t == t_val].sort_values('block_idx')
        mean  = sub['mean'].values * scale
        std   = sub['std'].values  * scale
        x     = sub['block_idx'].values
        ax.fill_between(x, mean - std, mean + std,
                        color=color, alpha=0.18, linewidth=0, zorder=2)
        ax.plot(x, mean, color=color, linewidth=2.2, marker='o',
                ms=6, zorder=3, label=f"task t={t_val}")

    # ── Bảng mean ± std góc trên trái ───────────────────────────────────────
    lines = ["Block   mean  ±  std"]
    for _, row in agg_b.iterrows():
        m = row['mean'] * scale
        s = row['std']  * scale
        lines.append(f"  {int(row['block_idx'])}:  {m:.3f} ± {s:.3f}")
    ax.text(0.01, 0.99, "\n".join(lines),
            transform=ax.transAxes, fontsize=8.5,
            verticalalignment='top', horizontalalignment='left',
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      alpha=0.9, edgecolor='#cccccc'))

    ax.set_xlabel("Block (Layer)", fontsize=11)
    ax.set_ylabel(y_label,         fontsize=11)
    ax.set_title(title,            fontsize=12, fontweight='bold', pad=10)
    ax.set_xticks(blocks)
    ax.grid(True, linestyle='--', alpha=0.5, color='#ccccdd')
    # legend bên dưới bảng text
    ax.legend(fontsize=9.5, framealpha=0.9, loc='upper left',
              bbox_to_anchor=(0.01, 0.60))

    plt.tight_layout()
    plt.savefig(fname, dpi=180, bbox_inches='tight')
    print(f"Saved → {fname}")
    plt.show()


make_fig('eps', 1.0,
         "Magnitude",
         "Cross-Client Representation Drift · EPS\n(ResNet-18 hetero, 45 pairs client)",
         "/mnt/user-data/outputs/final3_eps.png")

make_fig('cka', 100.0,
         "Similarity %",
         "Cross-Client Representation Similarity · CKA×100\n(ResNet-18 hetero, 45 pairs client)",
         "/mnt/user-data/outputs/final3_cka.png")

print("Done!")