import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ──────────────────────────────────────────────
# 1. Load dữ liệu
# ──────────────────────────────────────────────
CSV_PATH = r"C:\Thu\FCL\outputs\neuron_heatmap.csv"          # ← đổi thành đường dẫn file CSV của bạn

df = pd.read_csv(CSV_PATH)

# Lấy các cột neuron (n0 → n511)
neuron_cols = [c for c in df.columns if c.startswith("n")]
data = df[neuron_cols].values       # shape: (125, 512)

print(f"Data shape: {data.shape}")  # kỳ vọng (125, 512)

# ──────────────────────────────────────────────
# 2. Chuẩn bị nhãn trục Y  (task-round)
# ──────────────────────────────────────────────
N_TASKS  = 5
N_ROUNDS = 25   # round per task → tổng = 125

# Tạo nhãn "T1·R0", "T1·R1", ...
y_labels = [f"T{row['task']}·R{int(row['round_idx'])}" for _, row in df.iterrows()]

# Vị trí tick (cứ 5 round đánh dấu 1 lần để không bị rối)
tick_every = 5
y_tick_pos    = list(range(0, len(y_labels), tick_every))
y_tick_labels = [y_labels[i] for i in y_tick_pos]

# Đường kẻ phân cách giữa các task
task_boundaries = []
tasks = df["task"].values
for i in range(1, len(tasks)):
    if tasks[i] != tasks[i - 1]:
        task_boundaries.append(i - 0.5)

# ──────────────────────────────────────────────
# 3. Vẽ heatmap
# ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(20, 10))

# Dùng diverging colormap vì giá trị âm/dương
vmax = np.percentile(np.abs(data), 98)   # clip outlier nhẹ
im = ax.imshow(
    data,
    aspect="auto",
    cmap="RdBu_r",
    vmin=-vmax,
    vmax=vmax,
    interpolation="nearest",
)

# ── Colorbar ──
cbar = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
cbar.set_label("Activation value", fontsize=11)

# ── Trục X ──
ax.set_xlabel("Neuron dimension (0 – 511)", fontsize=12)
x_ticks = list(range(0, 512, 64))
ax.set_xticks(x_ticks)
ax.set_xticklabels([str(t) for t in x_ticks], fontsize=9)

# ── Trục Y ──
ax.set_ylabel("Training round  (Task · Round)", fontsize=12)
ax.set_yticks(y_tick_pos)
ax.set_yticklabels(y_tick_labels, fontsize=7)

# ── Đường kẻ phân cách task ──
for yb in task_boundaries:
    ax.axhline(y=yb, color="yellow", linewidth=1.2, linestyle="--", alpha=0.8)

# ── Nhãn task ở giữa mỗi dải ──
task_ids    = df["task"].unique()
task_starts = [np.where(tasks == t)[0][0] for t in task_ids]
task_ends   = [np.where(tasks == t)[0][-1] for t in task_ids]

for t, ts, te in zip(task_ids, task_starts, task_ends):
    mid = (ts + te) / 2
    ax.text(
        -22, mid, f"Task {t}",
        va="center", ha="right",
        fontsize=9, color="#333333", fontweight="bold",
    )

# ── Tiêu đề ──
ax.set_title(
    "ResNet-18 Neuron Activation Heatmap  |  512-dim × 125 rounds  (5 tasks × 25 rounds)",
    fontsize=14, fontweight="bold", pad=14,
)

plt.tight_layout()
plt.savefig("heatmap_resnet18.png", dpi=180, bbox_inches="tight")
print("Đã lưu: heatmap_resnet18.png")
plt.show()