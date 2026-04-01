# import wandb
# import pandas as pd

# api = wandb.Api()

# runs = {
#     "ours": "ducthu2003/TARGET/to7336gy",
#     "finetune": "ducthu2003/TARGET/qoue8ueb",
#     "lwf": "ducthu2003/TARGET/tjsya4ed",
#     "ewc": "ducthu2003/TARGET/xhg3yluy",
#     "icarl": "ducthu2003/TARGET/57cahmi0",
# }
# task_key = "Task_4, accuracy"

# dfs = []

# for model_name, run_id in runs.items():

#     print(f"Downloading {model_name}")

#     run = api.run(run_id)

#     rows = []

#     for row in run.scan_history():
#         if task_key in row:
#             rows.append({
#                 "_step": row["_step"],
#                 model_name: row[task_key]
#             })

#     df = pd.DataFrame(rows)

#     dfs.append(df)

# # merge theo step
# final_df = dfs[0]

# for df in dfs[1:]:
#     final_df = pd.merge(final_df, df, on="_step", how="outer")

# # sort step
# final_df = final_df.sort_values("_step")
# # bỏ dòng không có accuracy
# final_df = final_df.dropna(subset=["ours","finetune","lwf","ewc","icarl"], how="all")
# # reset index
# final_df = final_df.reset_index(drop=True)

# print("\nPreview:")
# print(final_df.head())

# final_df.to_csv("Task4_accuracy_all_models.csv", index=False)

# print("\nSaved to Task4_accuracy_all_models.csv")


import wandb
import pandas as pd

api = wandb.Api()
run = api.run("ducthu2003/TARGET/ccdok8rl")

history = run.history(keys=["Task_1_acc"])
df = pd.DataFrame(history)
print(df.head())

df.to_csv("Task_1_acc.csv", index=False)
print("Saved to Task_1_acc.csv")

import pandas as pd
import matplotlib.pyplot as plt

# Từ điển ánh xạ label → màu
colors = {
    'Task 0': 'blue',
    # 'FFA-LoRA': 'orange',
    # 'FedSA-LoRA': 'red',
    # 'FLoRA-CA': 'black',
}

# Custom labels bạn muốn vẽ
custom_labels = ['Task 0']

# Dữ liệu
df = pd.read_csv('Task_1_acc.csv')
x = df.iloc[:, 0]
max_columns = [col for col in df.columns if col.endswith('Task_1_acc')]

# Map label to columns theo thứ tự custom_labels
label_to_column = dict(zip(custom_labels, max_columns))

# Sắp xếp các label theo thứ tự trong `colors`
sorted_labels = [label for label in colors if label in custom_labels]

# Nếu có label không trong `colors`, thêm vào cuối
other_labels = [label for label in custom_labels if label not in colors]
final_labels = sorted_labels + other_labels

# Cấu hình smoothing
ema_span = 30
std_window = 3
plt.rcParams.update({'font.size': 18})

# Vẽ hình
plt.figure(figsize=(8, 6))
for label in final_labels:
    if label not in label_to_column:
        print(f"⚠️ Label '{label}' không khớp với bất kỳ cột dữ liệu nào.")
        continue

    col = label_to_column[label]
    color = colors.get(label, 'black')  # fallback nếu label không có màu

    ema = df[col].ewm(span=ema_span, adjust=False).mean()
    std = df[col].rolling(window=std_window, min_periods=1).std()

    linestyle = '--' if label == 'STAMP' else '-'  # Dùng nét đứt cho STAMP
    plt.plot(x, ema, label=label, linewidth=2.5, color=color, linestyle=linestyle)
    plt.fill_between(x, ema - std, ema + std, alpha=0.2, color=color)

# Giao diện
plt.xlabel('Task Steps')
plt.ylabel('Acc')
plt.title('Task 0 Accuracy')
plt.legend()
plt.grid(True)
plt.xlim(left=0, right=4)
plt.ylim(bottom=70, top=80)
plt.tight_layout()
plt.savefig("MNLI-grad.pdf", bbox_inches='tight')
plt.show()
