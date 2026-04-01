import pandas as pd
import matplotlib.pyplot as plt

colors = {
    'finetune': 'blue',
    'ours': 'green',
    'lwf': 'red',
    'ewc': 'orange',
    'icarl': 'purple'
}

df = pd.read_csv('Task_accuracy_all_models.csv')


# trục X = số class sau mỗi task
x = [20, 40, 60, 80, 100]

plt.rcParams.update({'font.size': 16})
plt.figure(figsize=(6,5))

for label, color in colors.items():

    if label not in df.columns:
        continue

    plt.plot(
        x,
        df[label],
        label=label,
        color=color,
        marker='x',      # dấu X tại mỗi task
        markersize=8,
        linewidth=2
    )

plt.xlabel("Classes")
plt.ylabel("Accuracy")
plt.title("5 tasks")

plt.legend()
plt.grid(True)

plt.xlim(20,100)
plt.ylim(10,80)

plt.tight_layout()
plt.savefig("tasks_accuracy.pdf")
plt.show()