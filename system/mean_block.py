import pandas as pd
import os

base_path = r"C:\Thu\FCL"
base_name = r"outputs\cosine_similarity.csv"

total = 0
for block in range(5):
    file_path = os.path.join(base_path, base_name.format(block))
    df = pd.read_csv(file_path)
    mean_val = df["value"].mean()
    print(f"block{block}: {mean_val:.6f}")
    total += mean_val

print(f"\nTổng trung bình: {total:.6f}")
print(f"Trung bình chung: {total / 5:.6f}")