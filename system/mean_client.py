import pandas as pd
import numpy as np

# Đọc dữ liệu
df = pd.read_csv('/home/ghostm211/Thu/FCL_3/outputs/client_representation_drift_2_10Client_-hetero-ResNet18.csv')

# Chỉ lấy các cặp cross-client (client1 != client2)
cross = df[df['client1'] != df['client2']].copy()

# Tính trung bình theo block_idx (gộp luôn cả t và tất cả cặp client)
metrics = ['cka', 'sigma', 'eps', 'cosine_similarity', 'align@10', 'align@20']
grouped = cross.groupby('block_idx')[metrics].mean().reset_index()

# In kết quả
print(f"{'Block':<8} {'CKA':>8} {'Sigma':>10} {'Eps':>12} {'CosSim':>10} {'Align@10':>10} {'Align@20':>10}")
print("-" * 72)
for _, row in grouped.iterrows():
    print(f"block {int(row['block_idx']):<3} "
          f"{row['cka']:>8.4f} "
          f"{row['sigma']:>10.3f} "
          f"{row['eps']:>12.6f} "
          f"{row['cosine_similarity']:>10.4f} "
          f"{row['align@10']:>10.4f} "
          f"{row['align@20']:>10.4f}")

# Lưu ra file CSV
output_path = '/home/ghostm211/Thu/FCL_3/outputs/cross_client_avg_by_block_all_tasks.csv'
grouped.to_csv(output_path, index=False)
print(f"\n[Saved] {output_path}")