# import wandb
# import pandas as pd
# import argparse
# import os

# def runwandb(entity, project, run_id, method, block, out_dir):
#     api = wandb.Api()
#     run = api.run(f"{entity}/{project}/runs/{run_id}")

#     rows = list(run.scan_history(page_size=10000))  # fix thiếu data
#     df = pd.DataFrame(rows)
    
#     print(f"  Raw rows từ W&B: {len(df)}")

#     overlap_cols = [
#         f"{block}/task0/{method}",
#         f"{block}/task1/{method}",
#         f"{block}/task2/{method}",
#         f"{block}/task3/{method}",
#         f"{block}/task4/{method}",
#     ]

#     cols_exist = [c for c in overlap_cols if c in df.columns]

#     if not cols_exist:
#         print(f"[ERROR] Không tìm thấy cột nào cho {block}/{method}")
#         return None

#     # Melt về format cũ: round_global + block4/{method}
#     df_sub = df[["round_global"] + cols_exist].copy()
#     df_melt = df_sub.melt(id_vars="round_global", value_vars=cols_exist,
#                           var_name="task", value_name=f"block4/{method}")

#     df_melt = df_melt.dropna(subset=[f"block4/{method}"])
#     df_melt = df_melt.sort_values("round_global").reset_index(drop=True)

#     df_final = df_melt[["round_global", f"block4/{method}"]]

#     return df_final


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--entity",  default="ducthu2003")
#     parser.add_argument("--project", default="Representation Drift Measurement")
#     parser.add_argument("--run_id",  default="jcabxqbj")
#     parser.add_argument("--method",  default="forgetting")
#     parser.add_argument("--block",   default="block4")
#     parser.add_argument("--out_dir", default=r"C:\Thu\FCL\material_experiment\dynamic")
#     args = parser.parse_args()

#     print(f"\n{'='*60}")
#     print(f"Entity: {args.entity} | Project: {args.project}")
#     print(f"Run ID: {args.run_id} | Method: {args.method} | Block: {args.block}")
#     print(f"{'='*60}\n")

#     df = runwandb(args.entity, args.project, args.run_id, args.method, args.block, args.out_dir)

#     if df is None:
#         print("[ERROR] Download thất bại")
#         exit(1)

#     print(df)
#     print(f"\nShape: {df.shape}")

#     os.makedirs(args.out_dir, exist_ok=True)
#     out_path = os.path.join(args.out_dir, f"{args.method}_{args.block}.csv")
#     df.to_csv(out_path, index=False)
#     print(f"✅ Saved: {out_path}")
# import wandb


# Phiên bản chuẩn downdata từ W&B, xử lý dữ liệu và lưu thành CSV với format: pair, round, value
"""
Download metrics từ W&B với format chuẩn: pair, round, value
Loại bỏ tất cả việc xử lý khác - chỉ download dữ liệu
"""

import re
import os
import sys
import argparse
import pandas as pd
import wandb

def download_from_wandb(entity, project, run_id, method, block):
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/runs/{run_id}")
    
    # Tăng page_size để không bỏ sót rows
    rows = list(run.scan_history(page_size=1000000))
    df = pd.DataFrame(rows)
    
    overlap_cols = [
        f"{block}/pair_0_1/{method}",
        f"{block}/pair_0_2/{method}",
        f"{block}/pair_0_3/{method}",
        f"{block}/pair_0_4/{method}",
        f"{block}/pair_1_2/{method}",
        f"{block}/pair_1_3/{method}",
        f"{block}/pair_1_4/{method}",
        f"{block}/pair_2_3/{method}",
        f"{block}/pair_2_4/{method}",
        f"{block}/pair_3_4/{method}",
        # f"{block}/{method}/pair0_1",
        # f"{block}/{method}/pair0_2",
        # f"{block}/{method}/pair0_3",
        # f"{block}/{method}/pair0_4",
        # f"{block}/{method}/pair1_2",
        # f"{block}/{method}/pair1_3",
        # f"{block}/{method}/pair1_4",
        # f"{block}/{method}/pair2_3",
        # f"{block}/{method}/pair2_4",
        # f"{block}/{method}/pair3_4",
    ]
    
    cols_exist = [c for c in overlap_cols if c in df.columns]
    
    if not cols_exist:
        print(f"[ERROR] Không tìm thấy cột nào cho {block}/{method}")
        return None
    
    print(f"✓ Tìm thấy {len(cols_exist)} cột pair")
    print(f"  Raw rows từ W&B: {len(df)}")
    
    # ─────────────────────────────────────────────────────────────
    # Xử lý từng pair riêng biệt thay vì melt chung
    # Vì W&B log sparse: mỗi row chỉ có data của 1 số pair
    # ─────────────────────────────────────────────────────────────
    results = []
    
    for col in cols_exist:
        # Tách pair name
        pair_name = re.search(r'pair_\d+_\d+', col).group(0)
        #pair_name = re.search(r'pair\d+_\d+', col).group(0)
        # Lấy rows có giá trị cho pair này (không dropna chung)
        df_pair = df[["round", col]].copy()
        df_pair = df_pair.dropna(subset=[col])  # chỉ drop NaN của cột này
        df_pair = df_pair.rename(columns={col: "value"})
        df_pair["pair"] = pair_name
        
        print(f"  {pair_name}: {len(df_pair)} rounds")
        results.append(df_pair)
    
    if not results:
        print("[ERROR] Không có dữ liệu")
        return None
    
    df_result = pd.concat(results, ignore_index=True)
    
    # Làm sạch
    df_result["round"] = pd.to_numeric(df_result["round"], errors='coerce')
    df_result["value"] = pd.to_numeric(df_result["value"], errors='coerce')
    df_result = df_result.dropna(subset=["round", "value"])
    df_result = df_result.drop_duplicates(subset=['pair', 'round'], keep='first')
    
    df_result = df_result[["pair", "round", "value"]].copy()
    df_result = df_result.astype({"pair": str, "round": int, "value": float})
    df_result = df_result.sort_values(["pair", "round"]).reset_index(drop=True)
    
    return df_result

def save_to_csv(df, output_path):
    """Lưu dataframe thành CSV"""
    if df is None or len(df) == 0:
        print("[ERROR] Dataframe rỗng, không thể lưu")
        return False
    
    # Tạo directory nếu cần
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"✓ Đã lưu: {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Pairs: {df['pair'].nunique()}")
    print(f"  Rounds: {df['round'].nunique()}")
    
    return True


def print_stats(df, method, block):
    """In thống kê dữ liệu"""
    if df is None or len(df) == 0:
        return
    
    print(f"\n{'='*70}")
    print(f"📊 THỐNG KÊ: {block} / {method}")
    print(f"{'='*70}")
    print(f"Shape:        {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"Pairs:        {df['pair'].nunique()} pairs")
    print(f"Rounds:       {df['round'].nunique()} rounds")
    print(f"Value range:  [{df['value'].min():.4f}, {df['value'].max():.4f}]")
    print(f"Value mean:   {df['value'].mean():.4f} ± {df['value'].std():.4f}")
    print(f"\nPairs: {sorted(df['pair'].unique())}")
    print(f"Rounds: {sorted(df['round'].unique())}")
    print(f"\nMẫu dữ liệu:")
    print(df.head(10).to_string(index=False))
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download metrics từ W&B với format chuẩn (pair, round, value)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Download gap_eps từ block4
  python download_wandb.py --entity ducthu2003 --project "Representation Drift" \\
                            --run_id q7re8n2t --method gap_eps --block block4

  # Download eps_curr từ block0, lưu vào file cụ thể
  python download_wandb.py --entity ducthu2003 --project "Representation Drift" \\
                            --run_id q7re8n2t --method eps_curr --block block0 \\
                            --output data_eps_curr.csv
        """
    )
    
    parser.add_argument("--entity", required=True, help="W&B entity name")
    parser.add_argument("--project", required=True, help="W&B project name")
    parser.add_argument("--run_id", required=True, help="W&B run ID")
    parser.add_argument("--method", required=True, help="Metric name (ví dụ: gap_eps, eps_curr)")
    parser.add_argument("--block", required=True, help="Block name (ví dụ: block0, block1, block4)")
    parser.add_argument("--output", default=None, help="Output CSV path (mặc định: {block}_{method}.csv)")
    parser.add_argument("--client", default="Client0", help="Client name (mặc định: Client0)")
    
    args = parser.parse_args()
    
    # Nếu không chỉ định output, tạo tên mặc định
    if args.output is None:
        args.output = f"{args.client}_{args.block}_{args.method}.csv"
    
    print(f"\n{'='*70}")
    print(f"🚀 DOWNLOAD DỮ LIỆU TỪ W&B")
    print(f"{'='*70}")
    print(f"Entity:   {args.entity}")
    print(f"Project:  {args.project}")
    print(f"Run ID:   {args.run_id}")
    print(f"Method:   {args.method}")
    print(f"Block:    {args.block}")
    print(f"Output:   {args.output}")
    print(f"{'='*70}\n")
    
    # Download dữ liệu
    print(f"📥 Đang download...")
    df = download_from_wandb(
        entity=args.entity,
        project=args.project,
        run_id=args.run_id,
        method=args.method,
        block=args.block
    )
    
    if df is None:
        print("[ERROR] Download thất bại")
        sys.exit(1)
    
    # In thống kê
    print_stats(df, args.method, args.block)
    
    # Lưu dữ liệu
    print(f"💾 Đang lưu dữ liệu...")
    if save_to_csv(df, args.output):
        print(f"\n✅ Hoàn tất!")
    else:
        print(f"\n❌ Lỗi khi lưu dữ liệu")
        sys.exit(1)

# # import pandas as pd

# # api = wandb.Api()
# # run = api.run("ducthu2003/TARGET/ccdok8rl")

# # history = run.history(keys=["Task_1_acc"])
# # df = pd.DataFrame(history)
# # print(df.head())

# # df.to_csv("Task_1_acc.csv", index=False)
# # print("Saved to Task_1_acc.csv")

# # import pandas as pd
# # import matplotlib.pyplot as plt

# # # Từ điển ánh xạ label → màu
# # colors = {
# #     'Task 0': 'blue',
# #     # 'FFA-LoRA': 'orange',
# #     # 'FedSA-LoRA': 'red',
# #     # 'FLoRA-CA': 'black',
# # }

# # # Custom labels bạn muốn vẽ
# # custom_labels = ['Task 0']

# # # Dữ liệu
# # df = pd.read_csv('Task_1_acc.csv')
# # x = df.iloc[:, 0]
# # max_columns = [col for col in df.columns if col.endswith('Task_1_acc')]

# # # Map label to columns theo thứ tự custom_labels
# # label_to_column = dict(zip(custom_labels, max_columns))

# # # Sắp xếp các label theo thứ tự trong `colors`
# # sorted_labels = [label for label in colors if label in custom_labels]

# # # Nếu có label không trong `colors`, thêm vào cuối
# # other_labels = [label for label in custom_labels if label not in colors]
# # final_labels = sorted_labels + other_labels

# # # Cấu hình smoothing
# # ema_span = 30
# # std_window = 3
# # plt.rcParams.update({'font.size': 18})

# # # Vẽ hình
# # plt.figure(figsize=(8, 6))
# # for label in final_labels:
# #     if label not in label_to_column:
# #         print(f"⚠️ Label '{label}' không khớp với bất kỳ cột dữ liệu nào.")
# #         continue

# #     col = label_to_column[label]
# #     color = colors.get(label, 'black')  # fallback nếu label không có màu

# #     ema = df[col].ewm(span=ema_span, adjust=False).mean()
# #     std = df[col].rolling(window=std_window, min_periods=1).std()

# #     linestyle = '--' if label == 'STAMP' else '-'  # Dùng nét đứt cho STAMP
# #     plt.plot(x, ema, label=label, linewidth=2.5, color=color, linestyle=linestyle)
# #     plt.fill_between(x, ema - std, ema + std, alpha=0.2, color=color)

# # # Giao diện
# # plt.xlabel('Task Steps')
# # plt.ylabel('Acc')
# # plt.title('Task 0 Accuracy')
# # plt.legend()
# # plt.grid(True)
# # plt.xlim(left=0, right=4)
# # plt.ylim(bottom=70, top=80)
# # plt.tight_layout()
# # plt.savefig("MNLI-grad.pdf", bbox_inches='tight')
# # plt.show()
