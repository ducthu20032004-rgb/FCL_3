# # import wandb
# # import pandas as pd
# # import argparse
# # import os

# # def runwandb(entity, project, run_id, method, block, out_dir):
# #     api = wandb.Api()
# #     run = api.run(f"{entity}/{project}/runs/{run_id}")

# #     rows = list(run.scan_history(page_size=10000))  # fix thiếu data
# #     df = pd.DataFrame(rows)
    
# #     print(f"  Raw rows từ W&B: {len(df)}")

# #     overlap_cols = [
# #         f"{block}/task0/{method}",
# #         f"{block}/task1/{method}",
# #         f"{block}/task2/{method}",
# #         f"{block}/task3/{method}",
# #         f"{block}/task4/{method}",
# #     ]

# #     cols_exist = [c for c in overlap_cols if c in df.columns]

# #     if not cols_exist:
# #         print(f"[ERROR] Không tìm thấy cột nào cho {block}/{method}")
# #         return None

# #     # Melt về format cũ: round_global + block4/{method}
# #     df_sub = df[["round_global"] + cols_exist].copy()
# #     df_melt = df_sub.melt(id_vars="round_global", value_vars=cols_exist,
# #                           var_name="task", value_name=f"block4/{method}")

# #     df_melt = df_melt.dropna(subset=[f"block4/{method}"])
# #     df_melt = df_melt.sort_values("round_global").reset_index(drop=True)

# #     df_final = df_melt[["round_global", f"block4/{method}"]]

# #     return df_final


# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("--entity",  default="ducthu2003")
# #     parser.add_argument("--project", default="Representation Drift Measurement")
# #     parser.add_argument("--run_id",  default="jcabxqbj")
# #     parser.add_argument("--method",  default="forgetting")
# #     parser.add_argument("--block",   default="block4")
# #     parser.add_argument("--out_dir", default=r"C:\Thu\FCL\material_experiment\dynamic")
# #     args = parser.parse_args()

# #     print(f"\n{'='*60}")
# #     print(f"Entity: {args.entity} | Project: {args.project}")
# #     print(f"Run ID: {args.run_id} | Method: {args.method} | Block: {args.block}")
# #     print(f"{'='*60}\n")

# #     df = runwandb(args.entity, args.project, args.run_id, args.method, args.block, args.out_dir)

# #     if df is None:
# #         print("[ERROR] Download thất bại")
# #         exit(1)

# #     print(df)
# #     print(f"\nShape: {df.shape}")

# #     os.makedirs(args.out_dir, exist_ok=True)
# #     out_path = os.path.join(args.out_dir, f"{args.method}_{args.block}.csv")
# #     df.to_csv(out_path, index=False)
# #     print(f"✅ Saved: {out_path}")
# # import wandb


# Phiên bản chuẩn downdata từ W&B, xử lý dữ liệu và lưu thành CSV với format: pair, round, value
# """
# Download metrics từ W&B với format chuẩn: pair, round, value
# Loại bỏ tất cả việc xử lý khác - chỉ download dữ liệu
# """

import re
import os
import sys
import argparse
import pandas as pd
import wandb

def download_from_wandb(client, entity, project, run_id, method, block):
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/runs/{run_id}")
    
    # Tăng page_size để không bỏ sót rows
    rows = list(run.scan_history(page_size=1000000))
    df = pd.DataFrame(rows)
    
    overlap_cols = [
        f"{client}/{block}/pair_0_1/{method}",
        f"{client}/{block}/pair_0_2/{method}",
        f"{client}/{block}/pair_0_3/{method}",
        f"{client}/{block}/pair_0_4/{method}",
        f"{client}/{block}/pair_1_2/{method}",
        f"{client}/{block}/pair_1_3/{method}",
        f"{client}/{block}/pair_1_4/{method}",
        f"{client}/{block}/pair_2_3/{method}",
        f"{client}/{block}/pair_2_4/{method}",
        f"{client}/{block}/pair_3_4/{method}",
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
        #pair_name = re.search(r'pair_\d+_\d+', col).group(0)
        pair_name = re.search(r'pair\d+_\d+', col).group(0)
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
    parser.add_argument("--block", required=False, help="Block name (ví dụ: block0, block1, block4)")
    parser.add_argument("--output", default=None, help="Output CSV path (mặc định: {block}_{method}.csv)")
    parser.add_argument("--client", default="client0", help="Client name (mặc định: Client0)")
    parser.add_argument("--model",default = "FedDBE", help="Model name (mặc định: FedDBE)")
    
    args = parser.parse_args()
    
    # Nếu không chỉ định output, tạo tên mặc định
    if args.output is None:
        args.output = f"{args.client}_{args.block}_{args.method}_{args.model}.csv"
    
    print(f"\n{'='*70}")
    print(f"🚀 DOWNLOAD DỮ LIỆU TỪ W&B")
    print(f"{'='*70}")
    print(f"Entity:   {args.entity}")
    print(f"Project:  {args.project}")
    print(f"Run ID:   {args.run_id}")
    print(f"Method:   {args.method}")
    print(f"Output:   {args.output}")
    print(f"{'='*70}\n")
# Thay toàn bộ phần if __name__ == "__main__": từ dòng "print(f"📥 Đang download...")" trở xuống

    all_results = []

    print(f"📥 Đang download tất cả clients & blocks...")

    for client in ["client0", "client1", "client2", "client3", "client4"]:
        for block in ["block0", "block1", "block2", "block3", "block4"]:
            print(f"\n{'─'*50}")
            print(f"👤 Client: {client} | 📦 Block: {block}")
            print(f"{'─'*50}")
            
            df = download_from_wandb(
                client=client,
                entity=args.entity,
                project=args.project,
                run_id=args.run_id,
                method=args.method,
                block=block
            )
            
            if df is None:
                print(f"[WARNING] Bỏ qua {client}/{block} - không có dữ liệu")
                continue
            
            # Thêm cột định danh
            df.insert(0, "client", client)
            df.insert(1, "block", block)
            
            print(f"  ✓ {len(df)} rows")
            all_results.append(df)

    # Gom lại và lưu 1 file duy nhất
    if all_results:
        df_all = pd.concat(all_results, ignore_index=True)
        
        output_path = args.output if args.output else f"{args.method}_{args.model}_all.csv"
        save_to_csv(df_all, output_path)
        
        # Print stats tổng
        print(f"\n{'='*70}")
        print(f"📊 TỔNG KẾT")
        print(f"{'='*70}")
        print(f"Shape:    {df_all.shape}")
        print(f"Clients:  {sorted(df_all['client'].unique())}")
        print(f"Blocks:   {sorted(df_all['block'].unique())}")
        print(f"Pairs:    {sorted(df_all['pair'].unique())}")
        print(f"Rounds:   {df_all['round'].nunique()} unique rounds")
        print(f"{'='*70}")
    else:
        print("[ERROR] Không có dữ liệu nào được download")

    print(f"\n✅ Hoàn tất!")

# import pandas as pd
# from pathlib import Path
# import re

# # =====================================================
# # CONFIG
# # =====================================================

# MODE = "mean"      # "mean" hoặc "round24"

# # Chỉ sửa dòng này
# BASE_FILE = r"C:\Thu\FCL\Client0_block0_sigma_old.csv"

# PAIR_ORDER = [
#     "pair_0_1",
#     "pair_0_2",
#     "pair_0_3",
#     "pair_0_4",
#     "pair_1_2",
#     "pair_1_3",
#     "pair_1_4",
#     "pair_2_3",
#     "pair_2_4",
#     "pair_3_4",
# ]

# # =====================================================
# # AUTO GENERATE BLOCK FILES
# # =====================================================

# FILES = {}

# for block_id in range(5):
#     path = re.sub(
#         r"block\d+",
#         f"block{block_id}",
#         BASE_FILE
#     )
#     FILES[f"block{block_id}"] = path

# # =====================================================
# # READ DATA
# # =====================================================

# result = pd.DataFrame(index=PAIR_ORDER)

# for block_name, file_path in FILES.items():

#     df = pd.read_csv(file_path)

#     if MODE == "mean":

#         values = (
#             df.groupby("pair")["value"]
#             .mean()
#             .reindex(PAIR_ORDER)
#         )

#     elif MODE == "round24":

#         values = (
#             df[df["round"] == 24]
#             .set_index("pair")["value"]
#             .reindex(PAIR_ORDER)
#         )

#     else:
#         raise ValueError("MODE phải là 'mean' hoặc 'round24'")

#     result[block_name] = values

# # =====================================================
# # PRINT RESULT
# # =====================================================

# pd.set_option("display.max_columns", None)
# pd.set_option("display.width", 200)

# print("\n==============================")
# print(f"MODE = {MODE}")
# print("==============================\n")

# print(result.round(6))

# # =====================================================
# # BLOCK MEAN
# # =====================================================

# print("\n==============================")
# print("Mean of each block")
# print("==============================\n")

# for block in result.columns:
#     print(f"{block}: {result[block].mean():.6f}")

# import pandas as pd

# def print_avg_task_block(file_path):
#     df = pd.read_csv(file_path)

#     # ép kiểu số
#     num_cols = df.columns.difference(['client1', 'client2'])
#     df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')

#     metrics = ["cka", "sigma", "eps", "cosine_similarity", "align@10", "align@20"]

#     for m in metrics:
#         print("\n" + "="*80)
#         print(f"METRIC: {m}")
#         print("="*80)

#         grouped = (
#             df.groupby(["block_idx", "t"])[m]
#               .mean()
#               .reset_index()
#         )

#         pivot = grouped.pivot(index="block_idx", columns="t", values=m)

#         # sort cho đẹp
#         pivot = pivot.sort_index().sort_index(axis=1)

#         print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))


# if __name__ == "__main__":
#     file_path = r"C:\Thu\FCL\outputs\client_representation_drift-hetero-ResNet18.csv"
#     print_avg_task_block(file_path)

# import pandas as pd
# import os

# base_path = r"C:\Thu\FCL"
# base_name = "Client0_None_eps_old_FedTarget_block{}.csv"

# total = 0
# for block in range(5):
#     file_path = os.path.join(base_path, base_name.format(block))
#     df = pd.read_csv(file_path)
#     mean_val = df["value"].mean()
#     print(f"block{block}: {mean_val:.6f}")
#     total += mean_val

# print(f"\nTổng trung bình: {total:.6f}")
# print(f"Trung bình chung: {total / 5:.6f}")