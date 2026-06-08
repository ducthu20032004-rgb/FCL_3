import pandas as pd
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python avg_by_block.py <path_to_csv>")
        sys.exit(1)

    path = sys.argv[1]
    df = pd.read_csv(path)

    # Tính trung bình theo block trên tất cả client
    result = (
        df.groupby("block_idx")[["eps", "sigma", "cka"]]
        .mean()
        .rename(columns={
            "eps":   "avg_eps",
            "sigma": "avg_sigma",
            "cka":   "avg_cka",
        })
    )

    pd.set_option("display.float_format", "{:.6f}".format)
    print("\n=== Trung bình theo Block (trên tất cả client) ===\n")
    print(result.to_string())
    print()

if __name__ == "__main__":
    main()