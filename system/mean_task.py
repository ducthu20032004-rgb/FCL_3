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
        df.groupby("block")[["eps_old", "sigma_old", "align10"]]
        .mean()
        .rename(columns={
            "eps_old":   "avg_eps",
            "sigma_old": "avg_sigma",
            "align10": "CKNNA@10",
        })
    )

    pd.set_option("display.float_format", "{:.6f}".format)
    print("\n=== Trung bình theo Block (trên tất cả client) ===\n")
    print(result.to_string())
    print()

if __name__ == "__main__":
    main()