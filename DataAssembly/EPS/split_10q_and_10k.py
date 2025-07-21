import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import config
import pandas as pd


def split_10k_data_and_clean(original_csv: str, annual_csv: str, quarterly_csv: str) -> None:
    # Read the original data
    df = pd.read_csv(original_csv)

    # Filter 10-K and 10-Q rows
    df_10k = df[df['Form Type'] == '10-K (Annual report)'].copy()
    df_10q = df[df['Form Type'] != '10-K (Annual report)'].copy()

    # Drop quarterly columns from 10-K
    df_10k = df_10k.drop(columns=['quarterly_raw_eps', 'quarterly_diluted_eps'], errors='ignore')
    # Drop annual columns from 10-Q
    df_10q = df_10q.drop(columns=['annual_raw_eps', 'annual_diluted_eps'], errors='ignore')

     # In 10-Q: set annual columns to NaN
    for col in ['annual_raw_eps', 'annual_diluted_eps']:
        if col in df_10k.columns:
            df_10k[col] = pd.NA

    # Save 10-K rows to a separate CSV
    df_10k.to_csv(annual_csv, index=False)
    print(f"[INFO] Saved {len(df_10k)} 10-K rows to {annual_csv}")

    # Save the updated 10-Q rows back to the original CSV
    df_10q.to_csv(quarterly_csv, index=False)
    print(f"[INFO] Updated {original_csv} with {len(df_10q)} rows (10-K removed)")

if __name__ == "__main__":
    if not os.path.exists(config.ANNUAL_EPS_DATA_CSV):
        split_10k_data_and_clean(original_csv=config.EPS_DATA_CSV, 
                                 annual_csv=config.ANNUAL_EPS_DATA_CSV,
                                 quarterly_csv=config.QUARTERLY_EPS_DATA_CSV)
    else:
        print(f"[WARN] File {config.ANNUAL_EPS_DATA_CSV} exists already, are you sure you want to do this?")
