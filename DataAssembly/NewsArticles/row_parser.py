import csv
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import config
import random
csv.field_size_limit(sys.maxsize)

def parse_row_custom(line:str):
    """
    Parse a malformed CSV row into fields:
    [index, datetime, Article_title, Stock_symbol, Url, Publisher, Author, Article]
    Returns None if the row cannot be parsed correctly.
    """
    fields = []
    field = ''
    in_quotes = False
    i = 0
    length = len(line)

    while i < length:
        char = line[i]

        if char == '"':
            in_quotes = not in_quotes
            field += char  # retain quotes for internal processing
            i += 1
            continue

        if char == ',' and not in_quotes:
            # Handle empty field
            stripped = field.strip().strip('"')
            fields.append(None if stripped == '' else stripped)
            field = ''
            i += 1
            continue

        field += char
        i += 1

    # Add last field
    stripped = field.strip().strip('"')
    fields.append(None if stripped == '' else stripped)

    if len(fields) < 8:
        print(f"[WARN] Skipping malformed row with only {len(fields)} fields.")
        return None

    # Drop anything beyond the 8th field
    return fields[:8]

def sample_and_export_cleaned_csv(input_path, output_path, num_samples=3):
    print(f"[INFO] Parsing file: {input_path}")
    valid_rows = []
    total_lines = 0
    skipped_lines = 0

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8', newline='') as outfile:

        writer = csv.writer(outfile, quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for i, line in enumerate(infile):
            total_lines += 1
            try:
                row = next(csv.reader([line], quotechar='"', doublequote=True, strict=True))
            except csv.Error:
                skipped_lines += 1
                continue  # Skip malformed row

            writer.writerow(row)
            valid_rows.append(row)

            if i > 0 and i % 500_000 == 0:
                print(f"[INFO] Scanned {i:,} lines so far...")

    print("\n[SUMMARY]")
    print(f"[INFO] Total lines read      : {total_lines:,}")
    print(f"[INFO] Valid rows written    : {len(valid_rows):,}")
    print(f"[INFO] Malformed rows skipped: {skipped_lines:,}")

    print(f"\n[INFO] Sample of {num_samples} parsed rows:\n")
    for sample in random.sample(valid_rows, min(num_samples, len(valid_rows))):
        print("----- Sample Row -----")
        print(f"Index        : {sample[0]}")
        print(f"Date         : {sample[1]}")
        print(f"Title        : {sample[2]}")
        print(f"Symbol       : {sample[3]}")
        print(f"URL          : {sample[4]}")
        print(f"Publisher    : {sample[5]}")
        print(f"Author       : {sample[6]}")
        print(f"Article Text :\n{sample[7]}")
        if len(sample) > 8:
            print(f"Summaries    : {[c for c in sample[8:]]}")
        print()

import pandas as pd

def filter_news_by_eps_tickers(news_csv_path, eps_csv_path, output_csv_path):
    """
    Filters the news CSV to only include rows where 'Symbol' is in the list of EPS 'Ticker' values.

    Parameters:
        news_csv_path (str): Path to the cleaned news CSV (must have a 'Symbol' column).
        eps_csv_path (str): Path to the EPS data CSV (must have a 'Ticker' column).
        output_csv_path (str): Path where the filtered CSV will be saved.
    """
    print(f"[INFO] Reading EPS tickers from: {eps_csv_path}")
    eps_df = pd.read_csv(eps_csv_path)
    tickers = set(eps_df['Ticker'].dropna().unique())
    print(f"[INFO] Loaded {len(tickers):,} unique tickers.")

    print(f"[INFO] Reading news data from: {news_csv_path}")
    news_df = pd.read_csv(news_csv_path)
    before = len(news_df)
    print(f"[INFO] Original news rows: {before:,}")

    filtered_df = news_df[news_df['Stock_symbol'].isin(tickers)]
    after = len(filtered_df)
    print(f"[INFO] Filtered news rows: {after:,} (kept {after/before:.2%})")

    print(f"[INFO] Writing filtered data to: {output_csv_path}")
    filtered_df.to_csv(output_csv_path, index=False)
    print("[DONE] Filtering complete.")


# Example usage
if __name__ == "__main__":
    sample_and_export_cleaned_csv(
        config.NEWS_CSV_PATH_FORMATTED_ROWS,
        config.NEWS_CSV_PATH_CLEAN,
        num_samples=0
    )

    filter_news_by_eps_tickers(news_csv_path=config.NEWS_CSV_PATH_CLEAN,
                               eps_csv_path=config.EPS_DATA_CSV,
                               output_csv_path=config.NEWS_CSV_PATH_CLEAN)