import re
import random
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import config

import os
import subprocess
import shutil

def download_raw_data(target_path, wget_url, downloaded_filename=None):
    """
    Check if the target file exists. If not, use wget to download it and move it into place.

    Parameters:
        target_path (str): Final destination path for the file.
        wget_url (str): URL to download the file from.
        downloaded_filename (str, optional): The filename that wget will save. 
                                             If not provided, inferred from the URL.
    """
    if os.path.exists(target_path):
        print(f"[INFO] File already exists at: {target_path}")
        return

    print(f"[INFO] File not found. Downloading from: {wget_url}")
    
    # Run wget to download the file
    subprocess.run(["wget", wget_url], check=True)

    # Infer filename from URL if not provided
    if downloaded_filename is None:
        downloaded_filename = wget_url.split("/")[-1]

    if not os.path.exists(downloaded_filename):
        raise FileNotFoundError(f"Expected download file not found: {downloaded_filename}")

    # Ensure target directory exists
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    # Move the file
    shutil.move(downloaded_filename, target_path)
    print(f"[INFO] File downloaded and moved to: {target_path}")


def fix_newlines_by_index_pattern_streaming(input_path, output_path):
    """
    Step 1: Stream through the file to flatten newlines and rebuild correct row boundaries
    based on float index + UTC timestamp pattern.
    """
    print("[INFO] Flattening and reformatting CSV by row start pattern...")

    row_start_pattern = re.compile(r'^\d+\.\d{1},\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC')

    original_line_count = 0
    final_line_count = 0
    sample_lines = []

    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        buffer = ""

        for i, line in enumerate(infile):
            original_line_count += 1
            line = line.strip()

            if row_start_pattern.match(line):
                if buffer:
                    outfile.write(buffer + '\n')
                    final_line_count += 1
                    if len(sample_lines) < 10 and random.random() < 0.005:
                        sample_lines.append(buffer)
                buffer = line
            else:
                buffer += " " + line

            if i % 500_000 == 0 and i > 0:
                print(f"[INFO] Scanned {i:,} lines...")

        if buffer:
            outfile.write(buffer + '\n')
            final_line_count += 1
            if len(sample_lines) < 10:
                sample_lines.append(buffer)

    print(f"[INFO] Original line count: {original_line_count:,}")
    print(f"[INFO] Final line count after flattening: {final_line_count:,}")
    print("[INFO] Sample of cleaned lines:")
    for line in sample_lines:
        print(line.strip())

    print("[INFO] Row structure restored.")


def repair_fns_news_csv(original_path, final_output_path):
    """
    Master function: repair broken CSV by restoring row structure only.
    """
    if os.path.exists(config.NEWS_CSV_PATH_FORMATTED_ROWS):
        print(f"[INFO] File already exists at: {config.NEWS_CSV_PATH_FORMATTED_ROWS}")
        return 
    fix_newlines_by_index_pattern_streaming(original_path, final_output_path)


def analyse_csv(filepath, num_samples=3):
    """
    Prints total row count and N random rows from a CSV file (as plain lines).
    Assumes the file is already line-separated properly (e.g., cleaned).
    """
    print(f"[INFO] Analysing CSV: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total = len(lines)
    print(f"[INFO] Total rows: {total:,}")

    # print the first line
    print("First line:",  lines[0])

    print(f"[INFO] Random sample ({min(num_samples, total)} rows):")
    for index,line in enumerate(random.sample(lines, min(num_samples, total))):
        print(f"Line {index+1}:")
        print(line.strip())


# Example usage:
if __name__ == "__main__":

    download_raw_data(target_path=config.NEWS_CSV_PATH_ORIGIN,
                      wget_url="https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/nasdaq_exteral_data.csv",
                      downloaded_filename="nasdaq_exteral_data.csv")

    repair_fns_news_csv(config.NEWS_CSV_PATH_ORIGIN, config.NEWS_CSV_PATH_FORMATTED_ROWS)

    num_sample = 3
    print(f"Printing {num_sample} random samples from original dataset:")
    analyse_csv(config.NEWS_CSV_PATH_ORIGIN, num_samples=num_sample)
    
    print('-'*30)
    print(f"Printing {num_sample} random samples from cleaned dataset:")
    analyse_csv(config.NEWS_CSV_PATH_FORMATTED_ROWS, num_samples=num_sample)
