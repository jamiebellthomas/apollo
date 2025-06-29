import re
import random
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import config

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
    fix_newlines_by_index_pattern_streaming(original_path, final_output_path)


def analyse_csv(filepath, num_samples=10):
    """
    Prints total row count and N random rows from a CSV file (as plain lines).
    Assumes the file is already line-separated properly (e.g., cleaned).
    """
    print(f"[INFO] Analysing CSV: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total = len(lines)
    print(f"[INFO] Total rows: {total:,}")

    print(f"[INFO] Random sample ({min(num_samples, total)} rows):")
    for line in random.sample(lines, min(num_samples, total)):
        print(line.strip())


# Example usage:
if __name__ == "__main__":
    # Uncomment the following line to run cleaning
    repair_fns_news_csv(config.NEWS_CSV_PATH_ORIGIN, config.NEWS_CSV_PATH_CLEAN)

    num_sample = 3
    print(f"Printing {num_sample} random samples from original dataset:")
    analyse_csv(config.NEWS_CSV_PATH_ORIGIN, num_samples=num_sample)
    
    print('-'*30)
    print(f"Printing {num_sample} random samples from cleaned dataset:")
    analyse_csv(config.NEWS_CSV_PATH_CLEAN, num_samples=num_sample)
