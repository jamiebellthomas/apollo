import re
import random
import tempfile
import os

def fix_newlines_by_index_pattern_streaming(input_path, temp_path):
    """
    Step 1: Stream through the file to flatten newlines and rebuild correct row boundaries
    based on float index + UTC timestamp pattern.
    """
    print("[INFO] Flattening and reformatting CSV by row start pattern...")

    row_start_pattern = re.compile(r'^\d+\.\d{1},\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} UTC')

    original_line_count = 0
    final_line_count = 0
    sample_lines = []

    with open(input_path, 'r', encoding='utf-8') as infile, open(temp_path, 'w', encoding='utf-8') as outfile:
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


def fix_internal_quotes_streaming(temp_path, final_path):
    """
    Step 2: For each line, keep first and last quote, replace all others with apostrophes.
    """
    print("[INFO] Cleaning internal quotes in rows...")
    with open(temp_path, 'r', encoding='utf-8') as infile, open(final_path, 'w', encoding='utf-8') as outfile:
        for i, line in enumerate(infile):
            quote_indices = [j for j, char in enumerate(line) if char == '"']

            if len(quote_indices) < 2:
                outfile.write(line)
                continue

            first, last = quote_indices[0], quote_indices[-1]
            fixed_line = (
                line[:first + 1] +
                line[first + 1:last].replace('"', "'") +
                line[last:]
            )
            outfile.write(fixed_line)

            if i > 0 and i % 500_000 == 0:
                print(f"[INFO] Processed {i:,} rows...")

    print("[DONE] All internal quotes cleaned and structure repaired.")


def repair_fns_news_csv(original_path, final_output_path):
    """
    Master function: repair broken CSV and produce a clean output.
    """
    temp_path = original_path.replace(".csv", "_rowfixed_temp.csv")
    fix_newlines_by_index_pattern_streaming(original_path, temp_path)
    fix_internal_quotes_streaming(temp_path, final_output_path)

    # Optional: remove temp file
    try:
        os.remove(temp_path)
        print(f"[INFO] Removed temp file: {temp_path}")
    except Exception as e:
        print(f"[WARN] Could not remove temp file: {e}")


    import random

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
    # repair_fns_news_csv(
    #     "Data/nasdaq_exteral_data.csv",
    #     "Data/final_clean_news.csv"
    # )
    print("[INFO] CSV repair completed. Now analysing the cleaned CSV...")
    analyse_csv("Data/final_clean_news.csv", num_samples=10)

