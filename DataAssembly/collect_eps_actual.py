import os
import re
from bs4 import BeautifulSoup
from openai import OpenAI
from sec_edgar_downloader import Downloader
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
import json

# Setup OpenAI client
client = OpenAI(api_key=config.OPENAI_API_KEY)

def ensure_output_folder():
    if not os.path.exists(config.SEC_DOWNLOADS):
        os.makedirs(config.SEC_DOWNLOADS)
        print(f"[INFO] Created download directory at {config.SEC_DOWNLOADS}")
    else:
        print(f"[INFO] Using existing download directory: {config.SEC_DOWNLOADS}")

def download_10q_filings(ticker, start_year=2012, end_year=2024):
    dl = Downloader(
        company_name=config.COMPANY_NAME,
        email_address=config.EMAIL,
        download_folder=config.SEC_DOWNLOADS
    )
    print(f"[INFO] Attempting to download 10-Qs for {ticker} ({start_year}–{end_year})")
    try:
        dl.get("10-Q", ticker)
        print(f"[SUCCESS] Download complete for {ticker}")
    except Exception as e:
        print(f"[ERROR] Download failed for {ticker}: {e}")

def list_downloaded_filings(ticker):
    filings_path = os.path.join(config.SEC_DOWNLOADS, "sec-edgar-filings", ticker, "10-Q")
    if not os.path.exists(filings_path):
        print(f"[WARN] No filings found for {ticker}")
        return []
    return [os.path.join(filings_path, f) for f in os.listdir(filings_path) if os.path.isdir(os.path.join(filings_path, f))]

def extract_eps_and_date_from_text(text):
    lines = text.splitlines()
    keywords = ["earnings per share", "earnings per common share"]
    matched_chunks = []

    for i, line in enumerate(lines):
        lower_line = line.lower()
        if any(keyword in lower_line for keyword in keywords):
            chunk = lines[i:i+20]
            matched_chunks.append("\n".join(chunk))

    if not matched_chunks:
        print("[WARN] No relevant EPS phrases found.")
        return None

    snippet = "\n\n---\n\n".join(matched_chunks)

    prompt = (
        "From the following 10-Q filing extract:\n"
        "1. The filing date (in YYYY-MM-DD format)\n"
        "2. The Basic EPS for the current reported quarter only\n"
        "3. The Diluted EPS for the current reported quarter only\n"
        "Ignore any previous quarters or yearly comparisons.\n"
        "Return your response in JSON format with exactly these keys: 'filing_date', 'basic_eps', 'diluted_eps'.\n\n"
        f"FILING EXTRACT:\n{snippet}"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a financial data extraction assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        content = response.choices[0].message.content
        match = re.search(r'\{.*?\}', content, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        print(f"[ERROR] LLM extraction failed: {e}")
    return None

def clean_filing_html_and_extract_eps(ticker):
    print(f"[INFO] Cleaning HTML and extracting EPS for {ticker}")
    for filing_dir in list_downloaded_filings(ticker):
        txt_path = os.path.join(filing_dir, "full-submission.txt")
        if not os.path.exists(txt_path):
            print(f"[WARN] Missing file: {txt_path}")
            continue

        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()

        try:
            soup = BeautifulSoup(raw_text, "html.parser")
            cleaned_text = soup.get_text(separator="\n")
            cleaned_text = "\n".join(line for line in cleaned_text.splitlines() if line.strip())

            result = extract_eps_and_date_from_text(cleaned_text)
            if result:
                print(f"[DATA] {filing_dir} -> {result}")
        except Exception as e:
            print(f"[ERROR] Failed to process {txt_path}: {e}")

if __name__ == "__main__":
    ticker = "AAPL"
    ensure_output_folder()
    download_10q_filings(ticker)
    clean_filing_html_and_extract_eps(ticker)
