#!/usr/bin/env python3
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import config
import csv
import time
from typing import List, Dict, Any, Optional
import requests
import json
# ===== Settings (edit if needed) ============================================
INPUT_CSV_PATH = config.METADATA_CSV_FILEPATH
SYMBOL_COL = "Symbol"
START_YEAR = 2012
END_YEAR = 2024
SLEEP_SECONDS = 60.00     # increase to 0.3–0.5 if you see 429s
MAX_RETRIES = 6
EST_OUT = "Data/eps_estimates_quarterly_2012_2024.csv"
SURP_OUT = "Data/eps_surprises_quarterly_2012_2024.csv"
FINNHUB_API_KEY = config.FINNHUB_API_KEY
# ============================================================================

API_BASE = "https://finnhub.io/api/v1"

def read_symbols_from_csv(path: str, col_name: str) -> List[str]:
    symbols = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or col_name not in reader.fieldnames:
            sys.stderr.write(f"Column '{col_name}' not found in {path}. Columns: {reader.fieldnames}\n")
            sys.exit(1)
        for row in reader:
            s = (row.get(col_name) or "").strip().upper()
            if s:
                symbols.append(s)
    seen, uniq = set(), []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq

def load_completed_symbols(file_path: str, symbol_col: str) -> set:
    if not os.path.exists(file_path):
        return set()
    completed = set()
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sym = row.get(symbol_col)
            if sym:
                completed.add(sym.strip().upper())
    return completed

def in_year_range(item: Dict[str, Any], start_year: int, end_year: int) -> bool:
    y = None
    if isinstance(item.get("year"), int):
        y = item["year"]
    elif isinstance(item.get("period"), str) and len(item["period"]) >= 4:
        try:
            y = int(item["period"][:4])
        except ValueError:
            y = None
    return (y is not None) and (start_year <= y <= end_year)

def request_with_retry(url: str, params: Dict[str, Any], max_retries: int, context: str = "") -> Optional[requests.Response]:
    backoff = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r
            elif r.status_code == 429:
                wait_time = backoff
                print(f"[{context}] Rate limit hit (429). Waiting {wait_time:.1f}s before retry {attempt}/{max_retries}...")
                time.sleep(wait_time)
                backoff = min(backoff * 2, 30)
                continue
            elif r.status_code in (500, 502, 503, 504):
                wait_time = backoff
                print(f"[{context}] Server error {r.status_code}. Waiting {wait_time:.1f}s before retry {attempt}/{max_retries}...")
                time.sleep(wait_time)
                backoff = min(backoff * 2, 30)
                continue
            else:
                print(f"[{context}] Request failed with status {r.status_code}: {r.text}")
                return r
        except requests.RequestException as e:
            wait_time = backoff
            print(f"[{context}] Network error: {e}. Waiting {wait_time:.1f}s before retry {attempt}/{max_retries}...")
            time.sleep(wait_time)
            backoff = min(backoff * 2, 30)
    print(f"[{context}] All retries failed.")
    return None

# Fetch and flatten EPS estimates (quarterly only)
def fetch_quarterly_estimates(symbol: str) -> List[Dict[str, Any]]:
    url = f"{API_BASE}/stock/eps-estimate"
    params = {"symbol": symbol, "freq": "quarterly", "token": FINNHUB_API_KEY}
    r = request_with_retry(url, params, MAX_RETRIES)
    if r is None or r.status_code != 200:
        sys.stderr.write(f"[{symbol} estimates] request failed.\n")
        return []
    # print(f"Raw EPS estimates API response for {symbol}:")
    # print(json.dumps(r.json(), indent=2))
    payload = r.json()
    data = payload.get("data") or []
    rows = []
    for it in data:
        if in_year_range(it, START_YEAR, END_YEAR):
            rows.append({
                "symbol": symbol,
                "freq": "quarterly",
                "period": it.get("period"),
                "year": it.get("year"),
                "quarter": it.get("quarter"),
                "epsAvg": it.get("epsAvg"),
                "epsHigh": it.get("epsHigh"),
                "epsLow": it.get("epsLow"),
                "numberAnalysts": it.get("numberAnalysts"),
            })
    return rows

# Fetch and flatten EPS surprises
def fetch_quarterly_surprises(symbol: str) -> List[Dict[str, Any]]:
    url = f"{API_BASE}/stock/earnings"
    params = {"symbol": symbol, "token": FINNHUB_API_KEY}
    r = request_with_retry(url, params, MAX_RETRIES)
    if r is None or r.status_code != 200:
        sys.stderr.write(f"[{symbol} surprises] request failed.\n")
        return []
    # print(f"Raw EPS surprises API response for {symbol}:")
    # print(json.dumps(r.json(), indent=2))
    payload = r.json()
    data = payload if isinstance(payload, list) else payload.get("earnings") or []
    rows = []
    for it in data:
        if in_year_range(it, START_YEAR, END_YEAR):
            rows.append({
                "symbol": symbol,
                "period": it.get("period"),
                "year": it.get("year"),
                "quarter": it.get("quarter"),
                "actual": it.get("actual"),
                "estimate": it.get("estimate"),
                "surprise": it.get("surprise"),
                "surprisePercent": it.get("surprisePercent"),
            })
    return rows

def main():
    if not FINNHUB_API_KEY:
        sys.stderr.write("Set FINNHUB_API_KEY env var or paste your key into FINNHUB_API_KEY at the top.\n")
        sys.exit(1)

    symbols = read_symbols_from_csv(INPUT_CSV_PATH, SYMBOL_COL)

    # Load completed tickers from both CSVs
    completed_est = load_completed_symbols(EST_OUT, "symbol")
    completed_surp = load_completed_symbols(SURP_OUT, "symbol")

    est_fields = ["symbol", "freq", "period", "year", "quarter", "epsAvg", "epsHigh", "epsLow", "numberAnalysts"]
    surp_fields = ["symbol", "period", "year", "quarter", "actual", "estimate", "surprise", "surprisePercent"]

    # Open files in append mode
    est_file_exists = os.path.exists(EST_OUT)
    surp_file_exists = os.path.exists(SURP_OUT)

    with open(EST_OUT, "a", newline="", encoding="utf-8") as f_est, \
         open(SURP_OUT, "a", newline="", encoding="utf-8") as f_surp:

        est_writer = csv.DictWriter(f_est, fieldnames=est_fields)
        surp_writer = csv.DictWriter(f_surp, fieldnames=surp_fields)

        if not est_file_exists:
            est_writer.writeheader()
        if not surp_file_exists:
            surp_writer.writeheader()

        for i, sym in enumerate(symbols, 1):
            # Estimates
            if sym not in completed_est:
                print(f"[{i}/{len(symbols)}] Fetching EPS estimates for {sym}...")
                est_rows = fetch_quarterly_estimates(sym)
                for row in est_rows:
                    est_writer.writerow(row)
                f_est.flush()
                time.sleep(SLEEP_SECONDS)
            else:
                print(f"[{i}/{len(symbols)}] Skipping {sym} (already in estimates CSV)")

            # Surprises
            if sym not in completed_surp:
                print(f"[{i}/{len(symbols)}] Fetching EPS surprises for {sym}...")
                surp_rows = fetch_quarterly_surprises(sym)
                for row in surp_rows:
                    surp_writer.writerow(row)
                f_surp.flush()
                time.sleep(SLEEP_SECONDS)
            else:
                print(f"[{i}/{len(symbols)}] Skipping {sym} (already in surprises CSV)")

    print("Done.")

if __name__ == "__main__":
    main()