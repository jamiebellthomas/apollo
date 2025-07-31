import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

import json
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Any

import numpy as np
import pandas as pd
from SubGraph import SubGraph


# ---------- Utilities ----------
def _parse_date(s: str) -> datetime.date:
    return datetime.strptime(s, "%Y-%m-%d").date()


# ---------- Loaders ----------
def load_eps(csv_path: str) -> pd.DataFrame:
    """
    Load EPS CSV and return only the fields we need:
      ticker (str), er_date (YYYY-MM-DD str), predicted_eps (float), real_eps (float)
    """
    df = pd.read_csv(csv_path)
    df = df.rename(columns={
        "Ticker": "ticker",
        "Reporting Date": "er_date",
        # NOTE: adjust these when proper EPS target/forecast columns arrive
        "quarterly_raw_eps": "predicted_eps",
        "quarterly_diluted_eps": "real_eps",
    })
    df["ticker"] = df["ticker"].astype(str)
    df["er_date"] = pd.to_datetime(df["er_date"]).dt.strftime("%Y-%m-%d")
    for c in ("predicted_eps", "real_eps"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan
    return df[["ticker", "er_date", "predicted_eps", "real_eps"]]


def load_facts(jsonl_path: str, sentiment_min: float) -> pd.DataFrame:
    """
    Load facts.jsonl and keep only |sentiment| >= sentiment_min.
    Returns DataFrame with:
      fact_id, date, date_ordinal, tickers(list), sentiment, event_type,
      raw_text, source_article_index, event_cluster_id (if present)
    """
    rows: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            s = float(r.get("sentiment", 0.0))
            if abs(s) < sentiment_min:
                continue
            d = _parse_date(r["date"])
            rows.append({
                "fact_id": i,                                 # stable id
                "date": r["date"],                            # original string
                "date_ordinal": d.toordinal(),
                "tickers": list(r.get("tickers", [])),
                "sentiment": s,
                "event_type": r.get("event_type", ""),
                "raw_text": r.get("raw_text"),
                "source_article_index": r.get("source_article_index"),
                "event_cluster_id": r.get("event_cluster_id", None),  # <-- carry if annotated
            })
    facts = pd.DataFrame(rows)
    if facts.empty:
        facts = pd.DataFrame(columns=[
            "fact_id","date","date_ordinal","tickers","sentiment",
            "event_type","raw_text","source_article_index","event_cluster_id"
        ])
    # Keep fact_id as index + column for fast .loc and stable ids
    facts = facts.set_index("fact_id", drop=False)
    return facts


# ---------- Indexing for fast per-ticker window queries ----------
def build_ticker_index(facts: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Per-ticker, time-sorted view for fast window queries.
    {ticker: DataFrame[fact_id, date, date_ordinal, sentiment, event_type, raw_text, source_article_index]}
    (Exploded by ticker; we will recover the full original tickers list and cluster id from facts by fact_id.)
    """
    f2t = facts[[
        "fact_id","date","date_ordinal","tickers","sentiment",
        "event_type","raw_text","source_article_index"
    ]].explode("tickers").rename(columns={"tickers": "ticker"})
    f2t["ticker"] = f2t["ticker"].astype(str)

    by_ticker: Dict[str, pd.DataFrame] = {}
    for tkr, g in f2t.groupby("ticker", sort=False):
        by_ticker[tkr] = g.sort_values("date_ordinal").reset_index(drop=True)
    return by_ticker


# ---------- Selection logic ----------
def select_facts_for_instance(
    ticker: str,
    er_date_str: str,
    ticker_view: Dict[str, pd.DataFrame],
    facts_df: pd.DataFrame,   # canonical (un-exploded) facts DF
    window_days: int
) -> List[Dict[str, Any]]:
    """
    For (ticker, er_date), return list of fact dicts (original fields + delta_days + event_cluster_id if present).
    Inclusion: start = ER - window_days, end = ER (ER day excluded).
    """
    er_date = _parse_date(er_date_str)
    start_date = er_date - timedelta(days=window_days)

    g = ticker_view.get(ticker)
    if g is None or g.empty:
        return []

    ords = g["date_ordinal"].to_numpy()
    lo = np.searchsorted(ords, start_date.toordinal(), side="left")
    hi = np.searchsorted(ords, er_date.toordinal(),   side="left")  # exclude ER day
    if lo >= hi:
        return []

    subset = g.iloc[lo:hi].copy().sort_values("date_ordinal")

    fact_list: List[Dict[str, Any]] = []
    for _, row in subset.iterrows():
        d_ord = int(row["date_ordinal"])
        delta = int(er_date.toordinal() - d_ord)  # in [1, window_days]
        if delta < 1 or delta > window_days:
            continue

        fid = int(row["fact_id"])

        # Recover original full tickers list & cluster id from canonical facts df
        orig_tickers = facts_df.loc[fid, "tickers"]
        if not isinstance(orig_tickers, list):
            # robust fallback if serialized
            try:
                orig_tickers = json.loads(orig_tickers)
                if not isinstance(orig_tickers, list):
                    orig_tickers = [str(orig_tickers)]
            except Exception:
                orig_tickers = [str(orig_tickers)]
        cid = facts_df.loc[fid, "event_cluster_id"] if "event_cluster_id" in facts_df.columns else None
        if pd.isna(cid):
            cid = None

        obj: Dict[str, Any] = {
            "fact_id": fid,
            "date": row["date"],
            "tickers": orig_tickers,                     # <- full original tickers list
            "raw_text": row["raw_text"],
            "event_type": row["event_type"],
            "sentiment": float(row["sentiment"]),
            "delta_days": delta,
            "event_cluster_id": (None if cid is None else int(cid)),  # <- ALWAYS write the key
        }
        if pd.notna(row.get("source_article_index")):
            try:
                obj["source_article_index"] = int(row["source_article_index"])
            except Exception:
                obj["source_article_index"] = row["source_article_index"]

        fact_list.append(obj)

    return fact_list


# ---------- Assembly ----------
def build_subgraph_record(
    ticker: str,
    er_date: str,
    predicted_eps: float | None,
    real_eps: float | None,
    ticker_view: Dict[str, pd.DataFrame],
    facts_df: pd.DataFrame,
    window_days: int
) -> SubGraph:
    fact_list = select_facts_for_instance(
        ticker=ticker,
        er_date_str=er_date,
        ticker_view=ticker_view,
        facts_df=facts_df,
        window_days=window_days
    )
    return SubGraph(
        primary_ticker=ticker,
        reported_date=er_date,  # (matches your SubGraph field)
        predicted_eps=None if pd.isna(predicted_eps) else float(predicted_eps),
        real_eps=None if pd.isna(real_eps) else float(real_eps),
        fact_count=len(fact_list),
        fact_list=fact_list
    )


# ---------- Writers ----------
def write_jsonl(records: Iterable[SubGraph], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as out:
        for rec in records:
            out.write(rec.to_json_line() + "\n")


# ---------- Orchestrator ----------
def build_subgraphs_jsonl(
    eps_csv_path: str,
    facts_jsonl_path: str,
    out_path: str,
    window_days: int = config.WINDOW_DAYS,
    sentiment_min: float = config.SENTIMENT_MIN
) -> None:
    eps = load_eps(eps_csv_path)
    facts = load_facts(facts_jsonl_path, sentiment_min=sentiment_min)
    ticker_view = build_ticker_index(facts)

    def gen() -> Iterable[SubGraph]:
        for _, r in eps.iterrows():
            yield build_subgraph_record(
                ticker=r["ticker"],
                er_date=r["er_date"],
                predicted_eps=r["predicted_eps"],
                real_eps=r["real_eps"],
                ticker_view=ticker_view,
                facts_df=facts,              # pass canonical facts df
                window_days=window_days
            )

    write_jsonl(gen(), out_path)
    print(f"Done. Wrote {len(eps)} SubGraph lines to {out_path}")


# ---------- Entry point ----------
def main() -> None:
    build_subgraphs_jsonl(
        eps_csv_path=config.QUARTERLY_EPS_DATA_CSV,
        facts_jsonl_path=config.NEWS_FACTS,
        out_path=config.SUBGRAPHS_JSONL,
        window_days=config.WINDOW_DAYS,
        sentiment_min=config.SENTIMENT_MIN
    )

if __name__ == "__main__":
    main()
