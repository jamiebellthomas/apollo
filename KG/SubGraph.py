import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config 

# Set environment variable to disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import sqlite3
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
import torch
from torch_geometric.data import HeteroData
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

import numpy as np
import pandas as pd
from EdgeDecay import EdgeDecay

# Suppress Hugging Face warnings
import warnings
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="transformers")

FEATURE_NAMES = [
            # Momentum / returns (stock-only)
            "cumret_20", "cumret_60", "cumret_252",
            "mom_10_minus_60",
            # Moving-average gaps
            "ma_gap_20", "ma_gap_60",
            # Up-day share
            "up_pct_20", "up_pct_60",
            # Vol / downside / drawdown (stock-only)
            "vol_20", "vol_60",
            "downside_vol_20", "downside_vol_60",
            "max_dd_60", "max_dd_252",
            # Market-adjusted (vs SPY) & correlation
            "abn_sum_20",
            "beta_60", "alpha_60", "resid_vol_60",
            "corr_60",
            # Trend: price & abnormal (CAR slope over 60, bps/day)
            "slope_price_60", "slope_car_60_bps",
            # Pre-event information leakage proxy
            "CAR_pre20",
            # Technical
            "RSI_14",
            "MACD_diff", "MACD_signal",
            # 52w relative position
            "pct_to_52w_high", "pct_from_52w_low"
        ]



@dataclass
class SubGraph:
    primary_ticker: str
    reported_date: str
    eps_surprise: float | None
    fact_count: int
    fact_list: List[Dict[str, Any]]
    label: int


    def to_json_line(self) -> str:
        """Return a JSON string with exactly the specified top-level keys."""
        payload = {
            "primary_ticker": self.primary_ticker,
            "reported_date": self.reported_date,
            "eps_surprise": self.eps_surprise,
            "fact_count": self.fact_count,
            "fact_list": self.fact_list, 
            "label": self.label
        }
        return json.dumps(payload, ensure_ascii=False)
    
    @staticmethod
    def canonicalize_event_text(et: str, mode: str = "as_is") -> str:
        et = et.strip()
        if mode == "as_is":
            return et                         # keep underscores if that’s how you clustered
        if mode == "spaces":
            return et.replace("_", " ")
        if mode == "template_spaces":
            return f"event type: {et.replace('_',' ')}"
        if mode == "template_as_is":
            return f"event type: {et}"
        return et
    
    
    def encode(
        self,
        text_encoder: SentenceTransformer = None,
        *,
        fuse: str | None = None,                 # None | "concat"
        l2_normalize: bool = True,
        edge_decay=EdgeDecay(decay_days=config.WINDOW_DAYS, final_weight=0.05),# instance of EdgeDecay or None
        decay_type: str = "exponential",          # "linear"|"exponential"|"logarithmic"|"sigmoid"|"quadratic"
        event_encode_mode: str = "spaces"           # "as_is" | "spaces" | "template_spaces" | "template_as_is"
    ) -> Dict[str, Any]:
        """
        Build arrays for this SubGraph using a single encoder for both text and event representations:
        - fact_text_embeddings : (N, d)
        - fact_event_embeddings: (N, d)  (event_type encoded with same encoder)
        - fact_features        : (N, 2d) if fuse=="concat", else None
        - edge                 : {"sentiment": (N,), "weighting": (N,)}
        - company_features     : (1, p)  ([[1.0]] if None)

        Notes:
        • No centroid lookups. Event representation is an embedding of the event string (optionally augmented).
        • If event_type is missing, falls back to "other".
        • L2-normalization is applied row-wise if enabled.
        • Edge weighting uses provided EdgeDecay with selected decay_type.
        """
        if text_encoder is None:
            # Use local cache directory
            cache_dir = os.path.join(os.path.dirname(__file__), "model_cache")
            text_encoder = SentenceTransformer("all-mpnet-base-v2", cache_folder=cache_dir)
            
        facts = self.fact_list or []
        N = len(facts)

        # ---------------- Gather per-fact fields ----------------
        texts       = [(f.get("raw_text") or "") for f in facts]
        sentiments  = np.asarray([float(f.get("sentiment", 0.0)) for f in facts], dtype=np.float32)
        delta_days  = np.asarray([int(f.get("delta_days", 0))     for f in facts], dtype=np.int32)
        event_types = [str(f.get("event_type") or "other") for f in facts]

        # ---------------- Build event strings to encode ----------------
        event_texts = [self.canonicalize_event_text(et, mode=event_encode_mode) for et in event_types]
        # ---------------- Embed raw_text ----------------
        if N > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                text_vecs = text_encoder.encode(
                    texts,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                    show_progress_bar=False
                ).astype(np.float32, copy=False)
            d = text_vecs.shape[1]
        else:
            # best-effort dimension inference
            d = getattr(text_encoder, "get_sentence_embedding_dimension", lambda: 384)()
            text_vecs = np.zeros((0, d), dtype=np.float32)

        # ---------------- Embed event representations (same encoder) ----------------
        if N > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                event_vecs = text_encoder.encode(
                    event_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                    show_progress_bar=False
                ).astype(np.float32, copy=False)
            # Sanity: ensure same dimension
            if event_vecs.shape[1] != d:
                raise ValueError(f"Encoder produced mismatched dims: text d={d}, event d={event_vecs.shape[1]}")
        else:
            event_vecs = np.zeros((0, d), dtype=np.float32)

        # ---------------- Optional L2 normalization ----------------
        if l2_normalize:
            eps = 1e-8
            if text_vecs.size:
                norms = np.linalg.norm(text_vecs, axis=1, keepdims=True)
                text_vecs = text_vecs / (norms + eps)
            if event_vecs.size:
                norms = np.linalg.norm(event_vecs, axis=1, keepdims=True)
                event_vecs = event_vecs / (norms + eps)

        # ---------------- Optional fusion ----------------
        if fuse == "concat":
            fact_features = np.concatenate([text_vecs, event_vecs], axis=1) if N > 0 else np.zeros((0, 2 * d), dtype=np.float32)
        else:
            fact_features = None

        # ---------------- Edge weighting via EdgeDecay ----------------
        if edge_decay is None or N == 0:
            weighting = np.ones((N,), dtype=np.float32)
        else:
            dt = decay_type.lower()
            if   dt == "linear":       fn = edge_decay.linear
            elif dt == "exponential":  fn = edge_decay.exponential
            elif dt == "logarithmic":  fn = edge_decay.logarithmic
            elif dt == "sigmoid":      fn = edge_decay.sigmoid
            elif dt == "quadratic":    fn = edge_decay.quadratic
            else:
                raise ValueError(f"Unknown decay_type: {decay_type!r}")
            weighting = np.array([fn(int(dy)) for dy in delta_days], dtype=np.float32)
            weighting = np.clip(weighting, 0.0, 1.0)

        edge = {
            "sentiment": sentiments,   # (N,)
            "weighting": weighting,    # (N,)
        }

        # Return; keep an alias for backward compatibility if your code still expects "fact_event_centroids".
        return {
            "fact_text_embeddings": text_vecs,      # (N, d)
            "fact_event_embeddings": event_vecs,    # (N, d)
            "fact_features": fact_features,         # (N, 2d) iff fuse=="concat"
            "edge": edge,                           # {"sentiment": (N,), "weighting": (N,)}
        }

    
    def visualise_pricing(self):
        conn = sqlite3.connect(config.DB_PATH)
        cursor = conn.cursor()
        init_date = datetime.strptime(self.reported_date, "%Y-%m-%d")
        start_date = init_date - timedelta(days=2)  # 2 days before the reported date
        end_date = start_date + timedelta(days=60)

        query = f"""
        SELECT date, adjusted_close
        FROM {config.PRICING_TABLE_NAME}
        WHERE ticker = ?
        AND date BETWEEN ? AND ?
        ORDER BY date;
        """

        cursor.execute(query, (self.primary_ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")))
        rows = cursor.fetchall()
        conn.close()

        dates = [datetime.strptime(row[0][:10], "%Y-%m-%d") for row in rows]
        prices = [row[1] for row in rows]

        plt.rcParams['text.usetex'] = True  # enable LaTeX rendering

        plt.figure(figsize=(12, 6))
        plt.plot(dates, prices, marker='o')
        plt.title(rf"\textbf{{Pricing Data for {self.primary_ticker}}} from {start_date.date()} to {end_date.date()}")
        plt.xlabel(r"\textbf{Date}")
        plt.ylabel(r"\textbf{Adjusted Close Price}")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    

    @classmethod
    def load_valid_tickers(cls) -> set:
        """
        Loads valid tickers from the metadata CSV defined in config.METADATA_CSV_FILEPATH.

        Returns:
            A set of ticker symbols (strings) that are valid for graph construction.
        """
        # Use absolute path for metadata CSV
        csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), config.METADATA_CSV_FILEPATH)
        df = pd.read_csv(csv_path)
        return set(df['Symbol'].dropna().unique())
    


    def get_fact_node_features(
        self,
        text_encoder: SentenceTransformer = None,
        edge_decay=EdgeDecay(decay_days=config.WINDOW_DAYS, final_weight=0.05),
        decay_type: str = "exponential",
        fuse: str | None = "concat",
        l2_normalize: bool = True
    ) -> np.ndarray:
        """
        Generates fact node embeddings using the encode() method.

        Args:
            text_encoder: SentenceTransformer instance. If None, will use cached one.
            fuse: "concat" or None — determines whether to concatenate text and centroid embeddings.
            l2_normalize: Whether to apply row-wise L2 normalisation to embeddings.

        Returns:
            fact_features: (N_facts, d) NumPy array of fused fact embeddings.
        """
        if text_encoder is None:
            # Import here to avoid circular imports
            from run import get_cached_transformer
            text_encoder = get_cached_transformer()
            
        encoded = self.encode(
            text_encoder=text_encoder,
            fuse=fuse,
            l2_normalize=l2_normalize,
            edge_decay=edge_decay,
            decay_type=decay_type
        )
        return encoded['fact_features']
    

    def get_ticker_node_features(
        self,
        valid_tickers: set | None = None
    ) -> tuple[np.ndarray, dict[str, int]]:
        """
        Build per-ticker features from price history (up to the event date), market-adjusted vs SPY.
        ALWAYS includes the primary ticker (if present in valid_tickers and pricing exists).

        Returns:
            - ticker_features: (N_tickers, D) float32 array
            - ticker_index: dict[ticker -> row index]
        """
        import sqlite3, math
        import numpy as np
        import pandas as pd
        from datetime import timedelta
        import config

        FEATURE_NAMES = [
            "cumret_20","cumret_60","cumret_252","mom_10_minus_60",
            "ma_gap_20","ma_gap_60","up_pct_20","up_pct_60",
            "vol_20","vol_60","downside_vol_20","downside_vol_60",
            "max_dd_60","max_dd_252","abn_sum_20","beta_60","alpha_60",
            "resid_vol_60","corr_60","slope_price_60","slope_car_60_bps",
            "CAR_pre20","RSI_14","MACD_diff","MACD_signal",
            "pct_to_52w_high","pct_from_52w_low"
        ]
        D = len(FEATURE_NAMES)

        facts = self.fact_list or []
        event_date = pd.to_datetime(self.reported_date)

        # --- universe: union(mentioned tickers, primary ticker) ---
        mentioned = {t for f in facts for t in f.get("tickers", [])}
        primary = {self.primary_ticker} if getattr(self, "primary_ticker", None) else set()
        universe = (mentioned | primary)
        if valid_tickers:
            used_tickers = sorted(t for t in universe if t in valid_tickers)
        else:
            used_tickers = sorted(universe)

        ticker_index = {t: i for i, t in enumerate(used_tickers)}
        feats = np.zeros((len(ticker_index), D), dtype=np.float32)
        if not used_tickers:
            return feats, ticker_index

        # --- helpers ---
        def _open_conn():
            try:
                # Use absolute path for database
                db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), config.DB_PATH)
                return sqlite3.connect(db_path)
            except Exception:
                # Fallback to relative path
                return sqlite3.connect(config.DB_PATH)

        def _fetch(symbol: str, start: str, end: str) -> pd.DataFrame:
            q = f"""
                SELECT date, adjusted_close
                FROM {config.PRICING_TABLE_NAME}
                WHERE ticker = ? AND date BETWEEN ? AND ?
                ORDER BY date ASC;
            """
            with _open_conn() as conn:
                df = pd.read_sql_query(q, conn, params=(symbol, start, end))
            if df.empty:
                return df
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").rename(columns={"adjusted_close":"px"}).reset_index(drop=True)
            return df

        def _ema(s: pd.Series, span: int) -> pd.Series:
            return s.ewm(span=span, adjust=False).mean()

        def _safe_std(a: pd.Series) -> float:
            return float(a.std(ddof=0)) if len(a) > 1 else 0.0

        def _lin_slope(y: np.ndarray) -> float:
            if y.size < 2:
                return 0.0
            x = np.arange(y.size, dtype=float)
            a, b = np.polyfit(x, y.astype(float), 1)
            return float(a)

        def _segment(df_merged: pd.DataFrame, e_idx: int, L: int):
            end = e_idx
            start = max(0, end - (L - 1))
            return df_merged.iloc[start:end + 1].copy()

        # --- pull enough window to snap to next trading day ---
        fetch_start = (event_date - pd.Timedelta(days=420)).strftime("%Y-%m-%d")
        fetch_end   = (event_date + pd.Timedelta(days=7)).strftime("%Y-%m-%d")

        spy = _fetch("SPY", fetch_start, fetch_end)
        if spy.empty:
            # no benchmark: return zeros but keep indices
            return feats, ticker_index

        for tkr, row_i in ticker_index.items():
            px = _fetch(tkr, fetch_start, fetch_end)
            if px.empty:
                continue

            df = pd.merge(px, spy, on="date", how="inner", suffixes=("_s","_m"))
            if df.empty or len(df) < 20:
                continue

            pos = df.index[df["date"] >= event_date]
            if len(pos) == 0:
                # fallback: last pre-event day
                pre = df.index[df["date"] < event_date]
                if len(pre) == 0:
                    continue
                e_idx = int(pre[-1])
            else:
                e_idx = int(pos[0])
            if e_idx < 1:
                continue

            df["r_s"] = np.log(df["px_s"] / df["px_s"].shift(1))
            df["r_m"] = np.log(df["px_m"] / df["px_m"].shift(1))
            df = df.dropna(subset=["r_s","r_m"]).reset_index(drop=True)

            pos2 = df.index[df["date"] >= event_date]
            if len(pos2) == 0:
                pre = df.index[df["date"] < event_date]
                if len(pre) == 0:
                    continue
                e_idx = int(pre[-1])
            else:
                e_idx = int(pos2[0])

            df["abn"] = df["r_s"] - df["r_m"]
            df["car"] = df["abn"].cumsum()

            seg20  = _segment(df, e_idx, 20)
            seg60  = _segment(df, e_idx, 60)
            seg252 = _segment(df, e_idx, 252)
            seg10  = _segment(df, e_idx, 10)
            seg14  = _segment(df, e_idx, 14)

            def cumret(seg): return float(seg["r_s"].sum()) if len(seg) else 0.0
            cumret_20  = cumret(seg20)
            cumret_60  = cumret(seg60)
            cumret_252 = cumret(seg252)
            mom_10_minus_60 = (float(seg10["r_s"].sum()) if len(seg10) else 0.0) - cumret_60

            px0 = float(df.iloc[e_idx]["px_s"])
            ma20 = float(df["px_s"].iloc[max(0, e_idx-19):e_idx+1].mean()) if e_idx >= 19 else math.nan
            ma60 = float(df["px_s"].iloc[max(0, e_idx-59):e_idx+1].mean()) if e_idx >= 59 else math.nan
            ma_gap_20 = (px0 / ma20 - 1.0) if (ma20 and not math.isnan(ma20)) else 0.0
            ma_gap_60 = (px0 / ma60 - 1.0) if (ma60 and not math.isnan(ma60)) else 0.0

            up_pct_20 = float((seg20["r_s"] > 0).mean()) if len(seg20) else 0.0
            up_pct_60 = float((seg60["r_s"] > 0).mean()) if len(seg60) else 0.0

            vol_20 = _safe_std(seg20["r_s"])
            vol_60 = _safe_std(seg60["r_s"])
            downside_vol_20 = _safe_std(seg20["r_s"].where(seg20["r_s"] < 0, 0.0))
            downside_vol_60 = _safe_std(seg60["r_s"].where(seg60["r_s"] < 0, 0.0))

            def max_dd(seg):
                if len(seg) < 2:
                    return 0.0
                prices = seg["px_s"].to_numpy()
                run_max = np.maximum.accumulate(prices)
                dd = prices / run_max - 1.0
                return float(dd.min())
            max_dd_60  = max_dd(seg60)
            max_dd_252 = max_dd(seg252)

            abn_sum_20 = float(seg20["abn"].sum()) if len(seg20) else 0.0

            def ols_beta_alpha(seg):
                if len(seg) < 5:
                    return 0.0, 0.0, 0.0
                x = seg["r_m"].to_numpy()
                y = seg["r_s"].to_numpy()
                a, b = np.polyfit(x, y, 1)  # a=beta, b=alpha (per-day)
                resid = y - (a * x + b)
                return float(a), float(b), float(resid.std(ddof=0))
            beta_60, alpha_60, resid_vol_60 = ols_beta_alpha(seg60)

            corr_60 = float(seg60["r_s"].corr(seg60["r_m"])) if len(seg60) >= 2 else 0.0

            slope_price_60 = _lin_slope(np.log(seg60["px_s"].to_numpy())) if len(seg60) >= 2 else 0.0
            if len(seg60) >= 2:
                car60 = np.cumsum(seg60["abn"].to_numpy())
                slope_car_60_bps = _lin_slope(car60) * 10000.0
            else:
                slope_car_60_bps = 0.0

            if e_idx >= 20:
                pre_slice = df.iloc[e_idx-20:e_idx]
                CAR_pre20 = float(pre_slice["abn"].sum())
            else:
                CAR_pre20 = 0.0

            if len(seg14) >= 2:
                delta = seg14["px_s"].diff().dropna()
                gains = delta.clip(lower=0).mean()
                losses = (-delta.clip(upper=0)).mean()
                RSI_14 = 100.0 if losses == 0 else 100.0 - (100.0 / (1.0 + gains / losses))
            else:
                RSI_14 = 50.0

            close_series = df["px_s"].iloc[:e_idx+1]
            if len(close_series) >= 26:
                ema12 = _ema(close_series, 12)
                ema26 = _ema(close_series, 26)
                macd = ema12 - ema26
                MACD_diff = float(macd.iloc[-1])
                MACD_signal = float(_ema(macd, 9).iloc[-1]) if len(macd) >= 9 else 0.0
            else:
                MACD_diff = 0.0
                MACD_signal = 0.0

            if len(seg252) >= 2:
                px_hist = seg252["px_s"].to_numpy()
                hi, lo = float(px_hist.max()), float(px_hist.min())
                pct_to_52w_high = (px0 / hi - 1.0) if hi > 0 else 0.0
                pct_from_52w_low = (px0 / lo - 1.0) if lo > 0 else 0.0
            else:
                pct_to_52w_high = 0.0
                pct_from_52w_low = 0.0

            vec = np.array([
                cumret_20, cumret_60, cumret_252,
                mom_10_minus_60,
                ma_gap_20, ma_gap_60,
                up_pct_20, up_pct_60,
                vol_20, vol_60,
                downside_vol_20, downside_vol_60,
                max_dd_60, max_dd_252,
                abn_sum_20,
                beta_60, alpha_60, resid_vol_60,
                corr_60,
                slope_price_60, slope_car_60_bps,
                CAR_pre20,
                RSI_14,
                MACD_diff, MACD_signal,
                pct_to_52w_high, pct_from_52w_low
            ], dtype=np.float32)

            feats[row_i, :] = vec

        return feats, ticker_index




    

    def get_edges(
        self,
        ticker_index: dict[str, int],
        edge_decay: EdgeDecay = EdgeDecay(decay_days=config.WINDOW_DAYS, final_weight=0.05),
        decay_type: str = "exponential"
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build edge connectivity and attributes for fact → company edges.
        Uses type-specific node indices for proper heterogeneous graph structure.
        
        Returns:
            - edge_index: (2, N_edges) with type-specific indices
            - edge_attr: (N_edges, 2) with (sentiment, weight) pairs
        """
        facts = self.fact_list or []
        edge_src = []  # fact indices (0 to N_facts-1)
        edge_dst = []  # company indices (0 to N_companies-1) - type-specific!
        edge_attrs = []

        # Set up decay function
        if edge_decay is None:
            compute_weight = lambda d: 1.0
        else:
            decay_fn = getattr(edge_decay, decay_type.lower(), None)
            if not decay_fn:
                raise ValueError(f"Unknown decay type: {decay_type}")
            compute_weight = lambda d: float(np.clip(decay_fn(int(d)), 0.0, 1.0))

        # Iterate over facts and link to valid tickers
        for fact_idx, fact in enumerate(facts):
            sentiment = float(fact.get("sentiment", 0.0))
            delta_days = int(fact.get("delta_days", 0))
            weight = compute_weight(delta_days)

            for ticker in fact.get("tickers", []):
                if ticker in ticker_index:
                    # Use type-specific indices:
                    # - fact_idx: 0 to N_facts-1 (fact node index)
                    # - ticker_index[ticker]: 0 to N_companies-1 (company node index)
                    edge_src.append(fact_idx)
                    edge_dst.append(ticker_index[ticker])
                    edge_attrs.append((sentiment, weight))

        edge_index = np.array([edge_src, edge_dst], dtype=np.int64)         # shape (2, N_edges)
        edge_attr = np.array(edge_attrs, dtype=np.float32)                  # shape (N_edges, 2)
        return edge_index, edge_attr



    def to_numpy_graph(
        self,
        text_encoder: SentenceTransformer = None,
        edge_decay: EdgeDecay = EdgeDecay(decay_days=config.WINDOW_DAYS, final_weight=0.05),
        decay_type: str = "exponential",
        fuse: str | None = "concat",
        l2_normalize: bool = True
    ) -> dict[str, np.ndarray | dict[str, int]]:
        """
        Builds a complete NumPy-format graph for this SubGraph, ready for conversion
        to a heterogeneous GNN input.

        Args:
            text_encoder: SentenceTransformer instance. If None, will use cached one.
            fuse: Whether to concatenate text and centroid embeddings.
            l2_normalize: Whether to L2-normalise text and centroid embeddings.

        Returns:
            A dictionary with:
            - 'fact_features': (N_facts, d)
            - 'ticker_features': (N_tickers, d)
            - 'edge_index': (2, N_edges)
            - 'edge_attr': (N_edges, 2)
            - 'ticker_index': mapping from ticker symbol to row index
        """
        if text_encoder is None:
            # Import here to avoid circular imports
            from run import get_cached_transformer
            text_encoder = get_cached_transformer()
            
        valid_tickers = self.load_valid_tickers()

        fact_features = self.get_fact_node_features(
            text_encoder=text_encoder,
            edge_decay=edge_decay,
            decay_type=decay_type,
            fuse=fuse,
            l2_normalize=l2_normalize
        )

        ticker_features, ticker_index = self.get_ticker_node_features(
            valid_tickers=valid_tickers,
        )

        edge_index, edge_attr = self.get_edges(
            ticker_index=ticker_index,
            edge_decay=edge_decay,
            decay_type=decay_type
        )

        # Add fact metadata to graph_dict for hover tooltips
        fact_dates = [f.get("date", "N/A") for f in self.fact_list]
        fact_texts = [f.get("raw_text", "") for f in self.fact_list]
        fact_event_types = [f.get("event_type", "") for f in self.fact_list]
        fact_ids = [f.get("fact_id", -1) for f in self.fact_list]  # Add original fact_ids

        fact_sentiments = [float(f.get("sentiment", 0.0)) for f in self.fact_list]
        fact_decays = [float(f.get("delta_days", 0)) for f in self.fact_list]  # we'll apply decay function later

        # Compute decay weights using the same logic as in encode()
        decay_fn = getattr(edge_decay, decay_type.lower(), None)
        if decay_fn:
            fact_weights = [float(np.clip(decay_fn(int(d)), 0.0, 1.0)) for d in fact_decays]
        else:
            fact_weights = [1.0 for _ in fact_decays]

        return {
            "fact_features": fact_features,
            "ticker_features": ticker_features,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "ticker_index": ticker_index,
            "fact_dates": fact_dates,
            "fact_texts": fact_texts,
            "fact_sentiments": fact_sentiments,
            "fact_weights": fact_weights,
            "fact_event_types": fact_event_types,
            "fact_ids": fact_ids  # Add fact_ids to the graph data
        }



    def to_pyg_data(
        self,
        text_encoder = None,
        edge_decay = EdgeDecay(decay_days=config.WINDOW_DAYS, final_weight=0.05),
        decay_type: str = "exponential",
        fuse: Optional[str] = "concat",
        l2_normalize: bool = True
    ) -> HeteroData:
        """
        Convert this SubGraph into a PyTorch Geometric HeteroData object.
        
        Includes:
        - Fact and company nodes
        - Bi-directional edges with 'mentions' and 'mentioned_in' relations
        - Edge attributes containing sentiment and decayed weighting
        - Node features and graph-level label
        """
        if text_encoder is None:
            # Import here to avoid circular imports
            from run import get_cached_transformer
            text_encoder = get_cached_transformer()
            
        np_graph: Dict[str, np.ndarray] = self.to_numpy_graph(
            text_encoder=text_encoder,
            edge_decay=edge_decay,
            decay_type=decay_type,
            fuse=fuse,
            l2_normalize=l2_normalize
        )

        # Create HeteroData object
        hetero_data = HeteroData()

        # Ensure we have valid features
        fact_features = np_graph["fact_features"]
        ticker_features = np_graph["ticker_features"]
        
        if fact_features is None or fact_features.size == 0:
            fact_features = np.zeros((1, 768), dtype=np.float32)
        if ticker_features is None or ticker_features.size == 0:
            ticker_features = np.zeros((1, 27), dtype=np.float32)

        # Assign node features
        hetero_data['fact'].x = torch.tensor(fact_features, dtype=torch.float)
        hetero_data['company'].x = torch.tensor(ticker_features, dtype=torch.float)

        # Create edge index and attributes
        if np_graph["edge_index"] is not None and np_graph["edge_index"].size > 0:
            edge_index = torch.tensor(np_graph["edge_index"], dtype=torch.long)
            edge_attr = torch.tensor(np_graph["edge_attr"], dtype=torch.float)

            # Add forward edge: fact → company
            hetero_data['fact', 'mentions', 'company'].edge_index = edge_index
            hetero_data['fact', 'mentions', 'company'].edge_attr = edge_attr

            # Add reverse edge: company → fact
            hetero_data['company', 'mentioned_in', 'fact'].edge_index = edge_index.flip(0)
            hetero_data['company', 'mentioned_in', 'fact'].edge_attr = edge_attr.clone()
        else:
            # Create dummy edges to maintain structure
            hetero_data['fact', 'mentions', 'company'].edge_index = torch.zeros((2, 1), dtype=torch.long)
            hetero_data['fact', 'mentions', 'company'].edge_attr = torch.zeros((1, 2), dtype=torch.float)
            hetero_data['company', 'mentioned_in', 'fact'].edge_index = torch.zeros((2, 1), dtype=torch.long)
            hetero_data['company', 'mentioned_in', 'fact'].edge_attr = torch.zeros((1, 2), dtype=torch.float)

        # Add graph-level label
        hetero_data['graph_label'] = torch.tensor(self.label, dtype=torch.long)
        
        # Store original fact_ids for explainability
        if "fact_ids" in np_graph:
            hetero_data.fact_ids = np_graph["fact_ids"]
        
        # Store subgraph metadata for explainability
        hetero_data.primary_ticker = self.primary_ticker
        hetero_data.reported_date = self.reported_date
        hetero_data.eps_surprise = self.eps_surprise
        hetero_data.label = self.label
        hetero_data.fact_count = len(self.fact_list) if self.fact_list else 0

        return hetero_data



    def visualise_numpy_graph(
        self,
        graph_dict: dict,
        max_nodes: int = 30
    ):
        """
        Visualises the SubGraph graph with:
        - Node types (fact, ticker)
        - Edge color for sentiment
        - Edge thickness for weighted polarity
        - Clean legend instead of edge text labels
        """
        fact_features = graph_dict["fact_features"]
        ticker_index = graph_dict["ticker_index"]
        edge_index = graph_dict["edge_index"]
        edge_attr = graph_dict["edge_attr"]

        num_facts = fact_features.shape[0]
        num_tickers = len(ticker_index)

        if num_facts + num_tickers > max_nodes:
            print(f"Graph too large to visualise ({num_facts + num_tickers} nodes > max_nodes={max_nodes})")
            return

        G = nx.DiGraph()
        edge_colors = []
        edge_widths = []

        # Add fact nodes
        for i in range(num_facts):
            G.add_node(i, type="fact", label=str(i))

        # Add ticker nodes
        fact_offset = num_facts
        reverse_ticker_index = {v: k for k, v in ticker_index.items()}
        ticker_name_to_node = {}
        for i in range(num_tickers):
            ticker = reverse_ticker_index[i]
            node_idx = fact_offset + i
            G.add_node(node_idx, type="ticker", label=ticker)
            ticker_name_to_node[ticker] = node_idx

        primary_ticker_node = ticker_name_to_node.get(self.primary_ticker, None)

        # Add edges with attributes
        for i in range(edge_index.shape[1]):
            fact_idx = edge_index[0, i]
            ticker_idx = edge_index[1, i]
            sentiment, decay = edge_attr[i]

            source = fact_idx
            target = fact_offset + ticker_idx

            color = "green" if sentiment > 0 else "red" if sentiment < 0 else "gray"
            weighted_polarity = abs(sentiment * decay)
            width = 0.5 + 6 * weighted_polarity

            G.add_edge(source, target, color=color, weight=width)
            edge_colors.append(color)
            edge_widths.append(width)

        # Layout
        pos = nx.spring_layout(G, seed=42, k=0.8, iterations=100, center=(0, 0))
        if primary_ticker_node is not None:
            pos[primary_ticker_node] = np.array([0.0, 0.0])

        # Nodes
        fact_nodes = [n for n in G.nodes() if G.nodes[n]["type"] == "fact"]
        ticker_nodes = [n for n in G.nodes() if G.nodes[n]["type"] == "ticker"]
        node_colors = ["skyblue" if G.nodes[n]["type"] == "fact" else "lightgray" for n in G.nodes()]
        node_labels = {n: G.nodes[n]["label"] for n in G.nodes()}

        # Plot
        plt.figure(figsize=(12, 8))
        nx.draw_networkx_nodes(G, pos, nodelist=fact_nodes, node_shape='o', node_color='skyblue', label=r"Fact")
        nx.draw_networkx_nodes(G, pos, nodelist=ticker_nodes, node_shape='s', node_color='lightgray', label=r"Ticker")
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=9)
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths)

        # Legend handles
        legend_elements = [
            Line2D([0], [0], color='green', lw=3, label=r'Positive sentiment'),
            Line2D([0], [0], color='red', lw=3, label=r'Negative sentiment'),
            Line2D([0], [0], color='gray', lw=3, label=r'Neutral sentiment'),
            Line2D([0], [0], color='black', lw=1, label=r'Low polarity $\times$ recency'),
            Line2D([0], [0], color='black', lw=4, label=r'High polarity $\times$ recency')
        ]

        plt.legend(handles=legend_elements, loc='upper left')
        plt.title(f"SubGraph for {self.primary_ticker}\nColor = sentiment, Thickness = weighted polarity")
        plt.axis("off")
        plt.tight_layout()
        plt.show()


    def visualise_numpy_graph_interactive(
        self,
        graph_dict: dict,
        max_nodes: int = 30,
        save_dir: str = "Plots/KGs"
    ) -> go.Figure:
        """
        Visualises the SubGraph graph interactively using Plotly.
        - Arrows encode sentiment polarity and strength.
        - Nodes are facts or tickers.
        - Hover over a ticker to see price-based FEATURES.
        - Saved to Plots/KGs/{ticker}_{reported_date}.html
        """

        pio.renderers.default = "browser"

        fact_features = graph_dict["fact_features"]
        ticker_index  = graph_dict["ticker_index"]
        ticker_feats  = graph_dict.get("ticker_features", None)  # (N_tickers, D) where D=len(FEATURE_NAMES)
        edge_index    = graph_dict["edge_index"]
        edge_attr     = graph_dict["edge_attr"]
        fact_dates    = graph_dict.get("fact_dates", [""] * fact_features.shape[0])
        fact_texts    = graph_dict.get("fact_texts", [""] * fact_features.shape[0])

        num_facts   = fact_features.shape[0]
        num_tickers = len(ticker_index)

        if num_facts + num_tickers > max_nodes:
            print(f"Graph too large to visualise ({num_facts + num_tickers} nodes > max_nodes={max_nodes})")
            return

        # --- helper: nice formatting for ticker features ---
        def _fmt_feature(name: str, val: float) -> str:
            # Present percents, bps/day, etc.
            try:
                v = float(val)
            except Exception:
                return f"{name}: {val}"

            # basis points/day features
            if name in ("slope_car_60_bps",):
                return f"{name}: {v:.2f} bps/day"

            if name == "alpha_60":
                # per-day alpha; show as bps/day
                return f"{name}: {v*10000.0:.2f} bps/day"

            if name == "slope_price_60":
                # log-price slope per day; show in bps/day to compare with CAR slope
                return f"{name}: {v*10000.0:.2f} bps/day"

            # proportions/percents
            if name.startswith(("cumret_", "ma_gap_", "pct_", "abn_sum_20", "CAR_pre20")):
                return f"{name}: {v*100:.2f}%"

            if name.startswith("up_pct_"):
                return f"{name}: {v*100:.1f}%"

            # correlations / beta / vols
            if name in ("beta_60", "corr_60"):
                return f"{name}: {v:.3f}"

            if name in ("vol_20", "vol_60", "downside_vol_20", "downside_vol_60", "resid_vol_60"):
                return f"{name}: {v*100:.2f}%"

            if name == "RSI_14":
                return f"{name}: {v:.1f}"

            if name in ("MACD_diff", "MACD_signal"):
                return f"{name}: {v:.4f}"

            # default
            return f"{name}: {v:.4f}"

        # Build mapping index->ticker and vice versa
        reverse_ticker_index = {v: k for k, v in ticker_index.items()}

        # NetworkX graph
        G = nx.DiGraph()
        fact_offset = num_facts
        ticker_name_to_node = {}

        # Add fact nodes
        for i in range(num_facts):
            G.add_node(i, label=f"Fact {i}", type="fact", date=fact_dates[i], text=fact_texts[i])

        # Add ticker nodes (attach feature vector if provided)
        for i in range(num_tickers):
            ticker = reverse_ticker_index[i]
            node_idx = fact_offset + i
            tfeat = ticker_feats[i] if (ticker_feats is not None and i < len(ticker_feats)) else None
            G.add_node(node_idx, label=ticker, type="ticker", tfeat=tfeat)
            ticker_name_to_node[ticker] = node_idx

        primary_node = ticker_name_to_node.get(self.primary_ticker)

        # Add edges with attributes
        for i in range(edge_index.shape[1]):
            fact_idx = int(edge_index[0, i])
            ticker_idx = int(edge_index[1, i])
            sentiment, decay = edge_attr[i]
            src = fact_idx
            dst = fact_offset + ticker_idx
            weighted = float(sentiment) * float(decay)

            G.add_edge(
                src,
                dst,
                sentiment=float(sentiment),
                decay=float(decay),
                weighted_polarity=weighted,
                color="green" if sentiment > 0 else "red" if sentiment < 0 else "gray",
                width=1 + 6 * abs(weighted)
            )

        # Layout
        pos = nx.spring_layout(G, seed=42, k=1.2, iterations=100)
        if primary_node is not None:
            pos[primary_node] = np.array([0.0, 0.0])  # center

        # Build edge traces by colour
        edge_groups = {"green": [], "red": [], "gray": []}
        hover_groups = {"green": [], "red": [], "gray": []}

        for u, v, attr in G.edges(data=True):
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            color = attr["color"]

            fact_data = G.nodes[u]
            hover_text = f"<b>Sentiment:</b> {attr['sentiment']:.2f}<br>"
            hover_text += f"<b>Decay:</b> {attr['decay']:.2f}<br>"
            hover_text += f"<b>Weighted:</b> {attr['weighted_polarity']:.2f}<br>"
            hover_text += f"<b>Date:</b> {fact_data.get('date', 'N/A')}<br>"
            hover_text += f"<b>Text:</b> {fact_data.get('text', '')[:300]}"

            edge_groups[color].append(((x0, x1, None), (y0, y1, None)))
            hover_groups[color].append(hover_text)

        edge_traces = []
        for color in ["green", "red", "gray"]:
            xs, ys, texts = [], [], hover_groups[color]
            for (x0, x1, _), (y0, y1, _) in edge_groups[color]:
                xs += [x0, x1, None]
                ys += [y0, y1, None]

            trace = go.Scatter(
                x=xs,
                y=ys,
                line=dict(width=2, color=color),
                hoverinfo='text',
                mode='lines',
                text=texts,
                name=f"{color.capitalize()} edges"
            )
            edge_traces.append(trace)

        # Node traces
        node_x, node_y, node_text, node_color, hover_texts = [], [], [], [], []
        for node, data in G.nodes(data=True):
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(data['label'])
            node_color.append('lightgray' if data['type'] == 'ticker' else 'skyblue')

            if data['type'] == 'fact':
                idx = node  # fact nodes are 0-indexed
                sentiment = graph_dict["fact_sentiments"][idx]
                decay = graph_dict["fact_weights"][idx]
                event_type = graph_dict["fact_event_types"][idx]
                hover_texts.append(
                    f"<b>Fact</b><br>"
                    f"Date: {data.get('date', 'N/A')}<br>"
                    f"Raw Sentiment: {sentiment:.2f}<br>"
                    f"Decay Coefficient: {decay:.2f}<br>"
                    f"Event Type: {event_type}<br>"
                    f"Text: {data.get('text', '')[:300]}"
                )
            else:
                # Ticker hover with features
                tkr = data['label']
                feat_vec = data.get('tfeat', None)
                if feat_vec is None or len(FEATURE_NAMES) == 0:
                    hover_texts.append(f"<b>Ticker</b><br>{tkr}")
                else:
                    lines = [f"<b>Ticker</b>: {tkr}"]
                    # be defensive about length mismatches
                    D = min(len(FEATURE_NAMES), len(feat_vec))
                    for j in range(D):
                        lines.append(_fmt_feature(FEATURE_NAMES[j], float(feat_vec[j])))
                    hover_texts.append("<br>".join(lines))

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=node_text,
            textposition='bottom center',
            hoverinfo='text',
            hovertext=hover_texts,
            marker=dict(
                color=node_color,
                size=20,
                line=dict(width=1, color='black')
            )
        )

        # Create figure
        fig = go.Figure(
            data=edge_traces + [node_trace],
            layout=go.Layout(
                title=dict(text=f"SubGraph for {self.primary_ticker} on {self.reported_date} - Label:{self.label}", x=0.5),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )

        # Save
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{self.primary_ticker}_{self.reported_date}.html")
        fig.write_html(save_path)
        print(f"Graph saved to: {save_path}")

        fig.show()
        return fig





#############################################



def load_centroids_jsonl(path: str) -> Dict[int, np.ndarray]:
    """
    Read config.CLUSTER_CENTROIDS JSONL → {cluster_id: np.float32[dim]}.
    Each line is: {"cluster_id": int, "cluster_name": str, "centroid": [floats...]}
    """
    id2centroid: Dict[int, np.ndarray] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            cid = int(obj["cluster_id"])
            vec = np.asarray(obj["centroid"], dtype=np.float32)
            id2centroid[cid] = vec
    return id2centroid

def load_first_with_exact_facts(jsonl_path: str, n: int = 5) -> dict:

    """
    Silly little utility to find the first subgraph with exactly n facts.
    """
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip bad lines silently
            facts = obj.get("fact_list") or []
            if len(facts) == n:
                print(f"Found at line {line_no}: "
                      f"{obj.get('primary_ticker')} {obj.get('reported_date')} with {len(facts)} facts")
                return obj
    raise ValueError(f"No subgraph with exactly {n} facts found in {jsonl_path}.")


if __name__ == "__main__":
    # ---------- test ----------
    # 1) pick a subgraph with exactly 5 facts
    obj = load_first_with_exact_facts(config.SUBGRAPHS_JSONL, n=125)
    sg = SubGraph(**obj)
    # sg.visualise_pricing()  # Visualise the pricing data for the subgraph

    # 2) resources
    encoder = SentenceTransformer("all-mpnet-base-v2")
    decay = EdgeDecay(decay_days=config.WINDOW_DAYS, final_weight=0.10)

    # 3) encode (choose fuse=None or "concat")
    out = sg.to_numpy_graph(
        text_encoder=encoder,
        edge_decay=decay,
        decay_type="exponential",
        fuse="concat",
        l2_normalize=True
    )

    # # 4) visualise
    # sg.visualise_numpy_graph(graph_dict=out, 
    #                          max_nodes=2000)

    # # 5) convert to PyG HeteroData
    # pyg_data = sg.to_pyg_data()
    # print(pyg_data)

    # 6) visualise interactive plot
    fig = sg.visualise_numpy_graph_interactive(graph_dict=out,
                                               max_nodes=3000)
    

    
