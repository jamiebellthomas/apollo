import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config 

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer


import numpy as np
import pandas as pd
from EdgeDecay import EdgeDecay

@dataclass
class SubGraph:
    primary_ticker: str
    reported_date: str
    predicted_eps: float | None
    real_eps: float | None
    fact_count: int
    fact_list: List[Dict[str, Any]]


    def to_json_line(self) -> str:
        """Return a JSON string with exactly the specified top-level keys."""
        payload = {
            "primary_ticker": self.primary_ticker,
            "reported_date": self.reported_date,
            "predicted_eps": self.predicted_eps,
            "real_eps": self.real_eps,
            "fact_count": self.fact_count,
            "fact_list": self.fact_list,  
        }
        return json.dumps(payload, ensure_ascii=False)
    
    
    def encode(
        self,
        text_encoder,
        id2centroid: Dict[int, np.ndarray],
        *,
        fuse: str | None = None,               # None | "concat"
        l2_normalize: bool = True,
        edge_decay=None,                        # instance of EdgeDecay or None
        decay_type: str = "exponential",        # "linear"|"exponential"|"logarithmic"|"sigmoid"|"quadratic"
        company_features: np.ndarray | None = None
    ) -> Dict[str, Any]:
        """
        Build arrays for this SubGraph:
        - fact_text_embeddings : (N, d_text)
        - fact_event_centroids : (N, d_event)
        - fact_features        : (N, d_text + d_event) if fuse=="concat", else None
        - edge                 : {"sentiment": (N,), "weighting": (N,)}  # weighting via EdgeDecay
        - company_features     : (1, p)  ([[1.0]] if None)

        Notes:
        • Uses the same sentence-transformer as clustering (e.g., "all-MiniLM-L6-v2").
        • Missing event_cluster_id → zero centroid row; weighting still computed from delta_days.
        • Learned projections/fusion should live in the model; this method only prepares inputs.
        """
        facts = self.fact_list or []
        N = len(facts)

        # ---------- Gather per-fact fields ----------
        texts      = [(f.get("raw_text") or "") for f in facts]
        sentiments = np.asarray([float(f.get("sentiment", 0.0)) for f in facts], dtype=np.float32)
        delta_days = np.asarray([int(f.get("delta_days", 0))     for f in facts], dtype=np.int32)
        cluster_ids = np.asarray(
            [(-1 if f.get("event_cluster_id") is None else int(f.get("event_cluster_id"))) for f in facts],
            dtype=np.int32
        )

        # ---------- Embed raw_text ----------
        if N > 0:
            text_vecs = text_encoder.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False
            ).astype(np.float32, copy=False)
            d_text = text_vecs.shape[1]
        else:
            # best-effort dimension inference
            d_text = getattr(text_encoder, "get_sentence_embedding_dimension", lambda: 384)()
            text_vecs = np.zeros((0, d_text), dtype=np.float32)

        # ---------- Lookup centroids ----------
        if id2centroid:
            example_vec = next(iter(id2centroid.values()))
            d_event = int(len(example_vec))
        else:
            d_event = d_text  # safe fallback
        centroids = np.zeros((N, d_event), dtype=np.float32)
        if N > 0 and id2centroid:
            for i, cid in enumerate(cluster_ids):
                if cid >= 0 and cid in id2centroid:
                    centroids[i] = id2centroid[cid]

        # --- L2-normalize text and centroid embeddings separately (row-wise) ---
        if l2_normalize:
            eps = 1e-8

            # Normalize raw_text embeddings: shape (N, d_text)
            if text_vecs.size:
                norms = np.linalg.norm(text_vecs, axis=1, keepdims=True)
                text_vecs = text_vecs / (norms + eps)

            # Normalize event-type centroids: shape (N, d_event)
            if centroids.size:
                norms = np.linalg.norm(centroids, axis=1, keepdims=True)
                centroids = centroids / (norms + eps)


        # ---------- Optional fusion ----------
        if fuse == "concat":
            if N > 0:
                fact_features = np.concatenate([text_vecs, centroids], axis=1)
            else:
                fact_features = np.zeros((0, d_text + d_event), dtype=np.float32)
        else:
            fact_features = None

        # ---------- Edge weighting via EdgeDecay ----------
        if edge_decay is None or N == 0:
            weighting = np.ones((N,), dtype=np.float32)
        else:
            dt = decay_type.lower()
            if dt == "linear":
                fn = edge_decay.linear
            elif dt == "exponential":
                fn = edge_decay.exponential
            elif dt == "logarithmic":
                fn = edge_decay.logarithmic
            elif dt == "sigmoid":
                fn = edge_decay.sigmoid
            elif dt == "quadratic":
                fn = edge_decay.quadratic
            else:
                raise ValueError(f"Unknown decay_type: {decay_type!r}")
            weighting = np.array([fn(int(d)) for d in delta_days], dtype=np.float32)
            weighting = np.clip(weighting, 0.0, 1.0)

        edge = {
            "sentiment": sentiments,   # (N,)
            "weighting": weighting,    # (N,)
        }

        # ---------- Company features ----------
        if company_features is None:
            comp_x = np.array([[1.0]], dtype=np.float32)
        else:
            comp_x = np.asarray(company_features, dtype=np.float32).reshape(1, -1)

        return {
            "fact_text_embeddings": text_vecs,   # (N, d_text)
            "fact_event_centroids": centroids,   # (N, d_event)
            "fact_features": fact_features,      # (N, d_text+d_event) iff fuse=="concat"
            "edge": edge,                        # {"sentiment": (N,), "weighting": (N,)}
            "company_features": comp_x,          # (1, p)
        }


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
    obj = load_first_with_exact_facts(config.SUBGRAPHS_JSONL, n=15)
    sg = SubGraph(**obj)

    # 2) resources
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    id2centroid = load_centroids_jsonl(config.CLUSTER_CENTROIDS)
    decay = EdgeDecay(decay_days=config.WINDOW_DAYS, final_weight=0.10)

    # 3) encode (choose fuse=None or "concat")
    out = sg.encode(
        text_encoder=encoder,
        id2centroid=id2centroid,
        fuse="concat",              # or None to keep separate matrices
        l2_normalize=True,
        edge_decay=decay,
        decay_type="exponential",   # "linear"|"exponential"|"logarithmic"|"sigmoid"|"quadratic"
        company_features=None
    )

    # 4) quick inspection
    print(f"Subgraph: {sg.primary_ticker} / {sg.reported_date}")
    print("fact_text_embeddings:", out["fact_text_embeddings"].shape)   # (5, 384)
    print("fact_event_centroids:", out["fact_event_centroids"].shape)   # (5, 384)
    print("fact_features (concat):", None if out["fact_features"] is None else out["fact_features"].shape)  # (5, 768)

    edge = out["edge"]

    print(f"\nSubgraph: {sg.primary_ticker} / ER date: {sg.reported_date}")
    print("---- Facts (sorted by delta_days ascending) ----")

    order = np.argsort([int(f.get("delta_days", 0)) for f in sg.fact_list])

    print(f"{'i':>2}  {'date':>10}  {'Δdays':>6}  {'weight':>8}  {'sent':>7}  {'cluster':>7}  {'event_type':>18}  text")
    for rank, i in enumerate(order):
        f   = sg.fact_list[i]
        dt  = f.get("date", "")
        dd  = int(f.get("delta_days", 0))
        w   = float(edge["weighting"][i]) if len(edge["weighting"]) > i else float("nan")
        s   = float(edge["sentiment"][i]) if len(edge["sentiment"]) > i else float("nan")
        cid = f.get("event_cluster_id", None)
        et  = (f.get("event_type") or "")
        # keep the table tidy
        if len(et) > 18:
            et = et[:17] + "…"
        txt = (f.get("raw_text") or "").replace("\n", " ")
        if len(txt) > 110:
            txt = txt[:107] + "..."
        print(f"{rank:>2}  {dt:>10}  {dd:>6}  {w:>8.4f}  {s:>+7.3f}  {str(cid):>7}  {et:>18}  {txt}")

