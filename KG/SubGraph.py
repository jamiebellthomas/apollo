import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config 

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
import sqlite3
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D


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
    label: int


    def to_json_line(self) -> str:
        """Return a JSON string with exactly the specified top-level keys."""
        payload = {
            "primary_ticker": self.primary_ticker,
            "reported_date": self.reported_date,
            "predicted_eps": self.predicted_eps,
            "real_eps": self.real_eps,
            "fact_count": self.fact_count,
            "fact_list": self.fact_list, 
            "label": self.label
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
        df = pd.read_csv(config.METADATA_CSV_FILEPATH)
        return set(df['Symbol'].dropna().unique())
    


    def get_fact_node_features(
        self,
        text_encoder,
        id2centroid: dict[int, np.ndarray],
        edge_decay=None,
        decay_type: str = "exponential",
        fuse: str | None = "concat",
        l2_normalize: bool = True
    ) -> np.ndarray:
        """
        Generates fact node embeddings using the encode() method.

        Args:
            fuse: "concat" or None — determines whether to concatenate text and centroid embeddings.
            l2_normalize: Whether to apply row-wise L2 normalisation to embeddings.

        Returns:
            fact_features: (N_facts, d) NumPy array of fused fact embeddings.
        """
        encoded = self.encode(
            text_encoder,
            id2centroid,
            fuse=fuse,
            l2_normalize=l2_normalize,
            edge_decay=edge_decay,
            decay_type=decay_type
        )
        return encoded['fact_features']
    

    def get_ticker_node_features(
        self,
        feature_dim: int,
        valid_tickers: set
    ) -> tuple[np.ndarray, dict[str, int]]:
        """
        Builds zero-initialised features for ticker nodes, filtered to valid tickers.

        Returns:
            - ticker_features: (N_tickers, feature_dim) array
            - ticker_index: dict mapping ticker symbols to row indices
        """
        facts = self.fact_list or []

        # Only tickers mentioned in facts *and* in the valid ticker universe
        used_tickers = sorted({t for f in facts for t in f.get("tickers", []) if t in valid_tickers})
        ticker_index = {ticker: i for i, ticker in enumerate(used_tickers)}

        ticker_features = np.zeros((len(ticker_index), feature_dim), dtype=np.float32)
        return ticker_features, ticker_index
    

    def get_edges(
        self,
        ticker_index: dict[str, int],
        edge_decay=None,
        decay_type: str = "exponential"
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Constructs edges from fact → ticker with [sentiment, decay_weight] as edge attributes.

        Returns:
            - edge_index: (2, N_edges) array of [fact_idx, ticker_idx]
            - edge_attr: (N_edges, 2) array of [sentiment, decay_weight]
        """
        facts = self.fact_list or []
        edge_src = []
        edge_dst = []
        edge_attrs = []

        # Select decay function
        if edge_decay is None:
            def compute_weight(d): return 1.0
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
                    edge_src.append(fact_idx)
                    edge_dst.append(ticker_index[ticker])
                    edge_attrs.append((sentiment, weight))

        edge_index = np.array([edge_src, edge_dst], dtype=np.int64)         # shape (2, N_edges)
        edge_attr = np.array(edge_attrs, dtype=np.float32)                  # shape (N_edges, 2)
        return edge_index, edge_attr



    def to_numpy_graph(
        self,
        text_encoder,
        id2centroid: dict[int, np.ndarray],
        edge_decay=None,
        decay_type: str = "exponential",
        fuse: str | None = "concat",
        l2_normalize: bool = True
    ) -> dict[str, np.ndarray | dict[str, int]]:
        """
        Builds a complete NumPy-format graph for this SubGraph, ready for conversion
        to a heterogeneous GNN input.

        Args:
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
        valid_tickers = self.load_valid_tickers()

        fact_features = self.get_fact_node_features(
            text_encoder=text_encoder,
            id2centroid=id2centroid,
            edge_decay=edge_decay,
            decay_type=decay_type,
            fuse=fuse,
            l2_normalize=l2_normalize
        )

        ticker_features, ticker_index = self.get_ticker_node_features(
            feature_dim=fact_features.shape[1],
            valid_tickers=valid_tickers
        )

        edge_index, edge_attr = self.get_edges(
            ticker_index=ticker_index,
            edge_decay=edge_decay,
            decay_type=decay_type
        )

        return {
            "fact_features": fact_features,
            "ticker_features": ticker_features,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "ticker_index": ticker_index
        }


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
    obj = load_first_with_exact_facts(config.SUBGRAPHS_JSONL, n=46)
    sg = SubGraph(**obj)
    sg.visualise_pricing()  # Visualise the pricing data for the subgraph

    # 2) resources
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    id2centroid = load_centroids_jsonl(config.CLUSTER_CENTROIDS)
    decay = EdgeDecay(decay_days=config.WINDOW_DAYS, final_weight=0.10)

    # 3) encode (choose fuse=None or "concat")
    out = sg.to_numpy_graph(
        text_encoder=encoder,
        id2centroid=id2centroid,
        edge_decay=decay,
        decay_type="exponential",
        fuse="concat",
        l2_normalize=True
    )

    # 4) visualise
    sg.visualise_numpy_graph(out, max_nodes=2000)

    
