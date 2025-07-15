import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import config
import json
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.lines import Line2D
from typing import Dict, List
from pathlib import Path
from matplotlib.colors import ListedColormap



def collect_event_types(file_path:str = config.NEWS_FACTS) -> set:

    event_types = set()
    with open(file_path) as f:
        for line in f:
            fact = json.loads(line)
            etype = fact.get("event_type")
            if etype:
                event_types.add(etype.strip().lower())

    # Remove 'other' from the set if it exists
    event_types.discard('other')
    event_types = list(event_types)
    print(f"Found {len(event_types)} unique event types.")



    return event_types


def get_embeddings(event_types:set) -> np.ndarray:

    model = SentenceTransformer("all-MiniLM-L6-v2")  # Or "yiyanghkust/finbert-embedding"
    embeddings = model.encode(event_types)
    return embeddings

def get_clusters(embeddings:np.ndarray, n_clusters:int) -> np.ndarray:
    kmeans = KMeans(n_clusters=n_clusters, 
                    random_state=42)
    labels = kmeans.fit_predict(embeddings)
    n_clusters = len(np.unique(labels))
    print(f"{n_clusters} clusters.")
    return labels

def reduce_dimensionality(embeddings:np.ndarray, perplexity:int) -> np.ndarray:
    tsne = TSNE(n_components=2, 
                perplexity=perplexity, 
                random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    return embeddings_2d

# ----------------------------------------------------------------------
# Utility: always return an n-colour ListedColormap, regardless of
#          Matplotlib version (new API if available, fallback otherwise)
# ----------------------------------------------------------------------
def _get_discrete_cmap(name: str, n: int):
    try:
        # Modern route (mpl ≥ 3.5): resample an existing colormap
        return plt.colormaps.get_cmap(name).resampled(n)
    except (AttributeError, TypeError):
        # Legacy route (works back to mpl 3.0)
        base = plt.get_cmap(name, 256)          # 256-level continuous map
        return ListedColormap(base(np.linspace(0, 1, n)))


def plot_two_dim_embeddings(
    embeddings_2d: np.ndarray,
    labels: np.ndarray,
    event_types: List[str],
    show_centroids: bool = True,
    outfile: str = "Plots/cluster_results/clusters.png",
) -> Dict[str, int]:
    """
    Scatter-plot 2-D embeddings coloured by cluster, annotate each point as
    (label, cluster-number), optionally mark centroids, save to *outfile*,
    and display the figure.

    Returns
    -------
    dict
        Mapping {event_type: cluster_label}.
    """
    # ------------------------------------------------------------------
    # Prepare colours and figure
    # ------------------------------------------------------------------
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    cmap = _get_discrete_cmap("nipy_spectral", n_clusters)

    fig, ax = plt.subplots(figsize=(12, 10))

    # ------------------------------------------------------------------
    # Scatter plot
    # ------------------------------------------------------------------
    ax.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=labels,
        cmap=cmap,
        s=50,
        alpha=0.9,
    )

    # Annotate each point with "(label, cluster)"
    for i, etype in enumerate(event_types):
        ax.annotate(
            f"({etype}, {labels[i]})",
            (embeddings_2d[i, 0], embeddings_2d[i, 1]),
            fontsize=8,
            alpha=0.7,
        )

    # ------------------------------------------------------------------
    # Optional centroids
    # ------------------------------------------------------------------
    if show_centroids:
        for k in unique_labels:
            pts = embeddings_2d[labels == k]
            cx, cy = pts.mean(axis=0)
            ax.scatter(
                cx,
                cy,
                marker="x",
                s=200,
                linewidths=3,
                color=cmap(k),
                zorder=5,
            )

    # ------------------------------------------------------------------
    # Legend – one handle per cluster
    # ------------------------------------------------------------------
    handles = [
        Line2D([0], [0], marker="o", linestyle="",
               color=cmap(k), label=f"Cluster {k}", markersize=10)
        for k in unique_labels
    ]
    ax.legend(handles=handles, title="Clusters")

    # ------------------------------------------------------------------
    # Final styling
    # ------------------------------------------------------------------
    ax.set_title("2-D representation of clusters using t-SNE")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.tight_layout()

    # ------------------------------------------------------------------
    # Save and show
    # ------------------------------------------------------------------
    out_path = Path(outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    # ------------------------------------------------------------------
    # Return mapping
    # ------------------------------------------------------------------
    return {etype: int(lbl) for etype, lbl in zip(event_types, labels)}


def main():
    event_types = collect_event_types()
    embeddings = get_embeddings(event_types = event_types)
    embeddings_2d = reduce_dimensionality(embeddings=embeddings,
                                          perplexity=20)
    labels = get_clusters(embeddings=embeddings,
                          n_clusters=13)
    plot_two_dim_embeddings(embeddings_2d=embeddings_2d,
                            labels=labels,
                            event_types=event_types,
                            show_centroids=False)


if __name__ == "__main__":
    main()