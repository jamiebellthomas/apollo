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
import csv


def collect_event_types(file_path: str = config.NEWS_FACTS) -> List[str]:
    event_types = set()
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            fact = json.loads(line)
            etype = fact.get("event_type")
            if etype:
                event_types.add(etype.strip().lower())
    # event_types.discard('other')  # remove "other" (optional)
    event_types = list(event_types)
    print(f"Found {len(event_types)} unique event types.")
    return event_types


def get_embeddings(event_types: List[str]) -> np.ndarray:
    model = SentenceTransformer("all-MiniLM-L6-v2")  # or "yiyanghkust/finbert-embedding"
    embeddings = model.encode(event_types, convert_to_numpy=True, normalize_embeddings=False)
    return embeddings.astype("float32", copy=False)


def get_clusters(embeddings: np.ndarray, n_clusters: int):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    n_clusters = len(np.unique(labels))
    print(f"{n_clusters} clusters.")
    return labels, kmeans  # return both so we can save centroids


def reduce_dimensionality(embeddings: np.ndarray, perplexity: int) -> np.ndarray:
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    return embeddings_2d


def _get_discrete_cmap(name: str, n: int):
    try:
        return plt.colormaps.get_cmap(name).resampled(n)
    except (AttributeError, TypeError):
        base = plt.get_cmap(name, 256)
        return ListedColormap(base(np.linspace(0, 1, n)))


def plot_two_dim_embeddings(
    embeddings_2d: np.ndarray,
    labels: np.ndarray,
    event_types: List[str],
    show_centroids: bool = True,
    outfile: str = "Plots/cluster_results/clusters.png",
) -> Dict[str, int]:
    
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    cmap = _get_discrete_cmap("nipy_spectral", n_clusters)
    fig, ax = plt.subplots(figsize=(12, 10))

    ax.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=labels,
        cmap=cmap,
        s=50,
        alpha=0.9,
    )

    # for i, etype in enumerate(event_types):
    #     ax.annotate(
    #         f"({etype}, {labels[i]})",
    #         (embeddings_2d[i, 0], embeddings_2d[i, 1]),
    #         fontsize=8,
    #         alpha=0.7,
    #     )
    
    for i, etype in enumerate(event_types):
        ax.annotate(
            f"({labels[i]})",
            (embeddings_2d[i, 0], embeddings_2d[i, 1]),
            fontsize=8,
            alpha=0.7,
        )



    if show_centroids:
        for k in unique_labels:
            pts = embeddings_2d[labels == k]
            cx, cy = pts.mean(axis=0)
            ax.scatter(cx, cy, marker="x", s=200, linewidths=3, color=cmap(k), zorder=5)

    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=cmap(k), label=f"Cluster {k}", markersize=10)
        for k in unique_labels
    ]
    if n_clusters < 26:
        ax.legend(handles=handles, title="Clusters")
        
        
    ax.set_title("2-D representation of clusters using t-SNE")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.tight_layout()

    out_path = Path(outfile)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

    return {etype: int(lbl) for etype, lbl in zip(event_types, labels)}


# ---------------- NEW: save map + centroids JSONL (to config.CLUSTER_CENTROIDS) + names CSV ----------------
def save_cluster_artifacts(
    event_types: List[str],
    embeddings: np.ndarray,
    labels: np.ndarray,
    kmeans: KMeans,
    out_dir: str = "artifacts",
    model_name: str = "all-MiniLM-L6-v2",
) -> Dict[str, int]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) event_type -> cluster_id map (JSON)
    etype_to_cluster = {et: int(c) for et, c in zip(event_types, labels)}
    (out / "event_cluster_map.json").write_text(
        json.dumps(
            {
                "version": 1,
                "model": model_name,
                "n_clusters": int(kmeans.n_clusters),
                "event_type_map": etype_to_cluster,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[save] event_cluster_map.json")

    # 2) simple names from nearest types (cosine on L2-normalized vectors)
    centroids = kmeans.cluster_centers_.astype("float32", copy=False)
    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    cen_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)

    names = []
    rows = []
    for cid in range(kmeans.n_clusters):
        sims = emb_norm @ cen_norm[cid]
        top = np.argsort(-sims)[:5]
        name = "|".join(event_types[i] for i in top)
        names.append(name)
        rows.append({"cluster_id": cid, "cluster_name": name, "size": int((labels == cid).sum())})

    with open(out / "event_clusters.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["cluster_id", "cluster_name", "size"])
        w.writeheader()
        w.writerows(rows)
    print(f"[save] event_clusters.csv")

    # 3) Centroids JSONL to config.CLUSTER_CENTROIDS (one line per cluster)
    Path(os.path.dirname(config.CLUSTER_CENTROIDS)).mkdir(parents=True, exist_ok=True)
    with open(config.CLUSTER_CENTROIDS, "w", encoding="utf-8") as fout:
        for cid in range(kmeans.n_clusters):
            rec = {
                "cluster_id": cid,
                "cluster_name": names[cid],
                "centroid": centroids[cid].tolist(),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[save] {config.CLUSTER_CENTROIDS}")

    # 4) vocabulary snapshot
    (out / "event_types.txt").write_text("\n".join(event_types), encoding="utf-8")
    print(f"[save] event_types.txt")

    return etype_to_cluster


# ---------------- NEW: in-place annotation of facts with event_cluster_id ----------------
def annotate_facts_with_clusters_inplace(
    path: str,
    etype_to_cluster: Dict[str, int],
    drop_unmapped: bool = False,
):
    """
    Rewrites *path* in place, adding 'event_cluster_id' (int or null) per fact.
    """
    src = Path(path)
    tmp = src.with_suffix(src.suffix + ".tmp")

    n_read = n_written = n_unmapped = 0
    with open(src, "r", encoding="utf-8") as fin, open(tmp, "w", encoding="utf-8") as fout:
        for line in fin:
            n_read += 1
            obj = json.loads(line)
            et = (obj.get("event_type") or "").strip().lower()
            cid = None if et == "other" else etype_to_cluster.get(et)
            if cid is None and drop_unmapped:
                continue
            if cid is None:
                n_unmapped += 1
            obj["event_cluster_id"] = (None if cid is None else int(cid))
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_written += 1

    os.replace(tmp, src)  # atomic replace
    print(f"[annotate] in-place: read={n_read}, written={n_written}, unmapped={n_unmapped}, dropped={n_read-n_written}, file={src}")


def main():
    event_types = collect_event_types()
    embeddings = get_embeddings(event_types=event_types)
    embeddings_2d = reduce_dimensionality(embeddings=embeddings, perplexity=45)

    labels, kmeans = get_clusters(embeddings=embeddings, n_clusters=60)

    plot_two_dim_embeddings(
        embeddings_2d=embeddings_2d,
        labels=labels,
        event_types=event_types,
        show_centroids=False,
    )

    etype_to_cluster = save_cluster_artifacts(
        event_types=event_types,
        embeddings=embeddings,
        labels=labels,
        kmeans=kmeans,
        out_dir="Data/Clusters",
        model_name="all-MiniLM-L6-v2",
    )

    annotate_facts_with_clusters_inplace(
        path=config.NEWS_FACTS,                 # write back to the SAME file
        etype_to_cluster=etype_to_cluster,
        drop_unmapped=False,                    # set True to drop 'other'/unseen
    )


if __name__ == "__main__":
    main()
