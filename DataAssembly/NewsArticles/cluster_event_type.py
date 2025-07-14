import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import config
import json
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def collect_event_types(file_path:str = config.NEWS_FACTS) -> set:

    event_types = set()
    with open(file_path) as f:
        for line in f:
            fact = json.loads(line)
            etype = fact.get("event_type")
            if etype:
                event_types.add(etype.strip().lower())
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
    return labels

def reduce_dimensionality(embeddings:np.ndarray) -> np.ndarray:
    tsne = TSNE(n_components=2, perplexity=7, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    return embeddings_2d

def plot_two_dim_embeddings(embeddings_2d:np.ndarray, labels:np.ndarray, event_types:list) -> dict:

    # Build mapping
    cluster_map = {etype: int(label) for etype, label in zip(event_types, labels)}

    # Plot
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=labels,
        cmap="tab10",   # This specifies the colormap used to colour the points according to their c= (the cluster labels).
        s=50,           # This controls the size of each scatter point.
        alpha=0.9      # This sets the transparency (opacity) of each point 1=filled 0=opaque
    )

    # Annotate each point with its event_type text
    for i, txt in enumerate(event_types):
        plt.annotate(txt,
                     (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                     fontsize=8,
                     alpha=0.7)

    plt.title("2d representation of clusters using TSNE")
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")
    handles, _ = scatter.legend_elements(prop="colors")
    legend_labels = [f"Cluster {i}" for i in np.unique(labels)]
    plt.legend(handles, legend_labels, title="Clusters")
    plt.tight_layout()
    plt.show()

    return cluster_map


def main():
    event_types = collect_event_types()
    embeddings = get_embeddings(event_types = event_types)
    embeddings_2d = reduce_dimensionality(embeddings=embeddings)
    labels = get_clusters(embeddings=embeddings,
                          n_clusters=8)
    plot_two_dim_embeddings(embeddings_2d=embeddings_2d,
                            labels=labels,
                            event_types=event_types)


if __name__ == "__main__":
    main()