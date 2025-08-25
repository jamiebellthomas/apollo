#!/usr/bin/env python3
"""
Comprehensive Diagnostics for HeteroGNN5 Model Performance
Expanded and hardened:
- Per-cluster/company calibration (ECE/MCE/Brier) + threshold suggestions
- Attention concentration (entropy, gini, top-k coverage), head diversity
- Robust cluster assignment via attention-summed contribution
- Stronger event-type interaction/conflict analysis
- Temporal drift by year/quarter with correct confusion decomposition
"""

import sys
import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import torch
from torch_geometric.loader import DataLoader  # FIX: correct import
import torch_geometric.nn as tgnn
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, average_precision_score, f1_score
)

from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------------------------------
# Local imports
# ------------------------------------------------------------------------------------
sys.path.append('..')
sys.path.append('../KG')
from HeteroGNN5 import HeteroAttnGNN


# ------------------------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------------------------
def parse_hyperparams_txt(path: Path) -> dict:
    hp = {}
    if not path.exists():
        return hp
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip(); v = v.strip()
        if v.lower() in ("true", "false"):
            hp[k] = (v.lower() == "true")
        elif v.lower() == "none":
            hp[k] = None
        else:
            try:
                if any(s in v.lower() for s in [".", "e-"]):
                    hp[k] = float(v)
                else:
                    hp[k] = int(v)
            except ValueError:
                hp[k] = v
    return hp


def split_list(dataset, train_ratio=0.8, val_ratio=0.2, seed=76369):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(dataset), generator=g).tolist()
    n = len(dataset)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = [dataset[i] for i in perm[:n_train]]
    val   = [dataset[i] for i in perm[n_train:n_train+n_val]]
    test  = [dataset[i] for i in perm[n_train+n_val:]]
    return train, val, test


def attach_y_and_meta(dataset, subgraphs):
    for g, sg in zip(dataset, subgraphs):
        g.y = g["graph_label"].float().view(-1)
        g.meta_primary_ticker = getattr(sg, "primary_ticker", None)


def apply_ticker_scaler_to_graphs(graphs, scaler):
    if scaler.get("identity", False):
        return
    for g in graphs:
        if 'company' in g.node_types and hasattr(g['company'], 'x'):
            x = g['company'].x
            if isinstance(x, torch.Tensor) and x.numel() > 0:
                x_w = torch.minimum(torch.maximum(x, scaler["low"]), scaler["high"])
                g['company'].x = (x_w - scaler["mean"]) / scaler["std"]


def fit_ticker_scaler(train_graphs, pct_low=1.0, pct_high=99.0):
    rows = []
    for g in train_graphs:
        if 'company' in g.node_types and hasattr(g['company'], 'x'):
            x = g['company'].x
            if isinstance(x, torch.Tensor) and x.numel() > 0:
                rows.append(x.detach().cpu())
    if not rows:
        print("[scaler] No company features; using identity.")
        return {"low": None, "high": None, "mean": None, "std": None, "identity": True}

    X = torch.cat(rows, dim=0)
    lo = torch.quantile(X, q=pct_low / 100.0, dim=0)
    hi = torch.quantile(X, q=pct_high / 100.0, dim=0)
    Xw = torch.minimum(torch.maximum(X, lo), hi)
    mean = Xw.mean(dim=0)
    std = Xw.std(dim=0, unbiased=False)
    std[std == 0] = 1.0
    print(f"[scaler] Fit on {X.shape[0]} rows, D={X.shape[1]} (winsorize p{pct_low}/{pct_high})")
    return {"low": lo.float(), "high": hi.float(), "mean": mean.float(), "std": std.float(), "identity": False}


def load_cached_data():
    print("Loading cached data...")
    with open('../KG/dataset_cache/training_cached_dataset_nf35_limall.pkl', 'rb') as f:
        train_cache = pickle.load(f)
    with open('../KG/dataset_cache/testing_cached_dataset_nf35_limall.pkl', 'rb') as f:
        test_cache = pickle.load(f)

    train_graphs = train_cache['graphs']
    test_graphs = test_cache['graphs']
    train_raw_sg = train_cache['raw_sg']
    test_raw_sg = test_cache['raw_sg']

    print(f"Loaded {len(train_graphs)} training graphs, {len(test_graphs)} test graphs")
    attach_y_and_meta(train_graphs, train_raw_sg)
    attach_y_and_meta(test_graphs, test_raw_sg)
    return train_graphs, test_graphs, train_raw_sg, test_raw_sg


# ------------------------------------------------------------------------------------
# Attention capture
# ------------------------------------------------------------------------------------
class AttentionExtractor:
    """
    Replaces model._attn_layer to capture attention weights without breaking forward.
    Stores per-layer dict[(src,rel,dst)] -> {alpha[E,H], edge_index[2,E], src_type, dst_type}
    """

    def __init__(self, model):
        self.model = model
        self.attention_weights = []
        self._orig = None

    def register(self):
        self._orig = self.model._attn_layer

        def hook(convs, x_dict, edge_index_dict, edge_attr_dict, collect_entropy_on=None):
            out = {nt: [] for nt in x_dict.keys()}
            entropy_term = x_dict[next(iter(x_dict))].new_tensor(0.0)
            layer_dump = {}

            for et, ei in edge_index_dict.items():
                conv = convs[str(et)]
                ea = edge_attr_dict[et]
                src_type, _, dst_type = et
                x_src, x_dst = x_dict[src_type], x_dict[dst_type]

                # keep computation graph for alpha (no detach here)
                y, (edge_index_used, alpha) = conv(
                    (x_src, x_dst),
                    ei,
                    edge_attr=ea,
                    return_attention_weights=True
                )
                out[dst_type].append(y)

                layer_dump[et] = {
                    'alpha': alpha,  # keep on device for analysis; convert later
                    'edge_index': edge_index_used,
                    'src_type': src_type,
                    'dst_type': dst_type
                }

                if self.model.entropy_reg_weight > 0.0 and collect_entropy_on is not None and et == collect_entropy_on:
                    dst = edge_index_used[1]
                    eps = 1e-9
                    h_sum = 0.0
                    for h in range(alpha.size(1)):
                        a = alpha[:, h].clamp(min=eps)
                        neg_a_log_a = -(a * torch.log(a))
                        h_node = tgnn.global_add_pool(neg_a_log_a, dst)
                        h_sum = h_sum + h_node.mean()
                    entropy_term = entropy_term + (h_sum / alpha.size(1))

            x_new = {}
            for nt, parts in out.items():
                x_new[nt] = torch.stack(parts, dim=0).sum(dim=0) if parts else x_dict[nt]

            self.model.extra_loss = self.model.entropy_reg_weight * entropy_term
            self.attention_weights.append(layer_dump)
            return x_new

        self.model._attn_layer = hook
        print("Registered attention capture hook")

    def remove(self):
        if self._orig is not None:
            self.model._attn_layer = self._orig
            self._orig = None
            print("Removed attention capture hook")

    def clear(self):
        self.attention_weights = []


# ------------------------------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------------------------------
class ComprehensiveDiagnostics:
    def __init__(self, model_dir, results_dir="../Results/heterognn5"):
        self.model_dir = Path(model_dir)
        self.results_dir = Path(results_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Data
        self.train_graphs, self.test_graphs, self.train_raw_sg, self.test_raw_sg = load_cached_data()
        self.fact_mapping = self.load_fact_mapping()

        # Clusters + encoder (cache embeddings for speed)
        self.centroids, self.cluster_ids = self.load_cluster_centroids()
        cache_dir = Path("../KG/model_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.sentence_model = SentenceTransformer('all-mpnet-base-v2', cache_folder=str(cache_dir))
        self._event_embed_cache = {}

        # Model + predictions + attention capture
        self.model, self.hp = self.load_model()
        self.attention_results = self.extract_attention_weights()

        # Categorize
        self.categorize_predictions()

    # ---------- data helpers ----------
    def load_fact_mapping(self):
        fact_mapping = {}
        subgraphs_path = "../Data/subgraphs.jsonl"
        if not os.path.exists(subgraphs_path):
            print("[warn] subgraphs.jsonl not found")
            return fact_mapping
        with open(subgraphs_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                for fact in data.get('fact_list', []):
                    fid = fact.get('fact_id')
                    if fid is None:
                        continue
                    fact_mapping[fid] = {
                        'event_type': fact.get('event_type', ''),
                        'raw_text': fact.get('raw_text', ''),
                        'primary_ticker': data.get('primary_ticker', ''),
                        'reported_date': data.get('reported_date', ''),
                        'sentiment': fact.get('sentiment', 0.0)
                    }
        print(f"Loaded {len(fact_mapping)} facts")
        return fact_mapping

    def load_cluster_centroids(self):
        centroids, cluster_ids = [], []
        with open("../Data/cluster_centroids.jsonl", 'r') as f:
            for line in f:
                data = json.loads(line)
                centroids.append(np.array(data['centroid']))
                cluster_ids.append(data['cluster_id'])
        return np.array(centroids), cluster_ids

    # ---------- model ----------
    def load_model(self):
        hp = parse_hyperparams_txt(self.model_dir / "hyperparameters.txt")
        train_set, val_set, _ = split_list(
            self.train_graphs,
            train_ratio=float(hp.get("train_ratio",0.8)),
            val_ratio=float(hp.get("val_ratio",0.2)),
            seed=int(hp.get("seed",76369))
        )
        scaler = fit_ticker_scaler(train_set)
        apply_ticker_scaler_to_graphs(train_set, scaler)
        apply_ticker_scaler_to_graphs(val_set, scaler)
        apply_ticker_scaler_to_graphs(self.test_graphs, scaler)

        sample_graph = train_set[0]
        metadata = (list(sample_graph.node_types), list(sample_graph.edge_types))

        model = HeteroAttnGNN(
            metadata=metadata,
            hidden_channels=int(hp.get("hidden_channels",1024)),
            num_layers=int(hp.get("num_layers",4)),
            heads=int(hp.get("heads",8)),
            time_dim=int(hp.get("time_dim",16)),
            feature_dropout=float(hp.get("feature_dropout",0.2)),
            edge_dropout=float(hp.get("edge_dropout",0.05)),
            final_dropout=float(hp.get("final_dropout",0.1)),
            readout=hp.get("readout","company"),
            funnel_to_primary=bool(hp.get("funnel_to_primary",False)),
            topk_per_primary=hp.get("topk_per_primary",15),
            attn_temperature=float(hp.get("attn_temperature",0.8)),
            entropy_reg_weight=float(hp.get("entropy_reg_weight",0.01)),
            time_bucket_edges=[0, 7, 30, 90, 9999] if hp.get("time_bucket_edges") else None,
            time_bucket_emb_dim=int(hp.get("time_bucket_emb_dim",8)),
            add_abs_sent=bool(hp.get("add_abs_sent",True)),
            add_polarity_bit=bool(hp.get("add_polarity_bit",True)),
            sentiment_jitter_std=float(hp.get("sentiment_jitter_std",0.1)),
            delta_t_jitter_frac=float(hp.get("delta_t_jitter_frac",0.05)),
        )

        init_loader = DataLoader([train_set[0]], batch_size=1, shuffle=False)
        sample_batch = next(iter(init_loader)).to(self.device)
        model.to(self.device)
        _ = model(sample_batch)

        state_dict = torch.load(self.model_dir / "model.pt", map_location=self.device)
        model.load_state_dict(state_dict, strict=False)
        return model, hp

    # ---------- attention + predictions ----------
    def extract_attention_weights(self, threshold=0.74):
        print("\n" + "="*60)
        print("EXTRACTING ATTENTION WEIGHTS")
        print("="*60)

        extractor = AttentionExtractor(self.model)
        extractor.register()

        batch_size = int(self.hp.get("batch_size", 32))
        test_loader = DataLoader(self.test_graphs, batch_size=2 * batch_size, shuffle=False)

        self.model.eval()
        results = []

        with torch.no_grad():
            for b_idx, batch in enumerate(test_loader):
                batch = batch.to(self.device)
                extractor.clear()
                logits = self.model(batch)
                probs = torch.sigmoid(logits)
                preds = (probs >= threshold).long()

                # sequential mapping is safe with shuffle=False
                bs = len(batch.y)
                start_idx = b_idx * test_loader.batch_size
                end_idx = min(start_idx + bs, len(self.test_raw_sg))
                batch_subgraphs = self.test_raw_sg[start_idx:end_idx]

                # copy attention weights out (to CPU) so we can free graph later
                layer_weights = []
                for layer in extractor.attention_weights:
                    lw = {}
                    for k, v in layer.items():
                        lw[k] = {
                            'alpha': v['alpha'].detach().cpu(),
                            'edge_index': v['edge_index'].detach().cpu(),
                            'src_type': v['src_type'],
                            'dst_type': v['dst_type'],
                        }
                    layer_weights.append(lw)

                for i in range(bs):
                    sg = batch_subgraphs[i] if i < len(batch_subgraphs) else None
                    # Map per-sample fact index to fact_id if available
                    idx_to_id = {}
                    if sg and getattr(sg, 'fact_list', None):
                        for fidx, fact in enumerate(sg.fact_list):
                            fid = fact.get('fact_id')
                            if fid is not None:
                                idx_to_id[fidx] = fid

                    results.append({
                        'batch_idx': b_idx,
                        'sample_idx': i,
                        'probability': probs[i].item(),
                        'predicted_label': int(preds[i].item()),
                        'actual_label': int(batch.y[i].item()),
                        'attention_weights': layer_weights,  # list per-layer dicts
                        'fact_index_to_id': idx_to_id,
                        'subgraph': sg
                    })

                print(f"Processed batch {b_idx+1}/{len(test_loader)}")

        extractor.remove()
        return results

    def categorize_predictions(self):
        r = self.attention_results
        self.tp_results = [x for x in r if x['predicted_label']==1 and x['actual_label']==1]
        self.fp_results = [x for x in r if x['predicted_label']==1 and x['actual_label']==0]
        self.tn_results = [x for x in r if x['predicted_label']==0 and x['actual_label']==0]
        self.fn_results = [x for x in r if x['predicted_label']==0 and x['actual_label']==1]
        print(f"TP={len(self.tp_results)} FP={len(self.fp_results)} TN={len(self.tn_results)} FN={len(self.fn_results)}")

    # ---------- attention helpers ----------
    def _dominant_cluster_for_sample(self, pred_result, top_k=10):
        """
        Assign a sample to the 'dominant cluster' by summing attention weights of top facts
        mapped to clusters; returns (cluster_id -> summed attention), dominant cluster id.
        """
        contrib = defaultdict(float)
        for fact_id, att in self.get_top_facts_for_prediction(pred_result, top_k=top_k):
            ft = self.fact_mapping.get(fact_id, {}).get('event_type', '')
            cid = self.get_cluster_for_event_type(ft)
            if cid is not None:
                contrib[cid] += float(att)
        if not contrib:
            return {}, None
        dom = max(contrib.items(), key=lambda kv: kv[1])[0]
        return contrib, dom

    def get_top_facts_for_prediction(self, pred_result, top_k=15):
        if not pred_result.get('attention_weights'):
            return []
        idx_to_id = pred_result['fact_index_to_id']
        fact_attn = defaultdict(float)

        for layer in pred_result['attention_weights']:
            key = ('fact','mentions','company')
            if key in layer:
                alpha = layer[key]['alpha']          # [E, H]
                edge_index = layer[key]['edge_index']# [2, E]
                avg = alpha.mean(dim=1)              # [E]
                src = edge_index[0]                  # fact indices
                for e in range(src.numel()):
                    fidx = int(src[e].item())
                    fid = idx_to_id.get(fidx, fidx)
                    fact_attn[fid] += float(avg[e].item())

        return sorted(fact_attn.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # ---------- cluster mapping ----------
    def get_cluster_for_event_type(self, event_type):
        if not event_type:
            return None
        if event_type in self._event_embed_cache:
            emb = self._event_embed_cache[event_type]
        else:
            emb = self.sentence_model.encode([event_type])[0]
            self._event_embed_cache[event_type] = emb
        dists = np.linalg.norm(self.centroids - emb, axis=1)
        idx = int(np.argmin(dists))
        return self.cluster_ids[idx]

    # ---------- metrics helpers ----------
    @staticmethod
    def _ece_mce_brier(probs, labels, n_bins=15):
        probs = np.asarray(probs)
        labels = np.asarray(labels)
        bins = np.linspace(0, 1, n_bins+1)
        ece = 0.0; mce = 0.0
        brier = np.mean((probs - labels)**2)
        for i in range(n_bins):
            lo, hi = bins[i], bins[i+1]
            mask = (probs >= lo) & (probs < hi) if i < n_bins-1 else (probs >= lo) & (probs <= hi)
            if mask.any():
                conf = probs[mask].mean()
                acc = labels[mask].mean()
                gap = abs(conf - acc)
                ece += gap * mask.mean()
                mce = max(mce, gap)
        return ece, mce, brier

    @staticmethod
    def _gini(x):
        x = np.asarray(x, dtype=np.float64)
        x = x[x>=0]
        if x.size == 0:
            return 0.0
        s = x.sum()
        if s == 0:
            return 0.0
        x_sorted = np.sort(x)
        n = x.size
        cum = np.cumsum(x_sorted)
        g = 1 + (1/n) - (2/(n*s)) * np.sum(cum)
        return float(g)

    # --------------------------------------------------------------------------------
    # 1) Cluster-level diagnostics (ratios + sentiment variance) – as before
    # --------------------------------------------------------------------------------
    def run_cluster_level_diagnostics(self):
        print("\n=== CLUSTER-LEVEL DIAGNOSTICS ===")
        cluster_data = {}
        for name, results in [('TP', self.tp_results), ('FP', self.fp_results),
                              ('TN', self.tn_results), ('FN', self.fn_results)]:
            counts = defaultdict(int)
            sents = defaultdict(list)
            for r in results:
                top_facts = self.get_top_facts_for_prediction(r)
                for fid, _ in top_facts:
                    fi = self.fact_mapping.get(fid)
                    if not fi: 
                        continue
                    cid = self.get_cluster_for_event_type(fi['event_type'])
                    if cid is None: 
                        continue
                    counts[cid] += 1
                    sents[cid].append(fi['sentiment'])
            cluster_data[name] = {
                'counts': dict(counts),
                'sentiments': {k: np.asarray(v) for k,v in sents.items()}
            }

        all_cids = set(cluster_data['TP']['counts']) | set(cluster_data['FN']['counts'])
        ratios = []
        for cid in all_cids:
            tp = cluster_data['TP']['counts'].get(cid, 0)
            fn = cluster_data['FN']['counts'].get(cid, 0)
            tot = tp + fn
            if tot > 0:
                ratios.append({'cluster_id': cid, 'tp_count': tp, 'fn_count': fn,
                               'ratio': tp/tot, 'total': tot})
        ratios.sort(key=lambda x: x['ratio'])

        # sentiment variance
        sent_var = []
        for cid in all_cids:
            tp_s = cluster_data['TP']['sentiments'].get(cid, np.array([]))
            fn_s = cluster_data['FN']['sentiments'].get(cid, np.array([]))
            all_s = np.concatenate([tp_s, fn_s]) if tp_s.size or fn_s.size else np.array([])
            var = np.var(all_s) if all_s.size > 1 else 0.0
            sent_var.append({
                'cluster_id': cid, 'variance': var,
                'tp_mean': float(tp_s.mean()) if tp_s.size else 0.0,
                'fn_mean': float(fn_s.mean()) if fn_s.size else 0.0,
                'tp_n': int(tp_s.size), 'fn_n': int(fn_s.size)
            })
        sent_var.sort(key=lambda x: x['variance'], reverse=True)
        print("Worst clusters (by TP/(TP+FN)):")
        for i, c in enumerate(ratios[:10]):
            print(f"{i+1}. Cluster {c['cluster_id']}: ratio={c['ratio']:.3f} (TP={c['tp_count']}, FN={c['fn_count']})")
        return cluster_data, ratios, sent_var

    # --------------------------------------------------------------------------------
    # 2) Fact-level (unchanged core, prints top FN facts)
    # --------------------------------------------------------------------------------
    def run_fact_level_analysis(self):
        print("\n=== FACT-LEVEL INFLUENCE ANALYSIS ===")
        fn_high = []
        for r in self.fn_results:
            for fid, att in self.get_top_facts_for_prediction(r, top_k=5):
                fi = self.fact_mapping.get(fid)
                if not fi:
                    continue
                fn_high.append({
                    'fact_id': fid,
                    'attention_weight': att,
                    'event_type': fi['event_type'],
                    'sentiment': fi['sentiment'],
                    'raw_text': (fi['raw_text'][:100]+'...') if len(fi['raw_text'])>100 else fi['raw_text'],
                    'primary_ticker': getattr(r.get('subgraph', {}), 'primary_ticker','Unknown'),
                    'probability': r['probability']
                })
        print(f"High-attention FN facts: {len(fn_high)}")
        for i, f in enumerate(fn_high[:10]):
            print(f"{i+1}. Fact {f['fact_id']} att={f['attention_weight']:.3f} evt={f['event_type']} sent={f['sentiment']:.3f}")
        return fn_high

    # --------------------------------------------------------------------------------
    # 3) Company-level errors (unchanged core; adds net error rates)
    # --------------------------------------------------------------------------------
    def run_company_level_analysis(self):
        print("\n=== COMPANY-LEVEL ERROR ANALYSIS ===")
        stats = defaultdict(lambda: {'tp':0,'fp':0,'tn':0,'fn':0,'total':0})
        for r in self.attention_results:
            ticker = getattr(r.get('subgraph', {}),'primary_ticker','Unknown')
            if r['predicted_label']==1 and r['actual_label']==1: stats[ticker]['tp']+=1
            elif r['predicted_label']==1 and r['actual_label']==0: stats[ticker]['fp']+=1
            elif r['predicted_label']==0 and r['actual_label']==0: stats[ticker]['tn']+=1
            elif r['predicted_label']==0 and r['actual_label']==1: stats[ticker]['fn']+=1
            stats[ticker]['total']+=1
        rows=[]
        for tk,s in stats.items():
            if s['total']>0:
                rows.append({
                    'ticker': tk,
                    'total': s['total'],
                    'fn_rate': s['fn']/s['total'],
                    'fp_rate': s['fp']/s['total'],
                    'total_error_rate': (s['fn']+s['fp'])/s['total'],
                    **s
                })
        rows.sort(key=lambda x: x['total_error_rate'], reverse=True)
        for i,c in enumerate(rows[:15]):
            print(f"{i+1}. {c['ticker']}: err={c['total_error_rate']:.3f} (FN={c['fn_rate']:.3f}, FP={c['fp_rate']:.3f}) [N={c['total']}]")
        return rows

    # --------------------------------------------------------------------------------
    # 4) Attention flow/concentration stats (entropy + gini + top-k coverage + head var)
    # --------------------------------------------------------------------------------
    def run_attention_flow_analysis(self):
        print("\n=== ATTENTION FLOW ANALYSIS ===")

        def sample_stats(r):
            entropies=[]; ginis=[]; head_vars=[]
            topk_cov=[]
            for layer in r.get('attention_weights', []):
                key = ('fact','mentions','company')
                if key not in layer: continue
                a = layer[key]['alpha'].numpy()     # [E,H]
                # normalize per-head over edges
                eps=1e-12
                a = a + eps
                a = a / a.sum(axis=0, keepdims=True)
                # entropy per head
                ent = (-a*np.log(a)).sum(axis=0)    # [H]
                entropies.extend(ent.tolist())
                # gini over edges on averaged head
                avg = a.mean(axis=1)                # [E]
                ginis.append(self._gini(avg))
                # top-k coverage (how much mass in top 5 edges)
                k = min(5, avg.size) if avg.size else 0
                if k>0:
                    cov = np.sort(avg)[-k:].sum()
                    topk_cov.append(cov)
                # head diversity (variance across heads per edge, then mean)
                if a.shape[1]>1:
                    head_vars.append(np.mean(np.var(a, axis=1)))
            return {
                'entropy': np.mean(entropies) if entropies else 0.0,
                'gini': np.mean(ginis) if ginis else 0.0,
                'topk_cov': np.mean(topk_cov) if topk_cov else 0.0,
                'head_var': np.mean(head_vars) if head_vars else 0.0
            }

        def collect(rs):
            s={'entropy':[],'gini':[],'topk_cov':[],'head_var':[]}
            for r in rs:
                st = sample_stats(r)
                for k in s: s[k].append(st[k])
            return {k:(np.mean(v) if v else 0.0, np.std(v) if v else 0.0) for k,v in s.items()}

        tp = collect(self.tp_results)
        fn = collect(self.fn_results)

        print(f"TP entropy mean±std: {tp['entropy'][0]:.3f}±{tp['entropy'][1]:.3f}")
        print(f"FN entropy mean±std: {fn['entropy'][0]:.3f}±{fn['entropy'][1]:.3f}")
        print(f"TP gini: {tp['gini'][0]:.3f} | FN gini: {fn['gini'][0]:.3f}")
        print(f"TP top5 coverage: {tp['topk_cov'][0]:.3f} | FN top5 coverage: {fn['topk_cov'][0]:.3f}")
        print(f"TP head var: {tp['head_var'][0]:.3f} | FN head var: {fn['head_var'][0]:.3f}")
        return tp, fn

    # --------------------------------------------------------------------------------
    # 5) Calibration + global optimal threshold
    # --------------------------------------------------------------------------------
    def run_calibration_analysis(self, current_threshold=0.74):
        print("\n=== CALIBRATION ANALYSIS ===")
        tp_probs = [r['probability'] for r in self.tp_results]
        fn_probs = [r['probability'] for r in self.fn_results]
        fp_probs = [r['probability'] for r in self.fp_results]
        tn_probs = [r['probability'] for r in self.tn_results]

        def stats(arr): 
            return (np.mean(arr) if arr else float('nan'),
                    np.std(arr) if arr else float('nan'))
        print(f"TP prob mean±std: {stats(tp_probs)[0]:.3f}±{stats(tp_probs)[1]:.3f}")
        print(f"FN prob mean±std: {stats(fn_probs)[0]:.3f}±{stats(fn_probs)[1]:.3f}")
        print(f"FP prob mean±std: {stats(fp_probs)[0]:.3f}±{stats(fp_probs)[1]:.3f}")
        print(f"TN prob mean±std: {stats(tn_probs)[0]:.3f}±{stats(tn_probs)[1]:.3f}")

        all_probs = tp_probs + fn_probs + fp_probs + tn_probs
        all_labels = [1]*len(tp_probs) + [1]*len(fn_probs) + [0]*len(fp_probs) + [0]*len(tn_probs)

        thresholds = np.arange(0.1, 0.95, 0.01)
        f1s=[]
        for th in thresholds:
            preds = [1 if p>=th else 0 for p in all_probs]
            f1s.append(f1_score(all_labels, preds))
        opt = float(thresholds[int(np.argmax(f1s))])
        close = sum(p >= current_threshold-0.1 for p in fn_probs)
        print(f"FNs near threshold (≥{current_threshold-0.1:.2f}): {close}/{len(fn_probs)} ({(100*close/max(1,len(fn_probs))):.1f}%)")
        print(f"Optimal global threshold (F1): {opt:.3f}")
        try:
            auc = roc_auc_score(all_labels, all_probs)
            ap = average_precision_score(all_labels, all_probs)
            print(f"AUC: {auc:.3f} | PR-AUC: {ap:.3f}")
        except Exception:
            pass

        ece,mce,brier = self._ece_mce_brier(all_probs, all_labels, n_bins=15)
        print(f"ECE={ece:.3f} | MCE={mce:.3f} | Brier={brier:.3f}")
        return (tp_probs, fn_probs, fp_probs, tn_probs, opt, (ece,mce,brier))

    # --------------------------------------------------------------------------------
    # 6) Event-type interaction analysis (stronger)
    # --------------------------------------------------------------------------------
    def run_event_type_interaction_analysis(self):
        print("\n=== EVENT-TYPE INTERACTION ANALYSIS ===")

        def combos(results, top_k=5):
            c = defaultdict(int)
            for r in results:
                evts=[]
                for fid,_ in self.get_top_facts_for_prediction(r, top_k=top_k):
                    e = self.fact_mapping.get(fid,{}).get('event_type')
                    if e: evts.append(e)
                evts = sorted(set(evts))
                from itertools import combinations
                for k in [2,3]:
                    for combo in combinations(evts, min(k, len(evts))):
                        c[combo]+=1
            return c

        tp_c = combos(self.tp_results)
        fn_c = combos(self.fn_results)

        tp_only = set(tp_c) - set(fn_c)
        fn_only = set(fn_c) - set(tp_c)

        print(f"TP-only combos: {len(tp_only)} | FN-only combos: {len(fn_only)}")
        tp_only_sorted = sorted([(k,tp_c[k]) for k in tp_only], key=lambda x: x[1], reverse=True)
        fn_only_sorted = sorted([(k,fn_c[k]) for k in fn_only], key=lambda x: x[1], reverse=True)
        return tp_c, fn_c, tp_only_sorted[:10], fn_only_sorted[:10]

    # --------------------------------------------------------------------------------
    # 7) Temporal drift (by period) – fixed FN logic
    # --------------------------------------------------------------------------------
    def run_temporal_drift_analysis(self):
        print("\n=== TEMPORAL DRIFT ANALYSIS ===")
        def period_of(date_str):
            try:
                dt = datetime.strptime(date_str, '%Y-%m-%d')
                if dt.year <= 2018: return '2015-2018'
                if dt.year <= 2022: return '2019-2022'
                return '2023-2025'
            except: return None

        def group(results):
            g=defaultdict(list)
            for r in results:
                d = getattr(r.get('subgraph',{}), 'reported_date', '')
                p = period_of(d)
                if p: g[p].append(r)
            return g

        tp_by = group(self.tp_results)
        fn_by = group(self.fn_results)
        for p in ['2015-2018','2019-2022','2023-2025']:
            tp=len(tp_by.get(p,[])); fn=len(fn_by.get(p,[]))
            tot=tp+fn
            if tot>0:
                print(f"{p}: TP={tp} FN={fn} FN_rate={fn/tot:.3f}")
        return tp_by, fn_by

    # --------------------------------------------------------------------------------
    # 8) Case studies (unchanged printing)
    # --------------------------------------------------------------------------------
    def run_case_study_examples(self):
        print("\n=== CASE STUDY EXAMPLES ===")
        items=[]
        for r in self.fn_results:
            tf = self.get_top_facts_for_prediction(r, top_k=3)
            if tf: items.append((r, tf))
        items.sort(key=lambda x: x[1][0][1] if x[1] else 0, reverse=True)
        for i,(r,tf) in enumerate(items[:3]):
            tk = getattr(r.get('subgraph',{}),'primary_ticker','Unknown')
            dt = getattr(r.get('subgraph',{}),'reported_date','Unknown')
            print(f"\nExample {i+1} | {tk} @ {dt} | prob={r['probability']:.3f}")
            for j,(fid,att) in enumerate(tf):
                fi = self.fact_mapping.get(fid,{})
                print(f"  {j+1}. fact {fid} att={att:.3f} evt={fi.get('event_type')} sent={fi.get('sentiment')}")
        return items

    # --------------------------------------------------------------------------------
    # 9) Cluster misclassification deep-dive (uses dominant cluster)
    # --------------------------------------------------------------------------------
    def run_cluster_misclassification_analysis(self, focus_clusters=(40,51,55)):
        print("\n=== CLUSTER-SPECIFIC MISCLASSIFICATION ANALYSIS ===")
        out={}
        for cid in focus_clusters:
            tps=[]; fns=[]
            for r in self.tp_results:
                _, dom = self._dominant_cluster_for_sample(r)
                if dom==cid:
                    facts=[(fid,att,self.fact_mapping.get(fid,{})) for fid,att in self.get_top_facts_for_prediction(r)]
                    if facts: tps.append((r,facts))
            for r in self.fn_results:
                _, dom = self._dominant_cluster_for_sample(r)
                if dom==cid:
                    facts=[(fid,att,self.fact_mapping.get(fid,{})) for fid,att in self.get_top_facts_for_prediction(r)]
                    if facts: fns.append((r,facts))
            out[cid]={'tp_examples':tps[:3],'fn_examples':fns[:3]}
            print(f"Cluster {cid}: TP_ex={len(tps)} FN_ex={len(fns)}")
        return out

    # --------------------------------------------------------------------------------
    # 10) Sentiment conditioning
    # --------------------------------------------------------------------------------
    def get_avg_sentiment(self, result):
        vals=[]
        for fid,_ in self.get_top_facts_for_prediction(result):
            fi=self.fact_mapping.get(fid)
            if fi: vals.append(fi['sentiment'])
        return float(np.mean(vals)) if vals else 0.0

    def run_sentiment_conditioning_analysis(self):
        print("\n=== SENTIMENT CONDITIONING ANALYSIS ===")
        neg_fns=[{'prob':r['probability'],'avg_sent':self.get_avg_sentiment(r)}
                 for r in self.fn_results if self.get_avg_sentiment(r)<0]
        pos_fps=[{'prob':r['probability'],'avg_sent':self.get_avg_sentiment(r)}
                 for r in self.fp_results if self.get_avg_sentiment(r)>0]
        print(f"FNs with avg negative sentiment: {len(neg_fns)}")
        print(f"FPs with avg positive sentiment: {len(pos_fps)}")

        thresholds=[-0.5,-0.3,0.0,0.3,0.5]
        summary={}
        for t in thresholds:
            nf=sum(self.get_avg_sentiment(r)<t for r in self.fn_results)
            pf=sum(self.get_avg_sentiment(r)>t for r in self.fp_results)
            summary[t]={'negative_fns_rate': nf/max(1,len(self.fn_results)),
                        'positive_fps_rate': pf/max(1,len(self.fp_results))}
        return {'negative_sentiment_fns':neg_fns,
                'positive_sentiment_fps':pos_fps,
                'sentiment_threshold_analysis':summary}

    # --------------------------------------------------------------------------------
    # 11) Per-cluster threshold sweep (dominant cluster via attention)
    # --------------------------------------------------------------------------------
    def run_cluster_threshold_analysis(self):
        print("\n=== CLUSTER THRESHOLD ANALYSIS ===")
        bucket = defaultdict(lambda: {'probs':[], 'labels':[]})
        for r in self.attention_results:
            contrib, dom = self._dominant_cluster_for_sample(r)
            if dom is None: 
                continue
            bucket[dom]['probs'].append(r['probability'])
            bucket[dom]['labels'].append(r['actual_label'])

        out={}
        for cid, data in bucket.items():
            probs=np.array(data['probs']); labels=np.array(data['labels'])
            if probs.size<5: 
                continue
            best=(0.74, 0.0, 0, 0, 0) # th, f1, tp, fp, fn
            for th in np.arange(0.3,0.9,0.02):
                preds=(probs>=th).astype(int)
                tp=((preds==1)&(labels==1)).sum()
                fp=((preds==1)&(labels==0)).sum()
                fn=((preds==0)&(labels==1)).sum()
                prec=tp/max(1,tp+fp); rec=tp/max(1,tp+fn)
                f1=0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)
                if f1>best[1]: 
                    best=(th,f1,tp,fp,fn)
            # simple expected net gain if moving from 0.74 to best
            base_preds=(probs>=0.74).astype(int)
            base_tp=((base_preds==1)&(labels==1)).sum()
            base_fp=((base_preds==1)&(labels==0)).sum()
            delta_tp=best[2]-base_tp; delta_fp=best[3]-base_fp
            out[cid]={'optimal_threshold':float(best[0]), 'best_f1':float(best[1]),
                      'sample_count':int(probs.size),
                      'expected_delta_tp':int(delta_tp), 'expected_delta_fp':int(delta_fp)}
            print(f"Cluster {cid}: th*={best[0]:.2f} F1={best[1]:.3f} N={probs.size} ΔTP={delta_tp} ΔFP={delta_fp}")
        return out

    # --------------------------------------------------------------------------------
    # 12) Fact redundancy & contradiction
    # --------------------------------------------------------------------------------
    def run_fact_redundancy_analysis(self):
        print("\n=== FACT REDUNDANCY ANALYSIS ===")
        pair_counts=defaultdict(int); contradictions=[]
        for r in self.attention_results:
            fids=[fid for fid,_ in self.get_top_facts_for_prediction(r)[:10]]
            for i in range(len(fids)):
                for j in range(i+1,len(fids)):
                    a,b=sorted([fids[i],fids[j]])
                    pair_counts[(a,b)]+=1
                    fa=self.fact_mapping.get(a); fb=self.fact_mapping.get(b)
                    if fa and fb:
                        sa, sb = fa['sentiment'], fb['sentiment']
                        if (sa*sb)<-0.1 and abs(sa)>0.2 and abs(sb)>0.2:
                            contradictions.append({'a':fa,'b':fb,'result':r,
                                                   'sentiment_gap':abs(sa-sb)})
        contr_fn=[c for c in contradictions if c['result']['predicted_label']==0 and c['result']['actual_label']==1]
        contr_tp=[c for c in contradictions if c['result']['predicted_label']==1 and c['result']['actual_label']==1]
        rate = len(contr_fn)/max(1,len(contradictions))
        print(f"Contradictory pairs: {len(contradictions)} | FN share: {rate:.2f}")
        top_pairs=sorted(pair_counts.items(), key=lambda x:x[1], reverse=True)[:20]
        return {'frequent_pairs':top_pairs,
                'contradictory_pairs':contradictions,
                'contradictory_fns':contr_fn,
                'contradictory_tps':contr_tp,
                'contradiction_fn_rate':rate}

    # --------------------------------------------------------------------------------
    # 13) Company bias (adds direct/indirect ratio)
    # --------------------------------------------------------------------------------
    def run_company_bias_analysis(self):
        print("\n=== COMPANY BIAS ANALYSIS ===")
        # choose top offenders by error rate after we compute company errors
        offenders=['AMZN','CMG','BRO']
        out={}
        for co in offenders:
            subset=[r for r in self.attention_results 
                    if getattr(r.get('subgraph',{}),'primary_ticker','')==co]
            if not subset: 
                continue
            direct=0.0; indirect=0.0; errs=0
            for r in subset:
                for fid,att in self.get_top_facts_for_prediction(r):
                    txt=self.fact_mapping.get(fid,{}).get('raw_text','').lower()
                    if co.lower() in txt: direct+=att
                    else: indirect+=att
                errs += int(r['predicted_label']!=r['actual_label'])
            tot=direct+indirect
            out[co]={'direct_attention':direct,'indirect_attention':indirect,
                     'direct_ratio': (direct/tot if tot>0 else 0.0),
                     'sample_count':len(subset),
                     'error_rate': errs/max(1,len(subset))}
            print(f"{co}: direct_ratio={out[co]['direct_ratio']:.3f} err={out[co]['error_rate']:.3f} N={len(subset)}")
        return out

    # --------------------------------------------------------------------------------
    # 14) Calibration curves by cluster/company/sentiment
    # --------------------------------------------------------------------------------
    def run_calibration_curves_analysis(self):
        print("\n=== CALIBRATION CURVES ANALYSIS ===")
        by_cluster = defaultdict(lambda:{'probs':[],'labels':[]})
        by_company = defaultdict(lambda:{'probs':[],'labels':[]})
        by_sent = {'positive':[],'neutral':[],'negative':[]}

        for r in self.attention_results:
            # cluster (dominant)
            _, dom = self._dominant_cluster_for_sample(r)
            if dom is not None:
                by_cluster[dom]['probs'].append(r['probability'])
                by_cluster[dom]['labels'].append(r['actual_label'])
            # company
            co = getattr(r.get('subgraph',{}),'primary_ticker',None)
            if co:
                by_company[co]['probs'].append(r['probability'])
                by_company[co]['labels'].append(r['actual_label'])
            # sentiment bucket
            s = self.get_avg_sentiment(r)
            key = 'positive' if s>0.2 else 'negative' if s<-0.2 else 'neutral'
            by_sent[key].append((r['probability'], r['actual_label']))

        def metrics(probs, labels):
            ece,mce,brier = self._ece_mce_brier(probs, labels, n_bins=15)
            return {'calibration_error':ece, 'max_error':mce, 'brier':brier,
                    'sample_count':len(probs),
                    'mean_probability':float(np.mean(probs)) if len(probs) else 0.0,
                    'actual_positive_rate':float(np.mean(labels)) if len(labels) else 0.0}

        out={'by_cluster':{}, 'by_company':{}, 'by_sentiment':{}}
        for cid,data in by_cluster.items():
            if len(data['probs'])>=5:
                out['by_cluster'][cid]=metrics(data['probs'], data['labels'])
        for co,data in by_company.items():
            if len(data['probs'])>=3:
                out['by_company'][co]=metrics(data['probs'], data['labels'])
        for k,arr in by_sent.items():
            if arr:
                p,l = zip(*arr)
                out['by_sentiment'][k]=metrics(list(p), list(l))
        return out

    # --------------------------------------------------------------------------------
    # 15) Event combinations (unchanged core; returns conflicts)
    # --------------------------------------------------------------------------------
    def run_event_combination_exploration(self):
        print("\n=== EVENT COMBINATION EXPLORATION ===")
        result = {'tp_combinations':defaultdict(int),'fn_combinations':defaultdict(int),
                  'fp_combinations':defaultdict(int),'tn_combinations':defaultdict(int)}
        for name, res in [('tp',self.tp_results),('fn',self.fn_results),
                          ('fp',self.fp_results),('tn',self.tn_results)]:
            for r in res:
                evts=[]
                for fid,_ in self.get_top_facts_for_prediction(r):
                    e=self.fact_mapping.get(fid,{}).get('event_type')
                    if e: evts.append(e)
                evts=sorted(set(evts))
                from itertools import combinations
                for k in [2,3]:
                    for combo in combinations(evts, min(k,len(evts))):
                        result[f'{name}_combinations'][combo]+=1

        conflicts=[]
        for combo, fnc in result['fn_combinations'].items():
            tpc = result['tp_combinations'].get(combo,0)
            if fnc>tpc and fnc>=2:
                conflicts.append({'combination':combo, 'fn_count':fnc, 'tp_count':tpc,
                                  'conflict_ratio': fnc/float(fnc+tpc)})
        result['conflicting_pairs']=sorted(conflicts, key=lambda x:x['conflict_ratio'], reverse=True)
        return result

    # --------------------------------------------------------------------------------
    # 16) Enhanced temporal drift (fixed FN logic, by year/quarter)
    # --------------------------------------------------------------------------------
    def run_enhanced_temporal_analysis(self):
        print("\n=== ENHANCED TEMPORAL ANALYSIS ===")
        out={'by_year':defaultdict(lambda:{'tp':0,'fp':0,'fn':0,'tn':0}),
             'by_quarter':defaultdict(lambda:{'tp':0,'fp':0,'fn':0,'tn':0}),
             'pre_post_2020':{'pre':{'tp':0,'fp':0,'fn':0,'tn':0},
                              'post':{'tp':0,'fp':0,'fn':0,'tn':0}}}

        def add(date_str, r):
            try:
                dt=datetime.strptime(date_str,'%Y-%m-%d')
            except:
                return
            y=dt.year; q=f"{y}-Q{(dt.month-1)//3+1}"
            pdict = out['by_year'][y]; qdict = out['by_quarter'][q]
            # correct confusion increments
            if r['predicted_label']==1 and r['actual_label']==1:
                pdict['tp']+=1; qdict['tp']+=1
            elif r['predicted_label']==1 and r['actual_label']==0:
                pdict['fp']+=1; qdict['fp']+=1
            elif r['predicted_label']==0 and r['actual_label']==1:
                pdict['fn']+=1; qdict['fn']+=1
            else:
                pdict['tn']+=1; qdict['tn']+=1
            era = 'pre' if y<2020 else 'post'
            if r['predicted_label']==1 and r['actual_label']==1: out['pre_post_2020'][era]['tp']+=1
            elif r['predicted_label']==1 and r['actual_label']==0: out['pre_post_2020'][era]['fp']+=1
            elif r['predicted_label']==0 and r['actual_label']==1: out['pre_post_2020'][era]['fn']+=1
            else: out['pre_post_2020'][era]['tn']+=1

        for r in self.attention_results:
            d=getattr(r.get('subgraph',{}),'reported_date',None)
            if d: add(d, r)

        # derive rates
        for bucket in [out['by_year'], out['by_quarter']]:
            for k,v in bucket.items():
                tot = sum(v.values())
                if tot>0:
                    v['error_rate']=(v['fn']+v['fp'])/tot
                    v['fn_rate']= v['fn']/max(1,(v['tp']+v['fn']))
        return out

    # --------------------------------------------------------------------------------
    # 17) Per-company & per-cluster calibration/threshold report CSVs
    # --------------------------------------------------------------------------------
    def save_detailed_csvs(self, cluster_thresholds, calibration_curves):
        outdir = self.model_dir
        # cluster thresholds
        ct_rows=[]
        for cid,d in cluster_thresholds.items():
            ct_rows.append({'cluster_id':cid, **d})
        if ct_rows:
            pd.DataFrame(ct_rows).to_csv(outdir/'cluster_threshold_suggestions.csv', index=False)
        # calibration by cluster
        bc=[]
        for cid, m in calibration_curves.get('by_cluster',{}).items():
            bc.append({'cluster_id':cid, **m})
        if bc:
            pd.DataFrame(bc).to_csv(outdir/'calibration_by_cluster.csv', index=False)
        # calibration by company
        bco=[]
        for co, m in calibration_curves.get('by_company',{}).items():
            bco.append({'ticker':co, **m})
        if bco:
            pd.DataFrame(bco).to_csv(outdir/'calibration_by_company.csv', index=False)

    # --------------------------------------------------------------------------------
    # Orchestrator
    # --------------------------------------------------------------------------------
    def run_all_diagnostics(self):
        print("="*80)
        print(f"COMPREHENSIVE DIAGNOSTICS FOR MODEL: {self.model_dir.name}")
        print("="*80)

        results={}
        results['cluster'] = self.run_cluster_level_diagnostics()
        results['fact'] = self.run_fact_level_analysis()
        results['company'] = self.run_company_level_analysis()
        results['attention'] = self.run_attention_flow_analysis()
        results['calibration'] = self.run_calibration_analysis()
        results['interaction'] = self.run_event_type_interaction_analysis()
        results['temporal'] = self.run_temporal_drift_analysis()
        results['case_study'] = self.run_case_study_examples()

        # advanced
        results['cluster_misclassification'] = self.run_cluster_misclassification_analysis()
        results['sentiment_conditioning'] = self.run_sentiment_conditioning_analysis()
        results['cluster_thresholds'] = self.run_cluster_threshold_analysis()
        results['fact_redundancy'] = self.run_fact_redundancy_analysis()
        results['company_bias'] = self.run_company_bias_analysis()
        results['calibration_curves'] = self.run_calibration_curves_analysis()
        results['event_combinations'] = self.run_event_combination_exploration()
        results['enhanced_temporal'] = self.run_enhanced_temporal_analysis()

        # write text report (same structure as yours, kept concise here)
        out_txt = self.model_dir / "comprehensive_diagnostics.txt"
        with open(out_txt, 'w') as f:
            f.write("="*80+"\n")
            f.write(f"COMPREHENSIVE DIAGNOSTICS FOR MODEL: {self.model_dir.name}\n")
            f.write("="*80+"\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            TP=len(self.tp_results); FP=len(self.fp_results)
            TN=len(self.tn_results); FN=len(self.fn_results)
            acc=(TP+TN)/max(1,(TP+TN+FP+FN))
            prec=TP/max(1,(TP+FP)); rec=TP/max(1,(TP+FN))
            f.write("SUMMARY STATISTICS:\n")
            f.write(f"TP={TP} FP={FP} TN={TN} FN={FN}\n")
            f.write(f"Accuracy={acc:.3f} Precision={prec:.3f} Recall={rec:.3f}\n\n")

            # cluster worst
            _, ratios, sent_var = results['cluster']
            f.write("Worst clusters (TP/(TP+FN)):\n")
            for i,c in enumerate(ratios[:15]):
                f.write(f"{i+1}. Cluster {c['cluster_id']}: ratio={c['ratio']:.3f} (TP={c['tp_count']} FN={c['fn_count']})\n")
            f.write("\nSentiment variance (top 10):\n")
            for i,c in enumerate(sent_var[:10]):
                f.write(f"{i+1}. Cluster {c['cluster_id']}: var={c['variance']:.3f} TPμ={c['tp_mean']:.3f} FNμ={c['fn_mean']:.3f}\n")

            # attention stats
            tp_att, fn_att = results['attention']
            f.write("\nATTENTION CONCENTRATION:\n")
            f.write(f"TP entropy mean={tp_att['entropy'][0]:.3f} | FN entropy mean={fn_att['entropy'][0]:.3f}\n")
            f.write(f"TP gini={tp_att['gini'][0]:.3f} | FN gini={fn_att['gini'][0]:.3f}\n")
            f.write(f"TP top5cov={tp_att['topk_cov'][0]:.3f} | FN top5cov={fn_att['topk_cov'][0]:.3f}\n")
            f.write(f"TP headvar={tp_att['head_var'][0]:.3f} | FN headvar={fn_att['head_var'][0]:.3f}\n")

            # calibration global
            _,_,_,_, opt, (ece,mce,brier) = results['calibration']
            f.write("\nGLOBAL CALIBRATION:\n")
            f.write(f"ECE={ece:.3f} MCE={mce:.3f} Brier={brier:.3f} | Optimal threshold={opt:.3f}\n")

            # per-cluster thresholds
            f.write("\nCLUSTER THRESHOLD SUGGESTIONS:\n")
            for cid,d in results['cluster_thresholds'].items():
                f.write(f"Cluster {cid}: th*={d['optimal_threshold']:.2f} F1={d['best_f1']:.3f} N={d['sample_count']} ΔTP={d['expected_delta_tp']} ΔFP={d['expected_delta_fp']}\n")

            # conflicts
            f.write("\nFN-BIASED EVENT COMBINATIONS (top 10):\n")
            for i,p in enumerate(results['event_combinations']['conflicting_pairs'][:10]):
                f.write(f"{i+1}. {p['combination']}: FN={p['fn_count']} TP={p['tp_count']} ratio={p['conflict_ratio']:.2f}\n")

        # structured CSVs
        self.save_detailed_csvs(results['cluster_thresholds'], results['calibration_curves'])
        print(f"\n✅ Diagnostics written to:\n- {out_txt}\n- CSVs in {self.model_dir}")

        return results


# ------------------------------------------------------------------------------------
# Entry
# ------------------------------------------------------------------------------------
def main():
    results_dir = Path("../Results/heterognn5")
    model_dirs = [d for d in results_dir.iterdir() if d.is_dir() and (d / "model.pt").exists()]
    print(f"Found {len(model_dirs)} models to analyze")
    ok=0; fail=0
    for md in sorted(model_dirs):
        try:
            print("\n" + "="*80)
            print(f"ANALYZING: {md.name}")
            print("="*80)
            diag = ComprehensiveDiagnostics(md)
            _ = diag.run_all_diagnostics()
            ok+=1
            print(f"✅ Done: {md.name}")
        except Exception as e:
            fail+=1
            print(f"❌ Error on {md.name}: {e}")
    print("\n" + "="*80)
    print(f"SUMMARY | success={ok} | failed={fail} | total={len(model_dirs)}")

if __name__ == "__main__":
    main()
