import json
from typing import Iterator, List, Optional
from SubGraph import SubGraph
import config

class SubGraphDataLoader:
    """
    Loads SubGraph JSONL and keeps only entries with:
      • at least `min_facts` facts, and
      • eps_surprise > 0 (strictly positive)

    You can later call `filter_clusters(...)` to remove facts whose
    event_cluster_id is in a blacklist, and optionally drop subgraphs
    that fall below `min_facts` after filtering.
    """

    def __init__(
        self,
        min_facts: int,
        jsonl_path: Optional[str] = None,
        limit: Optional[int] = None,
        exclude_clusters: Optional[List[int]] = None,  # optional initial blacklist to apply right after load
    ):
        self.min_facts = int(min_facts)
        self.jsonl_path = jsonl_path or config.SUBGRAPHS_JSONL
        self.limit = None if limit is None else int(limit)
        self.items: List[SubGraph] = []
        self._load()

        # If you passed an initial blacklist, apply it now
        if exclude_clusters:
            self.filter_clusters(exclude_clusters, drop_below_min_facts=True)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[SubGraph]:
        return iter(self.items)

    def get(self, i: int) -> SubGraph:
        return self.items[i]

    # ---------- internal ----------
    @staticmethod
    def _normalize_keys(obj: dict) -> dict:
        # Handle minor naming drift safely
        if "reporting_date" in obj and "reported_date" not in obj:
            obj["reported_date"] = obj.pop("reporting_date")
        return obj

    def _load(self) -> None:
        count = 0
        kept = 0
        items: List[SubGraph] = []

        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                count += 1
                if count % 100 == 0 and count > 0:
                    print(f"[load] Processed {count} lines...")
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue  # skip bad lines, keep streaming

                obj = self._normalize_keys(obj)
                facts = obj.get("fact_list") or []

                # Strictly positive EPS surprise only
                eps = obj.get("eps_surprise", None)
                if eps is None or float(eps) <= 0.0:
                    continue

                if len(facts) >= self.min_facts:
                    try:
                        items.append(SubGraph(**obj))
                        kept += 1
                    except TypeError:
                        # schema mismatch; skip
                        continue

                    if self.limit is not None and kept >= self.limit:
                        break

        self.items = items
        print(f"[load] Read {count} lines, kept {kept} subgraphs "
              f"(eps_surprise>0 & fact_count≥{self.min_facts}) from {self.jsonl_path}")

    # ---------- public: cluster filtering ----------
    def filter_clusters(
        self,
        exclude_clusters: List[int],
        drop_below_min_facts: bool = True
    ) -> None:
        """
        Remove facts whose 'event_cluster_id' is in exclude_clusters.
        Optionally drop any SubGraph whose fact_count falls below min_facts after filtering.
        """
        excl = set(int(x) for x in exclude_clusters)
        new_items: List[SubGraph] = []

        for sg in self.items:
            # Keep only facts not in excluded clusters
            filtered_facts = []
            for f in (sg.fact_list or []):
                cid = f.get("event_cluster_id", None)
                if cid is not None and int(cid) in excl:
                    continue
                filtered_facts.append(f)

            # Update sg's facts/fact_count
            sg.fact_list = filtered_facts
            sg.fact_count = len(filtered_facts)

            # Decide to keep or drop
            if (not drop_below_min_facts) or (sg.fact_count >= self.min_facts):
                new_items.append(sg)

        dropped = len(self.items) - len(new_items)
        self.items = new_items
        print(f"[filter_clusters] Excluded clusters={sorted(excl)} "
              f"→ dropped {dropped} subgraphs; kept {len(self.items)}")
