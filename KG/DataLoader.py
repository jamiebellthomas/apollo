import json
from typing import Iterator, List, Optional
from SubGraph import SubGraph
import config

class DataLoader:
    """
    Load SubGraph JSONL and keep only entries with exactly `n_facts` facts.
    """

    def __init__(self, n_facts: int, jsonl_path: Optional[str] = None, limit: Optional[int] = None):
        self.n_facts = int(n_facts)
        self.jsonl_path = jsonl_path or config.SUBGRAPHS_JSONL
        self.items: List[SubGraph] = []
        self._load(limit=limit)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[SubGraph]:
        return iter(self.items)

    def get(self, i: int) -> SubGraph:
        return self.items[i]

    # ---------- internal ----------
    @staticmethod
    def _normalize_keys(obj: dict) -> dict:
        # handle minor naming drift safely
        if "reporting_date" in obj and "reported_date" not in obj:
            obj["reported_date"] = obj.pop("reporting_date")
        return obj

    def _load(self, limit: Optional[int]) -> None:
        count = 0
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = self._normalize_keys(json.loads(line))
                facts = obj.get("fact_list") or []
                if len(facts) == self.n_facts:
                    self.items.append(SubGraph(**obj))
                    count += 1
                    if limit is not None and count >= limit:
                        break
    
    def reload(self, new_limit: Optional[int] = None) -> None:
        """
        Reload the data with a new limit.
        """
        self.items.clear()
        self.n_facts = new_limit if new_limit is not None else self.n_facts
        self._load(limit=new_limit)
        print(f"[reload] Loaded {len(self.items)} items with exactly {self.n_facts} facts from {self.jsonl_path}.")