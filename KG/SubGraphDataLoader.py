import json
import random
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
    
    The loader now supports fixed data splits with separate training and testing items.
    """

    def __init__(
        self,
        min_facts: int,
        jsonl_path: Optional[str] = None,
        limit: Optional[int] = None,
        exclude_clusters: Optional[List[int]] = None,  # optional initial blacklist to apply right after load
        train_ratio: float = 0.75,
        val_ratio: float = 0.1,
        seed: int = 42,
        split_data: bool = True,  # Whether to split data into training and testing
    ):
        self.min_facts = int(min_facts)
        self.jsonl_path = jsonl_path or config.SUBGRAPHS_JSONL
        self.limit = None if limit is None else int(limit)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed
        self.split_data = split_data
        
        # Initialize separate lists for training and testing
        self.training_items: List[SubGraph] = []
        self.testing_items: List[SubGraph] = []
        self.items: List[SubGraph] = []  # Keep for backward compatibility
        
        self._load()

        # If you passed an initial blacklist, apply it now
        if exclude_clusters:
            self.filter_clusters(exclude_clusters, drop_below_min_facts=True)

    def __len__(self) -> int:
        """Return total number of items (training + testing)."""
        return len(self.training_items) + len(self.testing_items)

    def __iter__(self) -> Iterator[SubGraph]:
        """Iterate over all items (training + testing)."""
        return iter(self.training_items + self.testing_items)

    def get(self, i: int) -> SubGraph:
        """Get item by index (training items first, then testing items)."""
        if i < len(self.training_items):
            return self.training_items[i]
        else:
            return self.testing_items[i - len(self.training_items)]

    def get_training_items(self) -> List[SubGraph]:
        """Get all training items."""
        return self.training_items

    def get_testing_items(self) -> List[SubGraph]:
        """Get all testing items."""
        return self.testing_items

    def get_training_iter(self) -> Iterator[SubGraph]:
        """Iterate over training items only."""
        return iter(self.training_items)

    def get_testing_iter(self) -> Iterator[SubGraph]:
        """Iterate over testing items only."""
        return iter(self.testing_items)

    def split_training_validation(self, val_ratio: float = None, seed: int = None) -> tuple[List[SubGraph], List[SubGraph]]:
        """
        Split training items into training and validation sets.
        
        Args:
            val_ratio: Ratio for validation set (defaults to self.val_ratio)
            seed: Random seed for splitting (defaults to self.seed)
            
        Returns:
            Tuple of (training_items, validation_items)
        """
        if val_ratio is None:
            val_ratio = self.val_ratio
        if seed is None:
            seed = self.seed
            
        if len(self.training_items) < 2:
            raise ValueError(f"Need at least 2 training items to split; got {len(self.training_items)}")
            
        # Shuffle training items with fixed seed
        random.seed(seed)
        shuffled_training = self.training_items.copy()
        random.shuffle(shuffled_training)
        
        # Split into training and validation
        n_val = int(len(shuffled_training) * val_ratio)
        val_items = shuffled_training[:n_val]
        train_items = shuffled_training[n_val:]
        
        print(f"[split] Training items split: {len(train_items)} train, {len(val_items)} validation")
        return train_items, val_items

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
        all_items: List[SubGraph] = []

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
                        all_items.append(SubGraph(**obj))
                        kept += 1
                    except TypeError:
                        # schema mismatch; skip
                        continue

                    if self.limit is not None and kept >= self.limit:
                        break

        print(f"[load] Read {count} lines, kept {kept} subgraphs "
              f"(eps_surprise>0 & fact_count≥{self.min_facts}) from {self.jsonl_path}")

        if self.split_data and len(all_items) >= 2:
            # Split data into training and testing
            self._split_data(all_items)
        else:
            # Keep all items in training (for backward compatibility)
            self.training_items = all_items
            self.testing_items = []
            
        # Set items for backward compatibility
        self.items = self.training_items + self.testing_items

    def _split_data(self, all_items: List[SubGraph]) -> None:
        """Split all items into training and testing sets."""
        if len(all_items) < 2:
            print(f"[split] Not enough items ({len(all_items)}) to split, keeping all in training")
            self.training_items = all_items
            self.testing_items = []
            return
            
        # Shuffle with fixed seed
        random.seed(self.seed)
        shuffled_items = all_items.copy()
        random.shuffle(shuffled_items)
        
        # Split into training and testing - use 0.85 for training (leaving 0.15 for testing)
        n_train_val = int(len(shuffled_items) * 0.85)
        
        self.training_items = shuffled_items[:n_train_val]
        self.testing_items = shuffled_items[n_train_val:]
        
        print(f"[split] Data split: {len(self.training_items)} training, {len(self.testing_items)} testing")

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
        
        # Filter training items
        new_training_items: List[SubGraph] = []
        for sg in self.training_items:
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
                new_training_items.append(sg)
        
        # Filter testing items
        new_testing_items: List[SubGraph] = []
        for sg in self.testing_items:
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
                new_testing_items.append(sg)

        dropped_training = len(self.training_items) - len(new_training_items)
        dropped_testing = len(self.testing_items) - len(new_testing_items)
        total_dropped = dropped_training + dropped_testing
        
        self.training_items = new_training_items
        self.testing_items = new_testing_items
        
        # Update items for backward compatibility
        self.items = self.training_items + self.testing_items
        
        print(f"[filter_clusters] Excluded clusters={sorted(excl)} "
              f"→ dropped {total_dropped} subgraphs (training: {dropped_training}, testing: {dropped_testing}); "
              f"kept {len(self.items)} total")
