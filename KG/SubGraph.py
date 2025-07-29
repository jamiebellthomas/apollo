import json
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Any, Tuple

import numpy as np
import pandas as pd

@dataclass
class SubGraph:
    primary_ticker: str
    reported_date: str
    predicted_eps: float | None
    real_eps: float | None
    fact_list: List[Dict[str, Any]]

    def to_json_line(self) -> str:
        """Return a JSON string with exactly the specified top-level keys."""
        payload = {
            "primary_ticker": self.primary_ticker,
            "reported_date": self.reported_date,
            "predicted_eps": self.predicted_eps,
            "real_eps": self.real_eps,
            "fact_count": len(self.fact_list),
            "fact_list": self.fact_list,  
        }
        return json.dumps(payload, ensure_ascii=False)