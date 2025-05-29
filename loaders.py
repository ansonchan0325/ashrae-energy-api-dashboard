# loaders.py
import os
import pandas as pd
from functools import lru_cache

BASE_DIR = "models/monthly_metadata_parquet"

@lru_cache(maxsize=24)
def load_month(month: str) -> pd.DataFrame:
    """
    Load (and cache) the Parquet file for a month 'YYYY-MM'.
    Returns a DataFrame indexed by (building_id, meter, timestamp).
    """
    path = os.path.join(BASE_DIR, f"metadata_{month}.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No metadata file for month {month}")
    df = pd.read_parquet(path, engine="pyarrow")
    # set index if needed:
    if set(["building_id","meter", "timestamp"]) <= set(df.columns):
        return df.set_index(["building_id","meter","timestamp"])
    else:
        raise ValueError("Missing expected columns in metadata")
