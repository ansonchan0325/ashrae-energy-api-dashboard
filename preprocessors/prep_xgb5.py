import pandas as pd
import numpy as np
from loaders import load_month
# import sys
# sys.modules['numpy._core'] = np.core

# _building_features = (
#     pd.read_pickle('models/building_metadata_df.pkl')
#       .set_index(['building_id','meter','timestamp'])
# )

XGB5_FEATURES = [
    'square_feet', 'year_built', 'floor_count', 'hour', 'weekend',
    'building_median', 'air_temperature', 'precip_depth_1_hr',
    'air_temperature_mean_lag72', 'dew_temperature_mean_lag72',
    'air_temperature_max_lag3', 'air_temperature_min_lag3',
    'dew_temperature_mean_lag3', 'air_temperature_mean_lag3',
    'building_id_uid_enc', 'building_id-m_nunique', 'site_id_uid_enc',
    'DT_w_dew_temp', 'building_id', 'site_id', 'primary_use'
]

def preprocess_row_for_xgb5(raw: dict) -> pd.DataFrame:
    """
    Build an N×M DataFrame for the 5-fold XGBoost (meter 0) over an hourly range.

    raw must include:
      - building_id: int
      - meter:       int (0)
      - start_date:  str ("YYYY-MM-DD"), inclusive at 00:00:00
      - end_date:    str ("YYYY-MM-DD"), inclusive at 23:00:00

    Returns:
      DataFrame with rows for every hour in the full inclusive range, and
      columns matching XGB5_FEATURES.
    """
    start = pd.to_datetime(raw['start_date'])
    end   = pd.to_datetime(raw['end_date']) + pd.Timedelta(hours=23)
    idx   = pd.date_range(start, end, freq='h')

    # figure out all months in the requested range
    months = sorted({ts.to_period("M").strftime("%Y-%m") for ts in idx})

    # load & concatenate only those months
    dfs = [load_month(m) for m in months]
    _building_features = pd.concat(dfs)

    rows = []
    for ts in idx:
        feats = _building_features.loc[(raw['building_id'],raw['meter'],ts)].copy()
        feats['building_id'] = raw['building_id']  # ensure present
        feats['meter']       = raw['meter']  # ensure present
        feats['hour']        = np.int8(ts.hour)
        feats['weekend']     = np.int8(ts.weekday() >= 5)
        rows.append(feats[XGB5_FEATURES])

    return pd.DataFrame(rows).reset_index(drop=True)