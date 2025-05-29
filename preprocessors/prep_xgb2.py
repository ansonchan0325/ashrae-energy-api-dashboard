import pandas as pd
import numpy as np
from loaders import load_month

XGB2_FEATURES = [
    'square_feet', 'building_id', 'meter', 'air_temperature', 'year_built',
    'primary_use', 'floor_count', 'dew_temperature', 'DT_hour',
    'sea_level_pressure', 'DT_day_week', 'site_id', 'DT_day_month',
    'cloud_coverage', 'wind_speed', 'wind_direction',
    'g_meter_site_id_uid_enc', 'building_median', 'building_min',
    'air_temperature_min_lag3', 'air_temperature_min_lag18',
    'air_temperature_std_lag18', 'cloud_coverage_median_lag18'
]

def preprocess_row_for_xgb2(raw: dict) -> pd.DataFrame:
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
        feats['building_id']  = raw['building_id']  # ensure present
        feats['meter']        = raw['meter']  # ensure present
        feats['DT_hour']      = np.int8(ts.hour)
        feats['DT_day_week']  = np.int8(ts.weekday())
        feats['DT_day_month'] = np.int8(ts.day)
        rows.append(feats[XGB2_FEATURES])

    return pd.DataFrame(rows).reset_index(drop=True)