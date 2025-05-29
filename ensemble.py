import pandas as pd
import numpy as np
from preprocessors.prep_xgb2 import preprocess_row_for_xgb2
from preprocessors.prep_xgb5 import preprocess_row_for_xgb5
from preprocessors.prep_lgbm import preprocess_row_for_lgbm3
from model_code.xgb_2fold import predict_xgb2
from model_code.xgb_5fold import predict_xgb5
from model_code.lgbm import predict_lgbm3
from loaders import load_month
from typing import List, Dict, Any

# Ensemble weights & multiplier 
W1, W2, W3 = 0.45, 0.25, 0.3
ENSEMBLE_MULTIPLIER = 0.90

def run_ensemble(raw: dict) -> List[Dict[str,Any]]:
    # 1) Preprocess inputs
    df2 = preprocess_row_for_xgb2(raw)
    df5 = preprocess_row_for_xgb5(raw)
    df3 = preprocess_row_for_lgbm3(raw)

    # 2) Inference
    preds2 = predict_xgb2(df2)
    preds5 = predict_xgb5(df5, raw['meter'])
    preds3 = predict_lgbm3(df3)

    # 3) Blend & scale
    final = (W1*preds2 + W2*preds5 + W3*preds3) * ENSEMBLE_MULTIPLIER

    # 4) Build the hourly index
    start = pd.to_datetime(raw['start_date'])
    end   = pd.to_datetime(raw['end_date']) + pd.Timedelta(hours=23)
    idx   = pd.date_range(start, end, freq='h')

    # 5) Pre‐load each month’s metadata once
    months = sorted({ts.to_period("M").strftime("%Y-%m") for ts in idx})
    month_dfs = { m: load_month(m) for m in months }

    # 6) Assemble the output
    records = []
    for ts, p2, p5, p3, p in zip(idx, preds2, preds5, preds3, final):
        # figure out which month file we need
        m = ts.to_period("M").strftime("%Y-%m")
        df_meta = month_dfs[m]

        # lookup row_id
        try:
            rid = df_meta.at[(raw['building_id'], raw['meter'], ts), 'row_id']
        except KeyError:
            rid = None

        records.append({
            'building_id': raw['building_id'],
            'meter':       raw['meter'],
            'start_date':  raw['start_date'],
            'end_date':    raw['end_date'],
            'row_id':      rid,
            'timestamp':   ts.isoformat(),
            'xgb2':        float(p2),
            'xgb5':        float(p5),
            'lgbm3':       float(p3),
            'prediction':  float(p)
        })

    return records

    # # 6) Evaluate against leak data
    # df_eval = (
    #     df_records
    #       .dropna(subset=['row_id'])
    #       .merge(_leak_df, on='row_id', how='inner')
    # )
    # if not df_eval.empty:
    #     log_rmsle = np.sqrt(
    #         ((np.log1p(df_eval['prediction']) - np.log1p(df_eval['leaked_meter_reading']))**2)
    #          .mean()
    #     )
    # else:
    #     log_rmsle = None

    # # 7) Collect warnings
    # warnings = []
    # # missing row_id
    # missing = df_records[df_records['row_id'].isna()]
    # for _, row in missing.iterrows():
    #     warnings.append(f"Missing row_id for timestamp {row['timestamp']}")
    # # invalid predictions
    # invalid = df_records[df_records['prediction'] <= 0]
    # for _, row in invalid.iterrows():
    #     warnings.append(
    #         f"Non-positive prediction {row['prediction']} at "
    #         f"timestamp {row['timestamp']} (row_id={row['row_id']})"
    #     )

    # return {
    #     'predictions': df_records,
    #     'log_rmsle':   log_rmsle,
    #     'warnings':    warnings
    # }

    