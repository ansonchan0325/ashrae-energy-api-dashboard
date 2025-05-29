import pickle
import numpy as np
import pandas as pd

# ─── Load pre-trained lgbm models ────────────────────────────────────
lgbm_1 = pickle.load(open('models/lgbm_1.bin', 'rb'))
lgbm_2 = pickle.load(open('models/lgbm_2.bin', 'rb'))
lgbm_3 = pickle.load(open('models/lgbm_3.bin', 'rb'))
_lgbm_models = [lgbm_1, lgbm_2, lgbm_3]

# ─── Inference function ─────────────────────────────────────────────────────────
def predict_lgbm3(df_row: pd.DataFrame):
    """
    Predict using the lgbm ensemble.

    Args:
        df_row: Single-row DataFrame with features in the correct order.

    Returns:
        Float prediction (mean of fold predictions).
    """

    preds = [model.predict(df_row.values) for model in _lgbm_models]  # shape: [n_models, n_samples]
    mean_preds = np.mean(preds, axis=0)  # shape: [n_samples]
    return np.expm1(mean_preds)          # shape: [n_samples]

    # preds = [model.predict(df_row.values) for model in _lgbm_models]
    # return np.expm1(np.mean(preds))

