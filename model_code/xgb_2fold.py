import pickle
import numpy as np
import pandas as pd

# ─── Load pre-trained 2-fold XGBoost models ────────────────────────────────────
bst_fold0 = pickle.load(open('models/xgb_2fold_fold0.bin', 'rb'))
bst_fold1 = pickle.load(open('models/xgb_2fold_fold1.bin', 'rb'))
_xgb2_models = [bst_fold0, bst_fold1]

# ─── Inference function ─────────────────────────────────────────────────────────
def predict_xgb2(df_row: pd.DataFrame):
    """
    Predict using the 2-fold XGBoost ensemble.

    Args:
        df_row: Single-row DataFrame with features in the correct order.

    Returns:
        Float prediction (mean of fold predictions).
    """
    preds = [model.predict(df_row.values) for model in _xgb2_models]  # shape: [n_models, n_samples]
    mean_preds = np.mean(preds, axis=0)  # shape: [n_samples]
    return np.expm1(mean_preds)          # shape: [n_samples]

    # preds = [model.predict(df_row.values) for model in _xgb2_models]
    # return np.expm1(np.mean(preds))



