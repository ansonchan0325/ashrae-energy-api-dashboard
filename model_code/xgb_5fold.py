import pickle
import numpy as np
import pandas as pd

# ─── Meter-to-model-file mapping ────────────────────────────────────────────────
_MODEL_FILES = {
    0: [
        'models/meter2-xgb-meter0-fold0.bin',
        'models/meter2-xgb-meter0-fold1.bin',
        'models/meter2-xgb-meter0-fold2.bin',
        'models/meter2-xgb-meter0-fold3.bin',
        'models/meter2-xgb-meter0-fold4.bin'
    ],
    1: ['models/meter2-xgb-meter1.bin'],
    2: ['models/meter2-xgb-meter2.bin'],
    3: ['models/meter2-xgb-meter3.bin']
}

# ─── Load all models once at import ────────────────────────────────────────────
# _xgb5_models = {
#     meter: [pickle.load(open(path, 'rb')) for path in paths]
#     for meter, paths in _MODEL_FILES.items()
# }
_xgb5_models = {}
for meter, paths in _MODEL_FILES.items():
    models = []
    for path in paths:
        loaded = pickle.load(open(path, 'rb'))
        if isinstance(loaded, list):
            models.extend(loaded)  # flatten list of models
        else:
            models.append(loaded)
    _xgb5_models[meter] = models


# ─── Inference function ─────────────────────────────────────────────────────────
def predict_xgb5(df, meter):
    """
    Batch inference for the XGBoost meter-specific ensemble.

    Args:
        df (pd.DataFrame): N×M DataFrame matching the XGB5_FEATURES order.
        meter (int): Meter type key (0, 1, 2, or 3).

    Returns:
        np.ndarray: length N; predictions back-transformed via expm1.
    """
    models = _xgb5_models.get(meter)
    if not models:
        raise ValueError(f"No XGB5 model files configured for meter {meter}")

    # each model outputs log1p(meter_reading)
    preds = np.vstack([m.predict(df.values) for m in models]).T
    mean_log = preds.mean(axis=1)
    return np.expm1(mean_log)
