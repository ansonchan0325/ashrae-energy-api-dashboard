# download_models.py
import os
from google.cloud import storage

BUCKET = "my-energy-api-files"

# 1) Where in GCS your monthly Parquet files live (under “models/monthly_metadata_parquet/…”)
METADATA_PREFIX = "models/monthly_metadata_parquet/"

# 2) Your other model binaries (still under “models/” in your bucket)
MODEL_FILES = [
    "lgbm_1.bin",
    "lgbm_2.bin",
    "lgbm_3.bin",
    "meter2-xgb-meter0-fold0.bin",
    "meter2-xgb-meter0-fold1.bin",
    "meter2-xgb-meter0-fold2.bin",
    "meter2-xgb-meter0-fold3.bin",
    "meter2-xgb-meter0-fold4.bin",
    "meter2-xgb-meter1.bin",
    "meter2-xgb-meter2.bin",
    "meter2-xgb-meter3.bin",
    "xgb_2fold_fold0.bin",
    "xgb_2fold_fold1.bin"
]

def fetch_models():
    client = storage.Client()
    bucket = client.bucket(BUCKET)

    # ─── 1) Download monthly Parquet metadata ────────────────────────────────────
    meta_out = "models/monthly_metadata_parquet"
    os.makedirs(meta_out, exist_ok=True)

    print("Fetching monthly metadata slices…")
    for blob in bucket.list_blobs(prefix=METADATA_PREFIX):
        if not blob.name.endswith(".parquet"):
            continue

        # blob.name: "models/monthly_metadata_parquet/metadata_2017-01.parquet"
        filename = os.path.basename(blob.name)
        dest_path = os.path.join(meta_out, filename)

        if not os.path.exists(dest_path):
            print(f"  → Downloading {filename}")
            blob.download_to_filename(dest_path)
        else:
            print(f"  ✓ {filename} already exists, skipping.")

    # ─── 2) Download the model binaries ─────────────────────────────────────────
    bin_out = "models"
    os.makedirs(bin_out, exist_ok=True)

    print("Fetching model binaries…")
    for fname in MODEL_FILES:
        gcs_path = f"models/{fname}"
        dest_path = os.path.join(bin_out, fname)

        if not os.path.exists(dest_path):
            print(f"  → Downloading {fname}")
            bucket.blob(gcs_path).download_to_filename(dest_path)
        else:
            print(f"  ✓ {fname} already exists, skipping.")

if __name__ == "__main__":
    fetch_models()
