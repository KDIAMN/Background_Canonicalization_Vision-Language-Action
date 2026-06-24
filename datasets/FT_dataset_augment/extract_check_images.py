import io
import pandas as pd
from pathlib import Path
from PIL import Image

PARQUET_PATH = Path("/path/to/VLA_Adapter/datasets/FT_dataset_augment/bg_masked_libero_object/data/chunk-000/file-067.parquet")
OUT_DIR      = Path("/path/to/VLA_Adapter/datasets/FT_dataset_augment/check_before_augment/extract5")

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(PARQUET_PATH)
    print(f"총 row 수: {len(df)}")

    for i, entry in enumerate(df["observation.images.image"]):
        img = Image.open(io.BytesIO(entry["bytes"]))
        img.save(OUT_DIR / f"frame_{i:04d}.png")

    print(f"완료 → {OUT_DIR}  ({len(df)}장)")

if __name__ == "__main__":
    main()
