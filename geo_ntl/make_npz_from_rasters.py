from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling


# ------------------------------------------------------------
# Einstellungen
# ------------------------------------------------------------
PREDICTOR_PATH = Path("geo_ntl/data_raw/kyiv_predictors_130m.tif")

# HIER deinen lokalen LJ1-01 Rasterpfad eintragen
# Er sollte ungefähr dieselbe Region abdecken.
LJ1_PATH = Path("geo_ntl/data_raw/kyiv_lj1.tif")

OUT_DIR = Path("geo_ntl/data/train")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = 128
STRIDE = 128
MIN_VALID_RATIO = 0.95

# Muss exakt zur Bandreihenfolge im EE-Export passen
PREDICTOR_BANDS = [
    "bm",
    "gaia_year",
    "urban_mask",
    "building_height",
    "water_mask",
    "ndvi",
    "ndbi",
    "mndwi",
]


# ------------------------------------------------------------
# 1) Predictor-Raster laden
# ------------------------------------------------------------
with rasterio.open(PREDICTOR_PATH) as src:
    predictors = src.read().astype(np.float32)   # (C, H, W)
    dst_transform = src.transform
    dst_crs = src.crs
    dst_height = src.height
    dst_width = src.width

print("Predictor shape:", predictors.shape)
print("Predictor CRS:", dst_crs)
print("Predictor size:", dst_width, dst_height)

if predictors.shape[0] != len(PREDICTOR_BANDS):
    raise ValueError(
        f"Expected {len(PREDICTOR_BANDS)} predictor bands, got {predictors.shape[0]}"
    )

predictors = np.nan_to_num(predictors, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


# ------------------------------------------------------------
# 2) LJ1-01 Target laden und auf Predictor-Gitter reprojizieren
# ------------------------------------------------------------
with rasterio.open(LJ1_PATH) as src:
    # Falls dein LJ1 nur ein Band hat:
    lj1_src = src.read(1).astype(np.float32)

    lj1_resampled = np.full((dst_height, dst_width), np.nan, dtype=np.float32)

    reproject(
        source=lj1_src,
        destination=lj1_resampled,
        src_transform=src.transform,
        src_crs=src.crs,
        src_nodata=src.nodata,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )

valid_mask = np.isfinite(lj1_resampled).astype(np.float32)[None, ...]
target = np.nan_to_num(lj1_resampled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)[None, ...]

print("Target shape after reprojection:", target.shape)
print("Valid ratio:", valid_mask.mean())


# ------------------------------------------------------------
# 3) In Patches schneiden und als .npz speichern
# ------------------------------------------------------------
count = 0

for y in range(0, dst_height - PATCH_SIZE + 1, STRIDE):
    for x in range(0, dst_width - PATCH_SIZE + 1, STRIDE):
        target_patch = target[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
        valid_patch = valid_mask[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
        pred_patch = predictors[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]

        # Nur Patches behalten, die fast vollständig gültig sind
        if valid_patch.mean() < MIN_VALID_RATIO:
            continue

        sample = {
            "target": target_patch.astype(np.float32),
            "valid_mask": valid_patch.astype(np.float32),
        }

        for i, name in enumerate(PREDICTOR_BANDS):
            sample[name] = pred_patch[i:i+1].astype(np.float32)

        out_path = OUT_DIR / f"sample_{count:06d}.npz"
        np.savez_compressed(out_path, **sample)
        count += 1

print(f"Saved {count} patches to {OUT_DIR}")