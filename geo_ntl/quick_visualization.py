from pathlib import Path
import rasterio
import matplotlib.pyplot as plt
import numpy as np

tif_path = Path(r"C:\Users\cinoa\Desktop\NTL\Git Diffusion\diffusers\geo_ntl\quick_vis_data\kyiv_predictors_130m.tif")

with rasterio.open(tif_path) as src:
    print("Anzahl Bänder:", src.count)
    print("CRS:", src.crs)
    print("Größe:", src.width, "x", src.height)
    print("Bandnamen:", src.descriptions)

    data = src.read()   # shape: (bands, height, width)

band_names = [
    "bm",
    "gaia_year",
    "urban_mask",
    "building_height",
    "water_mask",
    "ndvi",
    "ndbi",
    "mndwi",
]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()

for i in range(data.shape[0]):
    band = data[i]

    # NaNs/Infs sauber behandeln
    band = np.nan_to_num(band, nan=0.0, posinf=0.0, neginf=0.0)

    im = axes[i].imshow(band, cmap="viridis")
    axes[i].set_title(f"{i+1}: {band_names[i]}")
    axes[i].axis("off")
    plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()