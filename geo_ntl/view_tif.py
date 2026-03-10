from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio


DEFAULT_BAND_NAMES = [
    "bm",
    "gaia_year",
    "urban_mask",
    "building_height",
    "water_mask",
    "ndvi",
    "ndbi",
    "mndwi",
]

DEFAULT_CMAPS = {
    "bm": "magma",
    "gaia_year": "plasma",
    "urban_mask": "gray",
    "building_height": "viridis",
    "water_mask": "Blues",
    "ndvi": "YlGn",
    "ndbi": "RdYlBu_r",
    "mndwi": "PuBu",
}


def robust_limits(arr: np.ndarray, lower: float = 2, upper: float = 98) -> tuple[float, float]:
    """Robuste Anzeigegrenzen über Perzentile."""
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return 0.0, 1.0

    vmin = np.percentile(valid, lower)
    vmax = np.percentile(valid, upper)

    if np.isclose(vmin, vmax):
        vmin = float(np.nanmin(valid))
        vmax = float(np.nanmax(valid))

    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    return float(vmin), float(vmax)


def get_band_names(src: rasterio.io.DatasetReader) -> list[str]:
    """Bandnamen aus dem TIFF lesen oder Fallback verwenden."""
    descriptions = list(src.descriptions) if src.descriptions else []
    descriptions = [d if d not in (None, "") else None for d in descriptions]

    if any(descriptions):
        return [
            descriptions[i] if i < len(descriptions) and descriptions[i] is not None else f"band_{i+1}"
            for i in range(src.count)
        ]

    return [
        DEFAULT_BAND_NAMES[i] if i < len(DEFAULT_BAND_NAMES) else f"band_{i+1}"
        for i in range(src.count)
    ]


def show_all_bands(tif_path: Path, save_png: Path | None = None) -> None:
    with rasterio.open(tif_path) as src:
        data = src.read().astype(np.float32)  # (bands, H, W)
        band_names = get_band_names(src)

        print("Datei:", tif_path)
        print("Anzahl Bänder:", src.count)
        print("Größe:", src.width, "x", src.height)
        print("CRS:", src.crs)
        print("Transform:", src.transform)
        print("Bandnamen:", band_names)

    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    n_bands = data.shape[0]
    ncols = 4
    nrows = int(np.ceil(n_bands / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.2 * nrows))
    axes = np.array(axes).reshape(-1)

    for i in range(len(axes)):
        ax = axes[i]

        if i >= n_bands:
            ax.axis("off")
            continue

        band = data[i]
        name = band_names[i]
        cmap = DEFAULT_CMAPS.get(name, "viridis")

        # Für binäre Masken feste Grenzen, sonst robuste Perzentile
        if name in {"urban_mask", "water_mask"}:
            vmin, vmax = 0, 1
        else:
            vmin, vmax = robust_limits(band)

        im = ax.imshow(band, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"{i+1}: {name}")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_png is not None:
        save_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_png, dpi=200, bbox_inches="tight")
        print("PNG gespeichert:", save_png)

    plt.show()


def show_one_band(tif_path: Path, band_index: int, save_png: Path | None = None) -> None:
    with rasterio.open(tif_path) as src:
        if band_index < 1 or band_index > src.count:
            raise ValueError(f"Bandindex muss zwischen 1 und {src.count} liegen.")

        band_names = get_band_names(src)
        band = src.read(band_index).astype(np.float32)
        name = band_names[band_index - 1]

        print("Datei:", tif_path)
        print("Band:", band_index, "-", name)
        print("CRS:", src.crs)
        print("Größe:", src.width, "x", src.height)

    band = np.nan_to_num(band, nan=0.0, posinf=0.0, neginf=0.0)
    cmap = DEFAULT_CMAPS.get(name, "viridis")

    if name in {"urban_mask", "water_mask"}:
        vmin, vmax = 0, 1
    else:
        vmin, vmax = robust_limits(band)

    plt.figure(figsize=(8, 8))
    plt.imshow(band, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(f"{band_index}: {name}")
    plt.axis("off")
    plt.colorbar()

    if save_png is not None:
        save_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_png, dpi=200, bbox_inches="tight")
        print("PNG gespeichert:", save_png)

    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mehrband-GeoTIFF visualisieren")
    parser.add_argument(
        "tif_path",
        type=str,
        help="Pfad zum GeoTIFF"
    )
    parser.add_argument(
        "--band",
        type=int,
        default=None,
        help="Optional: nur ein einzelnes Band anzeigen (1-basiert)"
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional: PNG speichern"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tif_path = Path(args.tif_path)
    save_png = Path(args.save) if args.save is not None else None

    if not tif_path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {tif_path}")

    if args.band is None:
        show_all_bands(tif_path, save_png=save_png)
    else:
        show_one_band(tif_path, args.band, save_png=save_png)


if __name__ == "__main__":
    main()