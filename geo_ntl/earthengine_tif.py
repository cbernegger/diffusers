from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import requests
from PIL import Image
from rasterio import Affine
from rasterio.io import MemoryFile
from rasterio.windows import Window

try:
    import ee
except ImportError:
    ee = None  # type: ignore[assignment]

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]


PROJECT_ID = "ninth-sol-478514-g1"

DEFAULT_TIF_DIR = Path(r"C:\Users\cinoa\Desktop\NTL git\lj1_lightglue_batch100_improved_tu\warped_tifs")

LAYER_ORDER = [
    "bm",
    "gaia_year",
    "urban_mask",
    "building_height",
    "water_mask",
    "ndvi",
    "ndbi",
    "mndwi",
]


@dataclass(frozen=True)
class RasterGrid:
    path: Path
    width: int
    height: int
    crs: str
    transform: tuple[float, float, float, float, float, float]
    bounds: tuple[float, float, float, float]
    count: int
    dtype: str
    nodata: float | int | None

    @property
    def stem(self) -> str:
        return self.path.stem

    @property
    def region(self) -> ee.Geometry:
        left, bottom, right, top = self.bounds
        return ee.Geometry.Rectangle(
            coords=[left, bottom, right, top],
            proj=self.crs,
            geodesic=False,
        )

    @property
    def nominal_scale(self) -> float:
        return max(abs(self.transform[0]), abs(self.transform[4]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Earth Engine predictor layers for every TIFF in a folder."
    )
    parser.add_argument(
        "--tif-path",
        type=Path,
        default=None,
        help="Optional path to a single warped TIFF to process.",
    )
    parser.add_argument(
        "--tif-dir",
        type=Path,
        default=DEFAULT_TIF_DIR,
        help="Folder with warped input TIFFs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help='Output folder. Defaults to a sibling folder named "layers".',
    )
    parser.add_argument(
        "--project-id",
        type=str,
        default=PROJECT_ID,
        help="Earth Engine project id.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit for quick testing.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=1.0,
        help="Pause between layer downloads to avoid hammering Earth Engine.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing copied TIFFs and downloaded layers.",
    )
    parser.add_argument(
        "--read-timeout-seconds",
        type=float,
        default=900.0,
        help="HTTP read timeout for Earth Engine downloads.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=512,
        help="Tile size in output pixels for Earth Engine downloads.",
    )
    return parser.parse_args()


def init_ee(project_id: str) -> None:
    if ee is None:
        raise RuntimeError(
            "The `earthengine-api` package is not installed in this Python environment. "
            "Install it with `pip install earthengine-api`, then authenticate and rerun."
        )

    try:
        ee.Initialize(project=project_id)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Earth Engine initialization failed. Authenticate first with "
            "`earthengine authenticate` or `ee.Authenticate()`, then rerun the script."
        ) from exc

    print("Earth Engine ok:", ee.Number(1).getInfo())


def build_layers(grid: RasterGrid) -> dict[str, ee.Image]:
    region = grid.region

    gaia = ee.Image("Tsinghua/FROM-GLC/GAIA/v10").select("change_year_index")

    gaia_year = (
        ee.Image.constant(2019)
        .subtract(gaia)
        .rename("gaia_year")
        .updateMask(gaia.gte(1))
        .clip(region)
    )

    urban_mask = gaia.gte(1).selfMask().rename("urban_mask").clip(region)

    building_height = (
        ee.Image("JRC/GHSL/P2023A/GHS_BUILT_H/2018")
        .select("built_height")
        .rename("building_height")
        .clip(region)
    )

    s2_collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate("2018-05-01", "2018-09-30")
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .select(["B3", "B4", "B8", "B11"])
    )

    s2_count = s2_collection.size().getInfo()
    if s2_count == 0:
        raise RuntimeError(
            "No Sentinel-2 SR images found for this TIFF extent in 2018-05-01..2018-09-30 "
            "with CLOUDY_PIXEL_PERCENTAGE < 20."
        )

    s2_2018 = (
        s2_collection
        .median()
        .clip(region)
    )

    ndvi = s2_2018.normalizedDifference(["B8", "B4"]).rename("ndvi")
    ndbi = s2_2018.normalizedDifference(["B11", "B8"]).rename("ndbi")
    mndwi = s2_2018.normalizedDifference(["B3", "B11"]).rename("mndwi")

    dw_2018 = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterBounds(region)
        .filterDate("2018-05-01", "2018-09-30")
        .select(["water"])
        .mean()
        .rename(["dw_water"])
        .clip(region)
    )

    water_mask = dw_2018.select("dw_water").gte(0.5).rename("water_mask")

    viirs_2018 = (
        ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG")
        .filterBounds(region)
        .filterDate("2018-01-01", "2019-01-01")
    )

    bm = (
        viirs_2018
        .select("avg_rad")
        .mean()
        .rename("bm")
        .clip(region)
    )

    layers = {
        "bm": bm.unmask(0).toFloat(),
        "gaia_year": gaia_year.unmask(0).toFloat(),
        "urban_mask": urban_mask.unmask(0).toFloat(),
        "building_height": building_height.unmask(0).toFloat(),
        "water_mask": water_mask.unmask(0).toFloat(),
        "ndvi": ndvi.unmask(0).toFloat(),
        "ndbi": ndbi.unmask(0).toFloat(),
        "mndwi": mndwi.unmask(0).toFloat(),
    }

    return layers


def sanitize_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def iter_tifs(tif_dir: Path) -> list[Path]:
    tif_paths = sorted(tif_dir.glob("*.tif"))
    tif_paths.extend(sorted(tif_dir.glob("*.tiff")))

    seen: set[Path] = set()
    unique_paths: list[Path] = []
    for tif_path in tif_paths:
        if tif_path not in seen:
            unique_paths.append(tif_path)
            seen.add(tif_path)

    return unique_paths


def read_raster_grid(tif_path: Path) -> RasterGrid:
    with rasterio.open(tif_path) as src:
        if src.crs is None:
            raise ValueError(f"Raster has no CRS: {tif_path}")

        transform = tuple(float(v) for v in list(src.transform)[:6])
        bounds = (
            float(src.bounds.left),
            float(src.bounds.bottom),
            float(src.bounds.right),
            float(src.bounds.top),
        )

        return RasterGrid(
            path=tif_path,
            width=src.width,
            height=src.height,
            crs=src.crs.to_string(),
            transform=transform,
            bounds=bounds,
            count=src.count,
            dtype=src.dtypes[0],
            nodata=src.nodata,
        )


def write_metadata(out_dir: Path, grid: RasterGrid, source_tif_copy: Path) -> None:
    metadata = {
        "source_warped_tif": str(grid.path),
        "copied_warped_tif": str(source_tif_copy),
        "width": grid.width,
        "height": grid.height,
        "crs": grid.crs,
        "transform": list(grid.transform),
        "bounds": list(grid.bounds),
        "band_count": grid.count,
        "dtype": grid.dtype,
        "nodata": grid.nodata,
        "layers": LAYER_ORDER,
    }

    metadata_path = out_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)


def make_browseable_luojia(source_tif_path: Path, out_dir: Path, overwrite: bool) -> None:
    display_tif_path = out_dir / "luojia_display.tif"
    preview_png_path = out_dir / "luojia_preview.png"

    with rasterio.open(source_tif_path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata

    valid_mask = np.isfinite(arr)
    if nodata is not None:
        valid_mask &= arr != nodata

    display_u8 = np.zeros(arr.shape, dtype=np.uint8)

    if np.any(valid_mask):
        valid = arr[valid_mask]
        positive = valid[valid > 0]

        if positive.size > 0:
            noise_floor = float(np.percentile(positive, 82))
            support_floor = float(np.percentile(positive, 88))
            high = float(np.percentile(positive, 99.8))

            noise_floor = max(noise_floor, 1.0)
            support_floor = max(support_floor, noise_floor + 1.0)
            high = max(high, support_floor + 1.0)

            suppressed = np.zeros_like(arr, dtype=np.float32)
            suppressed[valid_mask] = np.clip(arr[valid_mask] - noise_floor, 0.0, high - noise_floor)

            log_scale = np.log1p(high - noise_floor)
            normalized = np.zeros_like(arr, dtype=np.float32)
            normalized[valid_mask] = np.log1p(suppressed[valid_mask]) / max(log_scale, 1e-6)
            # Strongly suppress the low end so widespread dim noise does not dominate the view.
            normalized[valid_mask] = np.power(np.clip(normalized[valid_mask], 0.0, 1.0), 2.2)

            support_mask = np.zeros(arr.shape, dtype=np.uint8)
            support_mask[valid_mask] = (arr[valid_mask] >= support_floor).astype(np.uint8)

            if cv2 is not None:
                kernel = np.ones((3, 3), dtype=np.uint8)
                support_mask = cv2.morphologyEx(support_mask, cv2.MORPH_OPEN, kernel)
                support_mask = cv2.morphologyEx(support_mask, cv2.MORPH_CLOSE, kernel)

                num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(support_mask, connectivity=8)
                filtered_mask = np.zeros_like(support_mask)
                min_component_area = 12

                for label in range(1, num_labels):
                    area = stats[label, cv2.CC_STAT_AREA]
                    if area >= min_component_area:
                        filtered_mask[labels == label] = 1

                support_mask = cv2.dilate(filtered_mask, kernel, iterations=1)
                soft_mask = cv2.GaussianBlur(support_mask.astype(np.float32), (0, 0), sigmaX=1.0, sigmaY=1.0)
                soft_mask = np.clip(soft_mask, 0.0, 1.0)
            else:
                soft_mask = support_mask.astype(np.float32)

            normalized *= soft_mask
            display_u8[valid_mask] = np.round(np.clip(normalized[valid_mask], 0.0, 1.0) * 255.0).astype(np.uint8)

            if cv2 is not None:
                display_u8 = cv2.GaussianBlur(display_u8, (0, 0), sigmaX=0.8, sigmaY=0.8)
                display_u8[~valid_mask] = 0

    profile.update(
        dtype="uint8",
        count=1,
        nodata=0,
    )

    with rasterio.open(display_tif_path, "w", **profile) as dst:
        dst.write(display_u8, 1)
        dst.set_band_description(1, "luojia_display")

    Image.fromarray(display_u8, mode="L").save(preview_png_path)


def download_ee_image(
    image: ee.Image,
    grid: RasterGrid,
    out_path: Path,
    export_name: str,
    session: requests.Session,
    read_timeout_seconds: float,
    tile_size: int,
) -> None:
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")

    a, b, c, d, e, f = grid.transform
    base_image = image.clip(grid.region)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")

    profile = {
        "driver": "GTiff",
        "width": grid.width,
        "height": grid.height,
        "count": 1,
        "dtype": "float32",
        "crs": grid.crs,
        "transform": Affine(a, b, c, d, e, f),
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }

    with rasterio.open(tmp_path, "w", **profile) as dst:
        for row_off in range(0, grid.height, tile_size):
            tile_height = min(tile_size, grid.height - row_off)

            for col_off in range(0, grid.width, tile_size):
                tile_width = min(tile_size, grid.width - col_off)
                tile_c = c + (col_off * a) + (row_off * b)
                tile_f = f + (col_off * d) + (row_off * e)
                tile_transform = [a, b, tile_c, d, e, tile_f]

                params = {
                    "name": sanitize_name(f"{export_name}_{row_off}_{col_off}"),
                    "crs": grid.crs,
                    "crs_transform": tile_transform,
                    "dimensions": f"{tile_width}x{tile_height}",
                    "format": "GEO_TIFF",
                }

                url = base_image.getDownloadURL(params)
                response = session.get(url, stream=True, timeout=(30, read_timeout_seconds))
                try:
                    response.raise_for_status()
                except requests.HTTPError as exc:
                    detail = response.text.strip()
                    if not detail:
                        detail = "<empty response body>"
                    raise RuntimeError(
                        f"{exc}. Earth Engine response: {detail}. "
                        f"Tile row_off={row_off}, col_off={col_off}, size={tile_width}x{tile_height}"
                    ) from exc

                with MemoryFile(response.content) as memfile:
                    with memfile.open() as tile_src:
                        tile_data = tile_src.read(1).astype("float32")

                if tile_data.shape != (tile_height, tile_width):
                    raise RuntimeError(
                        f"Unexpected tile shape {tile_data.shape} for tile "
                        f"row_off={row_off}, col_off={col_off}, expected {(tile_height, tile_width)}"
                    )

                dst.write(tile_data, 1, window=Window(col_off, row_off, tile_width, tile_height))

    tmp_path.replace(out_path)


def process_tif(
    tif_path: Path,
    output_dir: Path,
    session: requests.Session,
    overwrite: bool,
    sleep_seconds: float,
    read_timeout_seconds: float,
    tile_size: int,
) -> None:
    grid = read_raster_grid(tif_path)
    tif_output_dir = output_dir / grid.stem
    tif_output_dir.mkdir(parents=True, exist_ok=True)

    copied_tif_path = tif_output_dir / tif_path.name
    if overwrite or not copied_tif_path.exists():
        shutil.copy2(tif_path, copied_tif_path)

    make_browseable_luojia(copied_tif_path, tif_output_dir, overwrite=overwrite)
    write_metadata(tif_output_dir, grid, copied_tif_path)

    print(f"\nProcessing {tif_path.name}")
    print(f"  Grid: {grid.width} x {grid.height} | {grid.crs}")
    print(f"  Layer order: {LAYER_ORDER}")

    missing_layers = [
        layer_name
        for layer_name in LAYER_ORDER
        if overwrite or not (tif_output_dir / f"{layer_name}.tif").exists()
    ]

    if not missing_layers:
        print("  All Earth Engine layers already exist. Generated/kept browseable LuoJia preview files.")
        return

    layers = build_layers(grid)

    for layer_name in LAYER_ORDER:
        out_path = tif_output_dir / f"{layer_name}.tif"

        if out_path.exists() and not overwrite:
            print(f"  Skipping existing layer: {out_path.name}")
            continue

        print(f"  Downloading layer: {layer_name}")
        try:
            download_ee_image(
                image=layers[layer_name],
                grid=grid,
                out_path=out_path,
                export_name=f"{grid.stem}_{layer_name}",
                session=session,
                read_timeout_seconds=read_timeout_seconds,
                tile_size=tile_size,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Layer `{layer_name}` failed: {exc}") from exc

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)


def main() -> None:
    args = parse_args()

    tif_path = args.tif_path.expanduser() if args.tif_path is not None else None
    tif_dir = args.tif_dir.expanduser()

    if tif_path is not None:
        if not tif_path.exists():
            raise FileNotFoundError(f"TIFF file not found: {tif_path}")
        tif_paths = [tif_path]
        default_output_dir = tif_path.parent.parent / "layers"
        print(f"Input TIFF file: {tif_path}")
    else:
        if not tif_dir.exists():
            raise FileNotFoundError(f"TIFF folder not found: {tif_dir}")

        tif_paths = iter_tifs(tif_dir)
        if args.max_files is not None:
            tif_paths = tif_paths[: args.max_files]

        if not tif_paths:
            raise FileNotFoundError(f"No .tif or .tiff files found in: {tif_dir}")

        default_output_dir = tif_dir.parent / "layers"
        print(f"Input TIFF folder: {tif_dir}")
        print(f"Found {len(tif_paths)} TIFF file(s).")

    output_dir = args.output_dir.expanduser() if args.output_dir is not None else default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output layers folder: {output_dir}")

    init_ee(args.project_id)
    failures: list[tuple[str, str]] = []
    with requests.Session() as session:
        for tif_path in tif_paths:
            try:
                process_tif(
                    tif_path=tif_path,
                    output_dir=output_dir,
                    session=session,
                    overwrite=args.overwrite,
                    sleep_seconds=args.sleep_seconds,
                    read_timeout_seconds=args.read_timeout_seconds,
                    tile_size=args.tile_size,
                )
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                failures.append((tif_path.name, str(exc)))
                print(f"  Failed: {tif_path.name}")
                print(f"  Reason: {exc}")

    if failures:
        print("\nFinished with failures:")
        for tif_name, reason in failures:
            print(f"  - {tif_name}: {reason}")
        raise SystemExit(1)

    print("\nFinished successfully.")


if __name__ == "__main__":
    main()
