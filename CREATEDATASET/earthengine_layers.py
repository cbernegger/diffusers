from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import ExitStack
import json
import math
import numpy as np
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import rasterio
import requests
from rasterio import Affine
from rasterio.io import MemoryFile
from rasterio.windows import Window

from paths import LAYERS_DIR, WARPED_TIFS_DIR, ensure_createdataset_dirs
from scene_assets import copy_source_tif_to_scene_dir

try:
    import ee
except ImportError:
    ee = None  # type: ignore[assignment]

STEP_NAME = "earthengine"
PROJECT_ID = "ninth-sol-478514-g1"
DEFAULT_TIF_DIR = WARPED_TIFS_DIR
DEFAULT_OUTPUT_DIR = LAYERS_DIR
THREAD_LOCAL = threading.local()

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


@dataclass(frozen=True)
class TileRequest:
    index: int
    row_off: int
    col_off: int
    tile_width: int
    tile_height: int
    tile_transform: list[float]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download Earth Engine predictor layers into CREATEDATASET/LAYERS."
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
        help="Folder with warped input TIFFs. Defaults to CREATEDATASET/WARPED_TIFS.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help='Output folder. Defaults to CREATEDATASET/LAYERS.',
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
        default=0.0,
        help="Optional pause after each scene download.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing layer TIFFs.",
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
    parser.add_argument(
        "--max-concurrent-tiles",
        type=int,
        default=4,
        help="Maximum number of tiles to download in parallel per scene.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


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


def write_metadata(out_dir: Path, grid: RasterGrid, scene_source_copy_path: Path) -> None:
    metadata_path = out_dir / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as fh:
            metadata = json.load(fh)
    else:
        metadata = {}

    merged_layers = list(metadata.get("layers", []))
    for layer_name in LAYER_ORDER:
        if layer_name not in merged_layers:
            merged_layers.append(layer_name)

    metadata.update(
        {
            "source_warped_tif": str(grid.path.resolve()),
            "source_scene_tif_copy": str(scene_source_copy_path.resolve()),
            "scene_id": grid.stem,
            "width": grid.width,
            "height": grid.height,
            "crs": grid.crs,
            "transform": list(grid.transform),
            "bounds": list(grid.bounds),
            "band_count": grid.count,
            "dtype": grid.dtype,
            "nodata": grid.nodata,
            "layers": merged_layers,
            "generator": STEP_NAME,
        }
    )

    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)


def build_stacked_layer_image(grid: RasterGrid, layer_names: list[str]) -> ee.Image:
    layers = build_layers(grid)
    return ee.Image.cat([layers[layer_name] for layer_name in layer_names]).rename(layer_names)


def iter_tile_requests(grid: RasterGrid, tile_size: int) -> list[TileRequest]:
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")

    a, b, c, d, e, f = grid.transform
    requests: list[TileRequest] = []
    tile_index = 0

    for row_off in range(0, grid.height, tile_size):
        tile_height = min(tile_size, grid.height - row_off)

        for col_off in range(0, grid.width, tile_size):
            tile_width = min(tile_size, grid.width - col_off)
            tile_c = c + (col_off * a) + (row_off * b)
            tile_f = f + (col_off * d) + (row_off * e)
            tile_transform = [a, b, tile_c, d, e, tile_f]
            requests.append(
                TileRequest(
                    index=tile_index,
                    row_off=row_off,
                    col_off=col_off,
                    tile_width=tile_width,
                    tile_height=tile_height,
                    tile_transform=tile_transform,
                )
            )
            tile_index += 1

    return requests


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def render_progress_line(
    *,
    prefix: str,
    completed: int,
    total: int,
    start_time: float,
) -> str:
    total = max(total, 1)
    width = 28
    filled = min(width, math.floor((completed / total) * width))
    bar = "#" * filled + "-" * (width - filled)
    percent = (completed / total) * 100.0
    elapsed = format_duration(time.perf_counter() - start_time)
    return f"{prefix} [{bar}] {completed}/{total} ({percent:5.1f}%) elapsed {elapsed}"


def print_progress_line(message: str, *, done: bool) -> None:
    print(message.ljust(120), end="\n" if done else "\r", flush=True)


def get_thread_session() -> requests.Session:
    session = getattr(THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        THREAD_LOCAL.session = session
    return session


def fetch_tile_data(
    *,
    base_image: ee.Image,
    grid: RasterGrid,
    tile_request: TileRequest,
    layer_names: list[str],
    read_timeout_seconds: float,
    session: requests.Session | None,
) -> tuple[TileRequest, np.ndarray]:
    params = {
        "name": sanitize_name(f"{grid.stem}_layers_{tile_request.row_off}_{tile_request.col_off}"),
        "crs": grid.crs,
        "crs_transform": tile_request.tile_transform,
        "dimensions": f"{tile_request.tile_width}x{tile_request.tile_height}",
        "format": "GEO_TIFF",
    }

    url = base_image.getDownloadURL(params)
    local_session = session if session is not None else get_thread_session()
    http_get = local_session.get
    response = http_get(url, stream=True, timeout=(30, read_timeout_seconds))

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = response.text.strip()
        if not detail:
            detail = "<empty response body>"
        raise RuntimeError(
            f"{exc}. Earth Engine response: {detail}. "
            f"Tile row_off={tile_request.row_off}, col_off={tile_request.col_off}, "
            f"size={tile_request.tile_width}x{tile_request.tile_height}"
        ) from exc

    with MemoryFile(response.content) as memfile:
        with memfile.open() as tile_src:
            tile_data = tile_src.read().astype("float32")

    if tile_data.ndim == 2:
        tile_data = tile_data[None, ...]

    expected_shape = (len(layer_names), tile_request.tile_height, tile_request.tile_width)
    if tile_data.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected tile shape {tile_data.shape} for tile "
            f"row_off={tile_request.row_off}, col_off={tile_request.col_off}, expected {expected_shape}"
        )

    return tile_request, tile_data


def write_tile_data(
    *,
    tile_request: TileRequest,
    tile_data: np.ndarray,
    datasets: dict[str, rasterio.io.DatasetWriter],
    layer_names: list[str],
) -> None:
    for band_index, layer_name in enumerate(layer_names):
        dst = datasets.get(layer_name)
        if dst is None:
            continue
        dst.write(
            tile_data[band_index],
            1,
            window=Window(
                tile_request.col_off,
                tile_request.row_off,
                tile_request.tile_width,
                tile_request.tile_height,
            ),
        )


def download_ee_layers(
    image: ee.Image,
    grid: RasterGrid,
    out_dir: Path,
    layer_names: list[str],
    session: requests.Session,
    read_timeout_seconds: float,
    tile_size: int,
    overwrite: bool,
    max_concurrent_tiles: int,
    progress_prefix: str,
) -> None:
    a, b, c, d, e, f = grid.transform
    base_image = image.clip(grid.region)
    tile_requests = iter_tile_requests(grid, tile_size)
    total_tiles = len(tile_requests)

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

    with ExitStack() as stack:
        datasets: dict[str, rasterio.io.DatasetWriter] = {}
        for layer_name in layer_names:
            out_path = out_dir / f"{layer_name}.tif"
            if out_path.exists() and not overwrite:
                continue

            tmp_path = out_path.with_suffix(out_path.suffix + ".part")
            dst = stack.enter_context(rasterio.open(tmp_path, "w", **profile))
            dst.set_band_description(1, layer_name)
            datasets[layer_name] = dst

        if not datasets:
            return

        start_time = time.perf_counter()
        completed = 0
        print_progress_line(
            render_progress_line(
                prefix=progress_prefix,
                completed=completed,
                total=total_tiles,
                start_time=start_time,
            ),
            done=False,
        )

        if max_concurrent_tiles <= 1 or total_tiles <= 1:
            for tile_request in tile_requests:
                resolved_request, tile_data = fetch_tile_data(
                    base_image=base_image,
                    grid=grid,
                    tile_request=tile_request,
                    layer_names=layer_names,
                    read_timeout_seconds=read_timeout_seconds,
                    session=session,
                )
                write_tile_data(
                    tile_request=resolved_request,
                    tile_data=tile_data,
                    datasets=datasets,
                    layer_names=layer_names,
                )
                completed += 1
                print_progress_line(
                    render_progress_line(
                        prefix=progress_prefix,
                        completed=completed,
                        total=total_tiles,
                        start_time=start_time,
                    ),
                    done=completed == total_tiles,
                )
        else:
            max_workers = min(max_concurrent_tiles, total_tiles)
            tile_iter = iter(tile_requests)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                active_futures: dict[Future[tuple[TileRequest, np.ndarray]], TileRequest] = {}

                def submit_next() -> None:
                    next_tile = next(tile_iter, None)
                    if next_tile is None:
                        return
                    future = executor.submit(
                        fetch_tile_data,
                        base_image=base_image,
                        grid=grid,
                        tile_request=next_tile,
                        layer_names=layer_names,
                        read_timeout_seconds=read_timeout_seconds,
                        session=None,
                    )
                    active_futures[future] = next_tile

                for _ in range(max_workers):
                    submit_next()

                while active_futures:
                    done_futures, _ = wait(active_futures, return_when=FIRST_COMPLETED)
                    for future in done_futures:
                        active_futures.pop(future, None)
                        resolved_request, tile_data = future.result()
                        write_tile_data(
                            tile_request=resolved_request,
                            tile_data=tile_data,
                            datasets=datasets,
                            layer_names=layer_names,
                        )
                        completed += 1
                        print_progress_line(
                            render_progress_line(
                                prefix=progress_prefix,
                                completed=completed,
                                total=total_tiles,
                                start_time=start_time,
                            ),
                            done=completed == total_tiles,
                        )
                        submit_next()

    for layer_name in layer_names:
        out_path = out_dir / f"{layer_name}.tif"
        tmp_path = out_path.with_suffix(out_path.suffix + ".part")
        if tmp_path.exists():
            tmp_path.replace(out_path)


def process_tif(
    tif_path: Path,
    output_dir: Path,
    session: requests.Session,
    overwrite: bool,
    sleep_seconds: float,
    read_timeout_seconds: float,
    tile_size: int,
    max_concurrent_tiles: int,
    scene_index: int,
    scene_count: int,
) -> None:
    grid = read_raster_grid(tif_path)
    tif_output_dir = output_dir / grid.stem
    tif_output_dir.mkdir(parents=True, exist_ok=True)
    scene_source_copy_path = copy_source_tif_to_scene_dir(
        tif_output_dir,
        tif_path,
        overwrite=overwrite,
    )
    write_metadata(tif_output_dir, grid, scene_source_copy_path)

    print(f"\n[{scene_index}/{scene_count}] Processing {tif_path.name}")
    print(f"  Grid: {grid.width} x {grid.height} | {grid.crs}")
    print(f"  Layer order: {LAYER_ORDER}")

    missing_layers = [
        layer_name
        for layer_name in LAYER_ORDER
        if overwrite or not (tif_output_dir / f"{layer_name}.tif").exists()
    ]

    if not missing_layers:
        print("  All Earth Engine layers already exist.")
        return

    stacked_layers = build_stacked_layer_image(grid, missing_layers)
    print(f"  Downloading {len(missing_layers)} missing layer(s) in one batched Earth Engine pass")
    print(f"  Tile size: {tile_size} | concurrent tiles: {max_concurrent_tiles}")

    try:
        download_ee_layers(
            image=stacked_layers,
            grid=grid,
            out_dir=tif_output_dir,
            layer_names=missing_layers,
            session=session,
            read_timeout_seconds=read_timeout_seconds,
            tile_size=tile_size,
            overwrite=overwrite,
            max_concurrent_tiles=max_concurrent_tiles,
            progress_prefix="  Download",
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Batched layer download failed: {exc}") from exc

    if sleep_seconds > 0:
        time.sleep(sleep_seconds)


def run(
    *,
    tif_path: Path | None,
    tif_dir: Path,
    output_dir: Path,
    project_id: str,
    max_files: int | None,
    sleep_seconds: float,
    overwrite: bool,
    read_timeout_seconds: float,
    tile_size: int,
    max_concurrent_tiles: int,
) -> Path:
    ensure_createdataset_dirs()

    resolved_tif_path = tif_path.expanduser().resolve() if tif_path is not None else None
    resolved_tif_dir = tif_dir.expanduser().resolve()
    resolved_output_dir = output_dir.expanduser().resolve()

    if resolved_tif_path is not None:
        if not resolved_tif_path.exists():
            raise FileNotFoundError(f"TIFF file not found: {resolved_tif_path}")
        tif_paths = [resolved_tif_path]
        print(f"Input TIFF file: {resolved_tif_path}")
    else:
        if not resolved_tif_dir.exists():
            raise FileNotFoundError(f"TIFF folder not found: {resolved_tif_dir}")

        tif_paths = iter_tifs(resolved_tif_dir)
        if max_files is not None:
            tif_paths = tif_paths[:max_files]

        if not tif_paths:
            raise FileNotFoundError(f"No .tif or .tiff files found in: {resolved_tif_dir}")

        print(f"Input TIFF folder: {resolved_tif_dir}")
        print(f"Found {len(tif_paths)} TIFF file(s).")

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output layers folder: {resolved_output_dir}")

    init_ee(project_id)
    failures: list[tuple[str, str]] = []
    with requests.Session() as session:
        total_scenes = len(tif_paths)
        for scene_index, current_tif_path in enumerate(tif_paths, start=1):
            try:
                process_tif(
                    tif_path=current_tif_path,
                    output_dir=resolved_output_dir,
                    session=session,
                    overwrite=overwrite,
                    sleep_seconds=sleep_seconds,
                    read_timeout_seconds=read_timeout_seconds,
                    tile_size=tile_size,
                    max_concurrent_tiles=max_concurrent_tiles,
                    scene_index=scene_index,
                    scene_count=total_scenes,
                )
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                failures.append((current_tif_path.name, str(exc)))
                print(f"  Failed: {current_tif_path.name}")
                print(f"  Reason: {exc}")

    if failures:
        print("\nFinished with failures:")
        for tif_name, reason in failures:
            print(f"  - {tif_name}: {reason}")
        raise SystemExit(1)

    print("\nFinished successfully.")
    return resolved_output_dir


def run_from_args(args: argparse.Namespace) -> Path:
    return run(
        tif_path=args.tif_path,
        tif_dir=args.tif_dir,
        output_dir=args.output_dir,
        project_id=args.project_id,
        max_files=args.max_files,
        sleep_seconds=args.sleep_seconds,
        overwrite=args.overwrite,
        read_timeout_seconds=args.read_timeout_seconds,
        tile_size=args.tile_size,
        max_concurrent_tiles=args.max_concurrent_tiles,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_from_args(args)


if __name__ == "__main__":
    main()
