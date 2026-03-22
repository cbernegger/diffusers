from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject, transform_bounds

try:
    import h5py
except ImportError:
    h5py = None

from paths import (
    BLACK_MARBLE_CACHE_DIR,
    LAYERS_DIR,
    WARPED_TIFS_DIR,
    ensure_createdataset_dirs,
)
from scene_assets import copy_source_tif_to_scene_dir

TOKEN_NEW = "eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6ImNiZXJuZWdnZXIiLCJleHAiOjE3NzcwMjM4OTYsImlhdCI6MTc3MTgzOTg5NiwiaXNzIjoiaHR0cHM6Ly91cnMuZWFydGhkYXRhLm5hc2EuZ292IiwiaWRlbnRpdHlfcHJvdmlkZXIiOiJlZGxfb3BzIiwiYWNyIjoiZWRsIiwiYXNzdXJhbmNlX2xldmVsIjozfQ.4tmXmupBqsqH8-2BGQA0Ad00OWJj1n-DjrDfcXJWSfpkSbEskV2aI_H3shakwh1-qHqzHToEFoZMI5ZcXNspZV-uqDETpXt7UNjj6JQvh6mJkL9b2DgOb2APbJoZGyQrhbyDxMweoQdNoa15QuduBtnx2vmZOW2bxx4xI-nmKp5J9DenjXbDRmI8S-WMDvjxpOfErCvTMB5cG8lqvIEMakB-VrdztjgKc6Zy-F_N1Ggh9ErS8DazhFHbxP5Vu1J5h6jgGLi6LG0fOfGCyixoP2M62Qw0D2l5OVQA1fFtCNEA6JWuXZQFMJBQp1nLH4XrE6IlhWoMi5PrEZfMMGUTEQ"
STEP_NAME = "black_marble_daily"
STANDARD_ARCHIVE_ROOT = "https://ladsweb.modaps.eosdis.nasa.gov/archive/allData"
NRT_ARCHIVE_ROOT = "https://nrt3.modaps.eosdis.nasa.gov/archive/allData"
DEFAULT_ARCHIVE_ROOT = STANDARD_ARCHIVE_ROOT
DEFAULT_COLLECTION = "5200"
DEFAULT_PRODUCT = "VNP46A2"
DEFAULT_VERSION = "002"
DEFAULT_FIELD_NAME = "DNB_BRDF-Corrected_NTL"
FALLBACK_FIELD_NAMES = (
    "DNB_BRDF-Corrected_NTL",
    "Gap_Filled_DNB_BRDF-Corrected_NTL",
)
TARGET_LAYER_NAME = "bm_daily"
BLACK_MARBLE_CRS = CRS.from_epsg(4326)
TILE_PIXEL_SIZE = 2400
TILE_DEGREES = 10.0
NRT_START_DATE = date(2025, 1, 1)


@dataclass(frozen=True)
class TileInfo:
    path: Path
    tile: str
    h: int
    v: int


@dataclass(frozen=True)
class DownloadSource:
    archive_root: str
    collection: str
    product: str
    version: str
    note: str | None = None


def require_h5py():
    if h5py is None:
        raise ModuleNotFoundError(
            "The Black Marble step requires `h5py`. Install it in your environment to run "
            "`black_marble_daily`."
        )
    return h5py


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and crop daily Black Marble data into CREATEDATASET/LAYERS."
    )
    parser.add_argument(
        "--tif-path",
        type=Path,
        default=None,
        help="Optional single warped TIFF to process.",
    )
    parser.add_argument(
        "--tif-dir",
        type=Path,
        default=WARPED_TIFS_DIR,
        help="Folder with warped TIFF inputs. Defaults to CREATEDATASET/WARPED_TIFS.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=LAYERS_DIR,
        help="Folder where layer outputs are written. Defaults to CREATEDATASET/LAYERS.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit for quick testing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing bm_daily layer TIFFs.",
    )
    parser.add_argument(
        "--laads-token",
        type=str,
        default=None,
        help="Optional LAADS / Earthdata bearer token. If omitted, env vars and TOKEN_NEW are checked.",
    )
    parser.add_argument(
        "--black-marble-cache-dir",
        type=Path,
        default=BLACK_MARBLE_CACHE_DIR,
        help="Cache directory for downloaded Black Marble HDF5 tiles.",
    )
    parser.add_argument(
        "--black-marble-archive-root",
        type=str,
        default=DEFAULT_ARCHIVE_ROOT,
        help="Base archive root for Black Marble downloads.",
    )
    parser.add_argument(
        "--black-marble-collection",
        type=str,
        default=DEFAULT_COLLECTION,
        help="Collection id for Black Marble downloads.",
    )
    parser.add_argument(
        "--black-marble-product",
        type=str,
        default=DEFAULT_PRODUCT,
        help="Black Marble product code.",
    )
    parser.add_argument(
        "--black-marble-version",
        type=str,
        default=DEFAULT_VERSION,
        help="Black Marble product version.",
    )
    parser.add_argument(
        "--black-marble-field",
        type=str,
        default=DEFAULT_FIELD_NAME,
        help="Preferred HDF5 data field name.",
    )
    parser.add_argument(
        "--black-marble-high-quality-only",
        action="store_true",
        help="Deprecated and ignored. Black Marble now uses all available pixels regardless of quality flags.",
    )
    parser.add_argument(
        "--black-marble-connect-timeout-seconds",
        type=float,
        default=30.0,
        help="HTTP connect timeout for LAADS downloads.",
    )
    parser.add_argument(
        "--black-marble-read-timeout-seconds",
        type=float,
        default=300.0,
        help="HTTP read timeout for LAADS downloads.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


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
    width = 24
    filled = min(width, math.floor((completed / total) * width))
    bar = "#" * filled + "-" * (width - filled)
    percent = (completed / total) * 100.0
    elapsed = format_duration(time.perf_counter() - start_time)
    return f"{prefix} [{bar}] {completed}/{total} ({percent:5.1f}%) elapsed {elapsed}"


def print_progress_line(message: str, *, done: bool) -> None:
    print(message.ljust(120), end="\n" if done else "\r", flush=True)


def resolve_laads_token(explicit_token: str | None) -> str:
    if explicit_token:
        return explicit_token.strip()

    for env_name in ("LAADS_TOKEN", "EARTHDATA_TOKEN"):
        env_value = os.environ.get(env_name)
        if env_value:
            return env_value.strip()

    if TOKEN_NEW.strip():
        return TOKEN_NEW.strip()

    raise RuntimeError(
        "No LAADS token found. Pass --laads-token, set LAADS_TOKEN / EARTHDATA_TOKEN, "
        "or set TOKEN_NEW in CREATEDATASET/black_marble_daily.py."
    )


def parse_scene_date(scene_name: str) -> date:
    match_14 = re.search(r"_(\d{14})_", scene_name)
    if match_14 is not None:
        return datetime.strptime(match_14.group(1)[:8], "%Y%m%d").date()

    match_8 = re.search(r"(\d{8})", scene_name)
    if match_8 is not None:
        return datetime.strptime(match_8.group(1), "%Y%m%d").date()

    raise ValueError(f"Could not extract acquisition date from scene name `{scene_name}`")


def scene_bbox_lonlat(tif_path: Path) -> tuple[float, float, float, float]:
    with rasterio.open(tif_path) as src:
        if src.crs is None:
            raise ValueError(f"Warped TIFF has no CRS: {tif_path}")

        west, south, east, north = transform_bounds(
            src.crs,
            BLACK_MARBLE_CRS,
            *src.bounds,
            densify_pts=21,
        )
        return float(west), float(south), float(east), float(north)


def tiles_for_bbox(west: float, south: float, east: float, north: float) -> list[str]:
    eps = 1e-9
    h_min = int((west + 180.0) // TILE_DEGREES)
    h_max = int(((east - eps) + 180.0) // TILE_DEGREES)
    v_min = int((90.0 - north) // TILE_DEGREES)
    v_max = int((90.0 - (south + eps)) // TILE_DEGREES)

    tiles: list[str] = []
    for v in range(v_min, v_max + 1):
        for h in range(h_min, h_max + 1):
            tiles.append(f"h{h:02d}v{v:02d}")
    return tiles


def parse_tile(tile_or_filename: str) -> tuple[int, int, str]:
    match = re.search(r"(h\d{2}v\d{2})", tile_or_filename)
    if match is None:
        raise ValueError(f"Could not parse tile id from `{tile_or_filename}`")

    tile = match.group(1)
    h = int(tile[1:3])
    v = int(tile[4:6])
    return h, v, tile


def tile_filename(
    acquisition_date: date,
    tile: str,
    *,
    product: str,
    version: str,
) -> str:
    doy = acquisition_date.timetuple().tm_yday
    return f"{product}.A{acquisition_date.year}{doy:03d}.{tile}.{version}.h5"


def build_day_directory_url(
    acquisition_date: date,
    *,
    archive_root: str,
    collection: str,
    product: str,
) -> str:
    doy = acquisition_date.timetuple().tm_yday
    return (
        f"{archive_root.rstrip('/')}/{collection}/{product}/"
        f"{acquisition_date.year}/{doy:03d}/"
    )


def tile_url(
    acquisition_date: date,
    filename: str,
    *,
    archive_root: str,
    collection: str,
    product: str,
) -> str:
    return (
        f"{build_day_directory_url(acquisition_date, archive_root=archive_root, collection=collection, product=product)}"
        f"{filename}"
    )


def resolve_download_source(
    acquisition_date: date,
    *,
    archive_root: str,
    collection: str,
    product: str,
    version: str,
) -> DownloadSource:
    normalized_product = product.strip()
    if normalized_product.upper() == "VNP46A2_NRT" and acquisition_date < NRT_START_DATE:
        return DownloadSource(
            archive_root=STANDARD_ARCHIVE_ROOT,
            collection="5200",
            product="VNP46A2",
            version="002",
            note=(
                f"{normalized_product} is a near-real-time product and is not suitable for "
                f"historical scenes from {acquisition_date.isoformat()}; switching to VNP46A2."
            ),
        )

    return DownloadSource(
        archive_root=archive_root,
        collection=collection,
        product=normalized_product,
        version=version,
    )


def find_cached_tile(
    cache_day_dir: Path,
    acquisition_date: date,
    tile: str,
    *,
    product: str,
    version: str,
) -> Path | None:
    day_code = f"A{acquisition_date.year}{acquisition_date.timetuple().tm_yday:03d}"
    patterns = (
        f"{product}.{day_code}.{tile}.{version}.h5",
        f"{product}.{day_code}.{tile}.{version}.*.h5",
        f"{product}.{day_code}.{tile}.*.h5",
    )
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(sorted(cache_day_dir.glob(pattern)))
    if not matches:
        return None
    return sorted(set(matches))[-1]


def list_day_filenames(
    *,
    session: requests.Session,
    acquisition_date: date,
    archive_root: str,
    collection: str,
    product: str,
    connect_timeout_seconds: float,
    read_timeout_seconds: float,
) -> list[str]:
    directory_url = build_day_directory_url(
        acquisition_date,
        archive_root=archive_root,
        collection=collection,
        product=product,
    )
    response = session.get(
        directory_url,
        timeout=(connect_timeout_seconds, read_timeout_seconds),
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"Could not list Black Marble day directory: {directory_url} ({exc})"
        ) from exc

    matches = set(
        re.findall(
            rf"({re.escape(product)}\.A\d{{7}}\.h\d{{2}}v\d{{2}}\.\d{{3}}(?:\.\d+)?\.h5)",
            response.text,
        )
    )
    if not matches:
        raise RuntimeError(
            f"No HDF5 files were listed in Black Marble directory: {directory_url}"
        )
    return sorted(matches)


def resolve_remote_tile_filename(
    filenames: list[str],
    *,
    acquisition_date: date,
    tile: str,
    product: str,
    version: str,
) -> str:
    day_code = f"A{acquisition_date.year}{acquisition_date.timetuple().tm_yday:03d}"
    prefix = f"{product}.{day_code}.{tile}."
    candidates = [name for name in filenames if name.startswith(prefix) and name.endswith(".h5")]
    if not candidates:
        raise RuntimeError(
            f"No Black Marble file was listed for tile `{tile}` on {acquisition_date.isoformat()} "
            f"with product `{product}`."
        )

    preferred = [
        name
        for name in candidates
        if name.endswith(f".{version}.h5") or f".{version}." in name
    ]
    selected_pool = preferred or candidates
    return sorted(selected_pool)[-1]


def download_tiles(
    *,
    token: str,
    acquisition_date: date,
    tiles: list[str],
    cache_dir: Path,
    archive_root: str,
    collection: str,
    product: str,
    version: str,
    connect_timeout_seconds: float,
    read_timeout_seconds: float,
    progress_prefix: str,
) -> list[TileInfo]:
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}"})

    total = len(tiles)
    start_time = time.perf_counter()
    downloaded_tiles: list[TileInfo] = []
    cache_day_dir = cache_dir / product / str(acquisition_date.year) / f"{acquisition_date.timetuple().tm_yday:03d}"
    cache_day_dir.mkdir(parents=True, exist_ok=True)
    day_filenames: list[str] | None = None

    for index, tile in enumerate(tiles, start=1):
        h, v, tile_code = parse_tile(tile)
        cached_tile = find_cached_tile(
            cache_day_dir,
            acquisition_date,
            tile_code,
            product=product,
            version=version,
        )
        if cached_tile is not None:
            destination = cached_tile
            filename = destination.name
        else:
            if day_filenames is None:
                day_filenames = list_day_filenames(
                    session=session,
                    acquisition_date=acquisition_date,
                    archive_root=archive_root,
                    collection=collection,
                    product=product,
                    connect_timeout_seconds=connect_timeout_seconds,
                    read_timeout_seconds=read_timeout_seconds,
                )

            filename = resolve_remote_tile_filename(
                day_filenames,
                acquisition_date=acquisition_date,
                tile=tile_code,
                product=product,
                version=version,
            )
            destination = cache_day_dir / filename

        if not destination.exists():
            url = tile_url(
                acquisition_date,
                filename,
                archive_root=archive_root,
                collection=collection,
                product=product,
            )
            tmp_path = destination.with_suffix(destination.suffix + ".part")
            response = session.get(
                url,
                stream=True,
                timeout=(connect_timeout_seconds, read_timeout_seconds),
            )
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                detail = response.text.strip()
                if detail:
                    detail = " ".join(detail.split())
                    detail = detail[:300]
                else:
                    detail = "<empty response body>"
                raise RuntimeError(
                    f"Black Marble tile download failed for `{filename}`: {exc}. "
                    f"Response: {detail}"
                ) from exc

            with tmp_path.open("wb") as fh:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        fh.write(chunk)
            tmp_path.replace(destination)

        downloaded_tiles.append(TileInfo(path=destination, tile=tile_code, h=h, v=v))
        print_progress_line(
            render_progress_line(
                prefix=progress_prefix,
                completed=index,
                total=total,
                start_time=start_time,
            ),
            done=index == total,
        )

    return downloaded_tiles


def find_data_fields_group(h5_file: h5py.File) -> h5py.Group:
    h5 = require_h5py()
    direct = "HDFEOS/GRIDS/VIIRS_Grid_DNB_2d/Data Fields"
    if direct in h5_file:
        return h5_file[direct]

    found: list[str] = []

    def visitor(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        if isinstance(obj, h5.Group) and name.endswith("Data Fields"):
            found.append(name)

    h5_file.visititems(visitor)
    if not found:
        raise KeyError("Could not find `Data Fields` group in the HDF5 tile.")
    return h5_file[found[0]]


def resolve_field_name(data_fields: h5py.Group, preferred_field: str) -> str:
    if preferred_field in data_fields:
        return preferred_field

    for fallback_name in FALLBACK_FIELD_NAMES:
        if fallback_name in data_fields:
            return fallback_name

    raise KeyError(
        f"Preferred field `{preferred_field}` was not found and no fallback field was available. "
        f"Available fields: {list(data_fields.keys())}"
    )


def read_scaled_dataset(dataset: h5py.Dataset) -> tuple[np.ndarray, str]:
    data = dataset[:].astype("float32")

    fill_value = dataset.attrs.get("_FillValue")
    if fill_value is not None:
        fill_value = np.array(fill_value).item()
        data[data == fill_value] = np.nan

    scale = float(np.atleast_1d(dataset.attrs.get("scale_factor", [1.0]))[0])
    offset = float(np.atleast_1d(dataset.attrs.get("offset", [0.0]))[0])
    data = data * scale + offset

    units = dataset.attrs.get("units", b"")
    if isinstance(units, (bytes, np.bytes_)):
        units = units.decode("utf-8", errors="ignore").strip()
    else:
        units = str(units)

    return data, units


def read_tile_data(
    tile_info: TileInfo,
    *,
    preferred_field: str,
    high_quality_only: bool,
) -> tuple[np.ndarray, str, str]:
    h5 = require_h5py()
    with h5.File(tile_info.path, "r") as h5_file:
        data_fields = find_data_fields_group(h5_file)
        primary_field_name = resolve_field_name(data_fields, preferred_field)
        data, units = read_scaled_dataset(data_fields[primary_field_name])
        resolved_field_name = primary_field_name

        gap_fill_field_name = "Gap_Filled_DNB_BRDF-Corrected_NTL"
        if primary_field_name != gap_fill_field_name and gap_fill_field_name in data_fields:
            gap_filled_data, _ = read_scaled_dataset(data_fields[gap_fill_field_name])
            missing_mask = ~np.isfinite(data)
            if np.any(missing_mask):
                data[missing_mask] = gap_filled_data[missing_mask]
                resolved_field_name = f"{primary_field_name} + {gap_fill_field_name} fallback"

    return data, units, resolved_field_name


def build_mosaic(
    tile_infos: list[TileInfo],
    *,
    preferred_field: str,
    high_quality_only: bool,
) -> tuple[np.ndarray, list[float], list[str], str, str]:
    if not tile_infos:
        raise ValueError("At least one Black Marble tile is required to build a mosaic.")

    arrays_by_position: dict[tuple[int, int], np.ndarray] = {}
    hs = sorted({tile_info.h for tile_info in tile_infos})
    vs = sorted({tile_info.v for tile_info in tile_infos})
    resolved_files: list[str] = []
    units = ""
    resolved_field_name = preferred_field

    for tile_info in tile_infos:
        array, tile_units, tile_field_name = read_tile_data(
            tile_info,
            preferred_field=preferred_field,
            high_quality_only=high_quality_only,
        )
        arrays_by_position[(tile_info.h, tile_info.v)] = array
        units = units or tile_units
        resolved_field_name = tile_field_name
        resolved_files.append(tile_info.path.name)

    empty_tile = np.full((TILE_PIXEL_SIZE, TILE_PIXEL_SIZE), np.nan, dtype="float32")
    rows: list[np.ndarray] = []
    for v in vs:
        row = [arrays_by_position.get((h, v), empty_tile) for h in hs]
        rows.append(np.hstack(row))
    mosaic = np.vstack(rows).astype("float32")

    extent = [
        -180.0 + TILE_DEGREES * min(hs),
        -180.0 + TILE_DEGREES * (max(hs) + 1),
        90.0 - TILE_DEGREES * (max(vs) + 1),
        90.0 - TILE_DEGREES * min(vs),
    ]
    return mosaic, extent, resolved_files, units, resolved_field_name


def crop_mosaic_to_bbox(
    mosaic: np.ndarray,
    extent: list[float],
    bbox: tuple[float, float, float, float],
) -> tuple[np.ndarray, list[float]]:
    west0, east0, south0, north0 = extent
    west, south, east, north = bbox

    nrows, ncols = mosaic.shape
    dx = (east0 - west0) / ncols
    dy = (north0 - south0) / nrows

    col_min = int(np.floor((west - west0) / dx))
    col_max = int(np.ceil((east - west0) / dx))
    row_min = int(np.floor((north0 - north) / dy))
    row_max = int(np.ceil((north0 - south) / dy))

    col_min = max(0, min(ncols - 1, col_min))
    col_max = max(1, min(ncols, col_max))
    row_min = max(0, min(nrows - 1, row_min))
    row_max = max(1, min(nrows, row_max))

    if col_max <= col_min or row_max <= row_min:
        raise ValueError("Black Marble crop bbox produced an empty slice.")

    crop = mosaic[row_min:row_max, col_min:col_max]

    west2 = west0 + col_min * dx
    east2 = west0 + col_max * dx
    north2 = north0 - row_min * dy
    south2 = north0 - row_max * dy

    return crop, [west2, east2, south2, north2]


def reproject_crop_to_scene(
    crop: np.ndarray,
    crop_extent: list[float],
    scene_tif_path: Path,
) -> tuple[np.ndarray, rasterio.profiles.Profile]:
    crop_transform = from_bounds(
        crop_extent[0],
        crop_extent[2],
        crop_extent[1],
        crop_extent[3],
        crop.shape[1],
        crop.shape[0],
    )

    with rasterio.open(scene_tif_path) as src:
        if src.crs is None:
            raise ValueError(f"Warped TIFF has no CRS: {scene_tif_path}")

        destination = np.full((src.height, src.width), np.nan, dtype="float32")
        reproject(
            source=crop,
            destination=destination,
            src_transform=crop_transform,
            src_crs=BLACK_MARBLE_CRS,
            src_nodata=np.nan,
            dst_transform=src.transform,
            dst_crs=src.crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            dtype="float32",
            count=1,
            nodata=np.nan,
            compress="deflate",
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )

    return destination, profile


def write_bm_daily_layer(
    out_path: Path,
    array: np.ndarray,
    profile: rasterio.profiles.Profile,
    *,
    acquisition_date: date,
    product: str,
    field_name: str,
    units: str,
) -> None:
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    with rasterio.open(tmp_path, "w", **profile) as dst:
        dst.write(array.astype("float32"), 1)
        dst.set_band_description(1, TARGET_LAYER_NAME)
        dst.update_tags(
            acquisition_date=acquisition_date.isoformat(),
            product=product,
            field_name=field_name,
            units=units,
            layer_name=TARGET_LAYER_NAME,
        )
    tmp_path.replace(out_path)


def update_scene_metadata(
    scene_dir: Path,
    *,
    scene_tif_path: Path,
    scene_source_copy_path: Path,
    acquisition_date: date,
    product: str,
    collection: str,
    version: str,
    archive_root: str,
    field_name: str,
    units: str,
    source_files: list[str],
) -> None:
    metadata_path = scene_dir / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        metadata = {
            "scene_id": scene_dir.name,
            "source_warped_tif": str(scene_tif_path.resolve()),
            "source_scene_tif_copy": str(scene_source_copy_path.resolve()),
            "layers": [],
        }

    metadata["source_warped_tif"] = str(scene_tif_path.resolve())
    metadata["source_scene_tif_copy"] = str(scene_source_copy_path.resolve())
    layers = list(metadata.get("layers", []))
    if TARGET_LAYER_NAME not in layers:
        layers.append(TARGET_LAYER_NAME)
    metadata["layers"] = layers
    metadata["black_marble_daily"] = {
        "acquisition_date": acquisition_date.isoformat(),
        "product": product,
        "collection": collection,
        "version": version,
        "archive_root": archive_root,
        "field_name": field_name,
        "units": units,
        "source_files": source_files,
    }

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def process_tif(
    *,
    tif_path: Path,
    output_dir: Path,
    cache_dir: Path,
    token: str,
    archive_root: str,
    collection: str,
    product: str,
    version: str,
    preferred_field: str,
    high_quality_only: bool,
    overwrite: bool,
    connect_timeout_seconds: float,
    read_timeout_seconds: float,
    scene_index: int,
    scene_count: int,
) -> None:
    scene_dir = output_dir / tif_path.stem
    scene_dir.mkdir(parents=True, exist_ok=True)
    scene_source_copy_path = copy_source_tif_to_scene_dir(
        scene_dir,
        tif_path,
        overwrite=overwrite,
    )
    out_path = scene_dir / f"{TARGET_LAYER_NAME}.tif"

    if out_path.exists() and not overwrite:
        print(f"\n[{scene_index}/{scene_count}] {tif_path.name}")
        print(f"  Skipping existing layer: {out_path.name}")
        return

    acquisition_date = parse_scene_date(tif_path.stem)
    source = resolve_download_source(
        acquisition_date,
        archive_root=archive_root,
        collection=collection,
        product=product,
        version=version,
    )
    bbox = scene_bbox_lonlat(tif_path)
    tiles = tiles_for_bbox(*bbox)

    print(f"\n[{scene_index}/{scene_count}] {tif_path.name}")
    print(f"  Daily Black Marble date: {acquisition_date.isoformat()}")
    if source.note:
        print(f"  Source note: {source.note}")
    print(
        f"  Download source: {source.product} | collection {source.collection} | version {source.version}"
    )
    print(f"  Needed tiles: {', '.join(tiles)}")

    tile_infos = download_tiles(
        token=token,
        acquisition_date=acquisition_date,
        tiles=tiles,
        cache_dir=cache_dir,
        archive_root=source.archive_root,
        collection=source.collection,
        product=source.product,
        version=source.version,
        connect_timeout_seconds=connect_timeout_seconds,
        read_timeout_seconds=read_timeout_seconds,
        progress_prefix="  Tiles",
    )

    mosaic, extent, source_files, units, resolved_field_name = build_mosaic(
        tile_infos,
        preferred_field=preferred_field,
        high_quality_only=high_quality_only,
    )
    crop, crop_extent = crop_mosaic_to_bbox(mosaic, extent, bbox)
    warped_layer, profile = reproject_crop_to_scene(crop, crop_extent, tif_path)

    write_bm_daily_layer(
        out_path,
        warped_layer,
        profile,
        acquisition_date=acquisition_date,
        product=source.product,
        field_name=resolved_field_name,
        units=units,
    )
    update_scene_metadata(
        scene_dir,
        scene_tif_path=tif_path,
        scene_source_copy_path=scene_source_copy_path,
        acquisition_date=acquisition_date,
        product=source.product,
        collection=source.collection,
        version=source.version,
        archive_root=source.archive_root,
        field_name=resolved_field_name,
        units=units,
        source_files=source_files,
    )

    print(f"  Wrote layer: {out_path}")
    if resolved_field_name != preferred_field:
        print(f"  Requested field `{preferred_field}` was adjusted to `{resolved_field_name}`.")


def run(
    *,
    tif_path: Path | None,
    tif_dir: Path,
    output_dir: Path,
    max_files: int | None,
    overwrite: bool,
    laads_token: str | None,
    black_marble_cache_dir: Path,
    black_marble_archive_root: str,
    black_marble_collection: str,
    black_marble_product: str,
    black_marble_version: str,
    black_marble_field: str,
    black_marble_high_quality_only: bool,
    black_marble_connect_timeout_seconds: float,
    black_marble_read_timeout_seconds: float,
) -> Path:
    ensure_createdataset_dirs()

    resolved_tif_path = tif_path.expanduser().resolve() if tif_path is not None else None
    resolved_tif_dir = tif_dir.expanduser().resolve()
    resolved_output_dir = output_dir.expanduser().resolve()
    resolved_cache_dir = black_marble_cache_dir.expanduser().resolve()

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
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)
    token = resolve_laads_token(laads_token)
    if black_marble_high_quality_only:
        print("Ignoring --black-marble-high-quality-only; using all Black Marble pixels regardless of quality flags.")

    failures: list[tuple[str, str]] = []
    for scene_index, current_tif_path in enumerate(tif_paths, start=1):
        try:
            process_tif(
                tif_path=current_tif_path,
                output_dir=resolved_output_dir,
                cache_dir=resolved_cache_dir,
                token=token,
                archive_root=black_marble_archive_root,
                collection=black_marble_collection,
                product=black_marble_product,
                version=black_marble_version,
                preferred_field=black_marble_field,
                high_quality_only=False,
                overwrite=overwrite,
                connect_timeout_seconds=black_marble_connect_timeout_seconds,
                read_timeout_seconds=black_marble_read_timeout_seconds,
                scene_index=scene_index,
                scene_count=len(tif_paths),
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
        max_files=args.max_files,
        overwrite=args.overwrite,
        laads_token=args.laads_token,
        black_marble_cache_dir=args.black_marble_cache_dir,
        black_marble_archive_root=args.black_marble_archive_root,
        black_marble_collection=args.black_marble_collection,
        black_marble_product=args.black_marble_product,
        black_marble_version=args.black_marble_version,
        black_marble_field=args.black_marble_field,
        black_marble_high_quality_only=args.black_marble_high_quality_only,
        black_marble_connect_timeout_seconds=args.black_marble_connect_timeout_seconds,
        black_marble_read_timeout_seconds=args.black_marble_read_timeout_seconds,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_from_args(args)


if __name__ == "__main__":
    main()
