from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.features import rasterize
from rasterio.warp import transform_bounds, transform_geom

from paths import LAYERS_DIR, WARPED_TIFS_DIR, ensure_createdataset_dirs
from scene_assets import copy_source_tif_to_scene_dir

STEP_NAME = "osm"
TARGET_LAYER_NAME = "osm_roads"
DEFAULT_TIF_DIR = WARPED_TIFS_DIR
DEFAULT_OUTPUT_DIR = LAYERS_DIR
DEFAULT_OVERPASS_URLS = (
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass-api.de/api/interpreter",
)
DEFAULT_OVERPASS_URL = DEFAULT_OVERPASS_URLS[0]
DEFAULT_HIGHWAY_PATTERN = (
    "motorway|motorway_link|trunk|trunk_link|primary|primary_link|"
    "secondary|secondary_link|tertiary|tertiary_link|unclassified|"
    "residential|living_street|road"
)
DEFAULT_CONNECT_TIMEOUT_SECONDS = 30.0
DEFAULT_READ_TIMEOUT_SECONDS = 180.0
DEFAULT_MAX_BBOX_DEGREES = 0.5
DEFAULT_REQUEST_PAUSE_SECONDS = 0.0
DEFAULT_RETRY_ATTEMPTS = 4
DEFAULT_MAX_CONCURRENT_REQUESTS = min(3, len(DEFAULT_OVERPASS_URLS))
DEFAULT_MAX_CONCURRENT_SCENES = 1
THREAD_LOCAL = threading.local()
PRINT_LOCK = threading.Lock()


@dataclass(frozen=True)
class RasterGrid:
    path: Path
    width: int
    height: int
    crs: str
    transform: rasterio.Affine
    bounds: tuple[float, float, float, float]
    nodata: float | int | None

    @property
    def stem(self) -> str:
        return self.path.stem


@dataclass(frozen=True)
class OverpassTileResult:
    tile_index: int
    bbox_lonlat: tuple[float, float, float, float]
    endpoint_url: str
    elements: list[dict]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and rasterize OpenStreetMap roads into CREATEDATASET/LAYERS."
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
        default=DEFAULT_TIF_DIR,
        help="Folder with warped TIFF inputs. Defaults to CREATEDATASET/WARPED_TIFS.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
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
        help="Overwrite existing osm_roads layer TIFFs.",
    )
    parser.add_argument(
        "--osm-overpass-url",
        type=str,
        default=None,
        help="Optional single Overpass API interpreter endpoint override.",
    )
    parser.add_argument(
        "--osm-overpass-urls",
        nargs="+",
        default=None,
        help="Optional list of Overpass API interpreter endpoints to use in parallel.",
    )
    parser.add_argument(
        "--osm-highway-pattern",
        type=str,
        default=DEFAULT_HIGHWAY_PATTERN,
        help="Regex of OSM highway tag values to include.",
    )
    parser.add_argument(
        "--osm-connect-timeout-seconds",
        type=float,
        default=DEFAULT_CONNECT_TIMEOUT_SECONDS,
        help="HTTP connect timeout for Overpass requests.",
    )
    parser.add_argument(
        "--osm-read-timeout-seconds",
        type=float,
        default=DEFAULT_READ_TIMEOUT_SECONDS,
        help="HTTP read timeout for Overpass requests.",
    )
    parser.add_argument(
        "--osm-max-bbox-degrees",
        type=float,
        default=DEFAULT_MAX_BBOX_DEGREES,
        help="Maximum lon/lat tile size per Overpass request for large scenes.",
    )
    parser.add_argument(
        "--osm-request-pause-seconds",
        type=float,
        default=DEFAULT_REQUEST_PAUSE_SECONDS,
        help="Pause between Overpass tile requests to reduce rate limiting.",
    )
    parser.add_argument(
        "--osm-retry-attempts",
        type=int,
        default=DEFAULT_RETRY_ATTEMPTS,
        help="Retry attempts per Overpass tile request on transient failures.",
    )
    parser.add_argument(
        "--osm-max-concurrent-requests",
        type=int,
        default=DEFAULT_MAX_CONCURRENT_REQUESTS,
        help="Maximum number of OSM tile requests to run in parallel.",
    )
    parser.add_argument(
        "--osm-max-concurrent-scenes",
        type=int,
        default=DEFAULT_MAX_CONCURRENT_SCENES,
        help="Maximum number of TIFF scenes to process in parallel for the OSM step.",
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
    suffix: str = "",
) -> str:
    total = max(total, 1)
    width = 24
    filled = min(width, math.floor((completed / total) * width))
    bar = "#" * filled + "-" * (width - filled)
    percent = (completed / total) * 100.0
    elapsed = format_duration(time.perf_counter() - start_time)
    suffix_text = f" | {suffix}" if suffix else ""
    return f"{prefix} [{bar}] {completed}/{total} ({percent:5.1f}%) elapsed {elapsed}{suffix_text}"


def safe_print(message: str = "") -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def print_progress_line(message: str, *, done: bool, interactive: bool) -> None:
    with PRINT_LOCK:
        print(
            message.ljust(120),
            end="\n" if done or not interactive else "\r",
            flush=True,
        )


def should_emit_progress_update(completed: int, total: int) -> bool:
    if completed <= 0:
        return False
    if completed == 1 or completed == total:
        return True
    step = max(1, total // 10)
    return completed % step == 0


def get_thread_session() -> requests.Session:
    session = getattr(THREAD_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": "CREATEDATASET-OSM/1.0"})
        THREAD_LOCAL.session = session
    return session


def normalize_overpass_urls(values: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        for part in value.split(","):
            candidate = part.strip()
            if not candidate or candidate in seen:
                continue
            normalized.append(candidate)
            seen.add(candidate)
    return normalized


def resolve_overpass_urls(
    explicit_urls: list[str] | None,
    explicit_url: str | None,
) -> list[str]:
    if explicit_urls:
        normalized = normalize_overpass_urls(explicit_urls)
        if normalized:
            return normalized

    if explicit_url:
        normalized = normalize_overpass_urls([explicit_url])
        if normalized:
            return normalized

    return list(DEFAULT_OVERPASS_URLS)


def read_raster_grid(tif_path: Path) -> RasterGrid:
    with rasterio.open(tif_path) as src:
        if src.crs is None:
            raise ValueError(f"Raster has no CRS: {tif_path}")

        return RasterGrid(
            path=tif_path,
            width=src.width,
            height=src.height,
            crs=src.crs.to_string(),
            transform=src.transform,
            bounds=(
                float(src.bounds.left),
                float(src.bounds.bottom),
                float(src.bounds.right),
                float(src.bounds.top),
            ),
            nodata=src.nodata,
        )


def clamp_lonlat_bounds(bounds: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    west, south, east, north = bounds
    west = max(-180.0, min(180.0, west))
    east = max(-180.0, min(180.0, east))
    south = max(-90.0, min(90.0, south))
    north = max(-90.0, min(90.0, north))
    return west, south, east, north


def scene_bbox_lonlat(grid: RasterGrid) -> tuple[float, float, float, float]:
    bounds = transform_bounds(
        grid.crs,
        "EPSG:4326",
        *grid.bounds,
        densify_pts=21,
    )
    return clamp_lonlat_bounds(tuple(float(v) for v in bounds))


def build_overpass_query(
    *,
    bbox_lonlat: tuple[float, float, float, float],
    highway_pattern: str,
) -> str:
    west, south, east, north = bbox_lonlat
    return (
        "[out:json][timeout:180];"
        "("
        f'way["highway"~"^({highway_pattern})$"]({south:.8f},{west:.8f},{north:.8f},{east:.8f});'
        ");"
        "out geom qt;"
    )


def iter_bbox_tiles(
    bbox_lonlat: tuple[float, float, float, float],
    max_bbox_degrees: float,
) -> list[tuple[float, float, float, float]]:
    if max_bbox_degrees <= 0:
        raise ValueError(f"osm_max_bbox_degrees must be positive, got {max_bbox_degrees}")

    west, south, east, north = bbox_lonlat
    tiles: list[tuple[float, float, float, float]] = []

    current_west = west
    while current_west < east:
        current_east = min(current_west + max_bbox_degrees, east)
        current_south = south
        while current_south < north:
            current_north = min(current_south + max_bbox_degrees, north)
            tiles.append((current_west, current_south, current_east, current_north))
            current_south = current_north
        current_west = current_east

    return tiles


def fetch_overpass_elements(
    *,
    session: requests.Session | None,
    overpass_url: str,
    bbox_lonlat: tuple[float, float, float, float],
    highway_pattern: str,
    connect_timeout_seconds: float,
    read_timeout_seconds: float,
    request_pause_seconds: float,
    retry_attempts: int,
) -> list[dict]:
    retry_attempts = max(retry_attempts, 1)
    query = build_overpass_query(
        bbox_lonlat=bbox_lonlat,
        highway_pattern=highway_pattern,
    )
    local_session = session if session is not None else get_thread_session()

    last_error: Exception | None = None
    for attempt in range(1, retry_attempts + 1):
        try:
            response = local_session.post(
                overpass_url,
                data={"data": query},
                timeout=(connect_timeout_seconds, read_timeout_seconds),
            )

            if response.status_code in {429, 502, 503, 504} and attempt < retry_attempts:
                retry_after = response.headers.get("Retry-After")
                if retry_after is not None:
                    try:
                        wait_seconds = float(retry_after)
                    except ValueError:
                        wait_seconds = request_pause_seconds * (2 ** (attempt - 1))
                else:
                    wait_seconds = request_pause_seconds * (2 ** (attempt - 1))
                time.sleep(max(wait_seconds, 0.0))
                continue

            response.raise_for_status()
            payload = response.json()
            elements = payload.get("elements")
            if not isinstance(elements, list):
                raise RuntimeError("Overpass response did not include an `elements` list.")

            return elements
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt >= retry_attempts:
                break
            time.sleep(max(request_pause_seconds * (2 ** (attempt - 1)), 0.0))

    if isinstance(last_error, requests.HTTPError):
        response = last_error.response
        detail = ""
        if response is not None:
            detail = response.text.strip()
        if not detail:
            detail = "<empty response body>"
        raise RuntimeError(
            f"{last_error}. Overpass response: {detail}. "
            f"BBox={bbox_lonlat}"
        ) from last_error

    if last_error is None:
        raise RuntimeError(f"Overpass request failed without a captured error. BBox={bbox_lonlat}")

    raise RuntimeError(f"{last_error}. BBox={bbox_lonlat}") from last_error


def fetch_tiled_overpass_elements(
    *,
    session: requests.Session,
    overpass_urls: list[str],
    bbox_lonlat: tuple[float, float, float, float],
    highway_pattern: str,
    connect_timeout_seconds: float,
    read_timeout_seconds: float,
    max_bbox_degrees: float,
    request_pause_seconds: float,
    retry_attempts: int,
    max_concurrent_requests: int,
    progress_prefix: str,
    interactive_progress: bool,
) -> list[dict]:
    bbox_tiles = iter_bbox_tiles(bbox_lonlat, max_bbox_degrees)
    unique_ways: dict[int, dict] = {}
    fallback_ways: list[dict] = []
    start_time = time.perf_counter()
    endpoint_urls = overpass_urls or list(DEFAULT_OVERPASS_URLS)

    def fetch_tile_request(
        *,
        tile_index: int,
        bbox_tile: tuple[float, float, float, float],
        preferred_endpoint_index: int,
        shared_session: requests.Session | None,
    ) -> OverpassTileResult:
        endpoint_count = len(endpoint_urls)
        last_error: Exception | None = None

        for offset in range(endpoint_count):
            endpoint_index = (preferred_endpoint_index + offset) % endpoint_count
            endpoint_url = endpoint_urls[endpoint_index]
            request_session = shared_session if shared_session is not None else None
            try:
                elements = fetch_overpass_elements(
                    session=request_session,
                    overpass_url=endpoint_url,
                    bbox_lonlat=bbox_tile,
                    highway_pattern=highway_pattern,
                    connect_timeout_seconds=connect_timeout_seconds,
                    read_timeout_seconds=read_timeout_seconds,
                    request_pause_seconds=request_pause_seconds,
                    retry_attempts=retry_attempts,
                )
                if request_pause_seconds > 0:
                    time.sleep(request_pause_seconds)
                return OverpassTileResult(
                    tile_index=tile_index,
                    bbox_lonlat=bbox_tile,
                    endpoint_url=endpoint_url,
                    elements=elements,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc

        if last_error is None:
            raise RuntimeError(
                f"All Overpass endpoints failed for tile {tile_index + 1} without a captured error."
            )

        raise RuntimeError(
            f"All Overpass endpoints failed for tile {tile_index + 1} "
            f"bbox={bbox_tile}: {last_error}"
        ) from last_error

    print_progress_line(
        render_progress_line(
            prefix=progress_prefix,
            completed=0,
            total=len(bbox_tiles),
            start_time=start_time,
            suffix=f"unique ways 0 | endpoints {len(endpoint_urls)}",
        ),
        done=False,
        interactive=interactive_progress,
    )

    def absorb_elements(result: OverpassTileResult) -> None:
        for element in result.elements:
            if element.get("type") != "way":
                continue

            way_id = element.get("id")
            if isinstance(way_id, int):
                unique_ways[way_id] = element
            else:
                fallback_ways.append(element)

    if max_concurrent_requests <= 1 or len(bbox_tiles) <= 1 or len(endpoint_urls) <= 1:
        completed = 0
        for tile_index, bbox_tile in enumerate(bbox_tiles):
            result = fetch_tile_request(
                tile_index=tile_index,
                bbox_tile=bbox_tile,
                preferred_endpoint_index=tile_index % len(endpoint_urls),
                shared_session=session,
            )
            absorb_elements(result)
            completed += 1
            if interactive_progress or should_emit_progress_update(completed, len(bbox_tiles)):
                print_progress_line(
                    render_progress_line(
                        prefix=progress_prefix,
                        completed=completed,
                        total=len(bbox_tiles),
                        start_time=start_time,
                        suffix=(
                            f"unique ways {len(unique_ways) + len(fallback_ways)} "
                            f"| last {result.endpoint_url}"
                        ),
                    ),
                    done=completed == len(bbox_tiles),
                    interactive=interactive_progress,
                )
        return list(unique_ways.values()) + fallback_ways

    max_workers = min(max_concurrent_requests, len(endpoint_urls), len(bbox_tiles))
    tile_iter = iter(enumerate(bbox_tiles))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        active_futures: dict[Future[OverpassTileResult], int] = {}

        def submit_next(worker_index: int) -> None:
            next_tile = next(tile_iter, None)
            if next_tile is None:
                return
            tile_index, bbox_tile = next_tile
            future = executor.submit(
                fetch_tile_request,
                tile_index=tile_index,
                bbox_tile=bbox_tile,
                preferred_endpoint_index=worker_index % len(endpoint_urls),
                shared_session=None,
            )
            active_futures[future] = worker_index

        for worker_index in range(max_workers):
            submit_next(worker_index)

        completed = 0
        while active_futures:
            done_futures, _ = wait(active_futures, return_when=FIRST_COMPLETED)
            for future in done_futures:
                worker_index = active_futures.pop(future)
                result = future.result()
                absorb_elements(result)
                completed += 1
                if interactive_progress or should_emit_progress_update(completed, len(bbox_tiles)):
                    print_progress_line(
                        render_progress_line(
                            prefix=progress_prefix,
                            completed=completed,
                            total=len(bbox_tiles),
                            start_time=start_time,
                            suffix=(
                                f"unique ways {len(unique_ways) + len(fallback_ways)} "
                                f"| last {result.endpoint_url}"
                            ),
                        ),
                        done=completed == len(bbox_tiles),
                        interactive=interactive_progress,
                    )
                submit_next(worker_index)

    return list(unique_ways.values()) + fallback_ways


def overpass_elements_to_geometries(elements: list[dict], target_crs: str) -> list[dict]:
    geometries: list[dict] = []
    for element in elements:
        if element.get("type") != "way":
            continue

        coords = element.get("geometry")
        if not isinstance(coords, list) or len(coords) < 2:
            continue

        line_coordinates: list[tuple[float, float]] = []
        for coord in coords:
            lon = coord.get("lon")
            lat = coord.get("lat")
            if lon is None or lat is None:
                continue
            line_coordinates.append((float(lon), float(lat)))

        if len(line_coordinates) < 2:
            continue

        geometry = {
            "type": "LineString",
            "coordinates": line_coordinates,
        }

        try:
            transformed = transform_geom(
                "EPSG:4326",
                target_crs,
                geometry,
                precision=-1,
            )
        except Exception:
            continue

        if not transformed:
            continue

        geometries.append(transformed)

    return geometries


def rasterize_roads(grid: RasterGrid, road_geometries: list[dict]) -> np.ndarray:
    if not road_geometries:
        return np.zeros((grid.height, grid.width), dtype=np.uint8)

    return rasterize(
        ((geometry, 1) for geometry in road_geometries),
        out_shape=(grid.height, grid.width),
        transform=grid.transform,
        fill=0,
        dtype="uint8",
        all_touched=True,
    ).astype(np.uint8)


def write_osm_layer(out_path: Path, array: np.ndarray, grid: RasterGrid, *, overpass_url: str) -> None:
    profile = {
        "driver": "GTiff",
        "width": grid.width,
        "height": grid.height,
        "count": 1,
        "dtype": "uint8",
        "crs": grid.crs,
        "transform": grid.transform,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "nodata": 0,
    }

    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    with rasterio.open(tmp_path, "w", **profile) as dst:
        dst.write(array.astype("uint8"), 1)
        dst.set_band_description(1, TARGET_LAYER_NAME)
        dst.update_tags(
            layer_name=TARGET_LAYER_NAME,
            source="OpenStreetMap",
            overpass_url=overpass_url,
            units="binary_mask",
        )
    tmp_path.replace(out_path)


def update_scene_metadata(
    scene_dir: Path,
    *,
    scene_tif_path: Path,
    scene_source_copy_path: Path,
    bbox_lonlat: tuple[float, float, float, float],
    overpass_urls: list[str],
    highway_pattern: str,
    matched_way_count: int,
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
    metadata["osm"] = {
        "layer_name": TARGET_LAYER_NAME,
        "overpass_url": overpass_urls[0] if overpass_urls else None,
        "overpass_urls": overpass_urls,
        "highway_pattern": highway_pattern,
        "bbox_lonlat": list(bbox_lonlat),
        "matched_way_count": matched_way_count,
        "all_touched": True,
    }

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def process_tif(
    *,
    tif_path: Path,
    output_dir: Path,
    overwrite: bool,
    overpass_urls: list[str],
    highway_pattern: str,
    connect_timeout_seconds: float,
    read_timeout_seconds: float,
    max_bbox_degrees: float,
    request_pause_seconds: float,
    retry_attempts: int,
    max_concurrent_requests: int,
    interactive_progress: bool,
    scene_index: int,
    scene_count: int,
) -> None:
    grid = read_raster_grid(tif_path)
    scene_dir = output_dir / tif_path.stem
    scene_dir.mkdir(parents=True, exist_ok=True)
    scene_source_copy_path = copy_source_tif_to_scene_dir(
        scene_dir,
        tif_path,
        overwrite=overwrite,
    )
    out_path = scene_dir / f"{TARGET_LAYER_NAME}.tif"

    scene_label = f"[{scene_index}/{scene_count}]"
    progress_prefix = f"  {scene_label} OSM tiles"

    safe_print(f"\n{scene_label} Processing {tif_path.name}")
    safe_print(f"  Grid: {grid.width} x {grid.height} | {grid.crs}")

    if out_path.exists() and not overwrite:
        safe_print(f"  Skipping existing layer: {out_path.name}")
        return

    bbox_lonlat = scene_bbox_lonlat(grid)
    west, south, east, north = bbox_lonlat
    safe_print(
        "  OSM bbox (lon/lat): "
        f"west={west:.5f}, south={south:.5f}, east={east:.5f}, north={north:.5f}"
    )
    safe_print(
        "  Requesting OSM roads in tiles up to "
        f"{max_bbox_degrees:.2f} degrees | endpoints {len(overpass_urls)} | "
        f"parallel requests {max_concurrent_requests}"
    )

    with requests.Session() as session:
        session.headers.update({"User-Agent": "CREATEDATASET-OSM/1.0"})
        elements = fetch_tiled_overpass_elements(
            session=session,
            overpass_urls=overpass_urls,
            bbox_lonlat=bbox_lonlat,
            highway_pattern=highway_pattern,
            connect_timeout_seconds=connect_timeout_seconds,
            read_timeout_seconds=read_timeout_seconds,
            max_bbox_degrees=max_bbox_degrees,
            request_pause_seconds=request_pause_seconds,
            retry_attempts=retry_attempts,
            max_concurrent_requests=max_concurrent_requests,
            progress_prefix=progress_prefix,
            interactive_progress=interactive_progress,
        )
    road_geometries = overpass_elements_to_geometries(elements, grid.crs)
    road_mask = rasterize_roads(grid, road_geometries)

    write_osm_layer(out_path, road_mask, grid, overpass_url=overpass_urls[0] if overpass_urls else "")
    update_scene_metadata(
        scene_dir,
        scene_tif_path=tif_path,
        scene_source_copy_path=scene_source_copy_path,
        bbox_lonlat=bbox_lonlat,
        overpass_urls=overpass_urls,
        highway_pattern=highway_pattern,
        matched_way_count=len(road_geometries),
    )

    road_pixels = int(np.count_nonzero(road_mask))
    safe_print(f"  Matched OSM ways: {len(road_geometries)}")
    safe_print(f"  Road pixels burned: {road_pixels}")
    safe_print(f"  Wrote layer: {out_path}")


def run(
    *,
    tif_path: Path | None,
    tif_dir: Path,
    output_dir: Path,
    max_files: int | None,
    overwrite: bool,
    osm_overpass_url: str,
    osm_overpass_urls: list[str] | None,
    osm_highway_pattern: str,
    osm_connect_timeout_seconds: float,
    osm_read_timeout_seconds: float,
    osm_max_bbox_degrees: float,
    osm_request_pause_seconds: float,
    osm_retry_attempts: int,
    osm_max_concurrent_requests: int,
    osm_max_concurrent_scenes: int,
) -> Path:
    ensure_createdataset_dirs()

    resolved_tif_path = tif_path.expanduser().resolve() if tif_path is not None else None
    resolved_tif_dir = tif_dir.expanduser().resolve()
    resolved_output_dir = output_dir.expanduser().resolve()

    if resolved_tif_path is not None:
        if not resolved_tif_path.exists():
            raise FileNotFoundError(f"TIFF file not found: {resolved_tif_path}")
        tif_paths = [resolved_tif_path]
        safe_print(f"Input TIFF file: {resolved_tif_path}")
    else:
        if not resolved_tif_dir.exists():
            raise FileNotFoundError(f"TIFF folder not found: {resolved_tif_dir}")

        tif_paths = iter_tifs(resolved_tif_dir)
        if max_files is not None:
            tif_paths = tif_paths[:max_files]

        if not tif_paths:
            raise FileNotFoundError(f"No .tif or .tiff files found in: {resolved_tif_dir}")

        safe_print(f"Input TIFF folder: {resolved_tif_dir}")
        safe_print(f"Found {len(tif_paths)} TIFF file(s).")

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    resolved_overpass_urls = resolve_overpass_urls(osm_overpass_urls, osm_overpass_url)
    safe_print(f"Output layers folder: {resolved_output_dir}")
    safe_print(f"Overpass endpoints ({len(resolved_overpass_urls)}): {', '.join(resolved_overpass_urls)}")
    safe_print(f"OSM scene workers: {max(1, osm_max_concurrent_scenes)}")

    failures: list[tuple[str, str]] = []
    total_scenes = len(tif_paths)
    interactive_progress = total_scenes == 1 and max(1, osm_max_concurrent_scenes) == 1

    if max(1, osm_max_concurrent_scenes) <= 1 or total_scenes <= 1:
        for scene_index, current_tif_path in enumerate(tif_paths, start=1):
            try:
                process_tif(
                    tif_path=current_tif_path,
                    output_dir=resolved_output_dir,
                    overwrite=overwrite,
                    overpass_urls=resolved_overpass_urls,
                    highway_pattern=osm_highway_pattern,
                    connect_timeout_seconds=osm_connect_timeout_seconds,
                    read_timeout_seconds=osm_read_timeout_seconds,
                    max_bbox_degrees=osm_max_bbox_degrees,
                    request_pause_seconds=osm_request_pause_seconds,
                    retry_attempts=osm_retry_attempts,
                    max_concurrent_requests=osm_max_concurrent_requests,
                    interactive_progress=interactive_progress,
                    scene_index=scene_index,
                    scene_count=total_scenes,
                )
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                failures.append((current_tif_path.name, str(exc)))
                safe_print(f"  Failed: {current_tif_path.name}")
                safe_print(f"  Reason: {exc}")
    else:
        max_workers = min(max(1, osm_max_concurrent_scenes), total_scenes)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_tif: dict[Future[None], Path] = {}
            for scene_index, current_tif_path in enumerate(tif_paths, start=1):
                future = executor.submit(
                    process_tif,
                    tif_path=current_tif_path,
                    output_dir=resolved_output_dir,
                    overwrite=overwrite,
                    overpass_urls=resolved_overpass_urls,
                    highway_pattern=osm_highway_pattern,
                    connect_timeout_seconds=osm_connect_timeout_seconds,
                    read_timeout_seconds=osm_read_timeout_seconds,
                    max_bbox_degrees=osm_max_bbox_degrees,
                    request_pause_seconds=osm_request_pause_seconds,
                    retry_attempts=osm_retry_attempts,
                    max_concurrent_requests=osm_max_concurrent_requests,
                    interactive_progress=False,
                    scene_index=scene_index,
                    scene_count=total_scenes,
                )
                future_to_tif[future] = current_tif_path

            for future in as_completed(future_to_tif):
                current_tif_path = future_to_tif[future]
                try:
                    future.result()
                except KeyboardInterrupt:
                    raise
                except Exception as exc:  # noqa: BLE001
                    failures.append((current_tif_path.name, str(exc)))
                    safe_print(f"  Failed: {current_tif_path.name}")
                    safe_print(f"  Reason: {exc}")

    if failures:
        safe_print("\nFinished with failures:")
        for tif_name, reason in failures:
            safe_print(f"  - {tif_name}: {reason}")
        raise SystemExit(1)

    safe_print("\nFinished successfully.")
    return resolved_output_dir


def run_from_args(args: argparse.Namespace) -> Path:
    return run(
        tif_path=args.tif_path,
        tif_dir=args.tif_dir,
        output_dir=args.output_dir,
        max_files=args.max_files,
        overwrite=args.overwrite,
        osm_overpass_url=args.osm_overpass_url,
        osm_overpass_urls=args.osm_overpass_urls,
        osm_highway_pattern=args.osm_highway_pattern,
        osm_connect_timeout_seconds=args.osm_connect_timeout_seconds,
        osm_read_timeout_seconds=args.osm_read_timeout_seconds,
        osm_max_bbox_degrees=args.osm_max_bbox_degrees,
        osm_request_pause_seconds=args.osm_request_pause_seconds,
        osm_retry_attempts=args.osm_retry_attempts,
        osm_max_concurrent_requests=args.osm_max_concurrent_requests,
        osm_max_concurrent_scenes=args.osm_max_concurrent_scenes,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_from_args(args)


if __name__ == "__main__":
    main()
