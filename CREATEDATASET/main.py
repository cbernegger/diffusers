from __future__ import annotations

import argparse
from pathlib import Path

import OSM
import black_marble_daily
import earthengine_layers
from paths import BLACK_MARBLE_CACHE_DIR, LAYERS_DIR, OSM_CACHE_DIR, WARPED_TIFS_DIR, ensure_createdataset_dirs


PIPELINE_STEPS = {
    earthengine_layers.STEP_NAME: earthengine_layers,
    black_marble_daily.STEP_NAME: black_marble_daily,
    OSM.STEP_NAME: OSM,
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the modular CREATEDATASET layer-generation pipeline."
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        default=list(PIPELINE_STEPS),
        help=f"Pipeline steps to run. Available: {', '.join(PIPELINE_STEPS)}",
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
        "--project-id",
        type=str,
        default=earthengine_layers.PROJECT_ID,
        help="Earth Engine project id for the Earth Engine layer step.",
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
        help="Pause between Earth Engine layer downloads.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing layer outputs.",
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
    parser.add_argument(
        "--laads-token",
        type=str,
        default=None,
        help="Optional LAADS / Earthdata bearer token for the Black Marble step.",
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
        default=black_marble_daily.DEFAULT_ARCHIVE_ROOT,
        help="Base archive root for Black Marble downloads.",
    )
    parser.add_argument(
        "--black-marble-collection",
        type=str,
        default=black_marble_daily.DEFAULT_COLLECTION,
        help="Collection id for Black Marble downloads.",
    )
    parser.add_argument(
        "--black-marble-product",
        type=str,
        default=black_marble_daily.DEFAULT_PRODUCT,
        help="Black Marble product code.",
    )
    parser.add_argument(
        "--black-marble-version",
        type=str,
        default=black_marble_daily.DEFAULT_VERSION,
        help="Black Marble product version.",
    )
    parser.add_argument(
        "--black-marble-field",
        type=str,
        default=black_marble_daily.DEFAULT_FIELD_NAME,
        help="Preferred Black Marble HDF5 field name.",
    )
    parser.add_argument(
        "--black-marble-high-quality-only",
        action="store_true",
        help="Mask Black Marble pixels where Mandatory_Quality_Flag is not zero.",
    )
    parser.add_argument(
        "--black-marble-connect-timeout-seconds",
        type=float,
        default=30.0,
        help="HTTP connect timeout for Black Marble downloads.",
    )
    parser.add_argument(
        "--black-marble-read-timeout-seconds",
        type=float,
        default=300.0,
        help="HTTP read timeout for Black Marble downloads.",
    )
    parser.add_argument(
        "--osm-overpass-url",
        type=str,
        default=None,
        help="Optional single Overpass API interpreter endpoint override for OSM road downloads.",
    )
    parser.add_argument(
        "--osm-overpass-urls",
        nargs="+",
        default=None,
        help="Optional list of Overpass API interpreter endpoints for OSM road downloads.",
    )
    parser.add_argument(
        "--osm-highway-pattern",
        type=str,
        default=OSM.DEFAULT_HIGHWAY_PATTERN,
        help="Regex of OSM highway values to include in the road layer.",
    )
    parser.add_argument(
        "--osm-connect-timeout-seconds",
        type=float,
        default=OSM.DEFAULT_CONNECT_TIMEOUT_SECONDS,
        help="HTTP connect timeout for OSM Overpass requests.",
    )
    parser.add_argument(
        "--osm-read-timeout-seconds",
        type=float,
        default=OSM.DEFAULT_READ_TIMEOUT_SECONDS,
        help="HTTP read timeout for OSM Overpass requests.",
    )
    parser.add_argument(
        "--osm-max-bbox-degrees",
        type=float,
        default=OSM.DEFAULT_MAX_BBOX_DEGREES,
        help="Maximum lon/lat tile size per OSM Overpass request.",
    )
    parser.add_argument(
        "--osm-request-pause-seconds",
        type=float,
        default=OSM.DEFAULT_REQUEST_PAUSE_SECONDS,
        help="Pause between OSM Overpass tile requests.",
    )
    parser.add_argument(
        "--osm-retry-attempts",
        type=int,
        default=OSM.DEFAULT_RETRY_ATTEMPTS,
        help="Retry attempts for OSM Overpass tile requests.",
    )
    parser.add_argument(
        "--osm-max-concurrent-requests",
        type=int,
        default=OSM.DEFAULT_MAX_CONCURRENT_REQUESTS,
        help="Maximum number of OSM tile requests to run in parallel.",
    )
    parser.add_argument(
        "--osm-max-concurrent-scenes",
        type=int,
        default=OSM.DEFAULT_MAX_CONCURRENT_SCENES,
        help="Maximum number of TIFF scenes to process in parallel for the OSM step.",
    )
    parser.add_argument(
        "--osm-cache-dir",
        type=Path,
        default=OSM_CACHE_DIR,
        help="Folder for cached raw OSM tile responses.",
    )
    parser.add_argument(
        "--osm-refresh-cache",
        action="store_true",
        help="Ignore existing cached OSM tile responses and redownload them.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    ensure_createdataset_dirs()
    args = parse_args(argv)

    for step_name in args.steps:
        module = PIPELINE_STEPS.get(step_name)
        if module is None:
            available = ", ".join(sorted(PIPELINE_STEPS))
            raise ValueError(f"Unknown pipeline step `{step_name}`. Available steps: {available}")

        print(f"\n=== Running step: {step_name} ===")
        module.run_from_args(args)


if __name__ == "__main__":
    main()
