from __future__ import annotations

from pathlib import Path


CREATEDATASET_ROOT = Path(__file__).resolve().parent
WARPED_TIFS_DIR = CREATEDATASET_ROOT / "WARPED_TIFS"
LAYERS_DIR = CREATEDATASET_ROOT / "LAYERS"
SCENE_WORK_DIR = CREATEDATASET_ROOT / "scene_work"
BLACK_MARBLE_CACHE_DIR = CREATEDATASET_ROOT / "BLACK_MARBLE_CACHE"


def ensure_createdataset_dirs() -> None:
    for path in (
        CREATEDATASET_ROOT,
        WARPED_TIFS_DIR,
        LAYERS_DIR,
        SCENE_WORK_DIR,
        BLACK_MARBLE_CACHE_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
