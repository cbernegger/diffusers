from __future__ import annotations

import shutil
from pathlib import Path


SOURCE_SCENE_FILENAME = "luojia_original.tif"


def copy_source_tif_to_scene_dir(scene_dir: Path, source_tif_path: Path, *, overwrite: bool) -> Path:
    source_path = source_tif_path.expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Source scene TIFF not found: {source_path}")

    destination = scene_dir / SOURCE_SCENE_FILENAME
    if destination.exists() and not overwrite:
        return destination.resolve()

    if destination.resolve() == source_path:
        return destination.resolve()

    tmp_path = destination.with_suffix(destination.suffix + ".part")
    shutil.copy2(source_path, tmp_path)
    tmp_path.replace(destination)
    return destination.resolve()
