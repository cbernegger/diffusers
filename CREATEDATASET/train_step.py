from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (REPO_ROOT, SRC_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from diffusers import DDPMScheduler, UNet2DModel
from paths import LAYERS_DIR, WARPED_TIFS_DIR, ensure_createdataset_dirs
from scene_assets import SOURCE_SCENE_FILENAME


KNOWN_LAYER_ORDER = (
    "bm",
    "bm_daily",
    "gaia_year",
    "urban_mask",
    "building_height",
    "water_mask",
    "ndvi",
    "ndbi",
    "mndwi",
)


@dataclass(frozen=True)
class SceneRecord:
    scene_id: str
    target_path: Path
    layer_paths: dict[str, Path]
    height: int
    width: int


class LayerSceneDataset(Dataset):
    def __init__(
        self,
        scenes: list[SceneRecord],
        layer_names: list[str],
        patch_size: int,
        seed: int,
    ) -> None:
        if not scenes:
            raise ValueError("No scene records were provided.")
        if not layer_names:
            raise ValueError("No layer names were discovered in LAYERS.")

        self.scenes = scenes
        self.layer_names = layer_names
        self.patch_size = patch_size
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        scene = self.scenes[idx]
        target = normalize_target(read_single_band(scene.target_path))
        valid_mask = np.isfinite(target).astype(np.float32)
        target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)

        layers: list[np.ndarray] = []
        for layer_name in self.layer_names:
            layer_path = scene.layer_paths.get(layer_name)
            if layer_path is None:
                layer = np.zeros((scene.height, scene.width), dtype=np.float32)
            else:
                layer = normalize_layer(layer_name, read_single_band(layer_path))
            if layer.shape != target.shape:
                raise ValueError(
                    f"Scene `{scene.scene_id}` layer `{layer_name}` has shape {layer.shape}, "
                    f"expected {target.shape}"
                )
            layers.append(layer)

        y, x = choose_patch_origin(valid_mask, self.patch_size, self.rng)
        target_patch = target[y : y + self.patch_size, x : x + self.patch_size][None, ...]
        valid_patch = valid_mask[y : y + self.patch_size, x : x + self.patch_size][None, ...]
        cond_patch = np.stack(
            [layer[y : y + self.patch_size, x : x + self.patch_size] for layer in layers],
            axis=0,
        ).astype(np.float32)

        return {
            "scene_id": scene.scene_id,
            "target": torch.from_numpy(target_patch.astype(np.float32)),
            "cond": torch.from_numpy(cond_patch),
            "valid_mask": torch.from_numpy(valid_patch.astype(np.float32)),
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one diffusion training step from CREATEDATASET/LAYERS."
    )
    parser.add_argument(
        "--layers-dir",
        type=Path,
        default=LAYERS_DIR,
        help="Layer root. Defaults to CREATEDATASET/LAYERS.",
    )
    parser.add_argument(
        "--warped-dir",
        type=Path,
        default=WARPED_TIFS_DIR,
        help="Warped TIFF root. Defaults to CREATEDATASET/WARPED_TIFS.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=128,
        help="Patch size used for the training step.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for the training step.",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Optional limit for quick testing.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--num-train-timesteps",
        type=int,
        default=1000,
        help="Number of diffusion timesteps in the scheduler.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for patch sampling and training noise.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def read_single_band(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def normalize_target(array: np.ndarray) -> np.ndarray:
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    array = np.clip(array, a_min=0.0, a_max=None)
    return np.log1p(array).astype(np.float32)


def normalize_layer(name: str, array: np.ndarray) -> np.ndarray:
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    name = name.lower()

    if name in {"bm", "bm_daily"}:
        return np.log1p(np.clip(array, a_min=0.0, a_max=None)).astype(np.float32)

    if name == "gaia_year":
        out = np.zeros_like(array, dtype=np.float32)
        valid = (array >= 1985) & (array <= 2018)
        out[valid] = (array[valid] - 1985.0) / (2018.0 - 1985.0)
        return out

    if name in {"urban_mask", "water_mask"}:
        return (array > 0.5).astype(np.float32)

    if name == "building_height":
        return np.clip(array, 0.0, 50.0).astype(np.float32) / 50.0

    if name in {"ndvi", "ndbi", "mndwi"}:
        clipped = np.clip(array, -1.0, 1.0)
        return ((clipped + 1.0) / 2.0).astype(np.float32)

    return array.astype(np.float32)


def choose_patch_origin(valid_mask: np.ndarray, patch_size: int, rng: random.Random) -> tuple[int, int]:
    height, width = valid_mask.shape
    if height < patch_size or width < patch_size:
        raise ValueError(
            f"Patch size {patch_size} exceeds raster size {(height, width)}."
        )

    max_y = height - patch_size
    max_x = width - patch_size
    best_y = 0
    best_x = 0
    best_score = -1.0

    for _ in range(16):
        y = rng.randint(0, max_y) if max_y > 0 else 0
        x = rng.randint(0, max_x) if max_x > 0 else 0
        score = float(valid_mask[y : y + patch_size, x : x + patch_size].mean())
        if score > best_score:
            best_y = y
            best_x = x
            best_score = score
        if score >= 0.95:
            break

    return best_y, best_x


def layer_sort_key(name: str) -> tuple[int, int | str]:
    if name in KNOWN_LAYER_ORDER:
        return (0, KNOWN_LAYER_ORDER.index(name))
    return (1, name)


def resolve_target_path(scene_dir: Path, warped_dir: Path) -> Path:
    scene_copy_path = scene_dir / SOURCE_SCENE_FILENAME
    if scene_copy_path.exists():
        return scene_copy_path.resolve()

    metadata_path = scene_dir / "metadata.json"
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as fh:
            metadata = json.load(fh)
        scene_copy_from_metadata = metadata.get("source_scene_tif_copy")
        if scene_copy_from_metadata:
            copied_target_path = Path(scene_copy_from_metadata).expanduser()
            if copied_target_path.exists():
                return copied_target_path.resolve()
        target_path = Path(metadata["source_warped_tif"]).expanduser()
        if target_path.exists():
            return target_path.resolve()

    for suffix in (".tif", ".tiff"):
        fallback = warped_dir / f"{scene_dir.name}{suffix}"
        if fallback.exists():
            return fallback.resolve()

    raise FileNotFoundError(
        f"Could not resolve warped TIFF for scene `{scene_dir.name}`. "
        f"Checked {metadata_path} and {warped_dir}."
    )


def discover_scenes(layers_dir: Path, warped_dir: Path, max_scenes: int | None) -> tuple[list[SceneRecord], list[str]]:
    scene_dirs = sorted(path for path in layers_dir.iterdir() if path.is_dir())
    if max_scenes is not None:
        scene_dirs = scene_dirs[:max_scenes]

    if not scene_dirs:
        raise FileNotFoundError(f"No scene folders found in {layers_dir}")

    layer_names: set[str] = set()
    scenes: list[SceneRecord] = []

    for scene_dir in scene_dirs:
        layer_paths = {
            path.stem: path.resolve()
            for path in sorted(scene_dir.glob("*.tif"))
            if path.name != SOURCE_SCENE_FILENAME
        }
        if not layer_paths:
            continue

        target_path = resolve_target_path(scene_dir, warped_dir)
        with rasterio.open(target_path) as target_src:
            height = target_src.height
            width = target_src.width

        for layer_path in layer_paths.values():
            with rasterio.open(layer_path) as layer_src:
                if layer_src.height != height or layer_src.width != width:
                    raise ValueError(
                        f"Scene `{scene_dir.name}` contains layer `{layer_path.name}` with size "
                        f"{(layer_src.height, layer_src.width)} but target size is {(height, width)}."
                    )

        layer_names.update(layer_paths)
        scenes.append(
            SceneRecord(
                scene_id=scene_dir.name,
                target_path=target_path,
                layer_paths=layer_paths,
                height=height,
                width=width,
            )
        )

    if not scenes:
        raise FileNotFoundError(f"No layer TIFFs found in scene folders under {layers_dir}")

    ordered_layer_names = sorted(layer_names, key=layer_sort_key)
    return scenes, ordered_layer_names


def main(argv: list[str] | None = None) -> None:
    ensure_createdataset_dirs()
    args = parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    layers_dir = args.layers_dir.expanduser().resolve()
    warped_dir = args.warped_dir.expanduser().resolve()
    scenes, layer_names = discover_scenes(layers_dir, warped_dir, args.max_scenes)

    dataset = LayerSceneDataset(
        scenes=scenes,
        layer_names=layer_names,
        patch_size=args.patch_size,
        seed=args.seed,
    )

    loader = DataLoader(
        dataset,
        batch_size=min(args.batch_size, len(dataset)),
        shuffle=True,
        num_workers=0,
    )

    batch = next(iter(loader))
    target = batch["target"]
    cond = batch["cond"]
    valid_mask = batch["valid_mask"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet2DModel(
        sample_size=args.patch_size,
        in_channels=1 + cond.shape[1],
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 256),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    ).to(device)

    scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    target = target.to(device)
    cond = cond.to(device)
    valid_mask = valid_mask.to(device)

    noise = torch.randn_like(target)
    timesteps = torch.randint(
        0,
        scheduler.config.num_train_timesteps,
        (target.shape[0],),
        device=device,
    ).long()

    noisy_target = scheduler.add_noise(target, noise, timesteps)
    model_input = torch.cat([noisy_target, cond], dim=1)
    noise_pred = model(model_input, timesteps).sample

    weighted_loss = ((noise_pred - noise) ** 2) * valid_mask
    loss = weighted_loss.sum() / valid_mask.sum().clamp_min(1.0)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    print("device:", device)
    print("scenes used:", len(scenes))
    print("layer names:", layer_names)
    print("model_input shape:", tuple(model_input.shape))
    print("noise_pred shape:", tuple(noise_pred.shape))
    print("loss:", float(loss.item()))


if __name__ == "__main__":
    main()
