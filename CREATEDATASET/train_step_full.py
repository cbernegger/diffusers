from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.enums import Resampling
from rasterio.windows import Window
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for candidate in (REPO_ROOT, SRC_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

try:
    import diffusers as diffusers_package
    from diffusers import DDPMScheduler, UNet2DModel
except ModuleNotFoundError as exc:
    missing_name = getattr(exc, "name", "an unknown dependency")
    raise RuntimeError(
        "Diffusers training dependencies are missing in the active Python environment. "
        "This script needs an environment that has both the geospatial stack "
        "(`numpy`, `rasterio`, `torch`) and the diffusers dependencies "
        f"(currently missing: `{missing_name}`)."
    ) from exc
from paths import CREATEDATASET_ROOT, LAYERS_DIR, WARPED_TIFS_DIR, ensure_createdataset_dirs
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
    "osm_roads",
)


@dataclass(frozen=True)
class SceneRecord:
    scene_id: str
    target_path: Path
    layer_paths: dict[str, Path]
    height: int
    width: int


@dataclass(frozen=True)
class TargetNormalizer:
    high_value: float
    percentile: float

    def encode(self, array: np.ndarray) -> np.ndarray:
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
        encoded = np.log1p(np.clip(array, a_min=0.0, a_max=None))
        encoded = np.clip(encoded / max(self.high_value, 1e-6), 0.0, 1.0)
        return (encoded * 2.0 - 1.0).astype(np.float32)

    def decode(self, array: np.ndarray) -> np.ndarray:
        decoded = np.clip((array + 1.0) / 2.0, 0.0, 1.0) * self.high_value
        return np.expm1(decoded).astype(np.float32)


class ConditionalPatchDataset(Dataset):
    def __init__(
        self,
        scenes: list[SceneRecord],
        layer_names: list[str],
        target_normalizer: TargetNormalizer,
        patch_size: int,
        patches_per_scene: int,
        seed: int,
        min_valid_fraction: float,
        patch_attempts: int,
    ) -> None:
        if not scenes:
            raise ValueError("No scene records were provided.")
        if not layer_names:
            raise ValueError("No layer names were discovered in LAYERS.")
        if patch_size <= 0:
            raise ValueError("patch_size must be positive.")
        if patches_per_scene <= 0:
            raise ValueError("patches_per_scene must be positive.")

        self.scenes = scenes
        self.layer_names = layer_names
        self.target_normalizer = target_normalizer
        self.patch_size = patch_size
        self.patches_per_scene = patches_per_scene
        self.min_valid_fraction = min_valid_fraction
        self.patch_attempts = max(1, patch_attempts)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.scenes) * self.patches_per_scene

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        scene = self.scenes[idx % len(self.scenes)]
        window, target_patch, valid_mask = choose_target_window(
            scene=scene,
            patch_size=self.patch_size,
            rng=self.rng,
            min_valid_fraction=self.min_valid_fraction,
            attempts=self.patch_attempts,
        )

        target_patch = self.target_normalizer.encode(target_patch)[None, ...]
        valid_mask = valid_mask[None, ...].astype(np.float32)

        layers: list[np.ndarray] = []
        for layer_name in self.layer_names:
            layer_path = scene.layer_paths.get(layer_name)
            if layer_path is None:
                layer = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
            else:
                layer = normalize_layer(layer_name, read_window(layer_path, window))
            layers.append(layer)

        cond_patch = np.stack(layers, axis=0).astype(np.float32)

        return {
            "scene_id": scene.scene_id,
            "origin_x": int(window.col_off),
            "origin_y": int(window.row_off),
            "target": torch.from_numpy(target_patch.astype(np.float32)),
            "cond": torch.from_numpy(cond_patch),
            "valid_mask": torch.from_numpy(valid_mask),
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a conditional DDPM that predicts Luojia radiance from aligned raster layers."
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
        "--output-dir",
        type=Path,
        default=CREATEDATASET_ROOT / "TRAINING" / "luojia_conditional_ddpm",
        help="Directory where checkpoints, previews, and logs are written.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=128,
        help="Patch size used for training and preview sampling.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size.",
    )
    parser.add_argument(
        "--patches-per-scene",
        type=int,
        default=64,
        help="Number of random training patches sampled per scene per epoch.",
    )
    parser.add_argument(
        "--val-patches-per-scene",
        type=int,
        default=8,
        help="Number of random validation patches sampled per scene per evaluation pass.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm.",
    )
    parser.add_argument(
        "--num-train-timesteps",
        type=int,
        default=1000,
        help="Number of diffusion timesteps in the scheduler.",
    )
    parser.add_argument(
        "--preview-inference-steps",
        type=int,
        default=100,
        help="Number of denoising steps used for preview sampling.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.25,
        help="Fraction of scenes held out for validation.",
    )
    parser.add_argument(
        "--min-valid-fraction",
        type=float,
        default=0.8,
        help="Minimum preferred fraction of valid target pixels in a sampled patch.",
    )
    parser.add_argument(
        "--patch-attempts",
        type=int,
        default=24,
        help="How many random patch candidates to try before picking the best one.",
    )
    parser.add_argument(
        "--target-high-percentile",
        type=float,
        default=99.5,
        help="Percentile of log1p(target) used as the upper normalization anchor.",
    )
    parser.add_argument(
        "--target-stat-max-side",
        type=int,
        default=512,
        help="Maximum side length used when downsampling target rasters to estimate target normalization stats.",
    )
    parser.add_argument(
        "--save-every-epochs",
        type=int,
        default=5,
        help="Checkpoint frequency in epochs.",
    )
    parser.add_argument(
        "--preview-every-epochs",
        type=int,
        default=5,
        help="Preview-sampling frequency in epochs.",
    )
    parser.add_argument(
        "--log-every-steps",
        type=int,
        default=25,
        help="Training log frequency in optimizer steps.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Dataloader worker count. Defaults to 0 for reproducible random window sampling.",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for splits, patch sampling, and noise.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_window(path: Path, window: Window) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1, window=window).astype(np.float32)


def read_target_window(path: Path, window: Window) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as src:
        masked = src.read(1, window=window, masked=True).astype(np.float32)
    valid_mask = (~masked.mask).astype(np.float32)
    filled = np.asarray(masked.filled(0.0), dtype=np.float32)
    return filled, valid_mask


def normalize_layer(name: str, array: np.ndarray) -> np.ndarray:
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    name = name.lower()

    if name in {"bm", "bm_daily"}:
        log_scaled = np.log1p(np.clip(array, a_min=0.0, a_max=None))
        return (np.clip(log_scaled / 8.0, 0.0, 1.0) * 2.0 - 1.0).astype(np.float32)

    if name == "gaia_year":
        out = np.zeros_like(array, dtype=np.float32)
        valid = (array >= 1985) & (array <= 2018)
        out[valid] = (array[valid] - 1985.0) / (2018.0 - 1985.0)
        return (out * 2.0 - 1.0).astype(np.float32)

    if name in {"urban_mask", "water_mask", "osm_roads"}:
        return ((array > 0.5).astype(np.float32) * 2.0 - 1.0).astype(np.float32)

    if name == "building_height":
        normalized = np.clip(array, 0.0, 80.0) / 80.0
        return (normalized * 2.0 - 1.0).astype(np.float32)

    if name in {"ndvi", "ndbi", "mndwi"}:
        return np.clip(array, -1.0, 1.0).astype(np.float32)

    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array, dtype=np.float32)

    lower = float(np.percentile(finite, 1.0))
    upper = float(np.percentile(finite, 99.0))
    if upper <= lower:
        return np.zeros_like(array, dtype=np.float32)

    normalized = np.clip((array - lower) / (upper - lower), 0.0, 1.0)
    return (normalized * 2.0 - 1.0).astype(np.float32)


def choose_target_window(
    scene: SceneRecord,
    patch_size: int,
    rng: random.Random,
    min_valid_fraction: float,
    attempts: int,
) -> tuple[Window, np.ndarray, np.ndarray]:
    if scene.height < patch_size or scene.width < patch_size:
        raise ValueError(
            f"Patch size {patch_size} exceeds raster size {(scene.height, scene.width)} "
            f"for scene `{scene.scene_id}`."
        )

    max_y = scene.height - patch_size
    max_x = scene.width - patch_size
    best_window: Window | None = None
    best_target: np.ndarray | None = None
    best_valid: np.ndarray | None = None
    best_score = -1.0

    for _ in range(max(1, attempts)):
        y = rng.randint(0, max_y) if max_y > 0 else 0
        x = rng.randint(0, max_x) if max_x > 0 else 0
        window = Window(col_off=x, row_off=y, width=patch_size, height=patch_size)
        target_patch, valid_mask = read_target_window(scene.target_path, window)
        score = float(valid_mask.mean())
        if score > best_score:
            best_window = window
            best_target = target_patch
            best_valid = valid_mask
            best_score = score
        if score >= min_valid_fraction:
            break

    assert best_window is not None
    assert best_target is not None
    assert best_valid is not None
    return best_window, best_target, best_valid


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


def split_scenes(scenes: list[SceneRecord], validation_fraction: float, seed: int) -> tuple[list[SceneRecord], list[SceneRecord]]:
    scenes = list(scenes)
    rng = random.Random(seed)
    rng.shuffle(scenes)

    if len(scenes) <= 1 or validation_fraction <= 0.0:
        return scenes, []

    val_count = max(1, int(round(len(scenes) * validation_fraction)))
    val_count = min(val_count, len(scenes) - 1)
    val_scenes = sorted(scenes[:val_count], key=lambda scene: scene.scene_id)
    train_scenes = sorted(scenes[val_count:], key=lambda scene: scene.scene_id)
    return train_scenes, val_scenes


def estimate_target_normalizer(
    scenes: list[SceneRecord],
    percentile: float,
    max_side: int,
) -> TargetNormalizer:
    sampled_values: list[np.ndarray] = []

    for scene in scenes:
        with rasterio.open(scene.target_path) as src:
            scale = max(src.height, src.width) / float(max_side) if max_side > 0 else 1.0
            if scale > 1.0:
                out_height = max(1, int(math.ceil(src.height / scale)))
                out_width = max(1, int(math.ceil(src.width / scale)))
                masked = src.read(
                    1,
                    masked=True,
                    out_shape=(out_height, out_width),
                    resampling=Resampling.nearest,
                ).astype(np.float32)
            else:
                masked = src.read(1, masked=True).astype(np.float32)

        values = masked.compressed().astype(np.float32)
        values = values[np.isfinite(values)]
        values = values[values > 0.0]
        if values.size == 0:
            continue
        sampled_values.append(np.log1p(values))

    if not sampled_values:
        raise ValueError("Could not estimate target normalization statistics from the training scenes.")

    merged = np.concatenate(sampled_values)
    high_value = float(np.percentile(merged, percentile))
    high_value = max(high_value, 1.0)
    return TargetNormalizer(high_value=high_value, percentile=percentile)


def build_model(patch_size: int, cond_channels: int) -> UNet2DModel:
    return UNet2DModel(
        sample_size=patch_size,
        in_channels=1 + cond_channels,
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
    )


def compute_batch_loss(
    model: UNet2DModel,
    scheduler: DDPMScheduler,
    target: torch.Tensor,
    cond: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    noise = torch.randn_like(target)
    timesteps = torch.randint(
        0,
        scheduler.config.num_train_timesteps,
        (target.shape[0],),
        device=target.device,
    ).long()

    noisy_target = scheduler.add_noise(target, noise, timesteps)
    model_input = torch.cat([noisy_target, cond], dim=1)
    noise_pred = model(model_input, timesteps).sample

    weighted_loss = ((noise_pred - noise) ** 2) * valid_mask
    return weighted_loss.sum() / valid_mask.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate_epoch(
    model: UNet2DModel,
    scheduler: DDPMScheduler,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    losses: list[float] = []

    for batch in loader:
        target = batch["target"].to(device)
        cond = batch["cond"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        loss = compute_batch_loss(model, scheduler, target, cond, valid_mask)
        losses.append(float(loss.item()))

    model.train()
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def sample_conditioned_patch(
    model: UNet2DModel,
    scheduler: DDPMScheduler,
    cond: torch.Tensor,
    num_inference_steps: int,
) -> torch.Tensor:
    inference_scheduler = DDPMScheduler.from_config(scheduler.config)
    inference_scheduler.set_timesteps(num_inference_steps, device=cond.device)

    sample = torch.randn(
        (cond.shape[0], 1, cond.shape[2], cond.shape[3]),
        device=cond.device,
        dtype=cond.dtype,
    )

    for timestep in inference_scheduler.timesteps:
        timestep_batch = torch.full(
            (cond.shape[0],),
            int(timestep),
            device=cond.device,
            dtype=torch.long,
        )
        model_input = torch.cat([sample, cond], dim=1)
        noise_pred = model(model_input, timestep_batch).sample
        sample = inference_scheduler.step(noise_pred, timestep, sample).prev_sample

    return sample


def save_preview(
    output_dir: Path,
    epoch_index: int,
    batch: dict[str, torch.Tensor | str | int],
    generated_target: torch.Tensor,
    target_normalizer: TargetNormalizer,
    layer_names: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    target_encoded = batch["target"][0].cpu().numpy().astype(np.float32)
    generated_encoded = generated_target[0].cpu().numpy().astype(np.float32)
    valid_mask = batch["valid_mask"][0].cpu().numpy().astype(np.float32)
    cond = batch["cond"][0].cpu().numpy().astype(np.float32)

    target_decoded = target_normalizer.decode(target_encoded[0])
    generated_decoded = target_normalizer.decode(generated_encoded[0])

    preview_path = output_dir / f"preview_epoch_{epoch_index:04d}.npz"
    np.savez_compressed(
        preview_path,
        scene_id=np.array([batch["scene_id"][0]]),
        origin_x=np.array([int(batch["origin_x"][0])]),
        origin_y=np.array([int(batch["origin_y"][0])]),
        layer_names=np.array(layer_names),
        cond=cond,
        valid_mask=valid_mask,
        target_encoded=target_encoded,
        generated_encoded=generated_encoded,
        target_decoded=target_decoded,
        generated_decoded=generated_decoded,
    )


def save_checkpoint(
    checkpoint_dir: Path,
    model: UNet2DModel,
    scheduler: DDPMScheduler,
    optimizer: torch.optim.Optimizer,
    state: dict[str, object],
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir / "unet", safe_serialization=False)
    scheduler.save_pretrained(checkpoint_dir / "scheduler")
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    with (checkpoint_dir / "training_state.json").open("w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)


def append_metrics(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def serialize_args(args: argparse.Namespace) -> dict[str, object]:
    serialized: dict[str, object] = {}
    for key, value in vars(args).items():
        serialized[key] = str(value) if isinstance(value, Path) else value
    return serialized


def main(argv: list[str] | None = None) -> None:
    ensure_createdataset_dirs()
    args = parse_args(argv)
    set_random_seed(args.seed)

    layers_dir = args.layers_dir.expanduser().resolve()
    warped_dir = args.warped_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scenes, layer_names = discover_scenes(layers_dir, warped_dir, args.max_scenes)
    train_scenes, val_scenes = split_scenes(scenes, args.validation_fraction, args.seed)
    target_normalizer = estimate_target_normalizer(
        train_scenes,
        percentile=args.target_high_percentile,
        max_side=args.target_stat_max_side,
    )

    train_dataset = ConditionalPatchDataset(
        scenes=train_scenes,
        layer_names=layer_names,
        target_normalizer=target_normalizer,
        patch_size=args.patch_size,
        patches_per_scene=args.patches_per_scene,
        seed=args.seed,
        min_valid_fraction=args.min_valid_fraction,
        patch_attempts=args.patch_attempts,
    )
    val_dataset = (
        ConditionalPatchDataset(
            scenes=val_scenes,
            layer_names=layer_names,
            target_normalizer=target_normalizer,
            patch_size=args.patch_size,
            patches_per_scene=args.val_patches_per_scene,
            seed=args.seed + 10_000,
            min_valid_fraction=args.min_valid_fraction,
            patch_attempts=args.patch_attempts,
        )
        if val_scenes
        else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(args.batch_size, len(train_dataset)),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=min(args.batch_size, len(val_dataset)),
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        if val_dataset is not None
        else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.patch_size, cond_channels=len(layer_names)).to(device)
    scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    config_path = output_dir / "run_config.json"
    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "args": serialize_args(args),
                "layer_names": layer_names,
                "train_scene_ids": [scene.scene_id for scene in train_scenes],
                "val_scene_ids": [scene.scene_id for scene in val_scenes],
                "target_normalizer": asdict(target_normalizer),
                "diffusers_module": str(Path(diffusers_package.__file__).resolve()),
            },
            fh,
            indent=2,
        )

    print("Using diffusers from:", Path(diffusers_package.__file__).resolve())
    print("Device:", device)
    print("Train scenes:", len(train_scenes), [scene.scene_id for scene in train_scenes])
    print("Val scenes:", len(val_scenes), [scene.scene_id for scene in val_scenes])
    print("Layer names:", layer_names)
    print("Target normalizer:", asdict(target_normalizer))

    metrics_path = output_dir / "metrics.jsonl"
    global_step = 0

    for epoch_index in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []

        for batch_index, batch in enumerate(train_loader, start=1):
            target = batch["target"].to(device)
            cond = batch["cond"].to(device)
            valid_mask = batch["valid_mask"].to(device)

            loss = compute_batch_loss(model, scheduler, target, cond, valid_mask)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            optimizer.step()

            global_step += 1
            train_losses.append(float(loss.item()))

            if global_step % args.log_every_steps == 0:
                print(
                    f"epoch={epoch_index} batch={batch_index}/{len(train_loader)} "
                    f"global_step={global_step} loss={loss.item():.6f}"
                )

        train_loss = float(np.mean(train_losses))
        val_loss = evaluate_epoch(model, scheduler, val_loader, device) if val_loader is not None else float("nan")

        metrics_row = {
            "epoch": epoch_index,
            "global_step": global_step,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        append_metrics(metrics_path, metrics_row)
        print(
            f"Epoch {epoch_index}/{args.epochs}: "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
        )

        should_preview = val_loader is not None and (
            epoch_index == 1 or epoch_index % args.preview_every_epochs == 0 or epoch_index == args.epochs
        )
        if should_preview:
            preview_batch = next(iter(val_loader))
            preview_cond = preview_batch["cond"][:1].to(device)
            generated = sample_conditioned_patch(
                model=model,
                scheduler=scheduler,
                cond=preview_cond,
                num_inference_steps=args.preview_inference_steps,
            )
            save_preview(
                output_dir=output_dir / "previews",
                epoch_index=epoch_index,
                batch=preview_batch,
                generated_target=generated.cpu(),
                target_normalizer=target_normalizer,
                layer_names=layer_names,
            )

        should_save = epoch_index % args.save_every_epochs == 0 or epoch_index == args.epochs
        if should_save:
            checkpoint_state = {
                "epoch": epoch_index,
                "global_step": global_step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "layer_names": layer_names,
                "target_normalizer": asdict(target_normalizer),
                "train_scene_ids": [scene.scene_id for scene in train_scenes],
                "val_scene_ids": [scene.scene_id for scene in val_scenes],
                "diffusers_module": str(Path(diffusers_package.__file__).resolve()),
            }
            save_checkpoint(
                checkpoint_dir=output_dir / f"checkpoint_epoch_{epoch_index:04d}",
                model=model,
                scheduler=scheduler,
                optimizer=optimizer,
                state=checkpoint_state,
            )

    print("Training finished. Outputs written to:", output_dir)


if __name__ == "__main__":
    main()
