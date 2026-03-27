from __future__ import annotations

import argparse
import json
import math
import random
import sys
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window

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
        "Diffusers inference dependencies are missing in the active Python environment. "
        "This script needs an environment that has both the geospatial stack "
        "(`numpy`, `rasterio`, `torch`) and the diffusers dependencies "
        f"(currently missing: `{missing_name}`)."
    ) from exc

from paths import CREATEDATASET_ROOT
from scene_assets import SOURCE_SCENE_FILENAME


@dataclass(frozen=True)
class TargetNormalizer:
    high_value: float
    percentile: float

    def decode(self, array: np.ndarray) -> np.ndarray:
        decoded = np.clip((array + 1.0) / 2.0, 0.0, 1.0) * self.high_value
        return np.expm1(decoded).astype(np.float32)


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


@dataclass(frozen=True)
class CheckpointBundle:
    checkpoint_dir: Path
    layer_names: list[str]
    target_normalizer: TargetNormalizer
    preview_inference_steps: int
    patch_size: int


@dataclass(frozen=True)
class SceneInputs:
    scene_dir: Path
    reference_path: Path
    layer_paths: dict[str, Path]
    height: int
    width: int
    crs: str | None
    transform: tuple[float, float, float, float, float, float]
    extra_layer_names: list[str]
    missing_layer_names: list[str]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a full Luojia-style GeoTIFF from a folder of aligned condition TIFFs. "
            "A CREATEDATASET/LAYERS/<scene> folder is valid input; the script ignores "
            "`luojia_original.tif` if it is present."
        )
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Checkpoint directory produced by train_step_full.py, e.g. checkpoint_epoch_0050.",
    )
    parser.add_argument(
        "--input-scene-dir",
        type=Path,
        required=True,
        help="Folder containing condition TIFFs. Can be an existing CREATEDATASET/LAYERS/<scene> folder.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output GeoTIFF path. Defaults to CREATEDATASET/INFERENCE/<scene>__generated_luojia.tif.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Override the number of denoising steps. Defaults to the training preview setting.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Patch stride for tiled inference. Defaults to half the model patch size.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of patches to sample in parallel during inference.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Execution device. Defaults to auto.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for diffusion sampling.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output GeoTIFF if it already exists.",
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


def resolve_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def default_output_path(input_scene_dir: Path) -> Path:
    inference_dir = CREATEDATASET_ROOT / "INFERENCE"
    return (inference_dir / f"{input_scene_dir.name}__generated_luojia.tif").resolve()


def resolve_patch_size(sample_size: int | list[int] | tuple[int, ...]) -> int:
    if isinstance(sample_size, int):
        return sample_size
    if isinstance(sample_size, (list, tuple)) and sample_size:
        if int(sample_size[0]) != int(sample_size[-1]):
            raise ValueError(f"Expected square sample size, got {sample_size}.")
        return int(sample_size[0])
    raise ValueError(f"Could not resolve patch size from sample_size={sample_size!r}.")


def load_checkpoint_bundle(checkpoint_dir: Path) -> CheckpointBundle:
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    state_path = checkpoint_dir / "training_state.json"
    config_path = checkpoint_dir.parent / "run_config.json"

    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}

    layer_names = state.get("layer_names") or config.get("layer_names")
    if not layer_names:
        raise FileNotFoundError(
            f"Could not find `layer_names` in {state_path} or {config_path}."
        )

    target_normalizer_data = state.get("target_normalizer") or config.get("target_normalizer")
    if not target_normalizer_data:
        raise FileNotFoundError(
            f"Could not find `target_normalizer` in {state_path} or {config_path}."
        )

    preview_inference_steps = int(
        config.get("args", {}).get("preview_inference_steps", 100)
    )

    model = UNet2DModel.from_pretrained(checkpoint_dir / "unet")
    patch_size = resolve_patch_size(model.config.sample_size)
    del model

    return CheckpointBundle(
        checkpoint_dir=checkpoint_dir,
        layer_names=list(layer_names),
        target_normalizer=TargetNormalizer(
            high_value=float(target_normalizer_data["high_value"]),
            percentile=float(target_normalizer_data["percentile"]),
        ),
        preview_inference_steps=preview_inference_steps,
        patch_size=patch_size,
    )


def discover_scene_inputs(
    scene_dir: Path,
    expected_layer_names: list[str],
    ignored_paths: set[Path] | None = None,
) -> SceneInputs:
    scene_dir = scene_dir.expanduser().resolve()
    ignored_paths = {path.resolve() for path in (ignored_paths or set())}

    layer_paths: dict[str, Path] = {}
    for tif_path in sorted(scene_dir.glob("*.tif")):
        resolved = tif_path.resolve()
        if resolved in ignored_paths:
            continue
        if tif_path.name == SOURCE_SCENE_FILENAME:
            continue
        layer_paths[tif_path.stem] = resolved

    if not layer_paths:
        raise FileNotFoundError(
            f"No condition TIFFs found in {scene_dir}. "
            "This folder can be a LAYERS scene directory as long as it has the auxiliary rasters."
        )

    extra_layer_names = sorted(name for name in layer_paths if name not in expected_layer_names)
    missing_layer_names = sorted(name for name in expected_layer_names if name not in layer_paths)

    reference_name = next((name for name in expected_layer_names if name in layer_paths), None)
    if reference_name is None:
        reference_name = sorted(layer_paths)[0]
    reference_path = layer_paths[reference_name]

    with rasterio.open(reference_path) as ref:
        height = ref.height
        width = ref.width
        crs = ref.crs.to_string() if ref.crs is not None else None
        transform = tuple(ref.transform)

    for layer_name, layer_path in layer_paths.items():
        with rasterio.open(layer_path) as src:
            if src.height != height or src.width != width:
                raise ValueError(
                    f"Layer `{layer_name}` in {scene_dir} has size {(src.height, src.width)} "
                    f"but the reference raster size is {(height, width)}."
                )

    return SceneInputs(
        scene_dir=scene_dir,
        reference_path=reference_path,
        layer_paths=layer_paths,
        height=height,
        width=width,
        crs=crs,
        transform=transform,
        extra_layer_names=extra_layer_names,
        missing_layer_names=missing_layer_names,
    )


def build_positions(full_size: int, patch_size: int, stride: int) -> list[int]:
    if stride <= 0:
        raise ValueError("stride must be positive.")

    if full_size <= patch_size:
        return [0]

    positions = list(range(0, full_size - patch_size + 1, stride))
    final_position = full_size - patch_size
    if positions[-1] != final_position:
        positions.append(final_position)
    return positions


def build_patch_weight(patch_size: int) -> np.ndarray:
    if patch_size <= 1:
        return np.ones((1, 1), dtype=np.float32)

    window_1d = np.hanning(patch_size).astype(np.float32)
    if float(window_1d.max()) <= 0.0:
        window_1d = np.ones_like(window_1d)
    else:
        window_1d = window_1d / window_1d.max()

    weight = np.outer(window_1d, window_1d).astype(np.float32)
    return np.clip(weight, 1e-3, None)


def read_condition_patch(src: rasterio.io.DatasetReader, window: Window, patch_size: int) -> np.ndarray:
    return src.read(
        1,
        window=window,
        boundless=True,
        fill_value=0.0,
        out_shape=(patch_size, patch_size),
    ).astype(np.float32)


def build_condition_batch(
    opened_sources: dict[str, rasterio.io.DatasetReader],
    positions: list[tuple[int, int]],
    layer_names: list[str],
    patch_size: int,
) -> np.ndarray:
    cond_batch: list[np.ndarray] = []
    for origin_y, origin_x in positions:
        window = Window(col_off=origin_x, row_off=origin_y, width=patch_size, height=patch_size)
        channels: list[np.ndarray] = []
        for layer_name in layer_names:
            src = opened_sources.get(layer_name)
            if src is None:
                patch = np.zeros((patch_size, patch_size), dtype=np.float32)
            else:
                patch = normalize_layer(layer_name, read_condition_patch(src, window, patch_size))
            channels.append(patch)
        cond_batch.append(np.stack(channels, axis=0))
    return np.stack(cond_batch, axis=0).astype(np.float32)


def write_geotiff(output_path: Path, array: np.ndarray, reference_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(reference_path) as ref:
        profile = ref.profile.copy()

    profile.update(
        count=1,
        dtype="float32",
        compress="lzw",
        predictor=3,
        nodata=None,
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(array.astype(np.float32), 1)


def write_metadata(
    metadata_path: Path,
    checkpoint: CheckpointBundle,
    scene_inputs: SceneInputs,
    output_path: Path,
    patch_size: int,
    stride: int,
    num_inference_steps: int,
    device: torch.device,
) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_dir": str(checkpoint.checkpoint_dir),
        "input_scene_dir": str(scene_inputs.scene_dir),
        "output_path": str(output_path),
        "reference_path": str(scene_inputs.reference_path),
        "height": scene_inputs.height,
        "width": scene_inputs.width,
        "crs": scene_inputs.crs,
        "transform": list(scene_inputs.transform),
        "patch_size": patch_size,
        "stride": stride,
        "num_inference_steps": num_inference_steps,
        "device": str(device),
        "layer_names_expected": checkpoint.layer_names,
        "layer_names_available": sorted(scene_inputs.layer_paths),
        "missing_layer_names": scene_inputs.missing_layer_names,
        "extra_layer_names": scene_inputs.extra_layer_names,
        "target_normalizer": asdict(checkpoint.target_normalizer),
        "diffusers_module": str(Path(diffusers_package.__file__).resolve()),
    }
    with metadata_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    set_random_seed(args.seed)

    checkpoint = load_checkpoint_bundle(args.checkpoint_dir)
    output_path = (
        args.output_path.expanduser().resolve()
        if args.output_path is not None
        else default_output_path(args.input_scene_dir)
    )
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Pass --overwrite to replace it."
        )

    scene_inputs = discover_scene_inputs(
        scene_dir=args.input_scene_dir,
        expected_layer_names=checkpoint.layer_names,
        ignored_paths={output_path} if output_path.exists() else set(),
    )

    device = resolve_device(args.device)
    num_inference_steps = (
        int(args.num_inference_steps)
        if args.num_inference_steps is not None
        else checkpoint.preview_inference_steps
    )
    patch_size = checkpoint.patch_size
    stride = args.stride if args.stride is not None else max(1, patch_size // 2)
    batch_size = max(1, int(args.batch_size))

    model = UNet2DModel.from_pretrained(checkpoint.checkpoint_dir / "unet").to(device)
    model.eval()
    scheduler = DDPMScheduler.from_pretrained(checkpoint.checkpoint_dir / "scheduler")

    y_positions = build_positions(scene_inputs.height, patch_size, stride)
    x_positions = build_positions(scene_inputs.width, patch_size, stride)
    patch_positions = [(y, x) for y in y_positions for x in x_positions]
    patch_weight = build_patch_weight(patch_size)

    accumulated = np.zeros((scene_inputs.height, scene_inputs.width), dtype=np.float64)
    weight_sum = np.zeros((scene_inputs.height, scene_inputs.width), dtype=np.float64)

    print("Using diffusers from:", Path(diffusers_package.__file__).resolve())
    print("Checkpoint:", checkpoint.checkpoint_dir)
    print("Input scene:", scene_inputs.scene_dir)
    print("Reference raster:", scene_inputs.reference_path)
    print("Output:", output_path)
    print("Scene size:", (scene_inputs.height, scene_inputs.width))
    print("Patch size:", patch_size)
    print("Stride:", stride)
    print("Inference steps:", num_inference_steps)
    print("Expected layers:", checkpoint.layer_names)
    print("Missing layers:", scene_inputs.missing_layer_names)
    print("Extra layers ignored:", scene_inputs.extra_layer_names)

    with ExitStack() as stack:
        opened_sources = {
            layer_name: stack.enter_context(rasterio.open(layer_path))
            for layer_name, layer_path in scene_inputs.layer_paths.items()
            if layer_name in checkpoint.layer_names
        }

        for batch_start in range(0, len(patch_positions), batch_size):
            batch_positions = patch_positions[batch_start : batch_start + batch_size]
            cond_batch = build_condition_batch(
                opened_sources=opened_sources,
                positions=batch_positions,
                layer_names=checkpoint.layer_names,
                patch_size=patch_size,
            )
            cond_tensor = torch.from_numpy(cond_batch).to(device)
            generated_encoded = sample_conditioned_patch(
                model=model,
                scheduler=scheduler,
                cond=cond_tensor,
                num_inference_steps=num_inference_steps,
            )
            generated_decoded = checkpoint.target_normalizer.decode(
                generated_encoded.detach().cpu().numpy()[:, 0]
            )

            for generated_patch, (origin_y, origin_x) in zip(generated_decoded, batch_positions):
                y_end = min(origin_y + patch_size, scene_inputs.height)
                x_end = min(origin_x + patch_size, scene_inputs.width)
                valid_height = y_end - origin_y
                valid_width = x_end - origin_x
                weight = patch_weight[:valid_height, :valid_width]
                accumulated[origin_y:y_end, origin_x:x_end] += (
                    generated_patch[:valid_height, :valid_width] * weight
                )
                weight_sum[origin_y:y_end, origin_x:x_end] += weight

            processed = batch_start + len(batch_positions)
            print(f"processed patches: {processed}/{len(patch_positions)}")

    result = accumulated / np.clip(weight_sum, 1e-6, None)
    result = np.clip(result, a_min=0.0, a_max=None).astype(np.float32)

    write_geotiff(output_path=output_path, array=result, reference_path=scene_inputs.reference_path)
    write_metadata(
        metadata_path=output_path.with_suffix(".json"),
        checkpoint=checkpoint,
        scene_inputs=scene_inputs,
        output_path=output_path,
        patch_size=patch_size,
        stride=stride,
        num_inference_steps=num_inference_steps,
        device=device,
    )
    print("Finished writing:", output_path)


if __name__ == "__main__":
    main()
