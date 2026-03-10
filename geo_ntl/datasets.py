from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class NormalizationConfig:
    # NTL / Black Marble
    log_target: bool = True
    log_bm: bool = True

    # Optional scaling after log1p:
    # out = log1p(x) / log1p(scale)
    # If None, only log1p is applied.
    target_log_scale: Optional[float] = None
    bm_log_scale: Optional[float] = None

    # GAIA year normalization
    gaia_min_year: int = 1985
    gaia_max_year: int = 2018

    # Building height normalization
    max_building_height: float = 50.0

    # Spectral index clipping
    index_min: float = -1.0
    index_max: float = 1.0

    # Convert NDVI/NDBI/MNDWI from [-1, 1] to [0, 1]
    rescale_indices_to_01: bool = True


class GeoNTLDataset(Dataset):
    """
    Expects one .npz file per patch.

    Recommended keys inside each .npz:
        target          -> LJ1-01 target, shape (H,W) or (1,H,W)
        bm              -> Black Marble / VIIRS low-res resampled to target grid
        gaia_year
        urban_mask
        building_height
        water_mask
        ndvi
        ndbi
        mndwi
        valid_mask      -> optional

    Returned sample:
        {
            "target": Tensor[1,H,W],
            "cond": Tensor[C,H,W],
            "valid_mask": Tensor[1,H,W],   # always returned
            "filename": str,
            "cond_names": list[str],
        }
    """

    DEFAULT_COND_KEYS: Tuple[str, ...] = (
        "bm",
        "gaia_year",
        "urban_mask",
        "building_height",
        "water_mask",
        "ndvi",
        "ndbi",
        "mndwi",
    )

    def __init__(
        self,
        root: str | Path,
        cond_keys: Optional[Sequence[str]] = None,
        normalization: Optional[NormalizationConfig] = None,
        recursive: bool = False,
        strict: bool = True,
        allow_missing_cond: bool = False,
    ) -> None:
        self.root = Path(root)
        self.cond_keys = list(cond_keys) if cond_keys is not None else list(self.DEFAULT_COND_KEYS)
        self.norm = normalization or NormalizationConfig()
        self.strict = strict
        self.allow_missing_cond = allow_missing_cond

        pattern = "**/*.npz" if recursive else "*.npz"
        self.files = sorted(self.root.glob(pattern))

        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str | List[str]]:
        file_path = self.files[idx]

        with np.load(file_path, allow_pickle=False) as data:
            if "target" not in data:
                raise KeyError(f"{file_path.name}: missing required key 'target'")

            target = self._prepare_array(data["target"], key="target", file_path=file_path)

            cond_arrays = []
            cond_names = []

            for key in self.cond_keys:
                if key not in data:
                    if self.allow_missing_cond:
                        continue
                    raise KeyError(f"{file_path.name}: missing conditioning key '{key}'")

                arr = self._prepare_array(data[key], key=key, file_path=file_path)
                self._check_same_hw(target, arr, key, file_path)
                cond_arrays.append(arr)
                cond_names.append(key)

            if not cond_arrays:
                raise ValueError(f"{file_path.name}: no conditioning arrays available")

            cond = np.concatenate(cond_arrays, axis=0).astype(np.float32)

            if "valid_mask" in data:
                valid_mask = self._prepare_array(data["valid_mask"], key="valid_mask", file_path=file_path)
                self._check_same_hw(target, valid_mask, "valid_mask", file_path)
                valid_mask = (valid_mask > 0.5).astype(np.float32)
            else:
                # Default: valid everywhere target is finite
                valid_mask = np.isfinite(target).all(axis=0, keepdims=True).astype(np.float32)

            # Clean any remaining NaN/Inf
            target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            cond = np.nan_to_num(cond, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            valid_mask = np.nan_to_num(valid_mask, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        sample = {
            "target": torch.from_numpy(target),
            "cond": torch.from_numpy(cond),
            "valid_mask": torch.from_numpy(valid_mask),
            "filename": file_path.name,
            "cond_names": cond_names,
        }
        return sample

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_array(self, arr: np.ndarray, key: str, file_path: Path) -> np.ndarray:
        arr = self._ensure_chw(arr, key=key, file_path=file_path).astype(np.float32)
        arr = self._normalize(arr, key=key)
        return arr

    def _ensure_chw(self, arr: np.ndarray, key: str, file_path: Path) -> np.ndarray:
        """
        Accepts:
            (H, W)      -> converts to (1, H, W)
            (1, H, W)   -> keeps as is
        Rejects:
            anything else
        """
        if arr.ndim == 2:
            arr = arr[None, ...]
        elif arr.ndim == 3:
            pass
        else:
            raise ValueError(
                f"{file_path.name}: key '{key}' must have shape (H,W) or (C,H,W), got {arr.shape}"
            )

        if self.strict and arr.shape[0] != 1:
            raise ValueError(
                f"{file_path.name}: key '{key}' must have exactly 1 channel for now, got {arr.shape[0]}"
            )

        return arr

    def _check_same_hw(self, ref: np.ndarray, arr: np.ndarray, key: str, file_path: Path) -> None:
        if ref.shape[-2:] != arr.shape[-2:]:
            raise ValueError(
                f"{file_path.name}: key '{key}' has spatial shape {arr.shape[-2:]}, "
                f"expected {ref.shape[-2:]}"
            )

    def _normalize(self, x: np.ndarray, key: str) -> np.ndarray:
        key = key.lower()

        if key == "target":
            x = np.clip(x, a_min=0.0, a_max=None)
            if self.norm.log_target:
                x = np.log1p(x)
            if self.norm.target_log_scale is not None:
                denom = np.log1p(self.norm.target_log_scale)
                if denom > 0:
                    x = x / denom
            return x.astype(np.float32)

        if key == "bm":
            x = np.clip(x, a_min=0.0, a_max=None)
            if self.norm.log_bm:
                x = np.log1p(x)
            if self.norm.bm_log_scale is not None:
                denom = np.log1p(self.norm.bm_log_scale)
                if denom > 0:
                    x = x / denom
            return x.astype(np.float32)

        if key == "gaia_year":
            # Keep nodata / zero as zero. Valid years -> [0,1]
            out = np.zeros_like(x, dtype=np.float32)
            valid = (x >= self.norm.gaia_min_year) & (x <= self.norm.gaia_max_year)
            denom = float(self.norm.gaia_max_year - self.norm.gaia_min_year)
            out[valid] = (x[valid] - self.norm.gaia_min_year) / denom
            return out

        if key in {"urban_mask", "water_mask", "valid_mask"}:
            return (x > 0.5).astype(np.float32)

        if key == "building_height":
            x = np.clip(x, 0.0, self.norm.max_building_height)
            x = x / self.norm.max_building_height
            return x.astype(np.float32)

        if key in {"ndvi", "ndbi", "mndwi"}:
            x = np.clip(x, self.norm.index_min, self.norm.index_max)
            if self.norm.rescale_indices_to_01:
                x = (x - self.norm.index_min) / (self.norm.index_max - self.norm.index_min)
            return x.astype(np.float32)

        # Fallback: no normalization
        return x.astype(np.float32)


if __name__ == "__main__":
    # Quick smoke test:
    # python geo_ntl/datasets.py
    ds = GeoNTLDataset(
        root="geo_ntl/data",
        cond_keys=("bm", "gaia_year", "urban_mask", "building_height", "water_mask", "ndvi", "ndbi", "mndwi"),
    )

    sample = ds[0]
    print("filename:", sample["filename"])
    print("target shape:", sample["target"].shape)
    print("cond shape:", sample["cond"].shape)
    print("valid_mask shape:", sample["valid_mask"].shape)
    print("cond names:", sample["cond_names"])