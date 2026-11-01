from __future__ import annotations

import hashlib
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.sparse.annotations import (
    generate_bbox_annotation,
    generate_point_annotation,
    generate_scribble_annotation,
)


SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}

SPARSE_TYPE_TO_INDEX = {
    "point": 0,
    "scribble": 1,
    "bbox": 2,
    "labelled": 3,
    "supervised": 4,
}


@dataclass(frozen=True)
class BreastSample:
    client_id: str
    split: str
    sample_id: str
    image_path: Path
    mask_path: Path


def stable_sample_seed(
    base_seed: int,
    client_id: str,
    split: str,
    sample_id: str,
    sparse_type: str,
) -> int:
    """
    Produce the same annotation seed for the same sample on every run.

    sparse_type is included so that assigning a different weak-label type to
    the same image does not reuse the random stream of another annotation type.
    """
    key = (
        f"{int(base_seed)}|{client_id}|{split}|"
        f"{sample_id}|{sparse_type}"
    )
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _list_files(directory: Path) -> List[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not files:
        raise RuntimeError(f"No supported image files found in: {directory}")

    return files


def _find_mask(
    image_path: Path,
    mask_dir: Path,
    mask_suffix: str,
) -> Path:
    if not mask_dir.is_dir():
        raise FileNotFoundError(
            f"Mask directory does not exist: {mask_dir}"
        )

    expected_stem = f"{image_path.stem}{mask_suffix}"
    candidates = sorted(
        path
        for path in mask_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and path.stem == expected_stem
    )

    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one mask for '{image_path.name}' with "
            f"stem '{expected_stem}', but found {len(candidates)} in "
            f"'{mask_dir}'."
        )

    return candidates[0]


def discover_samples(
    dataset_root: str | Path,
    client_id: str,
    split: str,
    mask_suffix: str = "",
) -> List[BreastSample]:
    """
    Expected structure:

        dataset_root/
          B1/
            train/
              image/
              mask/
            validation/
              image/
              mask/
            test/
              image/
              mask/
    """
    split_dir = Path(dataset_root) / str(client_id) / str(split)
    image_dir = split_dir / "image"
    mask_dir = split_dir / "mask"

    return [
        BreastSample(
            client_id=str(client_id),
            split=str(split),
            sample_id=image_path.stem,
            image_path=image_path,
            mask_path=_find_mask(
                image_path=image_path,
                mask_dir=mask_dir,
                mask_suffix=mask_suffix,
            ),
        )
        for image_path in _list_files(image_dir)
    ]


def _resize_image(
    image: Image.Image,
    size_hw: Tuple[int, int],
) -> Image.Image:
    height, width = size_hw
    return image.resize(
        (width, height),
        resample=Image.Resampling.BILINEAR,
    )


def _resize_binary(
    array: np.ndarray,
    size_hw: Tuple[int, int],
) -> np.ndarray:
    height, width = size_hw
    binary_image = Image.fromarray(
        (array > 0).astype(np.uint8) * 255
    )
    binary_image = binary_image.resize(
        (width, height),
        resample=Image.Resampling.NEAREST,
    )
    return (
        np.asarray(binary_image, dtype=np.uint8) > 0
    ).astype(np.uint8)


def _resize_sparse_binary(
    array: np.ndarray,
    size_hw: Tuple[int, int],
) -> np.ndarray:
    """
    Resize a sparse annotation without deleting isolated labelled pixels.

    Nearest-neighbour resizing can completely remove a one-pixel scribble
    during downsampling. For reductions, BOX resampling performs occupancy
    aggregation; thresholding at >0 preserves every source annotation cell.
    Upsampling continues to use nearest-neighbour interpolation.
    """
    source = (np.asarray(array) > 0).astype(np.uint8)
    source_height, source_width = source.shape
    target_height, target_width = size_hw

    image = Image.fromarray(source * 255)

    if (
        target_height < source_height
        or target_width < source_width
    ):
        resample = Image.Resampling.BOX
    else:
        resample = Image.Resampling.NEAREST

    resized = image.resize(
        (target_width, target_height),
        resample=resample,
    )

    output = (
        np.asarray(resized, dtype=np.uint8) > 0
    ).astype(np.uint8)

    # Absolute safeguard: preserve at least one labelled source pixel.
    if source.any() and not output.any():
        ys, xs = np.nonzero(source)
        index = len(xs) // 2
        target_x = min(
            target_width - 1,
            max(
                0,
                int(
                    (float(xs[index]) + 0.5)
                    * target_width
                    / source_width
                ),
            ),
        )
        target_y = min(
            target_height - 1,
            max(
                0,
                int(
                    (float(ys[index]) + 0.5)
                    * target_height
                    / source_height
                ),
            ),
        )
        output[target_y, target_x] = 1

    return output


def _image_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0

    if array.ndim != 3 or array.shape[2] != 3:
        raise RuntimeError(
            f"Expected RGB image with shape [H,W,3], got {array.shape}."
        )

    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def _binary_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(
        (array > 0).astype(np.float32)
    ).unsqueeze(0).contiguous()


def _validate_annotation(
    annotation: Mapping[str, np.ndarray],
    expected_shape: Tuple[int, int],
    sparse_type: str,
) -> Dict[str, np.ndarray]:
    required = ("positive", "negative", "box_region")
    missing = [key for key in required if key not in annotation]

    if missing:
        raise KeyError(
            f"{sparse_type} annotation generator did not return: {missing}"
        )

    output: Dict[str, np.ndarray] = {}

    for key in required:
        value = np.asarray(annotation[key])

        if value.shape != expected_shape:
            raise RuntimeError(
                f"{sparse_type} annotation '{key}' has shape "
                f"{value.shape}; expected {expected_shape}."
            )

        output[key] = (value > 0).astype(np.uint8)

    overlap = (output["positive"] > 0) & (output["negative"] > 0)
    if overlap.any():
        # Trusted positive and trusted negative seeds must never conflict.
        output["negative"][overlap] = 0

    if sparse_type != "bbox":
        # Point, scribble, and labelled samples must not carry a box.
        output["box_region"].fill(0)

    return output


class BreastWeakDataset(Dataset):
    """
    Breast-tumor dataset with deterministic sparse or labelled conditions.

    Important supervision rule
    --------------------------
    For point, scribble, and bbox modes, the dense segmentation mask is used
    only inside this dataset to simulate the requested sparse annotation. For
    labelled mode, the complete mask is intentionally supplied in the model
    condition and therefore acts as an oracle input. The dense mask is not
    returned separately for training unless return_dense_mask=True is set.
    Validation/test loaders should set return_dense_mask=True so Dice, IoU,
    and HD95 can be computed.

    Sparse annotations are generated at the original image/mask resolution
    and resized afterward with nearest-neighbor interpolation. For bbox
    clients, the original RGB image is used by Select(x, B) to choose a
    small image-supported foreground seed set inside each box.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        client_id: str,
        split: str,
        sparse_type: str,
        mask_suffix: str = "",
        image_size: Sequence[int] = (256, 256),
        base_seed: int = 42,
        cache_in_memory: bool = True,
        return_dense_mask: bool | None = None,
        point_config: Mapping[str, object] | None = None,
        scribble_config: Mapping[str, object] | None = None,
        bbox_config: Mapping[str, object] | None = None,
    ) -> None:
        sparse_type = str(sparse_type).lower()
        split = str(split).lower()

        if sparse_type not in SPARSE_TYPE_TO_INDEX:
            raise ValueError(
                f"Unsupported sparse_type '{sparse_type}'. "
                f"Choose from {tuple(SPARSE_TYPE_TO_INDEX)}."
            )

        if len(image_size) != 2:
            raise ValueError("image_size must be [height, width].")

        height, width = int(image_size[0]), int(image_size[1])
        if height <= 0 or width <= 0:
            raise ValueError(
                f"image_size values must be positive, got {(height, width)}."
            )

        self.client_id = str(client_id)
        self.split = split
        self.sparse_type = sparse_type
        self.condition_index = SPARSE_TYPE_TO_INDEX[sparse_type]
        self.image_size = (height, width)
        self.base_seed = int(base_seed)
        self.cache_in_memory = bool(cache_in_memory)

        # Safe default: dense masks are available for validation/test metrics,
        # but not exposed by the training dataset.
        if return_dense_mask is None:
            self.return_dense_mask = split in {
                "val",
                "valid",
                "validation",
                "test",
            }
        else:
            self.return_dense_mask = bool(return_dense_mask)

        self.point_config = dict(point_config or {})
        self.scribble_config = dict(scribble_config or {})
        self.bbox_config = dict(bbox_config or {})

        self.samples = discover_samples(
            dataset_root=dataset_root,
            client_id=self.client_id,
            split=self.split,
            mask_suffix=mask_suffix,
        )

        self._annotation_cache: MutableMapping[
            str,
            Dict[str, np.ndarray],
        ] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _cache_key(self, sample: BreastSample) -> str:
        return (
            f"{sample.client_id}/{sample.split}/"
            f"{sample.sample_id}/{self.sparse_type}/"
            f"{self.image_size[0]}x{self.image_size[1]}"
        )

    def _generate_annotation(
        self,
        sample: BreastSample,
        original_image: np.ndarray,
        original_dense_mask: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Generate the assigned weak annotation at original resolution.
        """
        if original_image.ndim != 3 or original_image.shape[2] != 3:
            raise RuntimeError(
                f"Expected RGB image [H,W,3], got "
                f"{original_image.shape}."
            )

        if original_dense_mask.ndim != 2:
            raise RuntimeError(
                f"Expected 2D mask, got "
                f"{original_dense_mask.shape}."
            )

        if original_image.shape[:2] != original_dense_mask.shape:
            raise RuntimeError(
                "Image and mask must be spatially aligned before "
                "weak-label generation."
            )

        item_seed = stable_sample_seed(
            base_seed=self.base_seed,
            client_id=sample.client_id,
            split=sample.split,
            sample_id=sample.sample_id,
            sparse_type=self.sparse_type,
        )

        if self.sparse_type == "point":
            annotation = generate_point_annotation(
                original_dense_mask,
                patch_size=int(
                    self.point_config.get("patch_size", 8)
                ),
            )

        elif self.sparse_type == "scribble":
            annotation = generate_scribble_annotation(
                original_dense_mask,
                rng=random.Random(item_seed),
                erosion_min=int(
                    self.scribble_config.get("erosion_min", 1)
                ),
                erosion_max=int(
                    self.scribble_config.get("erosion_max", 11)
                ),
                dilation_size=int(
                    self.scribble_config.get("dilation_size", 2)
                ),
            )

        elif self.sparse_type == "supervised":
            # Fully supervised client: the dense mask is NOT placed in the
            # model condition. Placeholder sparse maps are all zero; the dense
            # mask is returned separately as the training target.
            zeros = np.zeros_like(
                original_dense_mask,
                dtype=np.uint8,
            )
            annotation = {
                "positive": zeros.copy(),
                "negative": zeros.copy(),
                "box_region": zeros.copy(),
            }

        elif self.sparse_type == "labelled":
            dense = (
                np.asarray(original_dense_mask) > 0
            ).astype(np.uint8)

            annotation = {
                "positive": dense.copy(),
                "negative": (1 - dense).astype(np.uint8),
                "box_region": np.zeros_like(
                    dense,
                    dtype=np.uint8,
                ),
            }

        else:
            foreground_config = dict(
                self.bbox_config.get(
                    "foreground_seed",
                    {},
                )
            )

            annotation = generate_bbox_annotation(
                mask=original_dense_mask,
                image=original_image,
                rng=np.random.default_rng(item_seed),
                min_component_area=int(
                    self.bbox_config.get(
                        "min_component_area",
                        100,
                    )
                ),
                max_displacement=int(
                    self.bbox_config.get(
                        "max_displacement",
                        2,
                    )
                ),
                noisy=bool(
                    self.bbox_config.get(
                        "noisy",
                        True,
                    )
                ),
                number_of_points=int(
                    foreground_config.get(
                        "number_of_points",
                        4,
                    )
                ),
                patch_size=int(
                    foreground_config.get(
                        "patch_size",
                        3,
                    )
                ),
                center_weight=float(
                    foreground_config.get(
                        "center_weight",
                        1.0,
                    )
                ),
                gradient_weight=float(
                    foreground_config.get(
                        "gradient_weight",
                        1.0,
                    )
                ),
                inner_margin_fraction=float(
                    foreground_config.get(
                        "inner_margin_fraction",
                        0.15,
                    )
                ),
            )

        return _validate_annotation(
            annotation=annotation,
            expected_shape=original_dense_mask.shape,
            sparse_type=self.sparse_type,
        )

    def _load_or_generate_annotation(
        self,
        sample: BreastSample,
        original_image: np.ndarray,
        original_dense_mask: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        cache_key = self._cache_key(sample)

        if self.cache_in_memory and cache_key in self._annotation_cache:
            return {
                key: value.copy()
                for key, value in self._annotation_cache[
                    cache_key
                ].items()
            }

        annotation = self._generate_annotation(
            sample=sample,
            original_image=original_image,
            original_dense_mask=original_dense_mask,
        )

        if self.cache_in_memory:
            self._annotation_cache[cache_key] = {
                key: value.copy()
                for key, value in annotation.items()
            }

        return annotation

    def _resize_annotation(
        self,
        annotation: Mapping[str, np.ndarray],
        resized_dense_mask: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Resize a generated weak annotation to the network resolution.
        """
        if self.sparse_type in {"point", "scribble"}:
            positive = _resize_sparse_binary(
                annotation["positive"],
                self.image_size,
            )
        else:
            # Bbox and labelled annotations are dense regions.
            positive = _resize_binary(
                annotation["positive"],
                self.image_size,
            )

        box_region = _resize_binary(
            annotation["box_region"],
            self.image_size,
        )

        if resized_dense_mask.shape != self.image_size:
            raise RuntimeError(
                "resized_dense_mask has shape "
                f"{resized_dense_mask.shape}; expected "
                f"{self.image_size}."
            )

        if self.sparse_type == "bbox":
            # Do not use the dense mask to alter bbox foreground seeds.
            # GFedBridge bbox supervision remains Select(image, box).
            positive = (
                positive * box_region
            ).astype(np.uint8)

            negative = (
                1 - box_region
            ).astype(np.uint8)

        elif self.sparse_type == "supervised":
            # Fully supervised mode uses the dense mask only as a separate
            # target. Sparse maps and all condition channels stay zero.
            positive = np.zeros_like(
                resized_dense_mask,
                dtype=np.uint8,
            )
            negative = np.zeros_like(
                resized_dense_mask,
                dtype=np.uint8,
            )
            box_region.fill(0)

        elif self.sparse_type == "labelled":
            # Oracle/labelled-condition mode: the complete resized mask is
            # supplied as the condition and as dense positive/negative seeds.
            dense = (
                np.asarray(resized_dense_mask) > 0
            ).astype(np.uint8)

            positive = dense.copy()
            negative = (1 - dense).astype(np.uint8)
            box_region.fill(0)

        else:
            negative = _resize_sparse_binary(
                annotation["negative"],
                self.image_size,
            )
            box_region.fill(0)

            # Sparse maps and GT are resized independently. Occupancy-
            # preserving downsampling can move a boundary label by one output
            # pixel. Re-apply the class constraint at network resolution.
            dense = (
                np.asarray(resized_dense_mask) > 0
            ).astype(np.uint8)

            positive = (
                positive * dense
            ).astype(np.uint8)
            negative = (
                negative * (1 - dense)
            ).astype(np.uint8)

            # Preserve minimal positive supervision for a non-empty lesion.
            if dense.any() and not positive.any():
                candidates = np.argwhere(dense > 0)
                y, x = candidates[len(candidates) // 2]
                positive[int(y), int(x)] = 1

            if not dense.any():
                positive.fill(0)
                negative.fill(1)

        overlap = ((positive > 0) & (negative > 0))

        if overlap.any():
            negative[overlap] = 0

        return {
            "positive": positive,
            "negative": negative,
            "box_region": box_region,
        }

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.samples[index]

        with Image.open(sample.image_path) as handle:
            original_image = handle.convert("RGB")
            original_width, original_height = original_image.size

            original_image_array = np.asarray(
                original_image,
                dtype=np.uint8,
            )

            resized_image = _resize_image(
                original_image,
                self.image_size,
            )

        with Image.open(sample.mask_path) as handle:
            original_dense_mask = np.asarray(
                handle.convert("L"),
                dtype=np.uint8,
            )

        original_dense_mask = (
            original_dense_mask > 0
        ).astype(np.uint8)

        expected_mask_shape = (
            original_height,
            original_width,
        )

        if original_dense_mask.shape != expected_mask_shape:
            warnings.warn(
                (
                    f"Aligning mask for '{sample.sample_id}' from "
                    f"{original_dense_mask.shape} to "
                    f"{expected_mask_shape} using nearest-neighbour "
                    "interpolation."
                ),
                RuntimeWarning,
                stacklevel=2,
            )

            mask_image = Image.fromarray(
                original_dense_mask * 255
            )
            mask_image = mask_image.resize(
                (original_width, original_height),
                resample=Image.Resampling.NEAREST,
            )
            original_dense_mask = (
                np.asarray(mask_image, dtype=np.uint8) > 0
            ).astype(np.uint8)

        annotation = self._load_or_generate_annotation(
            sample=sample,
            original_image=original_image_array,
            original_dense_mask=original_dense_mask,
        )

        resized_dense_mask = _resize_binary(
            original_dense_mask,
            self.image_size,
        )

        annotation = self._resize_annotation(
            annotation,
            resized_dense_mask=resized_dense_mask,
        )

        positive = annotation["positive"]
        negative = annotation["negative"]
        box_region = annotation["box_region"]
        has_foreground = bool(positive.any())

        known_mask = ((positive > 0) | (negative > 0)).astype(np.uint8)
        sparse_target = positive.copy()

        # The model always receives three condition channels:
        #   channel 0: foreground labels
        #   channel 1: background labels
        #   channel 2: bbox region (zero for labelled mode)
        #
        # In labelled mode, channels 0 and 1 contain the complete foreground
        # and background masks. This is an oracle condition, not a standard
        # image-only supervised baseline.
        if self.sparse_type == "supervised":
            condition = np.zeros(
                (3, self.image_size[0], self.image_size[1]),
                dtype=np.float32,
            )
        else:
            condition = np.stack(
                [positive, negative, box_region],
                axis=0,
            ).astype(np.float32)

        output: Dict[str, object] = {
            "image": _image_tensor(resized_image),
            "condition": torch.from_numpy(condition).contiguous(),
            "positive": _binary_tensor(positive),
            "negative": _binary_tensor(negative),
            "box_region": _binary_tensor(box_region),
            "has_foreground": torch.tensor(has_foreground, dtype=torch.bool,),
            "known_mask": _binary_tensor(known_mask),
            "sparse_target": _binary_tensor(sparse_target),
            "condition_index": torch.tensor(self.condition_index, dtype=torch.long,),
            "client_id": sample.client_id,
            "split": sample.split,
            "sample_id": sample.sample_id,
            "case_name": sample.sample_id,
            "sparse_type": self.sparse_type,
            "original_size": torch.tensor([original_height, original_width], dtype=torch.int64,),
        }

        if self.return_dense_mask or self.sparse_type == "supervised":
            output["dense_mask"] = _binary_tensor(resized_dense_mask)

        return output