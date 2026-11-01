from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt


def _surface(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)

    eroded = binary_erosion(
        mask,
        structure=np.ones((3, 3)),
        border_value=0,
    )
    return np.logical_xor(mask, eroded)


def hd95_single(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, float] = (1.0, 1.0),
) -> float:
    """
    Symmetric 95th-percentile Hausdorff distance.

    Returns:
      0.0  : both masks empty
      inf  : exactly one mask empty
      value: HD95 in pixels when spacing=(1,1), otherwise in physical units
    """
    prediction = prediction.astype(bool)
    target = target.astype(bool)

    if not prediction.any() and not target.any():
        return 0.0

    if not prediction.any() or not target.any():
        return float("inf")

    prediction_surface = _surface(prediction)
    target_surface = _surface(target)

    distance_to_target = distance_transform_edt(
        ~target_surface,
        sampling=spacing,
    )
    distance_to_prediction = distance_transform_edt(
        ~prediction_surface,
        sampling=spacing,
    )

    prediction_to_target = distance_to_target[prediction_surface]
    target_to_prediction = distance_to_prediction[target_surface]

    all_surface_distances = np.concatenate(
        [prediction_to_target, target_to_prediction]
    )

    return float(np.percentile(all_surface_distances, 95))


@torch.no_grad()
def per_sample_segmentation_metrics(
    probability: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
    epsilon: float = 1e-6,
    spacing: tuple[float, float] = (1.0, 1.0),
) -> dict[str, list[float]]:
    prediction = (probability >= threshold).float()
    target = (target >= 0.5).float()

    dice_values: list[float] = []
    iou_values: list[float] = []
    hd95_values: list[float] = []

    for sample_index in range(prediction.shape[0]):
        pred = prediction[sample_index, 0]
        gt = target[sample_index, 0]

        intersection = float((pred * gt).sum().item())
        prediction_size = float(pred.sum().item())
        target_size = float(gt.sum().item())
        union = prediction_size + target_size - intersection

        dice = (
            2.0 * intersection + epsilon
        ) / (
            prediction_size + target_size + epsilon
        )

        iou = (
            intersection + epsilon
        ) / (
            union + epsilon
        )

        hd95 = hd95_single(
            pred.detach().cpu().numpy(),
            gt.detach().cpu().numpy(),
            spacing=spacing,
        )

        dice_values.append(float(dice))
        iou_values.append(float(iou))
        hd95_values.append(float(hd95))

    return {
        "dice": dice_values,
        "iou": iou_values,
        "hd95": hd95_values,
    }


def summarize_metric(
    values: list[float],
) -> tuple[float, float, int]:
    """
    Return mean, sample SD, and number of finite observations.

    Infinite HD95 values are excluded from the finite mean and SD. The caller
    should report the number of excluded infinite cases separately.
    """
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]

    if finite.size == 0:
        return float("nan"), float("nan"), 0

    mean = float(finite.mean())
    standard_deviation = (
        float(finite.std(ddof=1))
        if finite.size > 1
        else 0.0
    )

    return mean, standard_deviation, int(finite.size)


def format_mean_sd(
    values: list[float],
    decimals: int = 4,
) -> str:
    mean, standard_deviation, count = summarize_metric(values)

    if count == 0:
        return "nan ± nan"

    return (
        f"{mean:.{decimals}f} ± "
        f"{standard_deviation:.{decimals}f}"
    )