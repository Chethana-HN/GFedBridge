from __future__ import annotations

import random
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import dilation, footprint_rectangle, skeletonize


def _as_binary_2d(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)

    if array.ndim == 3:
        if array.shape[0] == 1:
            array = array[0]
        elif array.shape[-1] == 1:
            array = array[..., 0]
        else:
            raise ValueError(
                f"Expected a single-channel mask, got {array.shape}."
            )

    if array.ndim != 2:
        raise ValueError(
            f"Expected a 2D mask, got {array.shape}."
        )

    return (array > 0).astype(np.uint8)


def _empty_annotation(shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
    return {
        "positive": np.zeros(shape, dtype=np.uint8),
        "negative": np.zeros(shape, dtype=np.uint8),
        "box_region": np.zeros(shape, dtype=np.uint8),
    }


def _paint_patch(
    canvas: np.ndarray,
    center_x: int,
    center_y: int,
    patch_size: int,
    valid_region: np.ndarray | None = None,
) -> None:
    """
    Paint a square patch centred as closely as possible on (x, y).

    FedSLAG used top-left anchored 8x8 patches. Here we preserve the same
    patch-based point supervision while centring the patch and optionally
    clipping it to a valid foreground/background region.
    """
    patch_size = max(1, int(patch_size))
    height, width = canvas.shape

    before = (patch_size - 1) // 2
    after = patch_size - before

    x0 = max(0, int(center_x) - before)
    x1 = min(width, int(center_x) + after)
    y0 = max(0, int(center_y) - before)
    y1 = min(height, int(center_y) + after)

    if x0 >= x1 or y0 >= y1:
        return

    if valid_region is None:
        canvas[y0:y1, x0:x1] = 1
    else:
        region = valid_region[y0:y1, x0:x1] > 0
        canvas[y0:y1, x0:x1][region] = 1


def _component_contours(mask: np.ndarray) -> List[np.ndarray]:
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    return contours


def generate_point_annotation(
    mask: np.ndarray,
    patch_size: int = 8,
) -> Dict[str, np.ndarray]:
    """
    Generate FedSLAG-style point supervision.

    For each foreground contour:
      - four foreground patches are placed around the lesion interior;
      - four background patches are placed farther from the lesion;
      - foreground patches are clipped to foreground;
      - background patches are clipped to background.

    For empty masks, only deterministic background patches are produced.
    """
    binary = _as_binary_2d(mask)
    height, width = binary.shape

    positive = np.zeros_like(binary, dtype=np.uint8)
    negative = np.zeros_like(binary, dtype=np.uint8)
    background = (1 - binary).astype(np.uint8)

    contours = _component_contours(binary)

    if not contours:
        cx = width // 2
        cy = height // 2
        offset = max(8, min(height, width) // 8)

        for px, py in (
            (cx - offset, cy),
            (cx + offset, cy),
            (cx, cy - offset),
            (cx, cy + offset),
        ):
            _paint_patch(
                negative,
                px,
                py,
                patch_size,
                valid_region=background,
            )

        return {
            "positive": positive,
            "negative": negative,
            "box_region": np.zeros_like(binary),
        }

    for contour in contours:
        moments = cv2.moments(contour)

        if moments["m00"] != 0:
            cx = int(round(moments["m10"] / moments["m00"]))
            cy = int(round(moments["m01"] / moments["m00"]))
        else:
            x, y, w, h = cv2.boundingRect(contour)
            cx = x + w // 2
            cy = y + h // 2

        x, y, w, h = cv2.boundingRect(contour)

        # Same geometric idea as the FedSLAG implementation.
        cx_ratio = 0.7
        cy_ratio = 0.3
        cx_boundary = int(
            cx_ratio * cx
            + (1.0 - cx_ratio) * (x + w / 2.0)
        )
        cy_boundary = int(
            cy_ratio * cy
            + (1.0 - cy_ratio) * (y + h / 2.0)
        )

        foreground_centres = (
            (cx, cy_boundary - h // 4),
            (cx, cy_boundary + h // 4),
            (cx_boundary - w // 4, cy),
            (cx_boundary + w // 4, cy),
        )

        background_centres = (
            (cx, cy_boundary - h // 2),
            (cx, cy_boundary + h // 2),
            (cx_boundary - w // 2, cy),
            (cx_boundary + w // 2, cy),
        )

        for px, py in foreground_centres:
            _paint_patch(
                positive,
                px,
                py,
                patch_size,
                valid_region=binary,
            )

        for px, py in background_centres:
            _paint_patch(
                negative,
                px,
                py,
                patch_size,
                valid_region=background,
            )

    negative[positive > 0] = 0

    return {
        "positive": positive,
        "negative": negative,
        "box_region": np.zeros_like(binary),
    }


def _adaptive_erosion(
    region: np.ndarray,
    rng: random.Random,
    erosion_min: int,
    erosion_max: int,
    area_threshold: int = 400,
) -> np.ndarray:
    """
    FedSLAG-style random erosion with safeguards against deleting a component.
    """
    region = (region > 0).astype(np.uint8)

    if int(region.sum()) <= area_threshold:
        return region

    minimum = max(0, int(erosion_min))
    maximum = max(minimum, int(erosion_max))
    iterations = rng.randint(minimum, maximum)

    if iterations <= 0:
        return region

    structure = ndimage.generate_binary_structure(2, 2)
    eroded = ndimage.binary_erosion(
        region,
        structure=structure,
        iterations=iterations,
    )

    # Do not allow random erosion to erase an existing class completely.
    if not eroded.any():
        for reduced_iterations in range(iterations - 1, 0, -1):
            eroded = ndimage.binary_erosion(
                region,
                structure=structure,
                iterations=reduced_iterations,
            )
            if eroded.any():
                break

    if not eroded.any():
        return region

    return eroded.astype(np.uint8)


def _skeleton_for_region(
    region: np.ndarray,
    rng: random.Random,
    erosion_min: int,
    erosion_max: int,
    dilation_size: int,
) -> np.ndarray:
    """
    Generate one class scribble using FedSLAG's erode→skeletonize→dilate flow.

    Important correction:
    ``skeletonize`` returns a boolean array in current scikit-image versions.
    Comparing it with 255, as in the original script, can erase the skeleton.
    """
    region = (region > 0).astype(np.uint8)

    if not region.any():
        return np.zeros_like(region, dtype=np.uint8)

    working = _adaptive_erosion(
        region=region,
        rng=rng,
        erosion_min=erosion_min,
        erosion_max=erosion_max,
    )

    skeleton = skeletonize(
        working > 0,
        method="lee",
    )

    # Correct boolean handling: never compare the result to 255.
    skeleton = (np.asarray(skeleton) > 0).astype(np.uint8)

    if not skeleton.any():
        skeleton = skeletonize(
            region > 0,
            method="lee",
        )
        skeleton = (
            np.asarray(skeleton) > 0
        ).astype(np.uint8)

    # A non-empty class must not silently lose all supervision. This can
    # happen for very small components with some skeletonize versions.
    # Select one real class pixel nearest the component centroid as a
    # minimal one-pixel scribble fallback.
    if not skeleton.any():
        ys, xs = np.nonzero(region)
        if len(xs) > 0:
            centre_x = float(xs.mean())
            centre_y = float(ys.mean())
            index = int(
                np.argmin(
                    (xs.astype(np.float32) - centre_x) ** 2
                    + (ys.astype(np.float32) - centre_y) ** 2
                )
            )
            skeleton[int(ys[index]), int(xs[index])] = 1

    if int(dilation_size) > 1 and skeleton.any():
        skeleton = dilation(
            skeleton > 0,
            footprint_rectangle(
                (int(dilation_size), int(dilation_size))
            ),
        ).astype(np.uint8)

    # Scribbles must remain inside their class region.
    return (
        skeleton * region
    ).astype(np.uint8)


def _fallback_background_scribble(
    shape: Tuple[int, int],
    width: int = 2,
) -> np.ndarray:
    """
    Deterministic background scribble for an all-background image.

    A full-image background skeleton can be empty or dominated by borders,
    so use a sparse central cross only for the empty-lesion special case.
    """
    height, width_image = shape
    output = np.zeros(shape, dtype=np.uint8)

    # Scale stroke thickness with source resolution so a sparse cross does
    # not disappear when a large source image is reduced to 256x256.
    width = max(
        int(width),
        int(np.ceil(min(height, width_image) / 128.0)),
    )

    cy = height // 2
    cx = width_image // 2
    half_length = max(6, min(height, width_image) // 8)

    output[
        max(0, cy - width // 2):
        min(height, cy + (width + 1) // 2),
        max(0, cx - half_length):
        min(width_image, cx + half_length + 1),
    ] = 1

    output[
        max(0, cy - half_length):
        min(height, cy + half_length + 1),
        max(0, cx - width // 2):
        min(width_image, cx + (width + 1) // 2),
    ] = 1

    return output


def generate_scribble_annotation(
    mask: np.ndarray,
    rng: random.Random,
    erosion_min: int = 1,
    erosion_max: int = 11,
    dilation_size: int = 2,
) -> Dict[str, np.ndarray]:
    """
    Generate binary foreground/background scribbles.

    This follows FedSLAG's class-wise procedure:
      foreground mask → erode → skeletonize → dilate
      background mask → erode → skeletonize → dilate

    The resulting background scribble may be loop-like because it is the
    skeleton of the full background class. That is expected FedSLAG behaviour.
    """
    binary = _as_binary_2d(mask)
    foreground = binary
    background = (1 - binary).astype(np.uint8)

    if foreground.any():
        positive = _skeleton_for_region(
            region=foreground,
            rng=rng,
            erosion_min=erosion_min,
            erosion_max=erosion_max,
            dilation_size=dilation_size,
        )
    else:
        positive = np.zeros_like(binary, dtype=np.uint8)

    if foreground.any():
        negative = _skeleton_for_region(
            region=background,
            rng=rng,
            erosion_min=erosion_min,
            erosion_max=erosion_max,
            dilation_size=dilation_size,
        )
    else:
        negative = _fallback_background_scribble(
            binary.shape,
            width=max(1, int(dilation_size)),
        )

    positive = (
        positive * foreground
    ).astype(np.uint8)
    negative = (
        negative * background
    ).astype(np.uint8)
    negative[positive > 0] = 0

    return {
        "positive": positive,
        "negative": negative,
        "box_region": np.zeros_like(binary),
    }


def _connected_component_boxes(
    mask: np.ndarray,
    rng: np.random.Generator,
    min_component_area: int,
    max_displacement: int,
    noisy: bool,
) -> List[Tuple[int, int, int, int]]:
    """
    FedSLAG-style component bounding boxes with optional x/y/w/h perturbation.
    """
    binary = _as_binary_2d(mask)
    height, width = binary.shape

    number, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    boxes: List[Tuple[int, int, int, int]] = []

    for label_id in range(1, number):
        area = int(
            stats[label_id, cv2.CC_STAT_AREA]
        )
        if area < int(min_component_area):
            continue

        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])

        if noisy and int(max_displacement) > 0:
            displacement = int(max_displacement)
            x += int(
                rng.integers(
                    -displacement,
                    displacement + 1,
                )
            )
            y += int(
                rng.integers(
                    -displacement,
                    displacement + 1,
                )
            )
            w += int(
                rng.integers(
                    -displacement,
                    displacement + 1,
                )
            )
            h += int(
                rng.integers(
                    -displacement,
                    displacement + 1,
                )
            )

        x0 = int(np.clip(x, 0, width - 1))
        y0 = int(np.clip(y, 0, height - 1))
        x1 = int(np.clip(x + max(1, w), x0 + 1, width))
        y1 = int(np.clip(y + max(1, h), y0 + 1, height))

        boxes.append(
            (x0, y0, x1, y1)
        )

    return boxes


def _normalise_gray(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)

    if array.ndim == 3 and array.shape[2] == 3:
        gray = cv2.cvtColor(
            array.astype(np.uint8),
            cv2.COLOR_RGB2GRAY,
        )
    elif array.ndim == 2:
        gray = array
    else:
        raise ValueError(
            f"Expected RGB or grayscale image, got {array.shape}."
        )

    gray = gray.astype(np.float32)

    minimum = float(gray.min())
    maximum = float(gray.max())

    if maximum > minimum:
        gray = (
            gray - minimum
        ) / (
            maximum - minimum
        )
    else:
        gray = np.zeros_like(gray, dtype=np.float32)

    return gray


def _bbox_image_supported_foreground(
    image: np.ndarray,
    box_region: np.ndarray,
    rng: np.random.Generator,
    number_of_points: int,
    patch_size: int,
    center_weight: float,
    gradient_weight: float,
    inner_margin_fraction: float,
) -> np.ndarray:
    """
    Implement the paper's Select(x, B).

    Only image and box geometry are used here. The dense mask is not consulted.
    Candidate pixels are restricted to the inner box and ranked using:
      - preference for central box locations;
      - preference for low local gradient (region interior evidence);
      - deterministic tiny jitter for tie-breaking.
    """
    box = (box_region > 0).astype(np.uint8)
    output = np.zeros_like(box, dtype=np.uint8)

    ys, xs = np.nonzero(box)
    if len(xs) == 0:
        return output

    gray = _normalise_gray(image)

    gradient_x = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )
    gradient_y = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )
    gradient = np.sqrt(
        gradient_x ** 2 + gradient_y ** 2
    )

    gradient_max = float(gradient.max())
    if gradient_max > 0:
        gradient = gradient / gradient_max

    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1

    box_width = max(1, x1 - x0)
    box_height = max(1, y1 - y0)

    margin_x = int(
        round(
            box_width
            * float(inner_margin_fraction)
        )
    )
    margin_y = int(
        round(
            box_height
            * float(inner_margin_fraction)
        )
    )

    inner_x0 = min(x1 - 1, x0 + margin_x)
    inner_x1 = max(inner_x0 + 1, x1 - margin_x)
    inner_y0 = min(y1 - 1, y0 + margin_y)
    inner_y1 = max(inner_y0 + 1, y1 - margin_y)

    inner = np.zeros_like(box, dtype=np.uint8)
    inner[
        inner_y0:inner_y1,
        inner_x0:inner_x1,
    ] = 1
    inner *= box

    candidate_y, candidate_x = np.nonzero(inner)

    if len(candidate_x) == 0:
        candidate_y, candidate_x = np.nonzero(box)

    centre_x = (x0 + x1 - 1) / 2.0
    centre_y = (y0 + y1 - 1) / 2.0

    normaliser = max(
        1.0,
        np.hypot(
            box_width / 2.0,
            box_height / 2.0,
        ),
    )

    centre_distance = np.hypot(
        candidate_x.astype(np.float32) - centre_x,
        candidate_y.astype(np.float32) - centre_y,
    ) / normaliser

    scores = (
        float(center_weight) * centre_distance
        + float(gradient_weight)
        * gradient[candidate_y, candidate_x]
    )

    scores += rng.uniform(
        0.0,
        1e-6,
        size=scores.shape,
    )

    order = np.argsort(scores)
    chosen: List[Tuple[int, int]] = []
    minimum_spacing = max(1.0, float(patch_size) * 1.5)

    for index in order:
        x = int(candidate_x[index])
        y = int(candidate_y[index])

        if all(
            np.hypot(x - old_x, y - old_y)
            >= minimum_spacing
            for old_x, old_y in chosen
        ):
            chosen.append((x, y))

        if len(chosen) >= int(number_of_points):
            break

    if not chosen and len(order) > 0:
        first = int(order[0])
        chosen.append(
            (
                int(candidate_x[first]),
                int(candidate_y[first]),
            )
        )

    for x, y in chosen:
        _paint_patch(
            output,
            x,
            y,
            patch_size,
            valid_region=box,
        )

    return (
        output * box
    ).astype(np.uint8)


def generate_bbox_annotation(
    mask: np.ndarray,
    image: np.ndarray,
    rng: np.random.Generator,
    min_component_area: int = 100,
    max_displacement: int = 2,
    noisy: bool = True,
    number_of_points: int = 4,
    patch_size: int = 3,
    center_weight: float = 1.0,
    gradient_weight: float = 1.0,
    inner_margin_fraction: float = 0.15,
) -> Dict[str, np.ndarray]:
    """
    Generate paper-consistent bbox supervision.

    FedSLAG component boxes are retained, but GFedBridge does not treat the
    full box as foreground:

      box_region = noisy component boxes
      negative   = outside-box trusted background
      positive   = Select(image, box_region), a small image-supported seed set
    """
    binary = _as_binary_2d(mask)

    if np.asarray(image).shape[:2] != binary.shape:
        raise ValueError(
            "Image and mask must be aligned for bbox seed selection: "
            f"{np.asarray(image).shape[:2]} vs {binary.shape}."
        )

    boxes = _connected_component_boxes(
        mask=binary,
        rng=rng,
        min_component_area=min_component_area,
        max_displacement=max_displacement,
        noisy=noisy,
    )

    box_region = np.zeros_like(binary, dtype=np.uint8)

    for x0, y0, x1, y1 in boxes:
        box_region[y0:y1, x0:x1] = 1

    if not box_region.any():
        return _empty_annotation(binary.shape)

    positive = _bbox_image_supported_foreground(
        image=image,
        box_region=box_region,
        rng=rng,
        number_of_points=number_of_points,
        patch_size=patch_size,
        center_weight=center_weight,
        gradient_weight=gradient_weight,
        inner_margin_fraction=inner_margin_fraction,
    )

    positive = (
        positive * box_region
    ).astype(np.uint8)
    negative = (
        1 - box_region
    ).astype(np.uint8)

    return {
        "positive": positive,
        "negative": negative,
        "box_region": box_region,
    }