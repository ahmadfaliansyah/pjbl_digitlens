"""Digit segmentation for DigitLens.

This module finds candidate digit regions; it does NOT classify digits.
Each candidate can be sent to a trained model (for example, an MNIST CNN).

Public interface kept compatible with the original DigitLens pages:
- segment_digits(image, padding=8, min_area=None) -> SegmentationResult
- result.original_with_boxes
- result.binary_preview
- result.regions (each region has index, bbox, area, crop)
- result.background, result.threshold, result.warnings

Important:
`DigitRegion.crop` is the original-color crop for display.
`DigitRegion.model_input` is an optional 28x28 MNIST-style image
(white digit on black background) derived from the selected segmentation mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw


@dataclass
class DigitRegion:
    """One candidate digit region, in reading order."""

    index: int
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    area: int
    crop: Image.Image
    # Kept optional so existing code that only reads .crop remains compatible.
    model_input: Image.Image | None = None


@dataclass
class SegmentationResult:
    original_with_boxes: Image.Image
    binary_preview: Image.Image
    regions: List[DigitRegion]
    background: str
    threshold: float
    warnings: List[str]


def _to_gray_array(image: Image.Image) -> np.ndarray:
    """Convert RGB/RGBA input to an 8-bit grayscale array."""
    rgb = image.convert("RGB")
    return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2GRAY)


def _make_threshold_candidates(gray: np.ndarray) -> list[tuple[np.ndarray, float, bool]]:
    """Create possible masks for dark-on-light and light-on-dark digits.

    We test more than one threshold because Otsu can mistake a textured
    background (for example, printed banknote patterns) for the foreground.
    Each returned mask uses WHITE pixels for the possible digit foreground.
    """
    # Bilateral filtering reduces small texture/noise while preserving strong
    # edges such as the outline of printed digits.
    smooth = cv2.bilateralFilter(gray, d=7, sigmaColor=35, sigmaSpace=7)

    otsu_value, _ = cv2.threshold(
        smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Try Otsu and two stricter thresholds. Stricter thresholds can help
    # separate very dark ink from medium-gray security patterns.
    threshold_values = sorted(
        {
            int(round(otsu_value)),
            max(20, int(round(otsu_value - 20))),
            max(20, int(round(otsu_value - 40))),
        },
        reverse=True,
    )

    candidates: list[tuple[np.ndarray, float, bool]] = []
    open_kernel = np.ones((2, 2), dtype=np.uint8)
    close_kernel = np.ones((3, 3), dtype=np.uint8)

    for threshold_value in threshold_values:
        for dark_foreground in (True, False):
            threshold_type = (
                cv2.THRESH_BINARY_INV if dark_foreground else cv2.THRESH_BINARY
            )
            _, mask = cv2.threshold(smooth, threshold_value, 255, threshold_type)

            # Remove tiny isolated specks, then close very small gaps in strokes.
            # A small kernel is used to avoid joining neighboring digits.
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_OPEN, open_kernel, iterations=1
            )
            # A slightly larger closing kernel bridges narrow gaps caused by
            # printed security lines crossing a thick digit stroke.
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE, close_kernel, iterations=1
            )

            candidates.append((mask, float(threshold_value), dark_foreground))

    return candidates


def _components_from_mask(
    mask: np.ndarray,
    min_area: int | None,
) -> list[tuple[int, int, int, int, int]]:
    """Return plausible character-like connected components.

    Each tuple is (x, y, width, height, area). Large page/background regions,
    tiny specks, and long thin texture lines are rejected.
    """
    height, width = mask.shape
    image_area = width * height

    # A small image-relative minimum prevents noise from becoming a digit.
    effective_min_area = (
        int(min_area)
        if min_area is not None
        else max(12, int(image_area * 0.00002))
    )
    min_height = max(8, int(height * 0.04))
    min_width = max(2, int(width * 0.002))

    count, _, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    components: list[tuple[int, int, int, int, int]] = []

    for label in range(1, count):
        x, y, w, h, area = (int(value) for value in stats[label])

        if area < effective_min_area:
            continue
        if h < min_height or w < min_width:
            continue
        if h > height * 0.95:
            continue

        # Long background textures often connect to the image boundary,
        # whereas a photographed number normally has a little margin.
        # Rejecting edge-touching components prevents diagonal security lines
        # from becoming false "digits".
        edge_margin = max(1, int(min(width, height) * 0.003))
        if (
            x <= edge_margin
            or y <= edge_margin
            or x + w >= width - edge_margin
            or y + h >= height - edge_margin
        ):
            continue

        aspect_ratio = w / max(h, 1)
        density = area / max(w * h, 1)

        # A digit may be narrow (for example "1"), but a one-pixel-wide
        # diagonal security line is not a useful digit candidate.
        if aspect_ratio < 0.035 or aspect_ratio > 1.55:
            continue
        if density < 0.025:
            continue

        components.append((x, y, w, h, area))

    if not components:
        return []

    # Remove components much shorter than the typical character height.
    # This gets rid of small remnants without letting one giant background
    # component determine the filtering threshold.
    typical_height = float(median([component[3] for component in components]))
    height_floor = max(min_height, int(typical_height * 0.35))
    components = [
        component for component in components if component[3] >= height_floor
    ]

    return components


def _score_components(
    components: list[tuple[int, int, int, int, int]],
    image_shape: tuple[int, int],
) -> float:
    """Score a threshold candidate; higher scores favor digit-like regions."""
    if not components:
        return -1_000_000.0

    image_height, image_width = image_shape
    heights = np.asarray([item[3] for item in components], dtype=np.float32)
    areas = np.asarray([item[4] for item in components], dtype=np.float32)

    count = len(components)
    height_consistency = float(
        np.std(heights) / max(float(np.mean(heights)), 1.0)
    )
    total_coverage = float(np.sum(areas) / max(image_height * image_width, 1))
    typical_height_ratio = float(np.median(heights) / max(image_height, 1))

    # Reward components that are tall enough to plausibly be characters.
    # Many tiny pieces from a patterned background should score lower than
    # a small set of full-height digits.
    score = 2.0 * min(count, 10)
    score -= 1.0 * max(0, count - 10)
    score -= 3.0 * height_consistency
    score += 20.0 * typical_height_ratio
    score += min(total_coverage * 10.0, 2.0)

    # A single very wide region is usually a background/object, not one digit.
    if count == 1:
        x, y, w, h, _ = components[0]
        if w > image_width * 0.22 or h > image_height * 0.90:
            score -= 8.0

    return score


def _sort_in_reading_order(
    components: list[tuple[int, int, int, int, int]],
) -> list[tuple[int, int, int, int, int]]:
    """Sort components top-to-bottom by rows, then left-to-right per row."""
    if len(components) < 2:
        return components

    typical_height = float(median([item[3] for item in components]))
    row_tolerance = max(10.0, typical_height * 0.55)

    # Start from the topmost component and assign components to a row using
    # their vertical center. This supports more than one line of numbers.
    by_vertical_position = sorted(components, key=lambda item: item[1] + item[3] / 2)
    rows: list[list[tuple[int, int, int, int, int]]] = []

    for component in by_vertical_position:
        center_y = component[1] + component[3] / 2
        best_row = None
        best_distance = None

        for row in rows:
            row_center = float(
                np.mean([item[1] + item[3] / 2 for item in row])
            )
            distance = abs(center_y - row_center)
            if distance <= row_tolerance and (
                best_distance is None or distance < best_distance
            ):
                best_row = row
                best_distance = distance

        if best_row is None:
            rows.append([component])
        else:
            best_row.append(component)

    rows.sort(
        key=lambda row: float(np.mean([item[1] + item[3] / 2 for item in row]))
    )

    ordered: list[tuple[int, int, int, int, int]] = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda item: item[0]))

    return ordered


def _make_mnist_input(binary_digit: np.ndarray) -> Image.Image:
    """Convert a white-foreground digit mask into a 28x28 MNIST-style image."""
    ys, xs = np.where(binary_digit > 0)
    canvas = np.zeros((28, 28), dtype=np.uint8)

    if len(xs) == 0 or len(ys) == 0:
        return Image.fromarray(canvas, mode="L")

    # Tight crop around the digit, then fit inside a 20x20 area while
    # preserving its aspect ratio.
    digit = binary_digit[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    digit_height, digit_width = digit.shape

    scale = min(20.0 / max(digit_width, 1), 20.0 / max(digit_height, 1))
    new_width = max(1, int(round(digit_width * scale)))
    new_height = max(1, int(round(digit_height * scale)))

    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(
        digit, (new_width, new_height), interpolation=interpolation
    )

    x_offset = (28 - new_width) // 2
    y_offset = (28 - new_height) // 2
    canvas[
        y_offset : y_offset + new_height,
        x_offset : x_offset + new_width,
    ] = resized

    # MNIST preprocessing commonly centers the digit by its center of mass.
    moments = cv2.moments(canvas)
    if moments["m00"] > 0:
        center_x = moments["m10"] / moments["m00"]
        center_y = moments["m01"] / moments["m00"]
        shift_x = int(round(13.5 - center_x))
        shift_y = int(round(13.5 - center_y))
        transform = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        canvas = cv2.warpAffine(
            canvas,
            transform,
            (28, 28),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    return Image.fromarray(canvas, mode="L")


def _segment_once(
    image: Image.Image,
    *,
    padding: int = 8,
    min_area: int | None = None,
) -> tuple[SegmentationResult, float]:
    """Find and crop candidate digits from an uploaded image (satu percobaan).

    The function tries both dark-on-light and light-on-dark foreground masks.
    It evaluates several thresholds, rejects large background regions and
    thin texture lines, and orders the remaining components top-to-bottom,
    then left-to-right.

    It works best when digits are visibly separated. Touching digits,
    complex patterned backgrounds, strong perspective, and very blurry
    images can still need a dedicated ROI/cropping step or a more advanced
    text detector.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("Input segment_digits harus berupa objek PIL.Image.")
    if image.width < 2 or image.height < 2:
        raise ValueError("Gambar terlalu kecil untuk diproses.")
    if padding < 0:
        raise ValueError("Padding tidak boleh bernilai negatif.")
    if min_area is not None and min_area < 1:
        raise ValueError("min_area harus lebih besar dari 0.")

    gray = _to_gray_array(image)
    height, width = gray.shape

    # Generate alternatives instead of deciding foreground polarity from the
    # image border. The border can belong to a dark table while the target
    # object (e.g. a banknote) is light.
    threshold_candidates = _make_threshold_candidates(gray)

    best_mask: np.ndarray | None = None
    best_components: list[tuple[int, int, int, int, int]] = []
    best_threshold = 0.0
    best_dark_foreground = True
    best_score = -1_000_000.0

    for mask, threshold_value, dark_foreground in threshold_candidates:
        components = _components_from_mask(mask, min_area)
        score = _score_components(components, (height, width))

        if score > best_score:
            best_score = score
            best_mask = mask
            best_components = components
            best_threshold = threshold_value
            best_dark_foreground = dark_foreground

    if best_mask is None:
        # Defensive fallback; normally threshold_candidates always has entries.
        best_mask = np.zeros((height, width), dtype=np.uint8)

    ordered_components = _sort_in_reading_order(best_components)

    rgb = image.convert("RGB")
    boxed = rgb.copy()
    draw = ImageDraw.Draw(boxed)
    regions: list[DigitRegion] = []

    for index, (x, y, w, h, area) in enumerate(ordered_components, start=1):
        left = max(0, x - padding)
        top = max(0, y - padding)
        right = min(width, x + w + padding)
        bottom = min(height, y + h + padding)

        original_crop = rgb.crop((left, top, right, bottom))
        digit_mask = best_mask[y : y + h, x : x + w]
        model_input = _make_mnist_input(digit_mask)

        regions.append(
            DigitRegion(
                index=index,
                bbox=(x, y, w, h),
                area=area,
                crop=original_crop,
                model_input=model_input,
            )
        )

        draw.rectangle(
            (x, y, x + w - 1, y + h - 1),
            outline=(37, 99, 235),
            width=max(2, min(4, int(round(min(width, height) / 250)))),
        )
        draw.text(
            (x, max(0, y - 18)),
            str(index),
            fill=(37, 99, 235),
            stroke_width=1,
        )

    warnings: list[str] = []

    if not regions:
        warnings.append(
            "Belum ada digit yang terdeteksi. Coba crop area angka lebih dekat, "
            "gunakan gambar yang lebih tajam, atau tingkatkan kontras."
        )
    else:
        if len(regions) == 1:
            x, y, w, h = regions[0].bbox
            if w > h * 1.4:
                warnings.append(
                    "Satu area sangat lebar terdeteksi. Beberapa digit mungkin "
                    "saling menempel atau latar belakang masih ikut terbaca."
                )
            else:
                warnings.append(
                    "Hanya satu kandidat digit ditemukan. Pastikan gambar memang "
                    "berisi satu digit atau coba crop area angka lebih dekat."
                )

        if len(regions) > 12:
            warnings.append(
                "Terdeteksi banyak kandidat. Sebagian mungkin masih berupa noise "
                "atau pecahan karakter; periksa kotak deteksi sebelum klasifikasi."
            )

    # Preview convention: white background, black foreground. This is easier
    # for a person to inspect, while model_input uses MNIST's white-on-black
    # convention.
    binary_preview = Image.fromarray(255 - best_mask, mode="L")

    result = SegmentationResult(
        original_with_boxes=boxed,
        binary_preview=binary_preview,
        regions=regions,
        background="terang" if best_dark_foreground else "gelap",
        threshold=float(best_threshold),
        warnings=warnings,
    )
    return result, best_score


def _add_margin(image: Image.Image) -> Image.Image:
    """Beri margin seragam di sekeliling gambar (warna = median piksel tepi).

    `_components_from_mask` menolak komponen yang menyentuh tepi gambar agar
    garis tekstur latar tidak dianggap digit. Akibatnya, angka yang di-crop
    terlalu rapat kehilangan digit di tepi. Dengan margin, digit tersebut tidak
    lagi menyentuh tepi gambar.
    """
    rgb = np.asarray(image.convert("RGB"))
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    fill = tuple(int(v) for v in np.median(border, axis=0))
    margin = max(20, int(round(0.10 * max(image.width, image.height))))
    padded = Image.new("RGB", (image.width + 2 * margin, image.height + 2 * margin), fill)
    padded.paste(image.convert("RGB"), (margin, margin))
    return padded


def segment_digits(
    image: Image.Image,
    *,
    padding: int = 8,
    min_area: int | None = None,
) -> SegmentationResult:
    """Find and crop candidate digits from an uploaded image.

    Gambar diproses apa adanya. Jika gambar yang diberi margin menghasilkan
    kandidat digit yang lebih meyakinkan (misalnya angka di-crop rapat sampai
    menyentuh tepi), hasil dengan margin dipakai dan pengguna diberi peringatan.
    Lihat `_segment_once` untuk detail algoritma.
    """
    result, score = _segment_once(image, padding=padding, min_area=min_area)
    padded_result, padded_score = _segment_once(
        _add_margin(image), padding=padding, min_area=min_area
    )
    if padded_result.regions and padded_score > score + 0.5:
        padded_result.warnings.insert(
            0,
            "Angka tampak menyentuh tepi gambar, sehingga gambar diberi margin otomatis. "
            "Koordinat kotak mengacu pada gambar yang sudah diberi margin.",
        )
        return padded_result
    return result
