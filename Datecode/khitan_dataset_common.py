from __future__ import annotations

import json
import math
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
SPLITS = ("train", "val", "test")

BBox = Tuple[float, float, float, float]
Rect = Tuple[float, float, float, float]


def image_files(path: Path) -> List[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def prepare_split_dirs(output_root: Path, clear_output: bool = False) -> None:
    if clear_output and output_root.exists():
        shutil.rmtree(output_root)
    for split in SPLITS:
        (output_root / split / "images").mkdir(parents=True, exist_ok=True)


def save_split_labels(output_root: Path, labels: Dict[str, Dict[str, list]]) -> None:
    for split, payload in labels.items():
        path = output_root / split / "labels.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)


def split_name(index: int, total: int, train_ratio: float, val_ratio: float) -> str:
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    if index < train_end:
        return "train"
    if index < val_end:
        return "val"
    return "test"


def load_char_map(radicals_file: Path) -> Dict[str, str]:
    content = radicals_file.read_text(encoding="utf-8").replace("\n", "").replace(" ", "")
    return {char: str(index) for index, char in enumerate(content)}


def make_background(size: int) -> np.ndarray:
    return np.zeros((size, size, 3), dtype=np.uint8)


def layout_templates() -> List[List[Rect]]:
    sx = random.uniform(0.45, 0.55)
    sy = random.uniform(0.45, 0.55)
    h1 = random.uniform(0.30, 0.36)
    h2 = random.uniform(0.63, 0.70)
    return [
        [(0.0, 0.0, 1.0, 1.0)],
        [(0.0, 0.0, sx, 1.0), (sx, 0.0, 1.0 - sx, 1.0)],
        [(0.0, 0.0, sx, sy), (sx, 0.0, 1.0 - sx, sy), (0.0, sy, 1.0, 1.0 - sy)],
        [(0.0, 0.0, sx, sy), (sx, 0.0, 1.0 - sx, sy), (0.0, sy, sx, 1.0 - sy), (sx, sy, 1.0 - sx, 1.0 - sy)],
        [(0.0, 0.0, sx, h1), (sx, 0.0, 1.0 - sx, h1), (0.0, h1, sx, h2 - h1), (sx, h1, 1.0 - sx, h2 - h1), (0.0, h2, 1.0, 1.0 - h2)],
        [(0.0, 0.0, sx, h1), (sx, 0.0, 1.0 - sx, h1), (0.0, h1, sx, h2 - h1), (sx, h1, 1.0 - sx, h2 - h1), (0.0, h2, sx, 1.0 - h2), (sx, h2, 1.0 - sx, 1.0 - h2)],
        [(0.0, 0.0, 0.5, 0.25), (0.5, 0.0, 0.5, 0.25), (0.0, 0.25, 0.5, 0.25), (0.5, 0.25, 0.5, 0.25), (0.0, 0.50, 0.5, 0.25), (0.5, 0.50, 0.5, 0.25), (0.2, 0.75, 0.6, 0.25)],
    ]


def layout_for_length(length: int) -> List[Rect]:
    templates = layout_templates()
    if length < 1 or length > len(templates):
        raise ValueError(f"Unsupported layout length: {length}")
    return templates[length - 1]


def jitter_rect(rect: Rect, jitter: float = 0.03, expand: Tuple[float, float] = (0.01, 0.04)) -> Rect:
    rx, ry, rw, rh = rect
    dx = random.uniform(-jitter / 2, jitter / 2)
    dy = random.uniform(-jitter / 2, jitter / 2)
    ew = random.uniform(*expand)
    eh = random.uniform(*expand)
    return max(0.0, min(rx + dx, 1.0)), max(0.0, min(ry + dy, 1.0)), rw + ew, rh + eh


def sharpen_normalize(img: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)
    _, img = cv2.threshold(img, 50, 255, cv2.THRESH_TOZERO)
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)


def elastic_transform(image: np.ndarray, alpha: float, sigma: float) -> np.ndarray:
    h, w = image.shape[:2]
    dx = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1), (0, 0), sigma) * alpha
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    return cv2.remap(
        image,
        np.float32(x + dx),
        np.float32(y + dy),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def process_patch(
    patch: np.ndarray,
    target_w: int,
    target_h: int,
    *,
    ink_prob: float = 0.3,
    elastic_prob: float = 0.0,
    elastic_strength: float = 0.10,
    perspective_prob: float = 0.6,
    perspective_twist: float = 0.03,
    rotation_range: float = 3.0,
    noise_prob: float = 0.3,
    noise_sigma: float = 5.0,
    normal_fill: float = 0.85,
    bottom_fill: float = 0.35,
    is_bottom_single: bool = False,
    sharpen: bool = True,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    h0, w0 = patch.shape[:2]

    if random.random() < ink_prob:
        kernel = np.ones((2, 2), np.uint8)
        patch = cv2.erode(patch, kernel, iterations=1) if random.random() < 0.5 else cv2.dilate(patch, kernel, iterations=1)

    if elastic_prob > 0 and random.random() < elastic_prob:
        base = min(w0, h0)
        patch = elastic_transform(patch, base * elastic_strength, max(1.0, base * 0.08))

    if random.random() < perspective_prob:
        src = np.float32([[0, 0], [w0, 0], [0, h0], [w0, h0]])
        twist = w0 * perspective_twist
        dst = np.float32([
            [random.uniform(-twist, twist), random.uniform(-twist, twist)],
            [w0 + random.uniform(-twist, twist), random.uniform(-twist, twist)],
            [random.uniform(-twist, twist), h0 + random.uniform(-twist, twist)],
            [w0 + random.uniform(-twist, twist), h0 + random.uniform(-twist, twist)],
        ])
        matrix = cv2.getPerspectiveTransform(src, dst)
        patch = cv2.warpPerspective(patch, matrix, (w0, h0), borderValue=(0, 0, 0))

    angle = random.uniform(-rotation_range, rotation_range)
    matrix = cv2.getRotationMatrix2D((w0 // 2, h0 // 2), angle, 1.0)
    patch = cv2.warpAffine(patch, matrix, (w0, h0), borderValue=(0, 0, 0))

    if random.random() < noise_prob:
        noise = np.random.normal(0, noise_sigma, patch.shape).astype(np.int16)
        patch = np.clip(patch.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        _, mask = cv2.threshold(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), 10, 255, cv2.THRESH_TOZERO)
        patch = cv2.bitwise_and(patch, patch, mask=mask)

    scale = min(target_w / w0, target_h / h0)
    fit_w, fit_h = int(w0 * scale), int(h0 * scale)
    fill = bottom_fill if is_bottom_single else normal_fill
    new_w = max(1, min(int(target_w * fill + fit_w * (1 - fill)), target_w))
    new_h = max(1, min(int(target_h * fill + fit_h * (1 - fill)), target_h))

    resized = cv2.resize(patch, (new_w, new_h))
    if sharpen:
        resized = sharpen_normalize(resized)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    off_x = (target_w - new_w) // 2
    off_y = (target_h - new_h) // 2
    canvas[off_y : off_y + new_h, off_x : off_x + new_w] = resized
    return canvas, (off_x, off_y, new_w, new_h)


def blend_additive(bg: np.ndarray, patch: np.ndarray, x: int, y: int, threshold: int = 15) -> np.ndarray:
    h, w = patch.shape[:2]
    bh, bw = bg.shape[:2]
    w = min(w, bw - x)
    h = min(h, bh - y)
    if w <= 0 or h <= 0:
        return bg
    patch = patch[:h, :w]
    roi = bg[y : y + h, x : x + w]
    _, mask = cv2.threshold(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), threshold, 255, cv2.THRESH_TOZERO)
    bg[y : y + h, x : x + w] = cv2.add(roi, cv2.bitwise_and(patch, patch, mask=mask))
    return bg


def blend_alpha(bg: np.ndarray, patch: np.ndarray, x: int, y: int, threshold: int = 15) -> np.ndarray:
    h, w = patch.shape[:2]
    bh, bw = bg.shape[:2]
    w = min(w, bw - x)
    h = min(h, bh - y)
    if w <= 0 or h <= 0:
        return bg
    patch = patch[:h, :w]
    roi = bg[y : y + h, x : x + w]
    _, mask = cv2.threshold(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.GaussianBlur(mask, (3, 3), 0).astype(np.float32) / 255.0
    mask = np.repeat(mask[:, :, None], 3, axis=2)
    blended = patch.astype(np.float32) * mask + roi.astype(np.float32) * (1.0 - mask)
    bg[y : y + h, x : x + w] = np.clip(blended, 0, 255).astype(np.uint8)
    return bg


def global_style(img: np.ndarray, blur_prob: float = 0.5, noise_prob: float = 0.7, noise_range: Tuple[int, int] = (5, 15), brightness: bool = True) -> np.ndarray:
    if random.random() < blur_prob:
        img = cv2.GaussianBlur(img, (random.choice([3, 5]), random.choice([3, 5])), 0)
    if random.random() < noise_prob:
        noise = np.random.normal(0, random.randint(*noise_range), img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    if brightness:
        img = cv2.convertScaleAbs(img, alpha=random.uniform(0.8, 1.2), beta=random.randint(-20, 20))
    return img


def global_contrast_degradation(img: np.ndarray, blur_prob: float = 0.4, noise_prob: float = 0.5) -> np.ndarray:
    if random.random() < blur_prob:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    if random.random() < noise_prob:
        noise = np.random.normal(0, random.randint(3, 8), img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def weathered_mask(shape: Tuple[int, int, int], num_lines: int, num_dots: int) -> np.ndarray:
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if num_lines <= 0 and num_dots <= 0:
        return mask

    for _ in range(num_lines):
        points = [np.array([random.randint(0, w - 1), random.randint(0, h - 1)])]
        max_len = random.randint(30, max(31, int(max(h, w) / 2)))
        for _ in range(random.randint(1, 3)):
            dist = random.randint(10, max(10, max_len // 2))
            angle = random.uniform(0, 2 * math.pi)
            prev = points[-1]
            nx = max(0, min(int(prev[0] + dist * math.cos(angle)), w - 1))
            ny = max(0, min(int(prev[1] + dist * math.sin(angle)), h - 1))
            points.append(np.array([nx, ny]))
        radius = random.uniform(2.0, 6.0)
        offset = 0.0
        for p1, p2 in zip(points, points[1:]):
            dist = float(np.linalg.norm(p2 - p1))
            if dist == 0:
                continue
            for step in range(int(dist)):
                t = step / dist
                p = p1 * (1 - t) + p2 * t
                r = max(1, int(radius + math.sin((offset + step) * 0.15) * 2.0 + random.uniform(-1.0, 1.5)))
                cv2.circle(mask, (int(p[0]), int(p[1])), r, 255, -1)
            offset += dist

    for _ in range(num_dots):
        cx, cy = random.randint(0, w - 1), random.randint(0, h - 1)
        r = random.uniform(3.0, 9.0)
        for _ in range(random.randint(2, 4)):
            ox = cx + random.randint(-int(r * 0.4), int(r * 0.4))
            oy = cy + random.randint(-int(r * 0.4), int(r * 0.4))
            ax1 = max(1, int(r * random.uniform(0.6, 1.4)))
            ax2 = max(1, int(r * random.uniform(0.6, 1.4)))
            cv2.ellipse(mask, (ox, oy), (ax1, ax2), random.randint(0, 360), 0, 360, 255, -1)
        for _ in range(random.randint(0, 4)):
            sx = cx + random.randint(-22, 22)
            sy = cy + random.randint(-22, 22)
            sr = random.uniform(1.0, 3.5)
            ax1 = max(1, int(sr * random.uniform(0.5, 2.0)))
            ax2 = max(1, int(sr * random.uniform(0.4, 1.0)))
            cv2.ellipse(mask, (sx, sy), (ax1, ax2), random.randint(0, 360), 0, 360, 255, -1)

    noise = cv2.GaussianBlur(np.random.randint(0, 255, (h, w), dtype=np.uint8), (15, 15), 0)
    mask = cv2.bitwise_and(mask, mask, mask=(noise > 110).astype(np.uint8) * 255)
    _, mask = cv2.threshold(cv2.GaussianBlur(mask, (3, 3), 0), 127, 255, cv2.THRESH_BINARY)
    return mask


def normalized_label(class_id: int, bbox_x: int, bbox_y: int, width: int, height: int, img_size: int, order: int) -> dict:
    return {
        "id": int(class_id),
        "bbox": [
            (bbox_x + width / 2) / img_size,
            (bbox_y + height / 2) / img_size,
            width / img_size,
            height / img_size,
        ],
        "order": int(order),
    }
