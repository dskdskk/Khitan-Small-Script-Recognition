from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2

from khitan_dataset_common import image_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--center-format", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clear-output", action="store_true")
    return parser.parse_args()


def read_annotation(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict) and isinstance(payload.get("shapes"), list):
        items = []
        for shape in payload["shapes"]:
            points = shape.get("points", [])
            if not points:
                continue

            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            label = shape.get("label", shape.get("id", -1))

            items.append({
                "id": int(label),
                "bbox": [
                    min(xs),
                    min(ys),
                    max(xs) - min(xs),
                    max(ys) - min(ys),
                ],
                "bbox_format": "xywh",
            })

        return items

    return []


def find_annotation(input_dir: Path, image_path: Path) -> Path | None:
    candidates = [
        input_dir / f"{image_path.name}.json",
        input_dir / f"{image_path.stem}.json",
    ]
    return next((path for path in candidates if path.exists()), None)


def convert_bbox(
    bbox: list[float],
    width: int,
    height: int,
    center_format: bool,
    bbox_format: str | None = None,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox

    if all(isinstance(v, float) and 0.0 <= v <= 1.0 for v in bbox):
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height

    if bbox_format == "xywh" or not center_format:
        x, y, w, h = x1, y1, x2, y2
    else:
        x, y, w, h = x1 - x2 / 2, y1 - y2 / 2, x2, y2

    x = max(0, int(round(x)))
    y = max(0, int(round(y)))
    w = min(width - x, int(round(w)))
    h = min(height - y, int(round(h)))

    return x, y, w, h


def main() -> None:
    args = parse_args()

    if args.clear_output and args.output_dir.exists():
        shutil.rmtree(args.output_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_files(args.input_dir):
        annotation_path = find_annotation(args.input_dir, image_path)
        if annotation_path is None:
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        image_height, image_width = image.shape[:2]

        try:
            items = read_annotation(annotation_path)
        except (OSError, json.JSONDecodeError, ValueError):
            continue

        for item in items:
            class_id = item.get("id", -1)
            bbox = item.get("bbox", [])

            if class_id == -1 or len(bbox) != 4:
                continue

            x, y, w, h = convert_bbox(
                bbox,
                image_width,
                image_height,
                args.center_format,
                item.get("bbox_format"),
            )

            if w <= 2 or h <= 2:
                continue

            crop = image[y: y + h, x: x + w]

            class_dir = args.output_dir / str(int(class_id))
            class_dir.mkdir(parents=True, exist_ok=True)

            output_name = f"{image_path.stem}_{x}_{y}{image_path.suffix.lower()}"
            cv2.imwrite(str(class_dir / output_name), crop)


if __name__ == "__main__":
    main()

