from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from khitan_dataset_common import (
    SPLITS,
    blend_alpha,
    global_contrast_degradation,
    image_files,
    jitter_rect,
    layout_templates,
    make_background,
    normalized_label,
    prepare_split_dirs,
    save_split_labels,
    split_name,
    process_patch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic Khitan-layout samples from a radical image bank.")
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--num-images", type=int, default=100000)
    parser.add_argument("--train-ratio", type=float, default=0.95)
    parser.add_argument("--val-ratio", type=float, default=0.04)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--block-min", type=int, default=160)
    parser.add_argument("--block-max", type=int, default=220)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--clear-output", action="store_true")
    return parser.parse_args()


def scan_bank(bank_dir: Path) -> tuple[list[int], dict[int, list[Path]], list[float]]:
    samples: dict[int, list[Path]] = {}
    for folder in sorted(bank_dir.iterdir()):
        if not folder.is_dir():
            continue
        try:
            class_id = int(folder.name)
        except ValueError:
            continue
        files = image_files(folder)
        if files:
            samples[class_id] = files
    class_ids = sorted(samples)
    weights = [1.0 / math.sqrt(len(samples[class_id])) for class_id in class_ids]
    return class_ids, samples, weights


def choose_unique_ids(class_ids: list[int], weights: list[float], count: int) -> list[int]:
    selected: list[int] = []
    while len(selected) < count:
        for class_id in random.choices(class_ids, weights=weights, k=count * 2):
            if class_id not in selected:
                selected.append(class_id)
            if len(selected) == count:
                break
    return selected


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    class_ids, bank, weights = scan_bank(args.bank_dir)
    if not class_ids:
        raise RuntimeError(f"No valid class folders found in {args.bank_dir}")

    prepare_split_dirs(args.output_root, clear_output=args.clear_output)
    labels = {split: {} for split in SPLITS}

    for index in tqdm(range(args.num_images), desc="Generating"):
        split = split_name(index, args.num_images, args.train_ratio, args.val_ratio)
        filename = f"block_{index:06d}.png"
        image = make_background(args.img_size)
        block_w = random.randint(args.block_min, args.block_max)
        block_h = random.randint(args.block_min, args.block_max)
        start_x = (args.img_size - block_w) // 2
        start_y = (args.img_size - block_h) // 2

        layout = random.choice(layout_templates())
        chosen_ids = choose_unique_ids(class_ids, weights, len(layout))
        image_labels = []

        for order, (rect, class_id) in enumerate(zip(layout, chosen_ids)):
            rx, ry, rw, rh = jitter_rect(rect, jitter=0.05, expand=(0.03, 0.06))
            cell_x = int(rx * block_w)
            cell_y = int(ry * block_h)
            cell_w = int(rw * block_w) + 4
            cell_h = int(rh * block_h) + 2

            source_path = random.choice(bank[class_id])
            patch = cv2.imread(str(source_path))
            if patch is None:
                continue

            is_bottom = order == len(layout) - 1 and len(layout) in {3, 5, 7}
            patch, (ox, oy, patch_w, patch_h) = process_patch(
                patch,
                cell_w,
                cell_h,
                ink_prob=0.4,
                elastic_prob=0.7,
                elastic_strength=random.uniform(0.08, 0.12),
                perspective_prob=0.6,
                perspective_twist=0.03,
                noise_prob=0.3,
                normal_fill=0.85,
                bottom_fill=0.35,
                is_bottom_single=is_bottom,
            )

            paste_x = start_x + cell_x
            paste_y = start_y + cell_y
            image = blend_alpha(image, patch, paste_x, paste_y)
            image_labels.append(normalized_label(class_id, paste_x + ox, paste_y + oy, patch_w, patch_h, args.img_size, order))

        image = global_contrast_degradation(image)
        cv2.imwrite(str(args.output_root / split / "images" / filename), image)
        labels[split][filename] = image_labels

    save_split_labels(args.output_root, labels)
    print(json.dumps({"output_root": str(args.output_root), "classes": len(class_ids), "images": args.num_images}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
