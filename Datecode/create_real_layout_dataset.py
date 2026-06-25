from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from khitan_dataset_common import (
    SPLITS,
    blend_additive,
    global_style,
    image_files,
    jitter_rect,
    layout_for_length,
    load_char_map,
    make_background,
    normalized_label,
    prepare_split_dirs,
    process_patch,
    save_split_labels,
    split_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sequence-based synthetic layout samples from a radical image bank.")
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--radicals-file", type=Path, required=True)
    parser.add_argument("--sequences-file", type=Path, required=True)
    parser.add_argument("--augmentations-per-seq", type=int, default=10)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--block-min", type=int, default=160)
    parser.add_argument("--block-max", type=int, default=220)
    parser.add_argument("--train-ratio", type=float, default=0.90)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--clear-output", action="store_true")
    return parser.parse_args()


def read_sequences(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_tasks(sequences: list[str], char_map: dict[str, str], valid_ids: set[str], repeats: int) -> tuple[list[dict], int]:
    tasks = []
    skipped = 0
    for sequence in sequences:
        chars = list(sequence)
        valid = 1 <= len(chars) <= 7 and all(char_map.get(ch) in valid_ids for ch in chars)
        if not valid:
            skipped += 1
            continue
        for aug_id in range(repeats):
            tasks.append({"chars": chars, "sequence": sequence, "aug_id": aug_id})
    return tasks, skipped


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    char_map = load_char_map(args.radicals_file)
    valid_ids = {p.name for p in args.bank_dir.iterdir() if p.is_dir() and image_files(p)}
    sequences = read_sequences(args.sequences_file)
    tasks, skipped = build_tasks(sequences, char_map, valid_ids, args.augmentations_per_seq)
    random.shuffle(tasks)

    prepare_split_dirs(args.output_root, clear_output=args.clear_output)
    labels = {split: {} for split in SPLITS}

    for index, task in enumerate(tqdm(tasks, desc="Generating")):
        split = split_name(index, len(tasks), args.train_ratio, args.val_ratio)
        digest = int(hashlib.md5(task["sequence"].encode("utf-8")).hexdigest()[:8], 16)
        filename = f"seq_{digest % 100000:05d}_{index:06d}.png"
        image = make_background(args.img_size)
        block_w = random.randint(args.block_min, args.block_max)
        block_h = random.randint(args.block_min, args.block_max)
        start_x = (args.img_size - block_w) // 2
        start_y = (args.img_size - block_h) // 2

        try:
            layout = layout_for_length(len(task["chars"]))
        except ValueError:
            continue

        image_labels = []
        valid_image = True
        for order, (rect, char) in enumerate(zip(layout, task["chars"])):
            class_id = char_map[char]
            files = image_files(args.bank_dir / class_id)
            if not files:
                valid_image = False
                break
            source = cv2.imread(str(random.choice(files)))
            if source is None:
                valid_image = False
                break

            rx, ry, rw, rh = jitter_rect(rect)
            cell_x = int(rx * block_w)
            cell_y = int(ry * block_h)
            cell_w = int(rw * block_w) + 4
            cell_h = int(rh * block_h) + 2
            is_bottom = order == len(task["chars"]) - 1 and len(task["chars"]) in {3, 5, 7}

            patch, (ox, oy, patch_w, patch_h) = process_patch(
                source,
                cell_w,
                cell_h,
                ink_prob=0.3,
                elastic_prob=0.0,
                perspective_prob=0.6,
                perspective_twist=0.03,
                noise_prob=0.3,
                is_bottom_single=is_bottom,
            )

            paste_x = start_x + cell_x
            paste_y = start_y + cell_y
            image = blend_additive(image, patch, paste_x, paste_y, threshold=25)
            image_labels.append(normalized_label(int(class_id), paste_x + ox, paste_y + oy, patch_w, patch_h, args.img_size, order))

        if valid_image:
            image = global_style(image)
            cv2.imwrite(str(args.output_root / split / "images" / filename), image)
            labels[split][filename] = image_labels

    save_split_labels(args.output_root, labels)
    print(json.dumps({"output_root": str(args.output_root), "tasks": len(tasks), "skipped_sequences": skipped}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
