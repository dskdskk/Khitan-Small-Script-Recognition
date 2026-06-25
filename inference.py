#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import json
import argparse
import traceback

import cv2
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from model_layout_codebook import CodebookLayoutTransformer


# ==========================================
# Logger: write terminal output to both console and file
# ==========================================
class Logger(object):
    def __init__(self, filename="eval_results.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ==========================================
# Generator definition for visualization
# ==========================================
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return F.relu(x + self.conv(x))


class SimpleGenerator(nn.Module):
    def __init__(self, class_num=470, embed_dim=512):
        super().__init__()

        self.input_text = nn.Module()
        self.input_text.TextEmbeddings = nn.Parameter(
            torch.randn(class_num, embed_dim)
        )

        self.fc = nn.Linear(embed_dim, 512 * 4 * 4)

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            ResBlock(256),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            ResBlock(128),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, labels):
        idx = labels.squeeze()

        if idx.ndim == 0:
            idx = idx.unsqueeze(0)

        codes = self.input_text.TextEmbeddings[idx]
        x = self.fc(codes).view(-1, 512, 4, 4)
        return self.decoder(x)


# ==========================================
# Constants
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 470
SOS_TOKEN = 470
PAD_TOKEN = 472
MAX_SEQ_LEN = 10


# ==========================================
# Utility functions
# ==========================================
def compute_iou(box1, box2):
    b1_x1 = box1[0] - box1[2] / 2
    b1_y1 = box1[1] - box1[3] / 2
    b1_x2 = box1[0] + box1[2] / 2
    b1_y2 = box1[1] + box1[3] / 2

    b2_x1 = box2[0] - box2[2] / 2
    b2_y1 = box2[1] - box2[3] / 2
    b2_x2 = box2[0] + box2[2] / 2
    b2_y2 = box2[1] + box2[3] / 2

    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    return inter_area / (b1_area + b2_area - inter_area + 1e-6)


def apply_nms(results, iou_threshold=0.55):
    if not results:
        return []

    keep = []
    results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)

    for current in results_sorted:
        is_overlapping = False

        for kept_item in keep:
            if compute_iou(current["bbox"], kept_item["bbox"]) > iou_threshold:
                is_overlapping = True
                break

        if not is_overlapping:
            keep.append(current)

    return keep


def resize_keep_aspect(img, target_w, target_h, fill_color=0):
    if isinstance(img, Image.Image):
        img = np.array(img)

    h, w = img.shape[:2]

    if w == 0 or h == 0:
        return np.full((target_h, target_w), fill_color, dtype=np.uint8)

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    if new_w < 1:
        new_w = 1
    if new_h < 1:
        new_h = 1

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    canvas = np.full((target_h, target_w), fill_color, dtype=np.uint8)

    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    y_end = min(target_h, y_offset + new_h)
    x_end = min(target_w, x_offset + new_w)

    if y_end > y_offset and x_end > x_offset:
        canvas[y_offset:y_end, x_offset:x_end] = resized[
            : y_end - y_offset,
            : x_end - x_offset,
        ]

    return canvas


def edit_distance(s1, s2):
    if len(s1) < len(s2):
        return edit_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)

    for i, c1 in enumerate(s1):
        current_row = [i + 1]

        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))

        previous_row = current_row

    return previous_row[-1]


# ==========================================
# Evaluation
# ==========================================
def evaluate_prediction(
    preds,
    top_sequences,
    gt_list,
    iou_thresh=0.5,
    img_size=512,
    topk_list=None,
):
    if topk_list is None:
        topk_list = [1, 3, 5]

    correct_topk = {k: 0 for k in topk_list}
    is_perfect_topk = {k: False for k in topk_list}

    if gt_list is None:
        return correct_topk, 0, is_perfect_topk

    gt_pool = [g.copy() for g in gt_list]
    total_gt = len(gt_list)
    matched_gt_indices = set()

    gt_sequence_assigned = []
    all_boxes_matched = True

    for p in preds:
        pred_box_pixel = [c * img_size for c in p["bbox"]]

        best_iou = 0
        best_gt_idx = -1

        for i, g in enumerate(gt_pool):
            if i in matched_gt_indices:
                continue

            gt_box_val = g["bbox"]
            gt_box_pixel = [c * img_size for c in gt_box_val]
            iou = compute_iou(pred_box_pixel, gt_box_pixel)

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou >= iou_thresh and best_gt_idx != -1:
            matched_gt_indices.add(best_gt_idx)

            raw_gt_id = gt_pool[best_gt_idx].get(
                "label",
                gt_pool[best_gt_idx].get("id"),
            )

            try:
                gt_id = int(float(raw_gt_id))
            except Exception:
                gt_id = -1

            for k in topk_list:
                if gt_id in p["topk_ids"][:k]:
                    correct_topk[k] += 1

            gt_sequence_assigned.append(gt_id)

        else:
            all_boxes_matched = False

    if total_gt > 0 and len(preds) == total_gt and all_boxes_matched:
        for k in topk_list:
            if gt_sequence_assigned in top_sequences[:k]:
                is_perfect_topk[k] = True

    elif total_gt == 0 and len(preds) == 0:
        for k in topk_list:
            is_perfect_topk[k] = True

    return correct_topk, total_gt, is_perfect_topk


# ==========================================
# Inference and visualization wrapper
# ==========================================
class Restorer:
    def __init__(self, layout_ckpt, gen_ckpt, codebook_ckpt):
        self.layout_model = CodebookLayoutTransformer(
            codebook_path=codebook_ckpt,
            num_classes=NUM_CLASSES,
            max_seq_len=MAX_SEQ_LEN,
            sos_token=SOS_TOKEN,
            pad_token=PAD_TOKEN,
        ).to(DEVICE)

        self.layout_model.load_state_dict(torch.load(layout_ckpt, map_location=DEVICE))
        self.layout_model.eval()

        self.generator = SimpleGenerator(class_num=NUM_CLASSES).to(DEVICE)
        self.generator.load_state_dict(torch.load(gen_ckpt, map_location=DEVICE))
        self.generator.eval()

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def process_single_image(
        self,
        img_path,
        gt_list=None,
        iou_thresh=0.55,
        debug_boxes=False,
        max_k=5,
    ):
        try:
            raw_img = Image.open(img_path).convert("RGB")
            img_tensor = self.transform(raw_img).unsqueeze(0).to(DEVICE)

            results = []

            with torch.no_grad():
                pred_logits, pred_boxes = self.layout_model(img_tensor)

                logits = pred_logits[0]
                boxes = pred_boxes[0]

                probs = F.softmax(logits, dim=-1)
                log_probs = torch.log(probs + 1e-9)

                fetch_k = min(max_k + 3, probs.size(-1))

                topk_scores, topk_ids = torch.topk(probs, k=fetch_k, dim=-1)
                topk_log_probs, _ = torch.topk(log_probs, k=fetch_k, dim=-1)

                for i in range(topk_ids.size(0)):
                    rid_top1 = topk_ids[i, 0].item()
                    score_top1 = topk_scores[i, 0].item()

                    cand_list = []

                    for rank in range(fetch_k):
                        rid = topk_ids[i, rank].item()

                        if rid < NUM_CLASSES:
                            cand_list.append((rid, topk_log_probs[i, rank].item()))

                    cand_list = cand_list[:max_k]

                    if rid_top1 < NUM_CLASSES and len(cand_list) > 0:
                        results.append({
                            "id": rid_top1,
                            "topk_ids": [c[0] for c in cand_list],
                            "cand_list": cand_list,
                            "bbox": boxes[i].cpu().numpy(),
                            "score": score_top1,
                        })

            # Slot filtering by NMS.
            results = apply_nms(results, iou_threshold=iou_thresh)

            # Spatial ordering: group predictions into rows, then sort left-to-right within each row.
            if len(results) > 0:
                avg_h = sum([r["bbox"][3] for r in results]) / len(results)
                row_threshold = avg_h * 0.45

                results.sort(key=lambda x: x["bbox"][1])

                rows = []
                current_row = []
                last_y = None

                for r in results:
                    curr_y = r["bbox"][1]

                    if last_y is None:
                        current_row.append(r)
                        last_y = curr_y

                    else:
                        if abs(curr_y - last_y) < row_threshold:
                            current_row.append(r)
                            last_y = sum(x["bbox"][1] for x in current_row) / len(current_row)

                        else:
                            rows.append(current_row)
                            current_row = [r]
                            last_y = curr_y

                if current_row:
                    rows.append(current_row)

                sorted_results = []

                for row in rows:
                    row.sort(key=lambda x: x["bbox"][0])
                    sorted_results.extend(row)

                results = sorted_results

            # Beam search over ordered slots.
            box_cands = [r["cand_list"] for r in results]
            top_sequences = []

            if box_cands:
                beams = [(0.0, [])]

                for cands in box_cands:
                    new_beams = []

                    for b_score, b_seq in beams:
                        for cid, cscore in cands:
                            new_beams.append((b_score + cscore, b_seq + [cid]))

                    new_beams.sort(key=lambda x: x[0], reverse=True)
                    beams = new_beams[:max_k]

                top_sequences = [seq for score, seq in beams]

            # Visualization.
            canvas_size = 512
            header_h = 40
            display_k = min(max_k, len(top_sequences))

            gt_sequence_assigned = []
            is_gt_matched_spatially = False

            if gt_list is not None and len(gt_list) == len(results) and len(results) > 0:
                gt_pool = [g.copy() for g in gt_list]
                matched_gt_indices = set()
                all_matched = True

                for p in results:
                    best_iou = 0
                    best_gt_idx = -1

                    for i, g in enumerate(gt_pool):
                        if i in matched_gt_indices:
                            continue

                        iou = compute_iou(
                            [c * canvas_size for c in p["bbox"]],
                            [c * canvas_size for c in g["bbox"]],
                        )

                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = i

                    if best_iou >= iou_thresh and best_gt_idx != -1:
                        matched_gt_indices.add(best_gt_idx)

                        raw_gt = gt_pool[best_gt_idx].get(
                            "label",
                            gt_pool[best_gt_idx].get("id"),
                        )
                        gt_sequence_assigned.append(int(float(raw_gt)))

                    else:
                        all_matched = False

                if all_matched:
                    is_gt_matched_spatially = True

            grid_w = display_k * canvas_size
            grid_h = canvas_size + header_h

            combined = Image.new(
                "RGB",
                (canvas_size + grid_w, grid_h),
                color=(30, 30, 30),
            )

            raw_img_resized = raw_img.resize((canvas_size, canvas_size))
            combined.paste(raw_img_resized, (0, header_h))

            grid_draw = ImageDraw.Draw(combined)
            grid_draw.rectangle(
                [2, 2, canvas_size - 2, header_h - 2],
                fill=(50, 50, 50),
            )
            grid_draw.text((10, 12), "Original Image", fill="white")

            if debug_boxes:
                debug_draw = ImageDraw.Draw(combined)
                for item in results:
                    cx, cy, w, h = item["bbox"]
                    x1 = int((cx - w / 2) * canvas_size)
                    y1 = int((cy - h / 2) * canvas_size)
                    x2 = int((cx + w / 2) * canvas_size)
                    y2 = int((cy + h / 2) * canvas_size)
                    debug_draw.rectangle(
                        [x1, header_h + y1, x2, header_h + y2],
                        outline="red",
                        width=2,
                    )

            for rank in range(display_k):
                seq = top_sequences[rank]
                paste_x = canvas_size + rank * canvas_size

                is_hit = is_gt_matched_spatially and (seq == gt_sequence_assigned)
                header_color = (0, 180, 0) if is_hit else (80, 80, 80)

                grid_draw.rectangle(
                    [paste_x + 2, 2, paste_x + canvas_size - 2, header_h - 2],
                    fill=header_color,
                )

                title = f"Top {rank + 1} [HIT]" if is_hit else f"Top {rank + 1}"
                grid_draw.text((paste_x + 10, 12), title, fill="white")

                restored_canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

                for j, rid in enumerate(seq):
                    if j >= len(results):
                        break

                    item = results[j]
                    cx, cy, w, h = item["bbox"]

                    target_w = int(w * canvas_size)
                    target_h = int(h * canvas_size)

                    if target_w < 5 or target_h < 5:
                        continue

                    with torch.no_grad():
                        gen_out = self.generator(labels=torch.tensor([rid]).to(DEVICE))

                        if isinstance(gen_out, tuple):
                            gen_out = gen_out[0]

                    gen_numpy = gen_out.squeeze(0).cpu().numpy()
                    gen_numpy = np.transpose(gen_numpy, (1, 2, 0))
                    gen_numpy = (gen_numpy * 0.5 + 0.5) * 255
                    gen_numpy = gen_numpy.clip(0, 255).astype(np.uint8)

                    if gen_numpy.shape[2] == 3:
                        char_gray = cv2.cvtColor(gen_numpy, cv2.COLOR_RGB2GRAY)
                    else:
                        char_gray = gen_numpy.squeeze(2)

                    thresh_temp = (char_gray > 50).astype(np.uint8) * 255
                    coords = cv2.findNonZero(thresh_temp)

                    if coords is not None:
                        x_ink, y_ink, w_ink, h_ink = cv2.boundingRect(coords)

                        if w_ink > 5 and h_ink > 5:
                            pad = 4
                            x_crop = max(0, x_ink - pad)
                            y_crop = max(0, y_ink - pad)
                            w_crop = min(128 - x_crop, w_ink + 2 * pad)
                            h_crop = min(128 - y_crop, h_ink + 2 * pad)

                            char_gray = char_gray[
                                y_crop: y_crop + h_crop,
                                x_crop: x_crop + w_crop,
                            ]

                    char_resized = resize_keep_aspect(
                        char_gray,
                        target_w,
                        target_h,
                        fill_color=0,
                    )

                    _, mask = cv2.threshold(
                        char_resized,
                        127,
                        255,
                        cv2.THRESH_BINARY,
                    )

                    x1 = int((cx - w / 2) * canvas_size)
                    y1 = int((cy - h / 2) * canvas_size)

                    h_char, w_char = mask.shape

                    y_start = max(0, y1)
                    x_start = max(0, x1)
                    y_end = min(canvas_size, y1 + h_char)
                    x_end = min(canvas_size, x1 + w_char)

                    if y_end > y_start and x_end > x_start:
                        crop_y = y_start - y1
                        crop_x = x_start - x1

                        paint_mask = mask[
                            crop_y: crop_y + (y_end - y_start),
                            crop_x: crop_x + (x_end - x_start),
                        ] > 127

                        restored_canvas[y_start:y_end, x_start:x_end][paint_mask] = 255

                        grid_draw.text(
                            (paste_x + x1, header_h + y1 - 12),
                            str(rid),
                            fill="yellow",
                        )

                res_pil = Image.fromarray(restored_canvas).convert("RGB")
                combined.paste(res_pil, (paste_x, header_h))

            return combined, results, top_sequences

        except Exception:
            print(f"\nError processing {img_path}:")
            traceback.print_exc()
            return None, [], []


# ==========================================
# Main program
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="Inference and evaluation for Khitan Small Script layout recognition."
    )

    parser.add_argument("--input_dir", default=None)
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--layout_ckpt", required=True)
    parser.add_argument("--gen_ckpt", required=True)
    parser.add_argument("--codebook_ckpt", required=True)

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--labels_path", default=None)
    parser.add_argument(
        "--topk",
        nargs="+",
        type=int,
        default=[1, 3, 5],
        help="Top-K sequence evaluation.",
    )

    args = parser.parse_args()

    args.topk = sorted(list(set(args.topk)))
    max_k_needed = max(args.topk)

    if args.dataset_root:
        if not args.input_dir:
            args.input_dir = os.path.join(args.dataset_root, args.split, "images")

        if not args.labels_path:
            split_label_path = os.path.join(args.dataset_root, args.split, "labels.json")
            root_label_path = os.path.join(args.dataset_root, "labels.json")

            args.labels_path = (
                split_label_path
                if os.path.exists(split_label_path)
                else root_label_path
            )

    if args.input_dir is None:
        raise ValueError("Please provide either --input_dir or --dataset_root.")

    os.makedirs(args.output_dir, exist_ok=True)

    log_file_path = os.path.join(args.output_dir, "eval_results.log")
    sys.stdout = Logger(log_file_path)
    print(f"Evaluation log will be saved to: {log_file_path}\n")

    all_synth_labels = {}

    if args.labels_path and os.path.exists(args.labels_path):
        try:
            with open(args.labels_path, "r", encoding="utf-8") as f:
                all_synth_labels = json.load(f)
        except Exception:
            pass

    restorer = Restorer(args.layout_ckpt, args.gen_ckpt, args.codebook_ckpt)

    img_files = [
        f for f in os.listdir(args.input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    img_files.sort()

    metrics = {
        "total_images": 0,
        "total_gt_radicals": 0,
        "total_pred_radicals": 0,
        "sum_1_ned": 0.0,
    }

    for k in args.topk:
        metrics[f"correct_top{k}"] = 0
        metrics[f"perfect_blocks_top{k}"] = 0

    for img_name in tqdm(img_files):
        gt_list = None

        if img_name in all_synth_labels:
            gt_list = all_synth_labels[img_name]

        else:
            candidate_label_files = [
                img_name + ".json",
                os.path.splitext(img_name)[0] + ".json",
            ]

            for name in candidate_label_files:
                p = os.path.join(args.input_dir, name)

                if os.path.exists(p):
                    try:
                        with open(p, "r", encoding="utf-8") as f:
                            d = json.load(f)
                            if isinstance(d, list):
                                gt_list = d
                    except Exception:
                        pass

        res_img, results, top_sequences = restorer.process_single_image(
            os.path.join(args.input_dir, img_name),
            gt_list=gt_list,
            iou_thresh=args.iou,
            debug_boxes=args.debug,
            max_k=max_k_needed,
        )

        if res_img:
            res_img.save(os.path.join(args.output_dir, f"res_{img_name}"))

        if gt_list is not None:
            has_bbox = (
                len(gt_list) > 0
                and isinstance(gt_list[0], dict)
                and "bbox" in gt_list[0]
            )

            corr_dict, tot, is_perfect_dict = evaluate_prediction(
                results,
                top_sequences,
                gt_list,
                iou_thresh=args.iou,
                img_size=512,
                topk_list=args.topk,
            )

            metrics["total_images"] += 1
            metrics["total_pred_radicals"] += len(results)

            gt_seq = []

            for g in gt_list:
                try:
                    gt_seq.append(int(float(g.get("label", g.get("id", -1)))))
                except Exception:
                    pass

            pred_seq = top_sequences[0] if len(top_sequences) > 0 else []

            gt_len = len(gt_seq)
            pred_len = len(pred_seq)

            if max(gt_len, pred_len) > 0:
                ed = edit_distance(gt_seq, pred_seq)
                ned = ed / max(gt_len, pred_len)
                metrics["sum_1_ned"] += (1.0 - ned)
            else:
                metrics["sum_1_ned"] += 1.0

            # Pure sequence-level exact match, independent of box matching.
            if "pure_perfect_top1" not in metrics:
                metrics["pure_perfect_top1"] = 0
            if "pure_perfect_top3" not in metrics:
                metrics["pure_perfect_top3"] = 0
            if "pure_perfect_top5" not in metrics:
                metrics["pure_perfect_top5"] = 0

            gt_seq_str = ",".join(map(str, gt_seq))
            pred_seq_strs = [",".join(map(str, seq)) for seq in top_sequences]

            if len(pred_seq_strs) > 0 and gt_seq_str == pred_seq_strs[0]:
                metrics["pure_perfect_top1"] += 1
            if gt_seq_str in pred_seq_strs[:3]:
                metrics["pure_perfect_top3"] += 1
            if gt_seq_str in pred_seq_strs[:5]:
                metrics["pure_perfect_top5"] += 1

            if has_bbox:
                metrics["total_gt_radicals"] += tot

                for k in args.topk:
                    metrics[f"correct_top{k}"] += corr_dict[k]

                    if is_perfect_dict[k]:
                        metrics[f"perfect_blocks_top{k}"] += 1

    print("\n" + "=" * 60)
    print("Sequence-level Top-K evaluation results")
    print("=" * 60)

    if metrics.get("total_gt_radicals", 0) > 0:
        total_gt = metrics["total_gt_radicals"]
        total_preds = metrics.get("total_pred_radicals", 0)

        print(f"Radical-level: total GT radicals = {total_gt}, total predictions = {total_preds}")

        for k in args.topk:
            recall = (metrics[f"correct_top{k}"] / total_gt) * 100

            if k == 1:
                print(
                    f"Radical Top-1 recall: {recall:.2f}% "
                    f"({metrics['correct_top1']}/{total_gt})"
                )

                precision = (
                    metrics["correct_top1"] / total_preds * 100
                    if total_preds > 0
                    else 0.0
                )
                print(f"Radical precision: {precision:.2f}%")

                f1_score = (
                    2 * (precision * recall) / (precision + recall)
                    if (precision + recall) > 0
                    else 0.0
                )
                print(f"Radical F1-score: {f1_score:.2f}%")

                avg_1_ned = (
                    metrics["sum_1_ned"] / metrics["total_images"] * 100
                    if metrics["total_images"] > 0
                    else 0.0
                )
                print(f"1-NED similarity: {avg_1_ned:.2f}%")

            else:
                print(
                    f"Radical Top-{k} hit rate: {recall:.2f}% "
                    f"({metrics[f'correct_top{k}']}/{total_gt})"
                )

        print("-" * 60)
        print(f"Block-level: total images = {metrics['total_images']}")

        for k in args.topk:
            block_acc = (
                metrics[f"perfect_blocks_top{k}"] / metrics["total_images"] * 100
                if metrics["total_images"] > 0
                else 0.0
            )

            if k == 1:
                print(
                    f"Sequence Top-1 exact accuracy with box matching: {block_acc:.2f}% "
                    f"({metrics['perfect_blocks_top1']}/{metrics['total_images']})"
                )
            else:
                print(
                    f"Sequence Top-{k} exact accuracy with box matching: {block_acc:.2f}% "
                    f"({metrics[f'perfect_blocks_top{k}']}/{metrics['total_images']})"
                )

        if "pure_perfect_top1" in metrics:
            print("-" * 60)
            total_images = metrics["total_images"]

            pure_top1 = metrics["pure_perfect_top1"] / total_images * 100 if total_images > 0 else 0.0
            pure_top3 = metrics["pure_perfect_top3"] / total_images * 100 if total_images > 0 else 0.0
            pure_top5 = metrics["pure_perfect_top5"] / total_images * 100 if total_images > 0 else 0.0

            print(
                f"Pure sequence Top-1 exact accuracy: {pure_top1:.2f}% "
                f"({metrics['pure_perfect_top1']}/{total_images})"
            )
            print(
                f"Pure sequence Top-3 exact accuracy: {pure_top3:.2f}% "
                f"({metrics['pure_perfect_top3']}/{total_images})"
            )
            print(
                f"Pure sequence Top-5 exact accuracy: {pure_top5:.2f}% "
                f"({metrics['pure_perfect_top5']}/{total_images})"
            )

    elif metrics["total_images"] > 0:
        block_acc = (
            metrics["perfect_blocks_top1"] / metrics["total_images"] * 100
            if metrics["total_images"] > 0
            else 0.0
        )
        print(
            f"Empty-image filtering accuracy: {block_acc:.2f}% "
            f"({metrics['perfect_blocks_top1']}/{metrics['total_images']})"
        )

    else:
        print("No valid ground-truth labels were detected. Only visual results were generated.")

    print("=" * 60)


if __name__ == "__main__":
    main()
'''
python inference.py \
  --input_dir /path/to/test/images \
  --output_dir ./results/inference_eval \
  --layout_ckpt /path/to/layout_model.pth \
  --gen_ckpt /path/to/generator.pth \
  --codebook_ckpt /path/to/codebook.pth \
  --iou 0.5 \
  --topk 1 3 5
'''