import os
import json
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw
from tqdm import tqdm
import numpy as np

from model_layout_codebook import CodebookLayoutTransformer


# ================= Global configuration =================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

NUM_CLASSES = 470
SOS_TOKEN = 470
EOS_TOKEN = 471
PAD_TOKEN = 472
RADICAL_START_ID = 0
MAX_SEQ_LEN = 10


def parse_args():
    parser = argparse.ArgumentParser(
        description="Degraded-style finetuning for the CodebookLayoutTransformer."
    )

    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory of degraded-style finetuning data. It should contain images/ and labels.json."
    )
    parser.add_argument(
        "--pretrained_ckpt",
        type=str,
        required=True,
        help="Path to the pretrained layout model checkpoint."
    )
    parser.add_argument(
        "--codebook_path",
        type=str,
        required=True,
        help="Path to the frozen codebook checkpoint."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Directory for saving finetuned checkpoints and visualization results."
    )

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=2)
    parser.add_argument("--vis_every", type=int, default=1)

    return parser.parse_args()


# ================= GIoU Loss =================
def generalized_iou_loss(pred_boxes, target_boxes):
    # Convert boxes from center format to corner format.
    p_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    p_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    p_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    p_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

    t_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
    t_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
    t_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
    t_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2

    # Intersection.
    inter_x1 = torch.max(p_x1, t_x1)
    inter_y1 = torch.max(p_y1, t_y1)
    inter_x2 = torch.min(p_x2, t_x2)
    inter_y2 = torch.min(p_y2, t_y2)

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # Union.
    p_area = (p_x2 - p_x1) * (p_y2 - p_y1)
    t_area = (t_x2 - t_x1) * (t_y2 - t_y1)
    union_area = p_area + t_area - inter_area + 1e-6

    iou = inter_area / union_area

    # Enclosing box for GIoU.
    enc_x1 = torch.min(p_x1, t_x1)
    enc_y1 = torch.min(p_y1, t_y1)
    enc_x2 = torch.max(p_x2, t_x2)
    enc_y2 = torch.max(p_y2, t_y2)

    enc_area = torch.clamp(enc_x2 - enc_x1, min=0) * torch.clamp(enc_y2 - enc_y1, min=0) + 1e-6

    giou = iou - (enc_area - union_area) / enc_area
    return (1 - giou).mean()


# ================= Dataset with strong degradation-style augmentation =================
class DegradedFinetuneDataset(Dataset):
    def __init__(self, root_dir):
        self.image_dir = os.path.join(root_dir, 'images') if os.path.exists(os.path.join(root_dir, 'images')) else root_dir
        self.img_files = [
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        self.all_labels = {}
        label_file = os.path.join(root_dir, 'labels.json')
        if os.path.exists(label_file):
            print("Found labels.json.")
            with open(label_file, 'r', encoding='utf-8') as f:
                self.all_labels = json.load(f)

        # Strong augmentation for degraded-style finetuning.
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomAffine(
                degrees=5,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                shear=5,
            ),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1,
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(3, sigma=(0.1, 2.0))],
                p=0.4,
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.1)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        print(f"Degraded-style finetuning dataset: {len(self.img_files)} images found.")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        fname = self.img_files[idx]
        img_path = os.path.join(self.image_dir, fname)

        gt_data = self.all_labels.get(fname)

        if gt_data is None:
            json_path = os.path.join(self.image_dir, fname + ".json")
            if not os.path.exists(json_path):
                json_path = os.path.join(self.image_dir, os.path.splitext(fname)[0] + ".json")

            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        gt_data = json.load(f)
                except Exception:
                    pass

        if gt_data is None:
            return (
                self.transform(Image.open(img_path).convert('RGB')),
                torch.full((MAX_SEQ_LEN,), PAD_TOKEN),
                torch.zeros(MAX_SEQ_LEN, 4),
            )

        try:
            img = Image.open(img_path).convert('RGB')
            w_orig, h_orig = img.size
            img_tensor = self.transform(img)

            boxes = []
            labels = []

            items = (
                gt_data['shapes']
                if isinstance(gt_data, dict) and 'shapes' in gt_data
                else (gt_data if isinstance(gt_data, list) else [])
            )

            for item in items:
                lbl = int(item.get('label', item.get('id', -1)))
                if lbl == -1:
                    continue

                if 'bbox' in item:
                    cx, cy, w, h = item['bbox']

                elif 'points' in item:
                    pts = item['points']
                    x_min = min(p[0] for p in pts)
                    x_max = max(p[0] for p in pts)
                    y_min = min(p[1] for p in pts)
                    y_max = max(p[1] for p in pts)

                    cx = (x_min + x_max) / 2 / w_orig
                    cy = (y_min + y_max) / 2 / h_orig
                    w = (x_max - x_min) / w_orig
                    h = (y_max - y_min) / h_orig

                else:
                    continue

                boxes.append([cx, cy, w, h])
                labels.append(lbl)

            final_labels = [SOS_TOKEN] + [l + RADICAL_START_ID for l in labels] + [EOS_TOKEN]
            final_boxes = [[0, 0, 0, 0]] + boxes + [[0, 0, 0, 0]]

            pad_len = MAX_SEQ_LEN - len(final_labels)
            if pad_len > 0:
                final_labels += [PAD_TOKEN] * pad_len
                final_boxes += [[0, 0, 0, 0]] * pad_len
            else:
                final_labels = final_labels[:MAX_SEQ_LEN]
                final_boxes = final_boxes[:MAX_SEQ_LEN]

            return (
                img_tensor,
                torch.tensor(final_labels),
                torch.tensor(final_boxes),
            )

        except Exception as e:
            print(f"Error loading sample {fname}: {e}")
            return (
                torch.zeros(3, 128, 128),
                torch.full((MAX_SEQ_LEN,), PAD_TOKEN),
                torch.zeros(MAX_SEQ_LEN, 4),
            )


# ================= Visualization =================
def validate_vis(model, val_loader, save_dir, epoch):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        try:
            img, _, _ = next(iter(val_loader))
        except StopIteration:
            return

        img = img[:8].to(DEVICE)
        batch_size = img.size(0)

        pred_logits, pred_boxes = model(img)
        pred_ids = pred_logits.argmax(-1)

        for i in range(batch_size):
            inv_img = img[i].cpu() * 0.5 + 0.5
            pil_img = transforms.ToPILImage()(inv_img)
            draw = ImageDraw.Draw(pil_img)

            for t in range(MAX_SEQ_LEN):
                pred_id = pred_ids[i, t].item()

                if pred_id < NUM_CLASSES:
                    cx, cy, w, h = pred_boxes[i, t].tolist()
                    x1, y1 = (cx - w / 2) * 128, (cy - h / 2) * 128
                    x2, y2 = (cx + w / 2) * 128, (cy + h / 2) * 128

                    draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
                    draw.text((x1, y1), str(pred_id), fill='red')

            pil_img.save(f"{save_dir}/ep{epoch}_sample{i}.png")

    model.train()


# ================= Finetuning loop =================
def finetune(args):
    os.makedirs(args.save_dir, exist_ok=True)

    dataset = DegradedFinetuneDataset(args.data_root)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    vis_loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
    )

    print("Start degraded-style finetuning.")

    model = CodebookLayoutTransformer(
        codebook_path=args.codebook_path,
        num_classes=NUM_CLASSES,
        max_seq_len=MAX_SEQ_LEN,
        sos_token=SOS_TOKEN,
        pad_token=PAD_TOKEN,
    ).to(DEVICE)

    if os.path.exists(args.pretrained_ckpt):
        print(f"Loading pretrained checkpoint: {args.pretrained_ckpt}")
        model.load_state_dict(torch.load(args.pretrained_ckpt, map_location=DEVICE))
    else:
        print(f"Pretrained checkpoint not found: {args.pretrained_ckpt}")
        return

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    criterion_cls = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    for epoch in range(args.epochs):
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Finetune Epoch {epoch + 1}")

        for imgs, tgt_labels, tgt_boxes in progress_bar:
            imgs = imgs.to(DEVICE)
            tgt_labels = tgt_labels.to(DEVICE)
            tgt_boxes = tgt_boxes.to(DEVICE)

            optimizer.zero_grad()

            pred_logits, pred_boxes = model(imgs)

            l_cls = criterion_cls(
                pred_logits.reshape(-1, NUM_CLASSES + 3),
                tgt_labels.reshape(-1),
            )

            mask = (
                (tgt_labels != PAD_TOKEN)
                & (tgt_labels != SOS_TOKEN)
                & (tgt_labels != EOS_TOKEN)
            )

            if mask.sum() > 0:
                l_giou = generalized_iou_loss(pred_boxes[mask], tgt_boxes[mask])
                l_l1 = F.l1_loss(pred_boxes[mask], tgt_boxes[mask])

                loss = l_cls * 3.0 + l_giou * 8.0 + l_l1 * 4.0

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                progress_bar.set_postfix({
                    'Cls': f"{l_cls.item():.2f}",
                    'Box': f"{l_giou.item():.2f}",
                })

            else:
                progress_bar.set_postfix({'Info': 'No labels'})

        if (epoch + 1) % args.vis_every == 0:
            validate_vis(model, vis_loader, f"{args.save_dir}/vis_log", epoch + 1)
            print(f"Visualization saved to: {args.save_dir}/vis_log")

        if (epoch + 1) % args.save_every == 0:
            save_path = os.path.join(args.save_dir, f"finetuned_ep{epoch + 1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Checkpoint saved: {save_path}")


if __name__ == '__main__':
    args = parse_args()
    finetune(args)
'''
python finetune_degraded.py \
  --data_root /home/qzh/data_hdd/home/qzh/dk/khitan_stylegan/A_khitan_实现/A_c/数据合成/synth_dataset_100k_real_加噪声版本/train \
  --pretrained_ckpt /home/qzh/data_hdd/home/qzh/dk/khitan_stylegan/A_khitan_实现/A_c/模型/checkpoints3/stage2_layout_codebook_合成数据_codebook_clean_query_10/layout_ep60.pth \
  --codebook_path /home/qzh/data_hdd/home/qzh/dk/khitan_stylegan/A_khitan_实现/A_c/模型/checkpoints/stage1_simple/codebook_only_ep50.pth \
  --save_dir /home/qzh/data_hdd/home/qzh/dk/khitan_stylegan/A_khitan_实现/A_c/模型/checkpoints3/stage3_finetune_codebook_clean_query_10 \
  --batch_size 64 \
  --epochs 60 \
  --lr 3e-5
'''