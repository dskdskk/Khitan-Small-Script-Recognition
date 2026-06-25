import os
import json
import argparse

import torch
import torch.nn as nn
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
        description="Train the layout recognizer with a frozen morphological codebook."
    )

    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Dataset root directory. It should contain train/images and labels.json."
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
        help="Directory for saving checkpoints and visualization results."
    )

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--vis_every", type=int, default=1)

    return parser.parse_args()


# ================= GIoU Loss =================
def generalized_iou_loss(pred_boxes, target_boxes):
    p_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    p_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    p_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    p_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

    t_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
    t_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
    t_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
    t_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2

    inter_x1 = torch.max(p_x1, t_x1)
    inter_y1 = torch.max(p_y1, t_y1)
    inter_x2 = torch.min(p_x2, t_x2)
    inter_y2 = torch.min(p_y2, t_y2)

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    p_area = (p_x2 - p_x1) * (p_y2 - p_y1)
    t_area = (t_x2 - t_x1) * (t_y2 - t_y1)

    union_area = p_area + t_area - inter_area + 1e-6
    iou = inter_area / union_area

    enc_x1 = torch.min(p_x1, t_x1)
    enc_y1 = torch.min(p_y1, t_y1)
    enc_x2 = torch.max(p_x2, t_x2)
    enc_y2 = torch.max(p_y2, t_y2)

    enc_area = torch.clamp(enc_x2 - enc_x1, min=0) * torch.clamp(enc_y2 - enc_y1, min=0) + 1e-6

    giou = iou - (enc_area - union_area) / enc_area
    return (1 - giou).mean()


# ================= Dataset =================
class LayoutDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.root_dir = os.path.join(root_dir, split)
        self.image_dir = os.path.join(self.root_dir, 'images')

        # Prefer labels.json under the split directory; fall back to root labels.json.
        label_file = os.path.join(self.root_dir, 'labels.json')
        if not os.path.exists(label_file):
            print(f"{label_file} not found. Trying labels.json under the dataset root.")
            label_file = os.path.join(root_dir, 'labels.json')

        print(f"Loading labels from: {label_file}")

        with open(label_file, 'r', encoding='utf-8') as f:
            self.all_labels = json.load(f)

        self.file_names = list(self.all_labels.keys())

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        fname = self.file_names[idx]

        if fname.endswith('.jpg') or fname.endswith('.png'):
            img_path = os.path.join(self.image_dir, fname)
        else:
            img_path = os.path.join(self.image_dir, f"{fname}.jpg")
            if not os.path.exists(img_path):
                img_path = img_path.replace('.jpg', '.png')

        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            return (
                torch.zeros(3, 128, 128),
                torch.zeros(MAX_SEQ_LEN).long(),
                torch.zeros(MAX_SEQ_LEN, 4),
            )

        w_orig, h_orig = img.size
        img_tensor = self.transform(img)

        raw_data = self.all_labels[fname]

        if isinstance(raw_data, dict) and 'shapes' in raw_data:
            items = raw_data['shapes']
        elif isinstance(raw_data, list):
            items = raw_data
        else:
            items = []

        boxes = []
        labels = []

        for item in items:
            if 'label' in item:
                label = int(item['label'])
            elif 'id' in item:
                label = int(item['id'])
            else:
                continue

            if 'bbox' in item:
                b = item['bbox']
                cx, cy, w, h = b[0], b[1], b[2], b[3]

            elif 'points' in item:
                pts = item['points']
                x_min = min(p[0] for p in pts)
                y_min = min(p[1] for p in pts)
                x_max = max(p[0] for p in pts)
                y_max = max(p[1] for p in pts)

                cx = (x_min + x_max) / 2 / w_orig
                cy = (y_min + y_max) / 2 / h_orig
                w = (x_max - x_min) / w_orig
                h = (y_max - y_min) / h_orig

            else:
                continue

            boxes.append([cx, cy, w, h])
            labels.append(label)

        final_labels = [SOS_TOKEN] + [l + RADICAL_START_ID for l in labels] + [EOS_TOKEN]
        final_boxes = [[0, 0, 0, 0]] + boxes + [[0, 0, 0, 0]]

        if len(final_labels) < MAX_SEQ_LEN:
            pad_len = MAX_SEQ_LEN - len(final_labels)
            final_labels += [PAD_TOKEN] * pad_len
            final_boxes += [[0, 0, 0, 0]] * pad_len
        else:
            final_labels = final_labels[:MAX_SEQ_LEN]
            final_boxes = final_boxes[:MAX_SEQ_LEN]

        return (
            img_tensor,
            torch.tensor(final_labels, dtype=torch.long),
            torch.tensor(final_boxes, dtype=torch.float32),
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

        img = img[:4].to(DEVICE)
        batch_size = img.size(0)

        pred_logits, pred_boxes = model(img)
        pred_ids = pred_logits.argmax(-1)

        for i in range(batch_size):
            inv_img = img[i].cpu() * 0.5 + 0.5
            pil_img = transforms.ToPILImage()(inv_img)
            draw = ImageDraw.Draw(pil_img)

            for t in range(MAX_SEQ_LEN):
                pred_id = pred_ids[i, t].item()

                # Only visualize valid radical/component predictions.
                if pred_id < NUM_CLASSES:
                    cx, cy, w, h = pred_boxes[i, t].tolist()
                    x1, y1 = (cx - w / 2) * 128, (cy - h / 2) * 128
                    x2, y2 = (cx + w / 2) * 128, (cy + h / 2) * 128

                    draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
                    draw.text((x1, y1), str(pred_id), fill='red')

            pil_img.save(f"{save_dir}/ep{epoch}_sample{i}.png")

    model.train()


# ================= Training loop =================
def train(args):
    os.makedirs(args.save_dir, exist_ok=True)

    train_loader = DataLoader(
        LayoutDataset(args.data_root, 'train'),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_path = os.path.join(args.data_root, 'val')
    if os.path.exists(val_path):
        val_loader = DataLoader(
            LayoutDataset(args.data_root, 'val'),
            batch_size=4,
            shuffle=False,
            num_workers=args.num_workers,
        )
        print("Validation set found.")
    else:
        print("No validation set found. Using training samples for visualization only.")
        val_loader = DataLoader(
            LayoutDataset(args.data_root, 'train'),
            batch_size=4,
            shuffle=True,
            num_workers=args.num_workers,
        )

    print("Initializing CodebookLayoutTransformer.")
    model = CodebookLayoutTransformer(
        codebook_path=args.codebook_path,
        num_classes=NUM_CLASSES,
        max_seq_len=MAX_SEQ_LEN,
        sos_token=SOS_TOKEN,
        pad_token=PAD_TOKEN,
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion_cls = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    print("Start training layout recognizer.")

    for epoch in range(args.epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

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
                l_l1 = nn.functional.l1_loss(pred_boxes[mask], tgt_boxes[mask])
            else:
                l_giou = torch.tensor(0.0).to(DEVICE)
                l_l1 = torch.tensor(0.0).to(DEVICE)

            loss = l_cls * 2.0 + l_giou * 2.0 + l_l1 * 5.0

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            progress_bar.set_postfix({
                'Cls': f"{l_cls.item():.2f}",
                'Box': f"{l_giou.item():.2f}",
            })

        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.save_dir, f"layout_ep{epoch + 1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

        if (epoch + 1) % args.vis_every == 0:
            validate_vis(model, val_loader, f"{args.save_dir}/vis", epoch + 1)
            print(f"Visualization saved for epoch {epoch + 1}.")


if __name__ == '__main__':
    args = parse_args()
    train(args)

'''
python train_layout.py \
  --data_root /home/qzh/data_hdd/home/qzh/dk/khitan_stylegan/A_khitan_实现/data/dataset_470/layout_dataset_synth_full_470 \
  --codebook_path /home/qzh/data_hdd/home/qzh/dk/khitan_stylegan/A_khitan_实现/A_c/模型/checkpoints/stage1_simple/codebook_only_ep50.pth \
  --save_dir /home/qzh/data_hdd/home/qzh/dk/khitan_stylegan/A_khitan_实现/A_c/模型/checkpoints3/stage2_layout_codebook_合成数据_codebook_clean_query_10 \
  --batch_size 32 \
  --epochs 60 \
  --lr 1e-4
'''