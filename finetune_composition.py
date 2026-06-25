import os
import json
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import numpy as np

from model_layout_codebook import CodebookLayoutTransformer


# ================= Global configuration =================
NUM_CLASSES = 470
SOS_TOKEN = 470
EOS_TOKEN = 471
PAD_TOKEN = 472
RADICAL_START_ID = 0
MAX_SEQ_LEN = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage III composition-aware / mixed-data finetuning for CodebookLayoutTransformer."
    )

    parser.add_argument(
        "--vocab_data_dir",
        type=str,
        required=True,
        help="Path to the standardized vocabulary/composition layout dataset."
    )
    parser.add_argument(
        "--random_data_dir",
        type=str,
        required=True,
        help="Path to the random degraded-style layout dataset."
    )
    parser.add_argument(
        "--pretrained_ckpt",
        type=str,
        required=True,
        help="Path to the pretrained checkpoint from the previous stage."
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
        help="Directory for saving Stage III finetuned checkpoints."
    )

    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--box_loss_weight", type=float, default=10.0)

    parser.add_argument("--backbone_lr", type=float, default=1e-5)
    parser.add_argument("--logic_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=2)

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

    return (1 - (iou - (enc_area - union_area) / enc_area)).mean()


def get_transforms():
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


# ================= Dataset =================
class UniversalJsonDataset(Dataset):
    def __init__(self, root_dir, transform=None, dataset_name="Unknown"):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "images")
        self.json_path = os.path.join(root_dir, "labels.json")
        self.transform = transform
        self.data_list = []

        if not os.path.exists(self.json_path):
            print(f"Label file not found: {self.json_path}")

        if os.path.exists(self.json_path):
            with open(self.json_path, "r", encoding="utf-8") as f:
                content = json.load(f)
                for fname, items in content.items():
                    self.data_list.append((fname, items))

        print(f"[{dataset_name}] Loaded {len(self.data_list)} samples.")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        fname, items = self.data_list[idx]
        img_path = os.path.join(self.img_dir, fname)

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        labels = [int(item.get("id", item.get("label", 0))) for item in items]
        boxes = [item.get("bbox", [0, 0, 0, 0]) for item in items]

        final_labels = [SOS_TOKEN] + [l + RADICAL_START_ID for l in labels] + [EOS_TOKEN]
        final_boxes = [[0, 0, 0, 0]] + boxes + [[0, 0, 0, 0]]

        if len(final_labels) < MAX_SEQ_LEN:
            pad_len = MAX_SEQ_LEN - len(final_labels)
            final_labels += [PAD_TOKEN] * pad_len
            final_boxes += [[0.0, 0.0, 0.0, 0.0]] * pad_len
        else:
            final_labels = final_labels[:MAX_SEQ_LEN]
            final_boxes = final_boxes[:MAX_SEQ_LEN]

        return (
            img,
            torch.tensor(final_labels, dtype=torch.long),
            torch.tensor(final_boxes, dtype=torch.float32),
        )


# ================= Stage III finetuning =================
def main(args):
    os.makedirs(args.save_dir, exist_ok=True)

    model = CodebookLayoutTransformer(
        codebook_path=args.codebook_path,
        num_classes=NUM_CLASSES,
        max_seq_len=MAX_SEQ_LEN,
        sos_token=SOS_TOKEN,
        pad_token=PAD_TOKEN,
    ).to(DEVICE)

    if os.path.exists(args.pretrained_ckpt):
        print(f"Loading pretrained checkpoint: {args.pretrained_ckpt}")
        model.load_state_dict(
            torch.load(args.pretrained_ckpt, map_location=DEVICE),
            strict=False,
        )
    else:
        print(f"Pretrained checkpoint not found: {args.pretrained_ckpt}")
        return

    backbone_params = []
    logic_params = []

    for name, param in model.named_parameters():
        if "backbone" in name or "resnet" in name:
            backbone_params.append(param)
        else:
            logic_params.append(param)

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": args.backbone_lr},
            {"params": logic_params, "lr": args.logic_lr},
        ],
        weight_decay=args.weight_decay,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,
        T_mult=2,
    )

    criterion_cls = nn.CrossEntropyLoss(
        ignore_index=PAD_TOKEN,
        label_smoothing=0.1,
    )

    ds_vocab = UniversalJsonDataset(
        args.vocab_data_dir,
        transform=get_transforms(),
        dataset_name="Vocab",
    )

    ds_random = UniversalJsonDataset(
        args.random_data_dir,
        transform=get_transforms(),
        dataset_name="Random",
    )

    train_loader = DataLoader(
        ConcatDataset([ds_vocab, ds_random]),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    for epoch in range(args.epochs):
        model.train()
        progress_bar = tqdm(
            train_loader,
            desc=f"Composition Finetune Epoch {epoch + 1}/{args.epochs}",
        )

        optimizer.zero_grad()

        epoch_total_loss = 0.0
        valid_batches = 0

        for i, (imgs, tgt_labels, tgt_boxes) in enumerate(progress_bar):
            imgs = imgs.to(DEVICE)
            tgt_labels = tgt_labels.to(DEVICE)
            tgt_boxes = tgt_boxes.to(DEVICE)

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
                l_box = l_giou + l_l1
            else:
                l_box = torch.tensor(0.0).to(DEVICE)

            loss = l_cls * 2.0 + args.box_loss_weight * l_box

            epoch_total_loss += loss.item()
            valid_batches += 1
            avg_loss = epoch_total_loss / valid_batches

            loss_to_backward = loss / args.accumulation_steps
            loss_to_backward.backward()

            if (i + 1) % args.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            progress_bar.set_postfix({"Avg Loss": f"{avg_loss:.4f}"})

        scheduler.step()

        current_lr = optimizer.param_groups[1]["lr"]
        print(
            f"Epoch {epoch + 1} finished. "
            f"Avg Loss: {avg_loss:.4f} | Current Logic LR: {current_lr:.6f}"
        )

        if (epoch + 1) % args.save_every == 0:
            save_path = os.path.join(args.save_dir, f"composition_ep{epoch + 1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Checkpoint saved: {save_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
'''
python finetune_composition.py \
  --vocab_data_dir /path/to/standardized_vocab_dataset/train \
  --random_data_dir /path/to/degraded_random_dataset/val \
  --pretrained_ckpt /path/to/stage2_or_degraded_finetuned_model.pth \
  --codebook_path /path/to/codebook.pth \
  --save_dir ./checkpoints/stage3_composition \
  --batch_size 50 \
  --accumulation_steps 4 \
  --epochs 60 \
  --box_loss_weight 10.0
'''
