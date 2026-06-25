#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_noise_robust_codebook.py

用途：
    用 clean 标准契丹部件 + real 真实拓片裁剪部件，训练一份
    SAGE-style / noise-aware 的 470×512 frozen codebook。

数据格式：
    clean_dir/
        0000.png
        0001.png
        ...
        0469.png

    real_dir/
        1/
            xxx.png
        2/
            xxx.png
        10/
            xxx.png
        ...

输出：
    out_dir/codebook_noiseaware_final.pth

该文件与当前 CodebookLayoutTransformer.load_codebook() 兼容：
    {'input_text.TextEmbeddings': Tensor[470, 512]}

clean 标准字 0000.png / 0001.png / ...
        ↓
CNN Encoder 提取 clean prototype

真实裁剪字 radical_bank_v4/10/*.png
        ↓
CNN Encoder 提取 noisy feature

训练目标：
同类 real feature 靠近 clean prototype
异类 real feature 远离其他 469 类 prototype

最后：
用训练好的 CNN 重新跑一遍 470 个 clean 标准字
导出新的 codebook_noiseaware_final.pth

python /home/qzh/data_hdd/home/qzh/dk/khitan_stylegan/A_khitan_实现/A_c/train_noise_robust_codebook.py \
  --clean_dir "/home/qzh/data_hdd/home/qzh/dk/khitan_stylegan/A_khitan_实现/A_c/数据合成/data_standard_radicals" \
  --real_dir "/home/qzh/data_hdd/home/qzh/dk/khitan_stylegan/A_khitan_实现/A_c/数据合成/radical_bank_v4" \
  --out_dir "/home/qzh/data_hdd/home/qzh/dk/khitan_stylegan/A_khitan_实现/A_c/模型/checkpoints3/stage1_noiseaware_codebook" \
  --epochs 40 \
  --batch_size 64 \
  --epoch_size 4096
python /home/qzh/data_hdd/home/qzh/dk/khitan_stylegan/A_khitan_实现/A_c/train_noise_robust_codebook.py \
  --clean_dir "/home/qzh/data_hdd/home/qzh/dk/khitan_stylegan/A_khitan_实现/A_c/数据合成/data_standard_radicals" \
  --real_dir "/home/qzh/data_hdd/home/qzh/dk/khitan_stylegan/A_khitan_实现/A_c/数据合成/radical_bank_merged" \
  --out_dir "/home/qzh/data_hdd/home/qzh/dk/khitan_stylegan/A_khitan_实现/A_c/模型/checkpoints3/stage1_noiseaware_codebook_bank_merged" \
  --epochs 50 \
  --clean_batch_size 64 \
  --real_batch_size 64 \
  --real_epoch_size 4096

"""



import os
import re
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_first_int(filename: str) -> Optional[int]:
    stem = Path(filename).stem
    m = re.search(r"\d+", stem)
    if m is None:
        return None
    return int(m.group(0))


def ensure_black_bg_white_fg(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    arr = np.array(img)
    if arr.mean() > 127:
        img = ImageOps.invert(img)
    return img


class PadToSquare:
    def __init__(self, fill=0):
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        side = max(w, h)
        canvas = Image.new("RGB", (side, side), (self.fill, self.fill, self.fill))
        canvas.paste(img, ((side - w) // 2, (side - h) // 2))
        return canvas


def build_transform(img_size=128, train=True):
    ops = [
        transforms.Lambda(ensure_black_bg_white_fg),
        PadToSquare(fill=0),
        transforms.Resize((img_size, img_size)),
    ]
    if train:
        ops += [
            transforms.RandomApply([
                transforms.RandomAffine(
                    degrees=6,
                    translate=(0.04, 0.04),
                    scale=(0.92, 1.08),
                    shear=4,
                    fill=0,
                )
            ], p=0.40),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
            ], p=0.12),
            transforms.ColorJitter(brightness=0.12, contrast=0.18),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(ops)


class CleanRadicalDataset(Dataset):
    def __init__(self, clean_dir: str, num_classes=470, transform=None):
        self.clean_dir = Path(clean_dir)
        self.num_classes = num_classes
        self.transform = transform

        self.label_to_files: Dict[int, List[Path]] = {i: [] for i in range(num_classes)}
        for p in sorted(self.clean_dir.iterdir()):
            if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
                continue
            label = parse_first_int(p.name)
            if label is None:
                continue
            if 0 <= label < num_classes:
                self.label_to_files[label].append(p)

        self.items: List[Tuple[Path, int]] = []
        for y in range(num_classes):
            for p in self.label_to_files[y]:
                self.items.append((p, y))

        missing = [i for i in range(num_classes) if len(self.label_to_files[i]) == 0]
        print(f"📊 Clean dataset: {len(self.items)} images, {num_classes - len(missing)}/{num_classes} classes covered.")
        if missing:
            print(f"⚠️ Missing clean classes: {missing[:30]}{' ...' if len(missing) > 30 else ''}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p, y = self.items[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(y, dtype=torch.long)


class RealCropDataset(Dataset):
    def __init__(self, real_dir: str, clean_dataset: CleanRadicalDataset, num_classes=470, transform=None):
        self.real_dir = Path(real_dir)
        self.clean_dataset = clean_dataset
        self.num_classes = num_classes
        self.transform = transform

        self.real_map: Dict[int, List[Path]] = {i: [] for i in range(num_classes)}
        for class_dir in sorted(self.real_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            try:
                y = int(class_dir.name)
            except ValueError:
                continue
            if not (0 <= y < num_classes):
                continue
            for p in sorted(class_dir.iterdir()):
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    self.real_map[y].append(p)

        # 只保留 clean 和 real 都有的类别，防止 real 标签在 clean 中不存在
        self.valid_labels = [
            y for y in range(num_classes)
            if len(self.real_map[y]) > 0 and len(clean_dataset.label_to_files.get(y, [])) > 0
        ]

        self.items: List[Tuple[Path, int]] = []
        for y in self.valid_labels:
            for p in self.real_map[y]:
                self.items.append((p, y))

        total_real = len(self.items)
        print(f"📊 Real crop dataset: {total_real} images, {len(self.valid_labels)}/{num_classes} classes have real crops.")

        if total_real == 0:
            raise RuntimeError("没有读取到真实裁剪图，请检查 real_dir。")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p, y = self.items[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(y, dtype=torch.long)


class InfiniteRealSamplerDataset(Dataset):
    """
    为了每个 epoch 有足够 real 对齐步数，随机重复采样 real crop。
    """
    def __init__(self, real_dataset: RealCropDataset, epoch_size=4096):
        self.real_dataset = real_dataset
        self.epoch_size = epoch_size

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        j = random.randrange(len(self.real_dataset))
        return self.real_dataset[j]


class RadicalEncoder(nn.Module):
    def __init__(self, embed_dim=512, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.proj = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embed_dim),
        )

    def forward(self, x):
        f = self.backbone(x).flatten(1)
        z = self.proj(f)
        return F.normalize(z, dim=-1)


@torch.no_grad()
def compute_clean_prototypes(encoder, clean_dataset, device, num_classes=470, batch_size=128, num_workers=4):
    encoder.eval()
    loader = DataLoader(clean_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    sums = None
    counts = torch.zeros(num_classes, dtype=torch.long)

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        z = encoder(imgs).detach().cpu()

        if sums is None:
            sums = torch.zeros(num_classes, z.shape[1], dtype=torch.float32)

        for i, y in enumerate(labels.cpu().tolist()):
            sums[y] += z[i]
            counts[y] += 1

    counts_safe = counts.clamp(min=1).unsqueeze(1).float()
    proto = sums / counts_safe
    proto = F.normalize(proto, dim=-1)

    missing = (counts == 0).nonzero(as_tuple=False).flatten().tolist()
    if missing:
        print(f"⚠️ Missing clean prototypes: {len(missing)} classes. First: {missing[:20]}")

    return proto, counts


@torch.no_grad()
def compute_real_prototypes(encoder, real_dataset, device, num_classes=470, img_size=128, batch_size=128, num_workers=4):
    encoder.eval()
    eval_tf = build_transform(img_size=img_size, train=False)

    # 重新建一个 eval 版本，避免 train augment
    class RealEval(Dataset):
        def __init__(self, items):
            self.items = items
        def __len__(self):
            return len(self.items)
        def __getitem__(self, idx):
            p, y = self.items[idx]
            img = Image.open(p).convert("RGB")
            return eval_tf(img), torch.tensor(y, dtype=torch.long)

    ds = RealEval(real_dataset.items)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    sums = None
    counts = torch.zeros(num_classes, dtype=torch.long)
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        z = encoder(imgs).detach().cpu()
        if sums is None:
            sums = torch.zeros(num_classes, z.shape[1], dtype=torch.float32)
        for i, y in enumerate(labels.cpu().tolist()):
            sums[y] += z[i]
            counts[y] += 1

    counts_safe = counts.clamp(min=1).unsqueeze(1).float()
    proto = sums / counts_safe
    proto = F.normalize(proto, dim=-1)
    return proto, counts


@torch.no_grad()
def evaluate_real_retrieval(encoder, real_dataset, prototypes, device, img_size=128, max_eval=3000):
    encoder.eval()
    eval_tf = build_transform(img_size=img_size, train=False)

    items = real_dataset.items.copy()
    random.shuffle(items)
    items = items[:max_eval]

    proto = prototypes.to(device)
    correct1, correct5, total = 0, 0, 0

    for p, y in items:
        img = Image.open(p).convert("RGB")
        img = eval_tf(img).unsqueeze(0).to(device)
        z = encoder(img)
        logits = z @ proto.t()
        top5 = logits.topk(k=5, dim=-1).indices[0].cpu().tolist()
        correct1 += int(top5[0] == y)
        correct5 += int(y in top5)
        total += 1

    return correct1 / max(total, 1) * 100, correct5 / max(total, 1) * 100


def export_codebook_tensor(encoder, clean_eval_ds, real_dataset, device, args):
    clean_proto, _ = compute_clean_prototypes(
        encoder, clean_eval_ds, device,
        num_classes=args.num_classes,
        batch_size=args.proto_batch_size,
        num_workers=args.num_workers,
    )

    if args.mix_real_alpha >= 0.999:
        final_proto = clean_proto
    else:
        real_proto, real_counts = compute_real_prototypes(
            encoder, real_dataset, device,
            num_classes=args.num_classes,
            img_size=args.img_size,
            batch_size=args.proto_batch_size,
            num_workers=args.num_workers,
        )
        alpha = args.mix_real_alpha
        final_proto = clean_proto.clone()
        has_real = real_counts > 0
        final_proto[has_real] = F.normalize(
            alpha * clean_proto[has_real] + (1.0 - alpha) * real_proto[has_real],
            dim=-1
        )

    return F.normalize(final_proto, dim=-1)


def train(args):
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"🚀 Device: {device}")

    clean_train_ds = CleanRadicalDataset(
        args.clean_dir,
        num_classes=args.num_classes,
        transform=build_transform(args.img_size, train=True),
    )
    clean_eval_ds = CleanRadicalDataset(
        args.clean_dir,
        num_classes=args.num_classes,
        transform=build_transform(args.img_size, train=False),
    )
    real_train_base = RealCropDataset(
        args.real_dir,
        clean_dataset=clean_eval_ds,
        num_classes=args.num_classes,
        transform=build_transform(args.img_size, train=True),
    )
    real_train_ds = InfiniteRealSamplerDataset(real_train_base, epoch_size=args.real_epoch_size)

    clean_loader = DataLoader(
        clean_train_ds,
        batch_size=args.clean_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    real_loader = DataLoader(
        real_train_ds,
        batch_size=args.real_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    encoder = RadicalEncoder(embed_dim=args.embed_dim, pretrained=not args.no_pretrained).to(device)
    optimizer = optim.AdamW(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 初始 clean prototypes
    prototypes, _ = compute_clean_prototypes(
        encoder, clean_eval_ds, device,
        num_classes=args.num_classes,
        batch_size=args.proto_batch_size,
        num_workers=args.num_workers,
    )

    best_top1 = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        if epoch == 1 or (epoch - 1) % args.update_proto_every == 0:
            prototypes, _ = compute_clean_prototypes(
                encoder, clean_eval_ds, device,
                num_classes=args.num_classes,
                batch_size=args.proto_batch_size,
                num_workers=args.num_workers,
            )
            print(f"🔄 Epoch {epoch}: clean prototypes updated.")

        encoder.train()
        proto_gpu = prototypes.to(device).detach()

        real_iter = iter(real_loader)
        running = {"loss": 0.0, "clean": 0.0, "real": 0.0, "align": 0.0}
        steps = 0

        pbar = tqdm(clean_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for clean_img, clean_y in pbar:
            try:
                real_img, real_y = next(real_iter)
            except StopIteration:
                real_iter = iter(real_loader)
                real_img, real_y = next(real_iter)

            clean_img = clean_img.to(device, non_blocking=True)
            clean_y = clean_y.to(device, non_blocking=True)
            real_img = real_img.to(device, non_blocking=True)
            real_y = real_y.to(device, non_blocking=True)

            z_clean = encoder(clean_img)
            z_real = encoder(real_img)

            # 1. Clean loss 覆盖全部 470 类，避免没有 real 的类别被遗忘
            logits_clean = (z_clean @ proto_gpu.t()) / args.tau
            loss_clean = F.cross_entropy(logits_clean, clean_y)

            # 2. Real loss 只在有真实 crop 的两百多类上做抗噪对齐
            logits_real = (z_real @ proto_gpu.t()) / args.tau
            loss_real = F.cross_entropy(logits_real, real_y)

            # 3. Real feature 显式靠近对应 clean prototype
            target_proto = proto_gpu[real_y]
            loss_align = 1.0 - F.cosine_similarity(z_real, target_proto, dim=-1).mean()

            loss = (
                args.w_clean * loss_clean
                + args.w_real * loss_real
                + args.w_align * loss_align
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.grad_clip)
            optimizer.step()

            running["loss"] += loss.item()
            running["clean"] += loss_clean.item()
            running["real"] += loss_real.item()
            running["align"] += loss_align.item()
            steps += 1

            pbar.set_postfix({
                "loss": f"{loss.item():.3f}",
                "Lc": f"{loss_clean.item():.3f}",
                "Lr": f"{loss_real.item():.3f}",
                "La": f"{loss_align.item():.3f}",
            })

        scheduler.step()

        prototypes, _ = compute_clean_prototypes(
            encoder, clean_eval_ds, device,
            num_classes=args.num_classes,
            batch_size=args.proto_batch_size,
            num_workers=args.num_workers,
        )
        top1, top5 = evaluate_real_retrieval(
            encoder, real_train_base, prototypes, device,
            img_size=args.img_size,
            max_eval=args.max_eval_real,
        )

        log = {
            "epoch": epoch,
            "loss": running["loss"] / max(steps, 1),
            "loss_clean": running["clean"] / max(steps, 1),
            "loss_real": running["real"] / max(steps, 1),
            "loss_align": running["align"] / max(steps, 1),
            "real_retrieval_top1": top1,
            "real_retrieval_top5": top5,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(log)

        print(f"📈 Epoch {epoch}: real crop retrieval Top-1={top1:.2f}%, Top-5={top5:.2f}%")

        latest_path = Path(args.out_dir) / "encoder_latest.pth"
        torch.save({"encoder": encoder.state_dict(), "epoch": epoch, "args": vars(args), "history": history}, latest_path)

        if top1 > best_top1:
            best_top1 = top1
            best_path = Path(args.out_dir) / "encoder_best.pth"
            torch.save({"encoder": encoder.state_dict(), "epoch": epoch, "args": vars(args), "history": history, "best_top1": best_top1}, best_path)

            best_proto = export_codebook_tensor(encoder, clean_eval_ds, real_train_base, device, args)
            codebook_path = Path(args.out_dir) / "codebook_noiseaware_best.pth"
            torch.save({
                "input_text.TextEmbeddings": best_proto.cpu(),
                "meta": {
                    "epoch": epoch,
                    "best_real_retrieval_top1": best_top1,
                    "shape": list(best_proto.shape),
                    "note": "Noise-aware codebook v2; clean loss covers all 470 classes; real loss covers classes with real crops.",
                }
            }, codebook_path)
            print(f"💾 Best codebook saved: {codebook_path}")

        with open(Path(args.out_dir) / "train_log.json", "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    final_proto = export_codebook_tensor(encoder, clean_eval_ds, real_train_base, device, args)
    final_path = Path(args.out_dir) / "codebook_noiseaware_final.pth"
    torch.save({
        "input_text.TextEmbeddings": final_proto.cpu(),
        "meta": {
            "shape": list(final_proto.shape),
            "note": "Final noise-aware codebook v2; compatible with CodebookLayoutTransformer.load_codebook().",
        }
    }, final_path)

    print("\n✅ Training finished.")
    print(f"📦 Final codebook: {final_path}")
    print(f"📦 Best  codebook: {Path(args.out_dir) / 'codebook_noiseaware_best.pth'}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_dir", type=str, required=True)
    parser.add_argument("--real_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--num_classes", type=int, default=470)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--embed_dim", type=int, default=512)

    parser.add_argument("--clean_batch_size", type=int, default=64)
    parser.add_argument("--real_batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--real_epoch_size", type=int, default=4096)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--tau", type=float, default=0.07)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--proto_batch_size", type=int, default=128)
    parser.add_argument("--max_eval_real", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--w_clean", type=float, default=1.0)
    parser.add_argument("--w_real", type=float, default=1.0)
    parser.add_argument("--w_align", type=float, default=0.5)

    parser.add_argument("--update_proto_every", type=int, default=1)

    parser.add_argument("--mix_real_alpha", type=float, default=1.0,
                        help="1.0: final codebook uses clean prototypes only. 0.8: 0.8 clean + 0.2 real for classes with real crops.")
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
