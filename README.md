# 契丹小字图像识别系统

## 项目概述

本项目实现了一套完整的**契丹小字（Khitan Small Script）图像识别系统**，基于深度学习技术对契丹小字拓片/合成图像进行原字（部件）的序列识别与空间定位。

系统的核心是一个 **Prototype-constrained Layout Transformer** 模型，结合冻结的形态学 Codebook（码本），在合成布局数据上训练，并通过退化风格微调和组合结构微调来提升对真实拓片风格图像的泛化能力。

---

## 项目结构

```text
契丹小字—核心源码/
│
├── model_layout_codebook.py          # 模型定义（核心）
├── train_noise_robust_codebook.py    # 阶段一：Codebook 训练
├── train_layout.py                   # 阶段二：布局识别训练
├── finetune_degraded.py              # 阶段三：退化风格微调
├── finetune_composition.py           # 阶段四：组合结构微调
├── inference.py                      # 推理与评估
│
├── assets/
│   └── radicals.txt                  # 字符/部件 ID 映射表（470类）
│
├── dataset/
│   ├── data_standard_radicals/       # 标准原字（部件）图像（0000.png ~ 0469.png）
│   ├── test_239/                     # 测试集（239张标注图像）
│   └── train_550/                    # 训练集（550张标注图像）
│
├── Datecode/                         # 数据合成与预处理工具
│   ├── build_radical_bank.py         # 从标注拓片图像中裁剪构建部件库
│   ├── synthesize_dynamic_layout.py  # 随机动态布局合成数据集
│   ├── create_real_layout_dataset.py # 基于真实序列的布局数据集合成
│   ├── create_real_dataset_by_degradation.py  # 退化等级消融实验数据集合成
│   ├── khitan_dataset_common.py      # 数据合成的公共工具函数库
│   └── README_usage.md               # 数据合成脚本使用说明
│
├── NotoSerifKhitanSmallScript-Regular.ttf  # 契丹小字字体文件
└── README.md                         # 本文档
```

---

## 整体流程 (Pipeline)

```
┌─────────────────────────────────────────────────────────────────────┐
│                      数据准备阶段 (Datecode/)                         │
│                                                                      │
│  标注拓片图像 ──► build_radical_bank.py ──► 部件裁剪库 (Radical Bank) │
│                                                                      │
│  部件裁剪库 ──► synthesize_dynamic_layout.py ──► 合成布局数据集        │
│  部件裁剪库 ──► create_real_layout_dataset.py ──► 真实序列数据集       │
│  部件裁剪库 ──► create_real_dataset_by_degradation.py ──► 退化数据集   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      模型训练阶段                                     │
│                                                                      │
│  阶段一: train_noise_robust_codebook.py                               │
│      用标准原字+真实拓片裁剪图像，训练抗噪 Codebook (470×512)          │
│                                    │                                 │
│  阶段二: train_layout.py                                             │
│      在合成布局数据上训练布局识别模型（Codebook 冻结）                  │
│                                    │                                 │
│  阶段三: finetune_degraded.py                                        │
│      在退化风格数据上微调，增强对噪声、模糊、断裂的鲁棒性              │
│                                    │                                 │
│  阶段四: finetune_composition.py                                     │
│      混合标准化组合+随机退化布局数据微调，提升组合识别和定位能力        │
│                                    │                                 │
│  最终评估: inference.py                                               │
│      加载训练好的模型进行推理与可视化评估                              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 模型架构 (`model_layout_codebook.py`)

### `CodebookLayoutTransformer` 类

模型由以下五个核心模块组成：

| 模块 | 说明 |
|------|------|
| **ResNet 视觉编码器** | 基于 ResNet-18（含 ImageNet 预训练权重），移除最后的全局池化层和全连接层，输出 `512×h×w` 的 2D 特征图 |
| **2D 行列位置编码** | 可学习的 Row/Column 位置嵌入，将空间坐标信息注入视觉特征 |
| **Transformer 解码器** | 4 层标准 Transformer Decoder，使用可学习的 Layout Query 从视觉记忆中解码出原字/部件槽位 |
| **冻结形态学 Codebook** | `470×512` 的冻结原型矩阵，将 Codebook 嵌入投影到 Transformer 特征空间，通过点积计算分类 logits |
| **辅助 Box 回归头** | 两层 MLP + Sigmoid，输出每个槽位的归一化边界框 `[cx, cy, w, h]`（范围 [0, 1]） |

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_classes` | 470 | 原字/部件类别数 |
| `d_model` | 256 | Transformer 隐藏维度 |
| `nhead` | 8 | 多头注意力头数 |
| `num_layers` | 4 | Transformer 解码器层数 |
| `max_seq_len` | 10 | 最大序列长度（即最多10个部件槽位） |
| `codebook_dim` | 512 | Codebook 嵌入维度 |
| `sos_token` | 470 | 序列起始标记 |
| `eos_token` | 471 | 序列结束标记 |
| `pad_token` | 472 | 填充标记 |

### 数据流

```
输入图像 [B, 3, H, W]
    │
    ▼
ResNet Backbone → 特征图 [B, 512, h, w]
    │
    ▼
1×1 Conv 投影 → [B, 256, h, w]
    │
    ├──► Flatten → [B, h×w, 256]
    │         +
    │    2D 位置编码 → Visual Memory [B, h×w, 256]
    │
    ▼
Transformer Decoder:
    Query (Learnable) + Visual Memory
    │
    ▼
Slot Features [B, N, 256]
    │
    ├──► Box Head (MLP+Sigmoid) → pred_boxes [B, N, 4]
    │
    └──► Codebook 点积分类 → pred_logits [B, N, 473]
         (470 原字 + SOS + EOS + PAD)
```

---

## 阶段一：形态学 Codebook 训练 (`train_noise_robust_codebook.py`)

### 目标

训练一个 470×512 的抗噪形态学 Codebook，作为后续所有训练阶段的**冻结原型**。核心思想是使 **同类真实拓片裁剪特征靠近其对应的 Clean 原型**，同时 **远离其他 469 类的原型**。

### 数据格式

```
clean_dir/                   # 标准原字图像（白字黑底或黑字白底）
    0000.png                 # 类别 0 的标准图像
    0001.png                 # 类别 1 的标准图像
    ...
    0469.png                 # 类别 469 的标准图像

real_dir/                    # 真实拓片裁剪图像（按类别分目录）
    0/
        crop_1.png
        crop_2.png
        ...
    1/
        crop_1.png
        ...
    ...
```

### 核心训练机制

| 损失函数 | 公式 | 作用 |
|----------|------|------|
| **Clean Loss (Lc)** | CrossEntropy(clean_feat @ prototypes / τ, label) | 覆盖全部 470 类，防止冷门类别遗忘 |
| **Real Loss (Lr)** | CrossEntropy(real_feat @ prototypes / τ, label) | 在有真实 crop 的类别上做抗噪对齐 |
| **Align Loss (La)** | 1 - CosineSimilarity(real_feat, target_proto) | 显式拉近 real 特征与对应 clean 原型 |

总损失：`L = w_clean × Lc + w_real × Lr + w_align × La`

### 数据增强

- 自动检测并转换黑底白字格式
- PadToSquare + Resize(128×128)
- 随机仿射变换（±6°, 平移4%, 缩放0.92-1.08, 剪切4°）
- 高斯模糊（概率12%）
- 色彩抖动（亮度±12%, 对比度±18%）
- ImageNet 标准化

### 评估指标

- **Real Crop Retrieval Top-1 / Top-5**：使用训练好的编码器对真实裁剪图进行最近原型检索，评估 Top-1 和 Top-5 准确率

### 使用示例

```bash
python train_noise_robust_codebook.py \
  --clean_dir "dataset/data_standard_radicals" \
  --real_dir "path/to/radical_bank" \
  --out_dir "./checkpoints/stage1_codebook" \
  --epochs 40 \
  --batch_size 64 \
  --epoch_size 4096
```

### 输出文件

| 文件 | 说明 |
|------|------|
| `encoder_best.pth` | 最佳检索表现的编码器权重 |
| `encoder_latest.pth` | 最新 epoch 的编码器权重 |
| `codebook_noiseaware_best.pth` | 最佳 epoch 对应的 Codebook |
| `codebook_noiseaware_final.pth` | 最终导出的 Codebook（兼容 `load_codebook()`） |
| `train_log.json` | 训练日志（每 epoch 的损失与指标） |

---

## 阶段二：布局识别训练 (`train_layout.py`)

### 目标

在合成布局数据上训练 `CodebookLayoutTransformer` 模型（**Codebook 冻结**），使模型同时学习：
- 原字/部件序列预测
- 原字/部件归一化位置预测（GIoU + L1 联合优化）
- 字块内部布局关系

### 数据集格式

```
data_root/
├── train/
│   ├── images/
│   │   ├── block_000001.png
│   │   └── ...
│   └── labels.json        # { "block_000001.png": [{"label": 0, "bbox": [cx, cy, w, h]}, ...] }
├── val/
│   ├── images/
│   └── labels.json
└── test/
    ├── images/
    └── labels.json
```

### 标注格式

每个标注对象包含：
```json
{
  "label": <类别ID: 0-469>,
  "bbox": [<center_x>, <center_y>, <width>, <height>]
}
```
所有 bbox 值为**归一化坐标**（范围 [0, 1]）。

### 损失函数

| 损失 | 权重 | 说明 |
|------|------|------|
| CrossEntropy（分类） | 2.0 | 忽略 PAD token，覆盖全部 473 个 token（470 原字 + 3 特殊标记） |
| GIoU Loss（定位） | 2.0 | 仅计算非特殊标记的槽位 |
| L1 Loss（定位） | 5.0 | 与 GIoU 联合使用提升定位精度 |

### 使用示例

```bash
python train_layout.py \
  --data_root "path/to/layout_dataset" \
  --codebook_path "./checkpoints/stage1_codebook/codebook_noiseaware_final.pth" \
  --save_dir "./checkpoints/stage2_layout" \
  --batch_size 32 \
  --epochs 60 \
  --lr 1e-4
```

### 可视化

每个 epoch 对验证集样本进行可视化：在输入图像上绘制预测的边界框（红色矩形）和预测类别 ID（红色文字），保存至 `{save_dir}/vis/`。

---

## 阶段三：退化风格微调 (`finetune_degraded.py`)

### 目标

在带有噪声、模糊、断裂、残损等退化风格的数据上继续微调模型，提高模型对**真实拓片风格图像**的适应能力。

### 强数据增强策略

| 增强操作 | 参数 | 概率 |
|----------|------|------|
| 随机仿射变换 | ±5°, 平移5%, 缩放0.95-1.05, 剪切5° | 100% |
| 色彩抖动 | 亮度±0.4, 对比度±0.4, 饱和度±0.4, 色相±0.1 | 100% |
| 高斯模糊 | kernel=3, sigma=[0.1, 2.0] | 40% |
| 随机灰度化 | - | 20% |
| 随机擦除 | scale=[0.02, 0.10] | 30% |

### 损失函数

与阶段二相同，但权重调整为：分类 3.0 + GIoU 8.0 + L1 4.0，更强调定位精度。

### 使用示例

```bash
python finetune_degraded.py \
  --data_root "path/to/degraded_dataset/train" \
  --pretrained_ckpt "./checkpoints/stage2_layout/layout_ep60.pth" \
  --codebook_path "./checkpoints/stage1_codebook/codebook_noiseaware_final.pth" \
  --save_dir "./checkpoints/stage3_degraded" \
  --batch_size 64 \
  --epochs 60 \
  --lr 3e-5
```

---

## 阶段四：组合结构微调 (`finetune_composition.py`)

### 目标

混合**标准化组合数据集**和**随机退化布局数据集**，进一步增强模型对复杂组合结构的识别能力和 Box 定位稳定性。

### 核心特性

- **双数据集混合训练**：同时使用 `vocab_data_dir`（标准化词汇组合）和 `random_data_dir`（随机退化布局），通过 `ConcatDataset` 合并
- **分层学习率**：Backbone（ResNet）使用较低的学习率（默认 `1e-5`），逻辑头（Transformer + Box Head）使用较高的学习率（默认 `1e-4`）
- **余弦退火热重启调度器**：`CosineAnnealingWarmRestarts(T_0=5, T_mult=2)`
- **梯度累积**：支持多步梯度累积（默认 `4` 步），等效更大 batch size
- **标签平滑**：CrossEntropy 使用 `label_smoothing=0.1`，防止过拟合

### 损失函数

`L = L_cls × 2.0 + box_loss_weight × (L_giou + L_l1)`，其中默认 `box_loss_weight=10.0`。

### 使用示例

```bash
python finetune_composition.py \
  --vocab_data_dir "path/to/standardized_vocab_dataset/train" \
  --random_data_dir "path/to/degraded_random_dataset/val" \
  --pretrained_ckpt "./checkpoints/stage3_degraded/finetuned_ep60.pth" \
  --codebook_path "./checkpoints/stage1_codebook/codebook_noiseaware_final.pth" \
  --save_dir "./checkpoints/stage4_composition" \
  --batch_size 50 \
  --accumulation_steps 4 \
  --epochs 60 \
  --box_loss_weight 10.0
```

---

## 推理与评估 (`inference.py`)

### 功能概述

推理脚本 `inference.py` 是整个系统的最终评估入口，执行以下流程：

```
输入图像
    │
    ▼
模型前向推理 → pred_logits (473类), pred_boxes (归一化坐标)
    │
    ▼
Softmax + 取 Top-K 候选（每个槽位）
    │
    ▼
Slot 过滤：滤除非原字类别的预测（只保留 class 0-469）
    │
    ▼
NMS 去重（IoU 阈值默认 0.55）
    │
    ▼
空间排序：按行分组 → 行内从左到右排序
    │
    ▼
Top-K Beam Search 序列解码（默认 K=5）
    │
    ▼
可视化输出：原始图像 + Top-K 序列重建结果
    │
    ▼
评估指标计算（若有 Ground Truth）
```

### 推理管道详解

#### 1. Slot 过滤
滤除预测为 SOS (470)、EOS (471)、PAD (472) 的槽位，仅保留原字/部件（0-469）预测。

#### 2. NMS 去重
按置信度降序排列所有有效预测，迭代保留与已保留结果 IoU ≤ 阈值的预测，剔除重复检测。

#### 3. 空间排序
- 按 `cy`（中心 y 坐标）升序排列
- 使用平均高度的 45% 作为行阈值，将相邻预测分组为"行"
- 每行内部按 `cx`（中心 x 坐标）升序排列
- 逐行拼接得到最终有序序列

#### 4. Beam Search 序列解码
每个有序槽位有 Top-K 候选（K 默认 = max_k + 3，最终截取 max_k），从第一个槽位展开 beam search，保留总分最高的 K 条完整序列路径。

#### 5. 可视化生成
生成一张网格图，包含：
- **左列**：原始输入图像（可选叠加检测框）
- **右列**：每个 Top-K 序列的还原示意图
  - 通过 `SimpleGenerator` 将每个预测类别 ID 生成对应的原字图像
  - 将生成的原字图像按预测的边界框位置拼贴还原
  - 若与 Ground Truth 完全匹配则标注绿色 `[HIT]`

### 评估指标

| 指标 | 说明 |
|------|------|
| **Radical Top-1 Recall** | 原字级别的 Top-1 召回率（需要 Box IoU ≥ 阈值匹配） |
| **Radical Precision** | 原字级别的精确率 |
| **Radical F1-score** | 原字级别的 F1 分数 |
| **Radical Top-K Hit Rate** | 原字级别的 Top-K 命中率 |
| **1-NED Similarity** | 1 减去归一化编辑距离（Normalized Edit Distance），衡量序列级相似度 |
| **Block Top-1 Exact Accuracy** | 整块（Block）级别精确匹配率（Box 匹配 + 序列完全一致） |
| **Pure Sequence Top-K Exact Accuracy** | 纯序列精确匹配率（不考虑 Box 匹配） |
| **Empty-image Filtering Accuracy** | 空图像正确识别为空的比例 |

### SimpleGenerator 说明

`inference.py` 中包含一个 `SimpleGenerator` 类，用于可视化时根据类别 ID 生成对应的原字图像。其结构与 `model_layout_codebook.py` 中的 `CodebookLayoutTransformer.load_codebook()` 兼容，可直接加载 Codebook 中的 `input_text.TextEmbeddings` 作为条件输入。

### 使用示例

```bash
# 基本推理（仅可视化，无 Ground Truth）
python inference.py \
  --input_dir "path/to/test/images" \
  --output_dir "./results/inference" \
  --layout_ckpt "./checkpoints/final_model.pth" \
  --gen_ckpt "path/to/generator.pth" \
  --codebook_ckpt "./checkpoints/stage1_codebook/codebook_noiseaware_final.pth"

# 带 Ground Truth 评估
python inference.py \
  --dataset_root "path/to/dataset" \
  --split "test" \
  --output_dir "./results/eval" \
  --layout_ckpt "./checkpoints/final_model.pth" \
  --gen_ckpt "path/to/generator.pth" \
  --codebook_ckpt "./checkpoints/stage1_codebook/codebook_noiseaware_final.pth" \
  --iou 0.5 \
  --topk 1 3 5 \
  --debug
```

### 关键参数

| 参数 | 说明 |
|------|------|
| `--input_dir` | 测试图像目录 |
| `--dataset_root` | 数据集根目录（含 train/val/test 分集和 labels.json） |
| `--split` | 数据集分集名（默认 `train`） |
| `--output_dir` | 结果输出目录 |
| `--layout_ckpt` | 布局识别模型权重路径 |
| `--gen_ckpt` | 生成器权重路径 |
| `--codebook_ckpt` | Codebook 权重路径 |
| `--iou` | Box 匹配 IoU 阈值（默认 0.50） |
| `--topk` | Top-K 评估列表（默认 1 3 5） |
| `--debug` | 启用调试框绘制（在原始图像上绘制检测框） |

### 输出文件

| 文件 | 说明 |
|------|------|
| `res_{image_name}` | 每张测试图像的可视化结果（网格拼接图） |
| `eval_results.log` | 完整的评估日志和控制台输出 |

---

## 数据合成管线 (`Datecode/`)

数据合成管线位于 `Datecode/` 子目录，提供从标注拓片图像到完整合成数据集的四步流程。详细使用说明见 `Datecode/README_usage.md`。

### 第一步：构建部件库 (`build_radical_bank.py`)

从标注拓片图像中裁剪出每个原字/部件的独立图像，按类别 ID 分目录存储。

```bash
python Datecode/build_radical_bank.py \
  --input-dir "dataset/test_239" \
  --output-dir "data/radical_bank" \
  --center-format \
  --clear-output
```

**标注格式**：支持 JSON 文件（`image.json` 或 `image.png.json`），每个标注对象包含 `id` 和 `bbox`（可选 `bbox_format`）。

### 第二步：随机动态布局合成 (`synthesize_dynamic_layout.py`)

从部件库随机采样原字，放置到 1~7 个部件的布局模板中，生成多样化合成数据集。包含：
- 弹性变形、透视扭曲、噪声等拓片风格模拟
- Alpha 混合粘贴，模拟真实拓片的墨水叠加效果
- 全局对比度退化

### 第三步：真实序列布局合成 (`create_real_layout_dataset.py`)

根据真实契丹小字字符序列文件生成合成样本（详见 `Datecode/README_usage.md`）。

### 第四步：退化等级消融合成 (`create_real_dataset_by_degradation.py`)

生成不同退化等级（L1_Mild / L2_Moderate / L3_Severe）的数据集，用于退化鲁棒性消融实验。

### 标注格式约定

所有合成数据集生成的 `labels.json` 使用统一的归一化中心-尺寸边界框格式：

```json
{
  "block_000001.png": [
    {
      "id": 0,
      "bbox": [0.35, 0.42, 0.18, 0.16]
    }
  ]
}
```

其中 `bbox = [center_x, center_y, width, height]`，所有值归一化到 [0, 1]。

---

## 特殊 Token 设计

| Token | ID | 说明 |
|-------|-----|------|
| 原字/部件 | 0 ~ 469 | 470 类契丹小字原字（部件） |
| SOS | 470 | 序列起始标记（Start of Sequence） |
| EOS | 471 | 序列结束标记（End of Sequence） |
| PAD | 472 | 填充标记（Padding，损失计算时忽略） |

**序列格式**：`[SOS, radical_1, radical_2, ..., radical_n, EOS, PAD, PAD, ...]`，固定长度 `MAX_SEQ_LEN=10`。

---

## 数据增强汇总

### Stage 1（Codebook 训练）
| 增强 | 参数 |
|------|------|
| 随机仿射 | ±6°, 平移4%, 缩放0.92-1.08, 剪切4° (p=0.40) |
| 高斯模糊 | σ=[0.1, 1.0] (p=0.12) |
| 色彩抖动 | 亮度±12%, 对比度±18% |

### Stage 2（布局训练）
| 增强 | 参数 |
|------|------|
| Resize | 128×128 |
| Normalize | μ=[0.5, 0.5, 0.5], σ=[0.5, 0.5, 0.5] |

### Stage 3（退化微调）
| 增强 | 参数 |
|------|------|
| 随机仿射 | ±5°, 平移5%, 缩放0.95-1.05, 剪切5° |
| 色彩抖动 | 亮度/对比度/饱和度±0.4, 色相±0.1 |
| 高斯模糊 | σ=[0.1, 2.0] (p=0.4) |
| 随机灰度 | p=0.2 |
| 随机擦除 | scale=[0.02, 0.10] (p=0.3) |

### Stage 4（组合微调）
| 增强 | 参数 |
|------|------|
| Resize | 128×128 |
| 色彩抖动 | 亮度/对比度±0.2 |
| 随机灰度 | p=0.1 |
| Normalize | μ=[0.5, 0.5, 0.5], σ=[0.5, 0.5, 0.5] |

---

## 损失函数汇总

| 损失函数 | 说明 | 使用阶段 |
|----------|------|----------|
| **CrossEntropy Loss** | 分类损失，忽略 PAD token；Stage 4 使用 label_smoothing=0.1 | Stage 1-4 |
| **Cosine Similarity Loss** | `1 - cos(z_real, target_proto)`，拉近真实特征与原型 | Stage 1 |
| **GIoU Loss** | Generalized IoU 损失，Box 回归 | Stage 2-4 |
| **L1 Loss** | Box 坐标的 L1 平滑损失 | Stage 2-4 |

---

## 依赖环境

主要依赖以下 Python 包：

```
torch >= 1.12
torchvision
numpy
opencv-python (cv2)
Pillow (PIL)
tqdm
```

---

## 文件类型说明

| 文件 | 格式 | 说明 |
|------|------|------|
| `labels.json` | JSON | `{ "image.png": [{"id/label": int, "bbox": [cx,cy,w,h]}, ...] }` |
| `codebook_*.pth` | PyTorch | `{"input_text.TextEmbeddings": Tensor[470, 512], "meta": {...}}` |
| `encoder_*.pth` | PyTorch | `{"encoder": state_dict, "epoch": int, "args": {...}, "history": [...]}` |
| `layout_*.pth` | PyTorch | `CodebookLayoutTransformer` 的 `state_dict` |
| `train_log.json` | JSON | `[{epoch, loss, loss_clean, loss_real, ...}, ...]` |

---

## 快速开始（3 步走）

### 1. 准备数据

```bash
# 构建部件库
python Datecode/build_radical_bank.py \
  --input-dir "dataset/test_239" \
  --output-dir "data/radical_bank" \
  --center-format

# 合成训练数据（生成 10 万张）
python Datecode/synthesize_dynamic_layout.py \
  --bank-dir "data/radical_bank" \
  --output-root "data/synth_dynamic" \
  --num-images 100000
```

### 2. 训练模型

```bash
# 阶段一：训练 Codebook（需准备 clean_dir 和 real_dir）
python train_noise_robust_codebook.py \
  --clean_dir "dataset/data_standard_radicals" \
  --real_dir "data/radical_bank" \
  --out_dir "./checkpoints/stage1"

# 阶段二：训练布局识别器
python train_layout.py \
  --data_root "data/synth_dynamic" \
  --codebook_path "./checkpoints/stage1/codebook_noiseaware_final.pth" \
  --save_dir "./checkpoints/stage2" \
  --epochs 60

# 阶段三：退化风格微调
python finetune_degraded.py \
  --data_root "path/to/degraded_data" \
  --pretrained_ckpt "./checkpoints/stage2/layout_ep60.pth" \
  --codebook_path "./checkpoints/stage1/codebook_noiseaware_final.pth" \
  --save_dir "./checkpoints/stage3"

# 阶段四：组合结构微调
python finetune_composition.py \
  --vocab_data_dir "path/to/vocab_data" \
  --random_data_dir "path/to/random_data" \
  --pretrained_ckpt "./checkpoints/stage3/finetuned_ep60.pth" \
  --codebook_path "./checkpoints/stage1/codebook_noiseaware_final.pth" \
  --save_dir "./checkpoints/stage4"
```

### 3. 推理评估

```bash
python inference.py \
  --dataset_root "path/to/test_dataset" \
  --split "test" \
  --output_dir "./results/eval" \
  --layout_ckpt "./checkpoints/stage4/composition_ep60.pth" \
  --gen_ckpt "path/to/generator.pth" \
  --codebook_ckpt "./checkpoints/stage1/codebook_noiseaware_final.pth" \
  --iou 0.5 \
  --topk 1 3 5
```

---

## 注意事项

- **Codebook 兼容性**：`model_layout_codebook.py` 中的 `load_codebook()` 方法支持多种格式，包括纯 Tensor、包含 `input_text.TextEmbeddings` 的字典，以及自动匹配形状的任意张量。
- **黑底白字预处理**：`train_noise_robust_codebook.py` 中的 `ensure_black_bg_white_fg()` 函数会自动检测图像亮度并反色，确保统一的黑底白字格式。
- **梯度累积**：阶段四支持 `--accumulation_steps`，可用于在显存受限的情况下模拟更大的 batch size。
- **分层学习率**：阶段四对 Backbone 和逻辑头使用不同的学习率，保护预训练视觉特征的同时允许分类/定位头充分学习。
- **所有路径均通过命令行参数传入**，不依赖任何硬编码的绝对路径。