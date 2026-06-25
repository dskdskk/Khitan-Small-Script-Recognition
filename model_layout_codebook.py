import torch
import torch.nn as nn
from torchvision import models


class CodebookLayoutTransformer(nn.Module):
    """
    Prototype-constrained layout Transformer for Khitan Small Script radical recognition.

    The model consists of:
    1. A ResNet visual encoder for extracting 2D visual features.
    2. Learnable row and column positional embeddings for layout-aware visual memory.
    3. Learnable layout queries and a Transformer decoder for radical slot prediction.
    4. A prototype-based classifier using a frozen morphological codebook.
    5. An auxiliary box regression head for spatial grounding.
    """

    def __init__(
        self,
        codebook_path: str,
        num_classes: int = 470,      # number of radical/component categories
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        max_seq_len: int = 10,
        codebook_dim: int = 512,
        max_pos_size: int = 50,
        sos_token: int = 470,
        eos_token: int = 471,
        pad_token: int = 472,
        pretrained_backbone: bool = True,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.d_model = d_model
        self.codebook_dim = codebook_dim
        self.max_seq_len = max_seq_len
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.num_special = 3  # SOS, EOS, PAD

        # ----------------------------------------------------------
        # 1. Visual encoder
        # ----------------------------------------------------------
        weights = models.ResNet18_Weights.DEFAULT if pretrained_backbone else None
        resnet = models.resnet18(weights=weights)

        # Remove average pooling and fully connected layers.
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Project ResNet feature channels to Transformer hidden dimension.
        self.conv_proj = nn.Conv2d(512, d_model, kernel_size=1)

        # ----------------------------------------------------------
        # 2. Frozen morphological codebook
        # ----------------------------------------------------------
        self.register_buffer(
            "codebook",
            torch.zeros(num_classes, codebook_dim),
            persistent=True,
        )
        self.load_codebook(codebook_path)

        # Project codebook embeddings to Transformer feature space.
        self.codebook_proj = nn.Linear(codebook_dim, d_model)

        # Learnable embeddings for special tokens: SOS, EOS, PAD.
        self.special_token_embed = nn.Parameter(
            torch.randn(self.num_special, d_model)
        )

        # ----------------------------------------------------------
        # 3. 2D positional encoding
        # ----------------------------------------------------------
        self.pos_embed_row = nn.Parameter(
            torch.randn(max_pos_size, d_model // 2)
        )
        self.pos_embed_col = nn.Parameter(
            torch.randn(max_pos_size, d_model // 2)
        )

        # ----------------------------------------------------------
        # 4. Layout Transformer decoder
        # ----------------------------------------------------------
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )

        # Learnable layout queries. Each query corresponds to a potential radical slot.
        self.query_embed = nn.Embedding(max_seq_len, d_model)

        # ----------------------------------------------------------
        # 5. Prediction heads
        # ----------------------------------------------------------
        self.box_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 4),
            nn.Sigmoid(),  # normalized box: cx, cy, w, h in [0, 1]
        )

    def load_codebook(self, path: str):
        """
        Load a frozen morphological codebook.

        Supported formats:
        1. Tensor with shape [num_classes, codebook_dim].
        2. Dict containing one of:
           - 'input_text.TextEmbeddings'
           - 'TextGenerator.input_text.TextEmbeddings'
           - any tensor matching [num_classes, codebook_dim]
        """
        ckpt = torch.load(path, map_location="cpu")
        embeds = None

        if isinstance(ckpt, torch.Tensor):
            embeds = ckpt

        elif isinstance(ckpt, dict):
            if "input_text.TextEmbeddings" in ckpt:
                embeds = ckpt["input_text.TextEmbeddings"]

            elif "TextGenerator.input_text.TextEmbeddings" in ckpt:
                embeds = ckpt["TextGenerator.input_text.TextEmbeddings"]

            else:
                for key, value in ckpt.items():
                    if isinstance(value, torch.Tensor) and value.shape == self.codebook.shape:
                        embeds = value
                        break

        if embeds is None:
            raise ValueError(
                "Failed to load codebook. No valid tensor was found in the checkpoint."
            )

        if embeds.shape != self.codebook.shape:
            raise ValueError(
                f"Invalid codebook shape: expected {tuple(self.codebook.shape)}, "
                f"but got {tuple(embeds.shape)}."
            )

        with torch.no_grad():
            self.codebook.copy_(embeds.float())

        print(f"Codebook loaded successfully from: {path}")

    def build_2d_positional_encoding(self, batch_size: int, h: int, w: int, device):
        """
        Build 2D positional encoding from feature-map grid coordinates.

        For each spatial position (row, col), the positional embedding is:
            pos(row, col) = concat(row_embedding(row), col_embedding(col))
        """
        if h > self.pos_embed_row.size(0) or w > self.pos_embed_col.size(0):
            raise ValueError(
                f"Feature map size ({h}, {w}) exceeds maximum positional size "
                f"({self.pos_embed_row.size(0)}, {self.pos_embed_col.size(0)})."
            )

        pos_row = self.pos_embed_row[:h].unsqueeze(1).repeat(1, w, 1)
        pos_col = self.pos_embed_col[:w].unsqueeze(0).repeat(h, 1, 1)

        pos = torch.cat([pos_row, pos_col], dim=-1)
        pos = pos.flatten(0, 1).unsqueeze(0).repeat(batch_size, 1, 1)
        return pos.to(device)

    def forward(self, img):
        """
        Args:
            img: input image tensor with shape [B, 3, H, W].

        Returns:
            pred_logits: slot-wise token logits, shape [B, N, num_classes + 3].
            pred_boxes: slot-wise normalized boxes, shape [B, N, 4].
        """

        # 1. Extract 2D visual feature map.
        features = self.backbone(img)          # [B, 512, h, w]
        features = self.conv_proj(features)    # [B, D, h, w]

        batch_size, channels, h, w = features.shape

        # 2. Flatten visual feature map into visual memory tokens.
        features_flat = features.flatten(2).permute(0, 2, 1)  # [B, h*w, D]

        # 3. Add 2D row-column positional encoding.
        pos_enc = self.build_2d_positional_encoding(
            batch_size=batch_size,
            h=h,
            w=w,
            device=features.device,
        )
        visual_memory = features_flat + pos_enc

        # 4. Decode radical slots using learnable layout queries.
        queries = self.query_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        slot_features = self.transformer_decoder(
            tgt=queries,
            memory=visual_memory,
        )  # [B, N, D]

        # 5. Auxiliary box prediction.
        pred_boxes = self.box_head(slot_features)

        # 6. Prototype-based token classification.
        radical_prototypes = self.codebook_proj(self.codebook)  # [470, D]

        token_prototypes = torch.cat(
            [radical_prototypes, self.special_token_embed],
            dim=0,
        )  # [473, D]

        pred_logits = torch.matmul(slot_features, token_prototypes.t())

        return pred_logits, pred_boxes

    def get_codebook_embeddings(self, labels: torch.Tensor):
        """
        Retrieve frozen codebook embeddings by radical labels.

        Args:
            labels: tensor of radical labels. Valid radical labels should be in [0, num_classes - 1].

        Returns:
            embeddings: codebook embeddings with shape [..., codebook_dim].
        """
        if torch.any(labels < 0) or torch.any(labels >= self.num_classes):
            raise ValueError(
                "get_codebook_embeddings only accepts radical labels in "
                f"[0, {self.num_classes - 1}]. Special tokens should be filtered before calling this function."
            )

        with torch.no_grad():
            return self.codebook[labels]