from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class GalleryContextEncoder(nn.Module):
    """Full self-attention over gallery embeddings; no ordering signals."""

    def __init__(self, dim: int, layers: int = 2, heads: int = 8,
                 ff_dim: int | None = None, dropout: float = 0.1,
                 residual: bool = True):
        super().__init__()
        heads = min(heads, dim)
        while dim % heads:
            heads -= 1
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=ff_dim or 4 * dim,
            dropout=dropout, batch_first=True, norm_first=True,
            activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.residual = residual

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None):
        # x: [batch, gallery, dim], mask=True means padded item.
        y = self.encoder(x, src_key_padding_mask=padding_mask)
        if self.residual:
            y = y + x
        y = self.norm(y)
        if padding_mask is not None:
            y = y.masked_fill(padding_mask.unsqueeze(-1), 0)
        return F.normalize(y, dim=-1)


class GalleryCLIP(nn.Module):
    def __init__(self, clip_model, layers=2, heads=8, ff_dim=None,
                 dropout=0.1, residual=True, use_gallery_context=True):
        super().__init__()
        self.clip = clip_model
        dim = clip_model.text_projection.shape[-1]
        self.gallery = GalleryContextEncoder(dim, layers, heads, ff_dim,
                                             dropout, residual)
        self.use_gallery_context = use_gallery_context
        self.logit_scale = clip_model.logit_scale

    def encode_images(self, images):
        return F.normalize(self.clip.encode_image(images), dim=-1)

    def contextualize(self, image_features, padding_mask=None):
        if not self.use_gallery_context:
            out = F.normalize(image_features, dim=-1)
            if padding_mask is not None:
                out = out.masked_fill(padding_mask.unsqueeze(-1), 0)
            return out
        return self.gallery(image_features, padding_mask)

    def encode_text(self, tokens):
        return F.normalize(self.clip.encode_text(tokens), dim=-1)

    def forward(self, images, tokens, padding_mask=None):
        image_features = self.encode_images(images)
        contextual = self.contextualize(image_features, padding_mask)
        text_features = self.encode_text(tokens)
        scale = self.logit_scale.exp().clamp(max=100)
        return scale * text_features @ contextual.transpose(-1, -2)
