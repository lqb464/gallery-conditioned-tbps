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
    def __init__(self, clip_model, num_classes=0, layers=2, heads=8, ff_dim=None,
                 dropout=0.1, residual=True, use_gallery_context=True,
                 temperature=0.02):
        super().__init__()
        self.clip = clip_model
        if hasattr(self.clip, "logit_scale"):
            self.clip.logit_scale.requires_grad_(False)
        dim = clip_model.text_projection.shape[-1]
        self.gallery = GalleryContextEncoder(dim, layers, heads, ff_dim,
                                             dropout, residual)
        self.use_gallery_context = use_gallery_context
        self.register_buffer("logit_scale", torch.tensor(1.0 / temperature))
        self.classifier = nn.Linear(dim, num_classes) if num_classes else None
        if self.classifier is not None:
            nn.init.normal_(self.classifier.weight, std=0.001)
            nn.init.constant_(self.classifier.bias, 0.0)

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

    def compute_loss(self, image_features, text_features, labels):
        """Original CLIP TBPS losses: similarity distribution matching + ID CE."""
        labels = labels.reshape(-1, 1)
        positives = labels.eq(labels.T).float()
        target = positives / positives.sum(dim=1, keepdim=True).clamp_min(1)
        # Keep log(target + epsilon) in fp32; epsilon underflows in fp16.
        with torch.autocast(device_type=image_features.device.type, enabled=False):
            image32 = F.normalize(image_features.float(), dim=-1)
            text32 = F.normalize(text_features.float(), dim=-1)
            scale = self.logit_scale.float().to(image_features.device).clamp(max=100)
            logits_i = scale * image32 @ text32.T
            logits_t = logits_i.T

            def sdm_direction(logits):
                pred = logits.softmax(dim=1)
                return (pred * (logits.log_softmax(dim=1) - (target + 1e-8).log())).sum(dim=1).mean()

            sdm = sdm_direction(logits_i) + sdm_direction(logits_t)
        if self.classifier is None:
            raise RuntimeError("num_classes is required to compute the CLIP TBPS ID loss")
        id_loss = 0.5 * (F.cross_entropy(self.classifier(image32), labels[:, 0]) +
                         F.cross_entropy(self.classifier(text32), labels[:, 0]))
        return sdm + id_loss, {"sdm": sdm.detach(), "id": id_loss.detach()}

    def forward(self, images, tokens, padding_mask=None):
        if images.ndim == 5:
            gallery_batch, gallery_size = images.shape[:2]
            images = images.reshape(gallery_batch * gallery_size, *images.shape[2:])
        else:
            gallery_batch, gallery_size = 1, images.shape[0]
        image_features = self.encode_images(images)
        image_features = image_features.reshape(gallery_batch, gallery_size, -1)
        contextual = self.contextualize(image_features, padding_mask)
        text_features = self.encode_text(tokens)
        scale = self.logit_scale.clamp(max=100)
        if text_features.ndim == 2 and gallery_batch == 1:
            return scale * text_features @ contextual.squeeze(0).T
        return scale * text_features @ contextual.transpose(-1, -2)
