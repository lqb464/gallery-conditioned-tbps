from __future__ import annotations

import argparse
from pathlib import Path
import torch
from tqdm import tqdm


@torch.no_grad()
def evaluate(model, dataset, device=None, batch_size=64, workers=4):
    from torch.utils.data import DataLoader
    device = device or next(model.parameters()).device
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    visual, labels = [], []
    for images, _tokens, pids in tqdm(loader, desc="gallery features"):
        visual.append(model.encode_images(images.to(device)))
        labels.extend(pids.tolist())
    gallery = model.contextualize(torch.cat(visual, 0).unsqueeze(0)).squeeze(0)
    # Score all reference captions, preserving every image's caption annotations.
    tokens, caption_labels = [], []
    for i, row in enumerate(dataset.records):
        for caption in row["captions"]:
            tokens.append(dataset.tokenizer([caption])[0]); caption_labels.append(i)
    text_features = []
    for start in range(0, len(tokens), batch_size):
        text_features.append(model.encode_text(torch.stack(tokens[start:start+batch_size]).to(device)))
    scores = torch.cat(text_features) @ gallery.T
    target = torch.tensor(labels, device=device)
    caption_pid = target[torch.tensor(caption_labels, device=device)]
    ranked = target[scores.argsort(dim=1, descending=True)]
    correct = ranked.eq(caption_pid[:, None])
    metrics = {f"text_to_image_r{k}": correct[:, :k].any(1).float().mean().item() for k in (1, 5, 10)}
    sorted_scores = scores.argsort(dim=1, descending=True)
    sorted_match = target[sorted_scores].eq(caption_pid[:, None])
    ranks = torch.arange(1, target.numel() + 1, device=device).float().unsqueeze(0)
    cumulative = sorted_match.cumsum(dim=1).float()
    positives = sorted_match.sum(dim=1).clamp_min(1)
    average_precision = ((cumulative / ranks) * sorted_match).sum(dim=1) / positives
    last_positive_rank = (sorted_match * ranks).amax(dim=1)
    inverse_negative_penalty = positives / last_positive_rank.clamp_min(1)
    metrics["text_to_image_mAP"] = average_precision.mean().item()
    metrics["text_to_image_mINP"] = inverse_negative_penalty.mean().item()
    # Image-to-text uses the best caption rank among all descriptions of that identity.
    image_scores = gallery @ torch.cat(text_features).T
    text_ids = caption_pid
    top_text = image_scores.argsort(dim=1, descending=True)
    text_rank_ids = text_ids[top_text]
    image_hit = text_rank_ids.eq(target[:, None])
    metrics.update({f"image_to_text_r{k}": image_hit[:, :k].any(1).float().mean().item() for k in (1, 5, 10)})
    metrics["mean_r1"] = metrics["text_to_image_r1"]
    return metrics


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data-root", required=True)
    ap.add_argument("--checkpoint", required=True); ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--batch-size", type=int, default=64); args = ap.parse_args()
    import open_clip
    from .data import PersonTextDataset, build_transforms, load_records
    from .model import GalleryCLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}. Training must finish successfully before evaluation.")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["args"]
    image_size = tuple(cfg["image_size"])
    clip, _, _ = open_clip.create_model_and_transforms(
        cfg["model"], pretrained=cfg["pretrained"], force_image_size=image_size,
        force_quick_gelu=True)
    model = GalleryCLIP(clip, num_classes=ckpt["num_classes"],
                        layers=cfg["gallery_layers"], heads=cfg["gallery_heads"],
                        ff_dim=cfg["gallery_ff_dim"], dropout=cfg["gallery_dropout"],
                        residual=cfg["gallery_residual"],
                        use_gallery_context=cfg["gallery_context"],
                        temperature=cfg["temperature"]).to(device)
    model.load_state_dict(ckpt["model"])
    ds = PersonTextDataset(load_records(args.data_root, args.split), build_transforms(False, image_size), open_clip.get_tokenizer(cfg["model"]))
    print(evaluate(model, ds, device=device, batch_size=args.batch_size))

if __name__ == "__main__": main()
