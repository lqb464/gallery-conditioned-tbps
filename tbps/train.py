from __future__ import annotations

import argparse, random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import open_clip

from .data import IdentityBatchSampler, PersonTextDataset, build_transforms, load_records
from .model import GalleryCLIP
from .evaluate import evaluate


def build_optimizer(model, lr, gallery_lr_factor=5.0, weight_decay=4e-5):
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "bias" in name:
            base = lr * 2.0
            if name.startswith("classifier."):
                base = lr * gallery_lr_factor
            decay = 0.0
        else:
            base = lr * (gallery_lr_factor if name.startswith("gallery.") or name.startswith("classifier.") else 1.0)
            decay = weight_decay
        groups.setdefault((base, decay), []).append(parameter)
    return torch.optim.Adam([{"params": ps, "lr": rate, "weight_decay": decay}
                             for (rate, decay), ps in groups.items()],
                            lr=lr, betas=(0.9, 0.999), eps=1e-3)


def build_scheduler(optimizer, epochs, warmup_epochs=5, warmup_factor=0.1):
    import math
    def factor(epoch):
        if epoch < warmup_epochs:
            alpha = epoch / max(1, warmup_epochs)
            return warmup_factor * (1 - alpha) + alpha
        ratio = min(1.0, (epoch - warmup_epochs) / max(1, epochs - warmup_epochs))
        return 0.5 * (1 + math.cos(math.pi * ratio))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def main():
    import yaml
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", default="configs/clip_tbps_rstpreid.yaml")
    known, _ = bootstrap.parse_known_args()
    with open(known.config) as f:
        defaults = yaml.safe_load(f)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=known.config)
    ap.add_argument("--data-root", required=True); ap.add_argument("--output-dir", default="runs/gallery")
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--accumulation-steps", type=int, default=1); ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--gallery-lr-factor", type=float, default=5.0); ap.add_argument("--weight-decay", type=float, default=4e-5)
    ap.add_argument("--warmup-epochs", type=int, default=5); ap.add_argument("--warmup-factor", type=float, default=0.1)
    ap.add_argument("--temperature", type=float, default=0.02); ap.add_argument("--num-instances", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--image-size", type=int, nargs=2, default=(384, 128))
    ap.add_argument("--gallery-layers", type=int, default=2); ap.add_argument("--gallery-heads", type=int, default=8)
    ap.add_argument("--gallery-ff-dim", type=int, default=2048); ap.add_argument("--gallery-dropout", type=float, default=0.1)
    ap.add_argument("--gallery-context", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--gallery-residual", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--model", default="ViT-B-16"); ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--seed", type=int, default=17)
    ap.set_defaults(**defaults)
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained,
                                                       force_image_size=tuple(args.image_size))
    tok = open_clip.get_tokenizer(args.model)
    train_transform = build_transforms(True, tuple(args.image_size), augmentation=True)
    eval_transform = build_transforms(False, tuple(args.image_size))
    train_ds = PersonTextDataset(load_records(args.data_root, "train"), train_transform, tok, training=True)
    val_rows = load_records(args.data_root, "val")
    val_ds = PersonTextDataset(val_rows, eval_transform, tok)
    model = GalleryCLIP(clip, num_classes=len(train_ds.pid_to_label),
                        layers=args.gallery_layers, heads=args.gallery_heads,
                        ff_dim=args.gallery_ff_dim, dropout=args.gallery_dropout,
                        residual=args.gallery_residual,
                        use_gallery_context=args.gallery_context,
                        temperature=args.temperature).to(device)
    batch_sampler = IdentityBatchSampler(train_ds, args.batch_size, args.num_instances)
    loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.workers, pin_memory=True)
    optim = build_optimizer(model, args.lr, args.gallery_lr_factor, args.weight_decay)
    scheduler = build_scheduler(optim, args.epochs, args.warmup_epochs, args.warmup_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True); best = -1
    for epoch in range(args.epochs):
        model.train(); optim.zero_grad(set_to_none=True); running = 0
        for step, (images, tokens, labels) in enumerate(tqdm(loader, desc=f"epoch {epoch+1}")):
            images, tokens, labels = images.to(device), tokens.to(device), labels.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                visual = model.encode_images(images)
                contextual = model.contextualize(visual.unsqueeze(0)).squeeze(0)
                text = model.encode_text(tokens)
                loss, loss_parts = model.compute_loss(contextual, text, labels)
                loss = loss / args.accumulation_steps
            scaler.scale(loss).backward(); running += loss.item() * args.accumulation_steps
            if (step + 1) % args.accumulation_steps == 0:
                scaler.step(optim); scaler.update(); optim.zero_grad(set_to_none=True)
        print(f"epoch={epoch+1} train_loss={running/max(1,len(loader)):.4f}")
        metrics = evaluate(model, val_ds, device=device, batch_size=args.batch_size, workers=args.workers)
        print("validation", metrics)
        score = metrics["text_to_image_r1"]
        state = {"model": model.state_dict(), "epoch": epoch+1, "metrics": metrics,
                 "num_classes": len(train_ds.pid_to_label), "args": vars(args)}
        torch.save(state, out / "last.pt")
        if score > best:
            best = score; torch.save(state, out / "best.pt")
        scheduler.step()

if __name__ == "__main__": main()
