from __future__ import annotations

import argparse, random
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import open_clip

from .data import IdentityBatchSampler, PersonTextDataset, load_records
from .model import GalleryCLIP
from .evaluate import evaluate


def multi_positive_clip_loss(logits, labels):
    # Symmetric supervised contrastive loss. Labels are only positives in loss,
    # and never affect gallery self-attention.
    pos = labels[:, None].eq(labels[None, :])
    logp_i = logits.log_softmax(dim=1)
    logp_t = logits.T.log_softmax(dim=1)
    return -0.5 * ((logp_i.masked_fill(~pos, 0).sum(1) / pos.sum(1)).mean() +
                   (logp_t.masked_fill(~pos.T, 0).sum(1) / pos.sum(0)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True); ap.add_argument("--output-dir", default="runs/gallery")
    ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--accumulation-steps", type=int, default=1); ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--context-lr", type=float, default=1e-4); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default="ViT-B-16"); ap.add_argument("--pretrained", default=" laion2b_s34b_b88".strip())
    ap.add_argument("--seed", type=int, default=17); args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tok = open_clip.get_tokenizer(args.model)
    model = GalleryCLIP(clip).to(device)
    train_ds = PersonTextDataset(load_records(args.data_root, "train"), preprocess, tok)
    val_rows = load_records(args.data_root, "val")
    val_ds = PersonTextDataset(val_rows, preprocess, tok)
    batch_sampler = IdentityBatchSampler(train_ds, args.batch_size)
    loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.workers, pin_memory=True)
    optim = torch.optim.AdamW([{"params": model.clip.parameters(), "lr": args.lr}, {"params": model.gallery.parameters(), "lr": args.context_lr}], weight_decay=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True); best = -1
    for epoch in range(args.epochs):
        model.train(); optim.zero_grad(set_to_none=True); running = 0
        for step, (images, tokens, labels) in enumerate(tqdm(loader, desc=f"epoch {epoch+1}")):
            images, tokens, labels = images.to(device), tokens.to(device), labels.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = model(images.unsqueeze(0), tokens).squeeze(0)
                loss = multi_positive_clip_loss(logits, labels) / args.accumulation_steps
            scaler.scale(loss).backward(); running += loss.item() * args.accumulation_steps
            if (step + 1) % args.accumulation_steps == 0:
                scaler.step(optim); scaler.update(); optim.zero_grad(set_to_none=True)
        print(f"epoch={epoch+1} train_loss={running/max(1,len(loader)):.4f}")
        metrics = evaluate(model, val_ds, device=device, batch_size=args.batch_size, workers=args.workers)
        print("validation", metrics)
        score = metrics["mean_r1"]
        state = {"model": model.state_dict(), "epoch": epoch+1, "metrics": metrics, "args": vars(args)}
        torch.save(state, out / "last.pt")
        if score > best:
            best = score; torch.save(state, out / "best.pt")

if __name__ == "__main__": main()
