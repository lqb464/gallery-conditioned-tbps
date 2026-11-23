# Gallery-Conditioned CLIP for Text-Based Person Search

Minimal RSTPReid implementation of Approach A, aligned to the [official CLIP TBPS/IRRA implementation](https://github.com/anosorae/IRRA): OpenAI CLIP ViT-B/16, 384×128 ReID images, 77-token text context, identity-balanced batch 64, SDM + ID losses, Adam at 1e-5, weight decay 4e-5, and cosine decay with five warmup epochs. The two-layer gallery Transformer is the new component and uses five times the base learning rate.

## Kaggle quick start

Open `notebooks/kaggle_rstpreid.ipynb` in Kaggle. Enable a GPU accelerator (T4 x2 is supported), attach the `hoanggv/tbps-benchmark` dataset, set `DATA_ROOT` to its mounted directory, then run all cells. Mixed precision is enabled. One T4 processes each complete gallery-context batch; the second GPU remains idle so attention can cover the whole group. Checkpoints and retrieval metrics are saved under `/kaggle/working/gallery_tbps/`.

The dataset loader accepts RSTPReid JSON annotation files with records containing `id`, `img_path`/`file_path`, `captions`, and `split`; it also accepts Hugging Face `datasets` parquet exports with those columns. Paths are resolved relative to `DATA_ROOT`.

## Method

Each training batch is a gallery group sampled as 16 identities × 4 image-caption pairs. The gallery encoder self-attends across the entire batch; identity labels affect SDM and ID losses only, never attention masks. Validation/test retrieval encodes the complete split gallery, contextualizes its features once, then ranks each caption against the contextualized gallery. A changed gallery composition therefore requires recomputing contextual features, while original CLIP image features can be cached.

The model is a research prototype. This checkout did not contain the pre-existing codebase or pretrained project checkpoint described in the task brief, so this implementation is built from scratch using OpenCLIP's public pretrained weights. No benchmark result is claimed until the Kaggle notebook is run.

## Local commands

```bash
pip install -r requirements.txt
python -m tbps.train --config configs/clip_tbps_rstpreid.yaml --data-root /path/to/RSTPReid --output-dir runs/gallery
python -m tbps.evaluate --data-root /path/to/RSTPReid --checkpoint runs/gallery/best.pt --split test
```

Training selects the best checkpoint by validation text-to-image R@1. Evaluation reports text-to-image R@1/5/10, mAP, and mINP, plus image-to-text R@1/5/10. The notebook is the recommended entry point for Kaggle.
