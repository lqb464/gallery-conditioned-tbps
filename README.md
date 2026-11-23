# Gallery-Conditioned CLIP for Text-Based Person Search

Minimal RSTPReid research implementation of Approach A: encode image and text with OpenCLIP, then apply permutation-equivariant full self-attention over the candidate gallery image embeddings before text-to-image ranking.

## Kaggle quick start

Open `notebooks/kaggle_rstpreid.ipynb` in Kaggle. Enable a GPU accelerator (T4 x2 is supported), attach the `hoanggv/tbps-benchmark` dataset, set `DATA_ROOT` to its mounted directory, then run all cells. The notebook uses mixed precision and gradient accumulation. It trains the CLIP encoders and gallery encoder jointly. Checkpoints and retrieval metrics are saved under `/kaggle/working/gallery_tbps/`.

The dataset loader accepts RSTPReid JSON annotation files with records containing `id`, `img_path`/`file_path`, `captions`, and `split`; it also accepts Hugging Face `datasets` parquet exports with those columns. Paths are resolved relative to `DATA_ROOT`.

## Method

Each training batch is a gallery group of images sampled from multiple identities. The gallery encoder self-attends across the entire batch; identity labels are used only for multi-positive image-text contrastive loss, never to construct attention masks. Validation/test retrieval encodes the complete split gallery, contextualizes its features once, then ranks each caption against the contextualized gallery. A changed gallery composition therefore requires recomputing contextual features, while original CLIP image features can be cached.

The model is a research prototype. This checkout did not contain the pre-existing codebase or pretrained project checkpoint described in the task brief, so this implementation is built from scratch using OpenCLIP's public pretrained weights. No benchmark result is claimed until the Kaggle notebook is run.

## Local commands

```bash
pip install -r requirements.txt
python -m tbps.train --data-root /path/to/RSTPReid --output-dir runs/gallery --epochs 10
python -m tbps.evaluate --data-root /path/to/RSTPReid --checkpoint runs/gallery/best.pt --split test
```

The training run selects the best checkpoint by validation image-to-text/text-to-image mean Recall@1. The notebook is the recommended entry point for Kaggle.
