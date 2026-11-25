# Gallery-Conditioned CLIP for Text-Based Person Retrieval

This repository incorporates IRRA implementation files copied from the supplied backup folder. The backup itself was left untouched. The copied source is under the MIT license; see `LICENSE` and the upstream README copy in `docs/`.

## What is being compared

- **IRRA baseline:** original model path, with `--gallery_context` omitted.
- **IRRA + Gallery Context:** the same pretrained CLIP image/text encoders and original IRRA SDM, ID, and MLM losses, with a two-layer, permutation-equivariant Transformer over the image gallery. SDM and ID train the contextualized image CLS′ against text EOS / identity labels; MLM remains on IRRA's original token-level cross-modal path.

Gallery membership/order is not given identity labels or positional embeddings. Thus attention is permutation-equivariant. The contextualized image embedding is compared to the original text embedding. The evaluation gallery must be fixed for contextual features to remain valid.

This is a research experiment, not a reported SOTA result. Training and evaluation have not been run in this environment because PyTorch/CUDA and the dataset are unavailable here.

## Kaggle

Open `notebooks/kaggle_rstpreid.ipynb` in Colab, extract RSTPReid so the root contains `RSTPReid/data_captions.json` and `RSTPReid/imgs/`, and enable GPU. The notebook trains the IRRA + Gallery Context model for 10 epochs with original `sdm+id+mlm` losses and evaluates both raw CLIP features (`t2i-base`) and contextualized gallery features (`t2i-hard-contrastive`) after each epoch. The final test cell reports the same two rows. The notebook explicitly clones `research/gallery-conditioned-representation` and calls root-level `train.py`.

The dataset root passed to IRRA should contain `RSTPReid/data_captions.json` and `RSTPReid/imgs/`. Adjust `DATA_ROOT` in the notebook if Kaggle mounts this dataset at a different path.

## Manual commands

Baseline:

```bash
python train.py --dataset_name RSTPReid --root_dir /path/to/data --output_dir runs \
  --name irra_baseline --val_dataset val --loss_names sdm+id+mlm --MLM \
  --sampler identity --num_instance 4 --batch_size 32 --num_workers 2 \
  --num_epoch 10 --seed 17
```

Gallery Context:

```bash
python train.py --dataset_name RSTPReid --root_dir /path/to/data --output_dir runs \
  --name gallery_context --val_dataset val --loss_names sdm+id+mlm --MLM \
  --sampler identity --num_instance 4 --batch_size 32 --num_workers 2 \
  --num_epoch 10 --seed 17 --gallery_context \
  --gallery_context_depth 2 --gallery_context_heads 8
```

The baseline is the default. Gallery Context requires identity sampling and `batch_size` divisible by `num_instance`. Inference command (replace the path with the timestamped run directory printed by training):

```bash
python test.py --config_file runs/RSTPReid/<run-name>/configs.yaml
```

The saved config restores whether Gallery Context is enabled.

## Source and attribution

The IRRA files originate from the backup supplied for this task and retain their upstream code headers/license notices. See [the upstream IRRA README copy](docs/IRRA_UPSTREAM_README.md). The accompanying license is retained in `LICENSE`.
