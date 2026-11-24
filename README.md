# Gallery-Conditioned CLIP for Text-Based Person Retrieval

This repository incorporates IRRA implementation files copied from the supplied backup folder. The backup itself was left untouched. The copied source is under the MIT license; see `LICENSE` and the upstream README copy in `docs/`.

## What is being compared

- **IRRA baseline:** original model path, with `--gallery_context` omitted.
- **IRRA + Gallery Context:** the same CLIP image/text encoders and IRRA SDM/ID losses, with a two-layer, permutation-equivariant Transformer over a set of images. During training it sees an identity-balanced batch; during evaluation it sees the complete gallery split.

Gallery membership/order is not given identity labels or positional embeddings. Thus attention is permutation-equivariant. The contextualized image embedding is compared to the original text embedding. The evaluation gallery must be fixed for contextual features to remain valid.

This is a research experiment, not a reported SOTA result. Training and evaluation have not been run in this environment because PyTorch/CUDA and the dataset are unavailable here.

## Kaggle

Open `notebooks/kaggle_rstpreid.ipynb`, attach Kaggle dataset `hoanggv/tbps-benchmark`, and enable GPU. The notebook first trains the original IRRA baseline and then the Gallery Context variant on RSTPReid. It uses one GPU (T4 x2 allocation is fine), batch size 32, and validates on the RSTPReid validation split. Test evaluation uses each best validation checkpoint.

The dataset root passed to IRRA should contain `RSTPReid/data_captions.json` and `RSTPReid/imgs/`. Adjust `DATA_ROOT` in the notebook if Kaggle mounts this dataset at a different path.

## Manual commands

Baseline:

```bash
python train.py --dataset_name RSTPReid --root_dir /path/to/data --output_dir runs \
  --name irra_baseline --val_dataset val --loss_names sdm+id \
  --sampler identity --num_instance 4 --batch_size 32 --num_workers 2
```

Gallery Context:

```bash
python train.py --dataset_name RSTPReid --root_dir /path/to/data --output_dir runs \
  --name gallery_context --val_dataset val --loss_names sdm+id \
  --sampler identity --num_instance 4 --batch_size 32 --num_workers 2 \
  --gallery_context --gallery_context_depth 2 --gallery_context_heads 8
```

The baseline is the default. Gallery Context requires identity sampling and `batch_size` divisible by `num_instance`. Inference command (replace the path with the timestamped run directory printed by training):

```bash
python test.py --config_file runs/RSTPReid/<run-name>/configs.yaml
```

The saved config restores whether Gallery Context is enabled.

## Source and attribution

The IRRA files originate from the backup supplied for this task and retain their upstream code headers/license notices. See [the upstream IRRA README copy](docs/IRRA_UPSTREAM_README.md). The accompanying license is retained in `LICENSE`.
