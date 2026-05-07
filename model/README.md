# model

Image–perturbation contrastive fine-tuning for cp-bg-bench.

## Layout

```
model/
  pyproject.toml         # package + ruff + pytest config
  pixi.toml              # conda + pytorch env, training tasks
  src/cp_bg_bench_model/ # importable package
  tests/                 # pytest suite
  config/                # Hydra configs (datamodule / model / training / paths / launcher / experiment)
  scripts/               # train / resume / inference / aggregate entry points
```

## Quickstart

```bash
cd model
pixi install
pixi run check              # ruff + pytest
pixi run model-train-fast   # 1-step Hydra fast_dev_run (requires data)
pixi run model-train        # full pretrain
```

Hydra entry point: `scripts/train.py` → `config/default.yaml`. Pick an
experiment via:

```bash
pixi run model-train experiment=01_JUMP_DINO_ECFP4_C
```

## Inference

```python
from cp_bg_bench_model import Cp_bg_benchModelPredictor, save_checkpoint

save_checkpoint(model, "predictor.pt")  # at end of training

predictor = Cp_bg_benchModelPredictor.load("predictor.pt", device="cuda")
embs = predictor.predict_batch(crops_uint8)  # (B, D) float32 numpy, L2-normed
```

`save_checkpoint` merges LoRA adapters by default and strips Lightning /
PEFT wrappers, so the resulting `.pt` is self-contained and PEFT-free at
load time.
