"""Regenerate the regression fixtures under ``tests/fixtures/parity/``.

Run with ``pixi run python tests/_generate_fixtures.py`` after intentional
behavioural changes (encoder rewrite, n-channel adapter tweak, etc.). The
fixtures lock current outputs so future refactors that drift get caught
by ``test_parity_with_fixtures``.

After regenerating, commit the updated ``.pt`` files alongside the code
change so CI has the new expected values.  If you don't regenerate,
``test_parity_with_fixtures`` will fail on the next run.

Each fixture stores the exact construction config + input + expected
embeddings. Determinism: ``torch.manual_seed(0)`` before every model
build *and* every input draw. CPU only — GPU non-determinism would
defeat the lock.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from cp_bg_bench_model._constants import DatasetEnum
from cp_bg_bench_model.models import CLIPModel

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "parity"


# --------------------------------------------------------------------------- #
# Combos                                                                      #
# --------------------------------------------------------------------------- #
# Each combo: (name, model_kwargs, input_factory). Inputs are deterministic.

COMBOS: list[dict[str, Any]] = [
    {
        "name": "densenet_5ch_precomputed",
        "model": {
            "embed_dim": 128,
            "image_encoder_name": "densenet",
            "perturbation_encoder_name": "precomputed",
            "loss": "CLIP",
            "image_size": 64,
            "in_channels": 5,
            "precomputed_in_dim": 64,
            "freeze_backbone_when_no_lora": False,
        },
        "image_shape": (4, 5, 64, 64),
        "mol_shape": (4, 64),
    },
    {
        "name": "densenet_6ch_precomputed",
        "model": {
            "embed_dim": 128,
            "image_encoder_name": "densenet",
            "perturbation_encoder_name": "precomputed",
            "loss": "CLIP",
            "image_size": 64,
            "in_channels": 6,
            "precomputed_in_dim": 64,
            "freeze_backbone_when_no_lora": False,
        },
        "image_shape": (4, 6, 64, 64),
        "mol_shape": (4, 64),
    },
    {
        "name": "densenet_5ch_ecfp",
        "model": {
            "embed_dim": 128,
            "image_encoder_name": "densenet",
            "perturbation_encoder_name": "ecfp",
            "loss": "CLIP",
            "image_size": 64,
            "in_channels": 5,
            "freeze_backbone_when_no_lora": False,
        },
        "image_shape": (4, 5, 64, 64),
        "mol_shape": (4, 2048),  # ECFP encoder via dense path
    },
]


def _build_and_run(combo: dict[str, Any]) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    model = CLIPModel(**combo["model"]).eval()

    torch.manual_seed(0)
    img = torch.randint(0, 256, combo["image_shape"], dtype=torch.uint8)
    mol = torch.randn(combo["mol_shape"], dtype=torch.float32)

    with torch.inference_mode():
        img_embed, mol_embed = model({DatasetEnum.IMG: img, DatasetEnum.PERTURBATION: mol})

    return {
        "config": combo["model"],
        "image_input": img,
        "mol_input": mol,
        "image_embed": img_embed.detach().clone(),
        "mol_embed": mol_embed.detach().clone(),
    }


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for combo in COMBOS:
        out = FIXTURE_DIR / f"{combo['name']}.pt"
        snapshot = _build_and_run(combo)
        torch.save(snapshot, out)
        print(f"wrote {out.name}  img={tuple(snapshot['image_embed'].shape)}  mol={tuple(snapshot['mol_embed'].shape)}")


if __name__ == "__main__":
    main()
