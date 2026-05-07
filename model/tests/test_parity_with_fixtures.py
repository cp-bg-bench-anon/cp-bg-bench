"""Regression-lock parity tests against snapshot fixtures.

Each fixture under ``tests/fixtures/parity/*.pt`` records a CLIPModel
config + deterministic inputs + the embeddings produced at fixture-
generation time. Rebuilding the model with the same seed and forwarding
the same inputs must reproduce those embeddings bit-for-bit (within
float32 tolerance). Drift here means a refactor changed behaviour —
either revert or regenerate via ``tests/_generate_fixtures.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from cp_bg_bench_model._constants import DatasetEnum
from cp_bg_bench_model.models import CLIPModel

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "parity"
FIXTURES = sorted(FIXTURE_DIR.glob("*.pt"))


@pytest.mark.parametrize("fixture_path", FIXTURES, ids=lambda p: p.stem)
def test_forward_matches_fixture(fixture_path: Path) -> None:
    snap = torch.load(fixture_path, weights_only=True)

    torch.manual_seed(0)
    model = CLIPModel(**snap["config"]).eval()

    with torch.inference_mode():
        img_out, mol_out = model({DatasetEnum.IMG: snap["image_input"], DatasetEnum.PERTURBATION: snap["mol_input"]})

    assert img_out.shape == snap["image_embed"].shape
    assert mol_out.shape == snap["mol_embed"].shape
    assert torch.allclose(img_out, snap["image_embed"], atol=1e-5, rtol=1e-5), (
        f"Image embedding drift in {fixture_path.name}: "
        f"max abs diff {(img_out - snap['image_embed']).abs().max().item():.3e}"
    )
    assert torch.allclose(mol_out, snap["mol_embed"], atol=1e-5, rtol=1e-5), (
        f"Molecule embedding drift in {fixture_path.name}: "
        f"max abs diff {(mol_out - snap['mol_embed']).abs().max().item():.3e}"
    )
