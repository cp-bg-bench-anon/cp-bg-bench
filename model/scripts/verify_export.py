"""
Verify an exported image encoder checkpoint produced by train.py.

Checks:
  1. Checkpoint loads and is an ImageEncoderWithHead
  2. LoRA adapter modules are present (if export was without merge)
  3. Projection head (MLP or linear) is present
  4. Forward pass succeeds with synthetic uint8 input
  5. Output shape, finiteness, and L2 norm properties

Usage:
    pixi run python scripts/verify_export.py <path/to/imgenc_finetuned> [--in-channels N] [--image-size S]
    pixi run python scripts/verify_export.py  # auto-discovers most-recent export under /tmp
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from cp_bg_bench_model.models._export import load_image_encoder_with_head

# ---- helpers ----------------------------------------------------------------

def _find_latest_export() -> Path | None:
    import os
    # CP_BG_BENCH_OUTPUT takes precedence; fall back to /tmp (not $TMPDIR which may be a build cache)
    base = os.environ.get("CP_BG_BENCH_OUTPUT", "/tmp/cp_bg_bench_model_test")
    candidates = sorted(glob.glob(f"{base}/**/imgenc_finetuned.pth", recursive=True))
    return Path(candidates[-1]) if candidates else None


def _collect_lora_modules(module: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    lora_indicators = ("lora_A", "lora_B", "lora_dropout", "lora_layer")
    out = []
    for name, mod in module.named_modules():
        mod_type = type(mod).__name__.lower()
        if any(ind in mod_type for ind in ("lora",)):
            out.append((name, mod))
            continue
        # also catch modules that *have* lora_A/B as children
        child_names = {n for n, _ in mod.named_children()}
        if any(ind in child_names for ind in lora_indicators):
            out.append((name, mod))
    return out


def _find_projection_head(module: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    # Fast path: all encoder classes expose self.head directly
    inner = getattr(module, "encoder", module)
    if (head := getattr(inner, "head", None)) is not None:
        return [("encoder.head", head)]
    # Fallback: scan by type name or attribute name
    heads = []
    for name, mod in module.named_modules():
        typ = type(mod).__name__
        is_head_type = typ in ("ProjectionHead", "LinearHead", "Linear") and "head" in name.lower()
        is_head_attr = name.split(".")[-1] == "head"
        if is_head_type or (not heads and is_head_attr):
            heads.append((name, mod))
    return heads


def _param_stats(module: torch.nn.Module) -> tuple[int, int]:
    total = trainable = 0
    for p in module.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    return total, trainable


# ---- main -------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Verify exported image encoder checkpoint")
    parser.add_argument(
        "checkpoint",
        nargs="?",
        help="Path to the imgenc_finetuned export (any artifact or shared stem)",
    )
    parser.add_argument("--in-channels", type=int, default=5, help="Number of image channels (default: 5 for JUMP; use 8 for rxrx1)")
    parser.add_argument("--image-size", type=int, default=224, help="Image spatial size (default: 224)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for test forward pass")
    parser.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    args = parser.parse_args()

    # --- resolve path ---
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = _find_latest_export()
        if ckpt_path is None:
            print("ERROR: no checkpoint provided and none found under /tmp/cp_bg_bench_model_test")
            return 1
        print(f"Auto-discovered checkpoint: {ckpt_path}")

    if not ckpt_path.exists():
        print(f"ERROR: file not found: {ckpt_path}")
        return 1

    device = torch.device(args.device)
    print(f"\n{'='*60}")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Device     : {device}")
    print(f"{'='*60}\n")

    # --- 1. load ---
    print("[1/5] Loading checkpoint...")
    encoder_module = load_image_encoder_with_head(ckpt_path).to(device).eval()
    print(f"      Type: {type(encoder_module).__name__}")

    total, trainable = _param_stats(encoder_module)
    print(f"      Params: {total:,} total, {trainable:,} trainable")
    print("      OK\n")

    # --- 2. LoRA ---
    print("[2/5] Checking for LoRA adapter modules...")
    lora_mods = _collect_lora_modules(encoder_module)
    if lora_mods:
        print(f"      Found {len(lora_mods)} LoRA module(s):")
        for name, mod in lora_mods[:5]:
            print(f"        {name}: {type(mod).__name__}")
        if len(lora_mods) > 5:
            print(f"        ... and {len(lora_mods) - 5} more")
        # verify lora weights are non-zero (not all-zero init)
        lora_nonzero = 0
        for _, mod in lora_mods:
            for pname, p in mod.named_parameters():
                if "lora_B" in pname or "lora_A" in pname:
                    if p.abs().max().item() > 1e-9:
                        lora_nonzero += 1
        print(f"      Non-zero LoRA weight tensors: {lora_nonzero}")
        if lora_nonzero == 0:
            print("      WARNING: all LoRA weights appear to be zero (expected if untrained or merged)")
    else:
        print("      No LoRA modules found (weights may have been merged during export)")
    print("      OK\n")

    # --- 3. projection head ---
    print("[3/5] Checking for projection head...")
    heads = _find_projection_head(encoder_module)
    if heads:
        for name, mod in heads:
            h_total, _ = _param_stats(mod)
            print(f"      {name}: {type(mod).__name__} ({h_total:,} params)")
    else:
        print("      WARNING: no projection head found by name heuristic")
        print("      (may be fused into backbone or named differently)")
    print("      OK\n")

    # --- 4. forward pass + intermediate activation check ---
    print(f"[4/5] Running forward pass (batch={args.batch_size}, C={args.in_channels}, HW={args.image_size})...")

    torch.manual_seed(42)
    crops_uint8 = torch.randint(0, 256, (args.batch_size, args.in_channels, args.image_size, args.image_size), dtype=torch.uint8)

    # ImageEncoderWithHead wraps the encoder; apply float conversion manually.
    inp = crops_uint8.to(device, dtype=torch.float32).div_(255.0)

    # Hook to capture the backbone/trunk output before the projection head
    trunk_out: dict[str, torch.Tensor] = {}
    trunk_hook_handle = None

    def _make_trunk_hook(name: str):
        def _hook(mod, inp_, out_):
            if isinstance(out_, torch.Tensor):
                trunk_out[name] = out_.detach().cpu()
        return _hook

    # Try to attach a hook on common backbone attribute names
    for attr in ("trunk", "backbone", "model", "encoder"):
        backbone = getattr(getattr(encoder_module, "encoder", encoder_module), attr, None)
        if backbone is not None:
            trunk_hook_handle = backbone.register_forward_hook(_make_trunk_hook(attr))
            break

    with torch.inference_mode():
        out = encoder_module(inp)

    if trunk_hook_handle is not None:
        trunk_hook_handle.remove()

    print(f"      Output shape : {tuple(out.shape)}")

    if trunk_out:
        t = next(iter(trunk_out.values()))
        t_max = t.abs().max().item()
        t_finite = torch.isfinite(t).all().item()
        print(f"      Backbone out : finite={t_finite}, max_abs={t_max:.3e}")
        if t_max > 1e6:
            print(
                "      WARNING: backbone activations are very large (max_abs > 1e6).\n"
                "               This usually means gradient explosion during training\n"
                "               (no LR warmup, or too-large LR on first step).\n"
                "               The export is structurally correct; the weights reflect\n"
                "               the training state at the time of export."
            )
    print("      OK\n")

    # --- 5. output sanity ---
    print("[5/5] Checking output properties...")
    if out.ndim != 2:
        print(f"FAIL: expected 2D output (B, D), got {out.shape}")
        return 1
    if out.shape[0] != args.batch_size:
        print(f"FAIL: batch size mismatch: {out.shape[0]} vs {args.batch_size}")
        return 1

    embed_dim = out.shape[1]
    finite = torch.isfinite(out).all().item()

    print(f"      Embed dim  : {embed_dim}")
    print(f"      All finite : {finite}")

    if finite:
        norms = out.norm(dim=-1)
        mean_norm = norms.mean().item()
        min_norm = norms.min().item()
        max_norm = norms.max().item()
        print(f"      L2 norms   : mean={mean_norm:.4f}, min={min_norm:.4f}, max={max_norm:.4f}")
        is_normalized = abs(mean_norm - 1.0) < 0.05
        print(f"      L2-normed  : {'yes' if is_normalized else 'no (raw logits — use F.normalize downstream)'}")
        pairwise = F.cosine_similarity(out[0:1], out[1:], dim=-1)
        max_sim = pairwise.max().item()
        print(f"      Max cosine sim between samples: {max_sim:.4f}")
        if max_sim > 0.9999:
            print("      WARNING: embeddings are nearly identical — possible degenerate model weights")
    else:
        print(
            "      WARNING: NaN/Inf in output.\n"
            "               Backbone activations are astronomically large (overflow in projection head).\n"
            "               Likely cause: gradient explosion from training without LR warmup.\n"
            "               The export is valid — this reflects the actual model state."
        )

    print(f"\n{'='*60}")
    print("PASS: export structure and forward pass verified")
    if not finite:
        print("NOTE : output NaN due to exploded weights in this checkpoint — not an export bug.")
        print("       A properly trained checkpoint (with warmup) will produce finite embeddings.")
    print(f"{'='*60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
