"""Unified single-cell embedding + well-level aggregation pipeline.

Combines ``embed_to_anndata.py`` and ``aggregate_embeddings.py`` into one
invocation. When given ``--run-dir``, all parameters are derived
automatically from the training run's saved Hydra config.

Usage::

    # Auto-derive everything from a training run directory
    pixi run model-infer --run-dir ${DATA_ROOT}

    # Override specific params
    pixi run model-infer --run-dir <dir> --fraction 0.5 --batch-size 128

    # Standalone (no training run)
    pixi run model-infer \\
        --checkpoint model.pt --dataset /data/crops \\
        --config-id 33_Rxrx3C_SubCell_ESM2_C --output-dir /out

    # Re-aggregate only (skip embedding)
    pixi run model-infer --run-dir <dir> --skip-embed

    # Embed only (skip aggregation)
    pixi run model-infer --run-dir <dir> --skip-aggregate
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

# Sibling scripts are not a package — make them importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# Default seed propagated to embed_to_anndata (stratified sampling) and
# downstream to aggregate_embeddings (Harmony). The CLI ``--seed`` flag still
# allows ad-hoc overrides.
RANDOM_SEED: int = 42


# ── Config resolution ────────────────────────────────────────────────────────


@dataclasses.dataclass
class RunConfig:
    """Parameters derived from a training run's ``.hydra/`` directory."""

    config_id: str
    checkpoint: Path
    dataset_path: Path
    channels: int
    image_size: int
    model_image_size: int
    control_col: str
    control_value: str
    group_cols: list[str]


def _resolve_interpolation(raw: str, paths: dict[str, str]) -> str:
    """Resolve ``${paths.root}`` and ``${paths.data}`` in a raw YAML string."""
    result = raw
    for key, val in paths.items():
        result = result.replace(f"${{paths.{key}}}", val)
    return result


def load_run_config(run_dir: Path) -> RunConfig:
    """Read ``.hydra/config.yaml`` and ``.hydra/hydra.yaml`` from *run_dir*."""
    import yaml

    hydra_dir = run_dir / ".hydra"
    if not hydra_dir.is_dir():
        raise FileNotFoundError(f"No .hydra/ directory in {run_dir}. Is this a valid training run directory?")

    with open(hydra_dir / "config.yaml") as f:
        config: dict = yaml.safe_load(f)

    with open(hydra_dir / "hydra.yaml") as f:
        hydra_meta = yaml.safe_load(f)

    choices = hydra_meta["hydra"]["runtime"]["choices"]
    config_id = choices.get("experiment")
    if not config_id:
        raise ValueError("Could not determine experiment name from hydra.yaml")

    export_stem = run_dir / "exports" / "imgenc_finetuned"
    checkpoint = export_stem.with_suffix(".pth")
    if not checkpoint.is_file():
        checkpoint = export_stem.with_suffix(".pt")
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"No encoder export found at {export_stem} "
            f"(checked .pth and .pt)"
        )

    # Saved config.yaml may contain unresolved ``${paths.root}`` in
    # ``paths.data``.  Resolve it manually from the concrete paths.root.
    paths_raw = config.get("paths", {})
    paths_resolved = {k: str(v) for k, v in paths_raw.items() if "${" not in str(v)}
    data_raw = str(paths_raw.get("data", ""))
    dataset_path = Path(_resolve_interpolation(data_raw, paths_resolved))

    channel_names = config.get("datamodule", {}).get("channel_names")
    channels = len(channel_names) if channel_names else config["model"]["litmodule"]["in_channels"]

    dm = config.get("datamodule", {})
    model_image_size: int = config["model"]["litmodule"]["image_size"]
    image_size: int = dm.get("data_image_size") or model_image_size

    control_keys = config.get("datamodule", {}).get("control_perturbation_keys")
    if control_keys:
        control_value = control_keys[0]
    else:
        atg = config.get("datamodule", {}).get("always_train_groups", [])
        control_value = atg[0] if atg else "EMPTY"

    return RunConfig(
        config_id=config_id,
        checkpoint=checkpoint,
        dataset_path=dataset_path,
        channels=channels,
        image_size=image_size,
        model_image_size=model_image_size,
        control_col="perturbation",
        control_value=control_value,
        group_cols=["source", "plate", "well"],
    )


# ── Argument resolution ──────────────────────────────────────────────────────


@dataclasses.dataclass
class ResolvedParams:
    """Fully resolved parameters for the embed + aggregate pipeline."""

    config_id: str
    checkpoint: Path
    dataset_path: Path
    output_dir: Path
    # embed
    fraction: float
    stratify_col: str
    channels: int
    image_size: int
    model_image_size: int
    batch_size: int
    device: str
    seed: int
    image_col: str
    # aggregate
    group_cols: list[str]
    control_col: str
    control_value: str
    n_pcs: int
    # pipeline control
    skip_embed: bool
    skip_aggregate: bool


def resolve_args(args: argparse.Namespace) -> ResolvedParams:
    """Merge CLI arguments with :class:`RunConfig` (if ``--run-dir`` given)."""
    run_cfg: RunConfig | None = None
    if args.run_dir is not None:
        run_cfg = load_run_config(args.run_dir)

    def _pick(cli_val, run_val, default):
        if cli_val is not None:
            return cli_val
        if run_val is not None:
            return run_val
        return default

    config_id = _pick(args.config_id, run_cfg and run_cfg.config_id, None)
    if not config_id:
        raise ValueError("--config-id is required when --run-dir is not provided")

    checkpoint = _pick(args.checkpoint, run_cfg and run_cfg.checkpoint, None)
    if not checkpoint:
        raise ValueError("--checkpoint is required when --run-dir is not provided")
    checkpoint = Path(checkpoint)

    dataset_cli = args.dataset
    dataset_derived = run_cfg.dataset_path if run_cfg else None
    if dataset_cli is not None:
        dataset_path = Path(dataset_cli)
        if not dataset_path.is_dir():
            raise FileNotFoundError(f"--dataset path does not exist: {dataset_path}")
    elif dataset_derived is not None:
        if not dataset_derived.is_dir():
            raise FileNotFoundError(
                f"Dataset path from training config does not exist: {dataset_derived}\n"
                f"The data may have moved. Re-run with --dataset <correct_path> to override."
            )
        dataset_path = dataset_derived
    else:
        raise ValueError("--dataset is required when --run-dir is not provided")

    if args.output_dir is not None:
        output_dir = args.output_dir
    elif args.run_dir is not None:
        output_dir = args.run_dir / "inference"
    else:
        raise ValueError("--output-dir is required when --run-dir is not provided")

    channels = _pick(args.channels, run_cfg and run_cfg.channels, 6)
    image_size = _pick(args.image_size, run_cfg and run_cfg.image_size, 224)
    model_image_size = _pick(
        args.model_image_size, run_cfg and run_cfg.model_image_size, image_size
    )
    control_col = _pick(args.control_col, run_cfg and run_cfg.control_col, "perturbation")
    control_value = _pick(args.control_value, run_cfg and run_cfg.control_value, "EMPTY")

    if args.group_cols is not None:
        group_cols = args.group_cols.split(",")
    elif run_cfg is not None:
        group_cols = run_cfg.group_cols
    else:
        group_cols = ["source", "plate", "well"]

    return ResolvedParams(
        config_id=config_id,
        checkpoint=checkpoint,
        dataset_path=dataset_path,
        output_dir=output_dir,
        fraction=args.fraction if args.fraction is not None else 1.0,
        stratify_col=args.stratify_col or "plate",
        channels=channels,
        image_size=image_size,
        model_image_size=model_image_size,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        image_col=args.image_col,
        group_cols=group_cols,
        control_col=control_col,
        control_value=control_value,
        n_pcs=args.n_pcs,
        skip_embed=args.skip_embed,
        skip_aggregate=args.skip_aggregate,
    )


# ── Pipeline orchestration ───────────────────────────────────────────────────


def run_inference(params: ResolvedParams) -> tuple[Path, Path | None]:
    """Run embed + aggregate.  Returns *(singlecell_path, aggregated_path)*."""
    from aggregate_embeddings import aggregate_embeddings
    from embed_to_anndata import embed_to_anndata

    params.output_dir.mkdir(parents=True, exist_ok=True)

    singlecell_path = params.output_dir / f"{params.config_id}_singlecell.h5ad"
    aggregated_path: Path | None = params.output_dir / f"{params.config_id}_aggregated.h5ad"

    # ── Step 1: Embed ────────────────────────────────────────────────────────
    if not params.skip_embed:
        log.info("=" * 60)
        log.info("Stage 1/2: Single-cell embedding")
        log.info("=" * 60)
        singlecell_path = embed_to_anndata(
            config_id=params.config_id,
            checkpoint=params.checkpoint,
            dataset_path=params.dataset_path,
            output_dir=params.output_dir,
            fraction=params.fraction,
            stratify_col=params.stratify_col,
            channels=params.channels,
            image_size=params.image_size,
            model_image_size=params.model_image_size,
            batch_size=params.batch_size,
            device=params.device,
            seed=params.seed,
            image_col=params.image_col,
        )
    else:
        log.info("Skipping embed step (--skip-embed)")
        if not singlecell_path.is_file():
            raise FileNotFoundError(f"--skip-embed but singlecell file not found: {singlecell_path}")

    # ── Step 2: Aggregate ────────────────────────────────────────────────────
    if not params.skip_aggregate:
        log.info("=" * 60)
        log.info("Stage 2/2: Well-level aggregation + batch correction")
        log.info("=" * 60)
        aggregated_path = aggregate_embeddings(
            input_path=singlecell_path,
            output_dir=params.output_dir,
            group_cols=params.group_cols,
            control_col=params.control_col,
            control_value=params.control_value,
            n_pcs=params.n_pcs,
        )
    else:
        log.info("Skipping aggregate step (--skip-aggregate)")
        aggregated_path = None

    # ── Summary ──────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Done.")
    log.info(f"  Single-cell : {singlecell_path}")
    if aggregated_path:
        log.info(f"  Aggregated  : {aggregated_path}")
    log.info("=" * 60)

    return singlecell_path, aggregated_path


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run single-cell embedding + well-level aggregation pipeline.",
    )

    src = p.add_argument_group("run-dir mode (derives all params from training run)")
    src.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Training run directory containing .hydra/ and exports/.",
    )

    standalone = p.add_argument_group("standalone mode (required without --run-dir)")
    standalone.add_argument("--checkpoint", type=Path, default=None)
    standalone.add_argument("--dataset", type=Path, default=None)
    standalone.add_argument("--config-id", default=None)

    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <run-dir>/inference/).",
    )

    embed = p.add_argument_group("embed overrides")
    embed.add_argument("--fraction", type=float, default=None)
    embed.add_argument("--stratify-col", default=None)
    embed.add_argument("--channels", type=int, default=None)
    embed.add_argument("--image-size", type=int, default=None,
                       help="On-disk crop size (default: from config or 224)")
    embed.add_argument("--model-image-size", type=int, default=None,
                       help="Model input size; resize if different from --image-size")
    embed.add_argument("--batch-size", type=int, default=256)
    embed.add_argument("--device", default="cuda")
    embed.add_argument("--seed", type=int, default=RANDOM_SEED)
    embed.add_argument("--image-col", default="cell")

    agg = p.add_argument_group("aggregate overrides")
    agg.add_argument("--group-cols", default=None, help="Comma-separated.")
    agg.add_argument("--control-col", default=None)
    agg.add_argument("--control-value", default=None)
    agg.add_argument("--n-pcs", type=int, default=50)

    pipeline = p.add_argument_group("pipeline control")
    pipeline.add_argument(
        "--skip-embed",
        action="store_true",
        help="Skip embedding; use existing singlecell.h5ad.",
    )
    pipeline.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Skip aggregation; produce only singlecell.h5ad.",
    )

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    params = resolve_args(args)

    log.info(f"config_id     : {params.config_id}")
    log.info(f"checkpoint    : {params.checkpoint}")
    log.info(f"dataset       : {params.dataset_path}")
    log.info(f"output_dir    : {params.output_dir}")
    log.info(f"channels      : {params.channels}")
    log.info(f"image_size    : {params.image_size} (disk) → {params.model_image_size} (model)")
    log.info(f"control       : {params.control_col} == {params.control_value}")
    log.info(f"group_cols    : {params.group_cols}")

    run_inference(params)


if __name__ == "__main__":
    main()
