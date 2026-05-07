from __future__ import annotations

# ---- runtime limits: MUST be before importing torch/numpy/torchvision ----
import os

def _configure_runtime() -> None:
    # If SLURM provides node-local scratch, use it (fast) for torch IPC files.
    # This avoids the network-filesystem TMPDIR trap.
    slurm_tmp = os.environ.get("SLURM_TMPDIR")
    if slurm_tmp:
        os.environ.setdefault("TMPDIR", slurm_tmp)

    # Hard cap thread pools (main proc + inherited by workers)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("BLIS_NUM_THREADS", "1")

    # Prevent HF tokenizers from spawning threads if you ever touch them
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_configure_runtime()

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
# ---- end runtime limits ----


os.environ.setdefault("HYDRA_FULL_ERROR", "1")

import shutil
from datetime import datetime

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback
from pytorch_lightning.loggers import Logger as PLLogger

# Optional: Hugging Face datasets cache disabling
try:
    from datasets import disable_caching as hf_disable_caching
except Exception:  # pragma: no cover
    hf_disable_caching = None

OmegaConf.register_new_resolver("eval", lambda s: eval(s, {}), replace=True)  # noqa: S307

# Default seed used when the Hydra config does not specify one. Pinned for
# reproducibility — the canonical value also appears in
# ``model/config/datamodule/default.yaml`` (``datamodule.seed``).
RANDOM_SEED: int = 42


# ---------- Utilities ----------

def _ensure_tmpdirs(cfg: DictConfig) -> str:
    for k in ("TMPDIR", "TMP", "TEMP"):
        if os.environ.get(k):
            tmp = os.environ[k]
            break
    else:
        tmp = (
            os.environ.get("SLURM_TMPDIR")
            or str(getattr(cfg.get("runtime", {}), "tmpdir", ""))
            or "/tmp"
        )
        os.makedirs(tmp, exist_ok=True)
        for k in ("TMPDIR", "TMP", "TEMP"):
            os.environ[k] = tmp

    place_hf = bool(getattr(cfg.get("runtime", {}), "place_hf_cache", True))
    if place_hf:
        hf_cache = os.path.join(tmp, "hf_cache")
        os.makedirs(hf_cache, exist_ok=True)
        os.environ.setdefault("HF_HOME", hf_cache)
        os.environ.setdefault("HF_DATASETS_CACHE", hf_cache)
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hf_cache)

    if bool(getattr(cfg.get("runtime", {}), "disable_hf_caching", False)) and hf_disable_caching is not None:
        hf_disable_caching()

    return tmp


def _stage_data_if_needed(cfg: DictConfig) -> str:
    paths = cfg.get("paths", {})
    src = str(getattr(paths, "data_src", "")) or str(getattr(paths, "data", ""))
    dst = str(getattr(paths, "data", "")) or src
    mode = str(getattr(paths, "data_stage_mode", "none")).lower()

    src_abs = os.path.abspath(src)
    dst_abs = os.path.abspath(dst)

    if mode not in {"none", "link", "copy"}:
        raise ValueError(f"Invalid cfg.paths.data_stage_mode='{mode}', expected one of: none, link, copy")

    if mode == "none" or src_abs == dst_abs:
        cfg.paths.data = src_abs
        return src_abs

    os.makedirs(os.path.dirname(dst_abs), exist_ok=True)

    if mode == "link":
        if os.path.islink(dst_abs) or os.path.exists(dst_abs):
            try:
                existing = os.path.realpath(dst_abs)
                if existing != src_abs:
                    raise FileExistsError(
                        f"Destination '{dst_abs}' exists and points to '{existing}', expected '{src_abs}'."
                    )
            except OSError:
                pass
        else:
            os.symlink(src_abs, dst_abs)
        cfg.paths.data = dst_abs
        return dst_abs

    if mode == "copy":
        shutil.copytree(src_abs, dst_abs, dirs_exist_ok=True)
        cfg.paths.data = dst_abs
        return dst_abs

    cfg.paths.data = src_abs
    return src_abs


def _apply_logging_env(cfg: DictConfig) -> None:
    default_base = str(getattr(cfg.paths, "output", "")) or "${DATA_ROOT}/projects/2024_cp_bg_bench_model"
    os.environ.setdefault("WANDB_DIR", f"{default_base}/wandb_runs")
    os.environ.setdefault("WANDB_CACHE_DIR", f"{default_base}/.cache/wandb")
    try:
        os.makedirs(os.environ["WANDB_DIR"], exist_ok=True)
        os.makedirs(os.environ["WANDB_CACHE_DIR"], exist_ok=True)
    except Exception:
        pass

    wb_env_cfg = cfg.logging.get("wandb", {}).get("env", {}) if cfg.get("logging") else {}
    wb_env = OmegaConf.to_container(wb_env_cfg, resolve=True) or {}
    for k, v in wb_env.items():
        if v is not None:
            os.environ[str(k)] = str(v)

    wandb_dir = os.environ.get("WANDB_DIR", "")
    if wandb_dir:
        try:
            os.makedirs(wandb_dir, exist_ok=True)
        except Exception:
            pass


def _build_callbacks(cfg: DictConfig) -> list[Callback]:
    import logging

    callbacks_cfg = cfg.training.get("callbacks", {}) if cfg.get("training") else {}
    callbacks: list[Callback] = []

    disable_disk = bool(cfg.logging.get("disable_local_disk", False))
    allow_ckpt = bool(cfg.logging.get("allow_checkpoints", not disable_disk))

    # Auto-disable val-dependent callbacks when no validation split is requested.
    try:
        val_frac = float(getattr(cfg.datamodule, "val_frac", 0.0))
    except Exception:
        val_frac = 0.0
    skip_val_callbacks = val_frac <= 0.0

    for name, cb_cfg in (callbacks_cfg or {}).items():
        if hasattr(cb_cfg, "enable") and not bool(cb_cfg.enable):
            continue
        if name == "model_checkpoint" and disable_disk and not allow_ckpt:
            continue
        if skip_val_callbacks:
            monitor = getattr(cb_cfg, "monitor", None)
            # Skip callbacks that watch validation metrics when there is no val split.
            if isinstance(monitor, str) and monitor.startswith("val_"):
                logging.getLogger("callbacks").info(
                    f"Skipping callback '{name}' because val_frac=0 disables validation."
                )
                continue
        callbacks.append(hydra.utils.instantiate(cb_cfg))

    return callbacks


def _build_loggers(cfg: DictConfig) -> list[PLLogger]:
    loggers: list[PLLogger] = []

    if bool(cfg.logging.get("enable_wandb", True)):
        wandb_save_dir = None if bool(cfg.logging.get("disable_local_disk", False)) else f"{cfg.paths.output}/wandb_logger"
        # Derive run name before instantiating the logger so W&B picks it up
        # at run creation time (post-hoc assignment doesn't reliably stick).
        run_name = cfg.logging.wandb.name or None
        if not run_name:
            try:
                hcfg = HydraConfig.get()
                exp_choice = hcfg.runtime.choices.get("experiment") or cfg.get("name", "run")
                ts = datetime.now().strftime("%Y%m%d%H%M")
                run_name = f"{exp_choice}_{ts}"
            except Exception:
                pass
        wandb_kwargs = {
            "_target_": "pytorch_lightning.loggers.wandb.WandbLogger",
            "project": cfg.logging.wandb.project,
            "name": run_name,
            "save_dir": wandb_save_dir,
        }
        wandb_logger = hydra.utils.instantiate(wandb_kwargs)
        # Log all resolved config values so overrides are retrievable in W&B.
        try:
            flat = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
            wandb_logger.experiment.config.update({"hydra_cfg": flat}, allow_val_change=True)
        except Exception:
            pass
        loggers.append(wandb_logger)

    if bool(cfg.logging.get("enable_csv", False)) and not bool(cfg.logging.get("disable_local_disk", False)):
        csv_kwargs = {
            "_target_": "pytorch_lightning.loggers.csv_logs.CSVLogger",
            "save_dir": cfg.logging.csv.save_dir,
            "name": cfg.logging.csv.name,
            "prefix": cfg.logging.csv.prefix,
        }
        csv_logger = hydra.utils.instantiate(csv_kwargs)
        loggers.append(csv_logger)

    return loggers


def _log_data_path_to_wandb(data_path: str, loggers: list[PLLogger]) -> None:
    """
    Record the resolved data path into the WandB run config so each job captures
    the exact dataset location used (especially important when staging/linking).
    """
    try:
        from pytorch_lightning.loggers import WandbLogger
    except Exception:
        return

    for lg in loggers or []:
        if isinstance(lg, WandbLogger):
            try:
                lg.experiment.config.update({"data_path": data_path}, allow_val_change=True)
            except Exception:
                pass


def _set_ipc_strategy(cfg: DictConfig) -> None:
    import multiprocessing as mp

    try:
        import torch

        try:
            # Keep IPC off the network filesystem; prefer shared memory/Fds.
            torch.multiprocessing.set_sharing_strategy("file_descriptor")
        except RuntimeError:
            pass
    except Exception:
        pass

    try:
        mp.set_start_method("forkserver", force=True)
    except RuntimeError:
        pass

    default_base = str(getattr(cfg.paths, "output", "")) or "${DATA_ROOT}/projects/2024_cp_bg_bench_model"
    os.environ.setdefault("WANDB_DIR", f"{default_base}/wandb_runs")
    os.environ.setdefault("WANDB_CACHE_DIR", f"{default_base}/.cache/wandb")
    try:
        os.makedirs(os.environ["WANDB_DIR"], exist_ok=True)
        os.makedirs(os.environ["WANDB_CACHE_DIR"], exist_ok=True)
    except Exception:
        pass


def _hydra_output_dir(default: str) -> str:
    try:
        return HydraConfig.get().runtime.output_dir
    except Exception:
        return default


def _best_ckpt_path(trainer) -> str | None:
    try:
        for cb in getattr(trainer, "checkpoint_callbacks", []) or []:
            path = getattr(cb, "best_model_path", "")
            if path:
                return path
    except Exception:
        pass
    return None


def _export_if_enabled(trainer, model, cfg) -> None:
    try:
        from cp_bg_bench_model.models._export import export_image_encoder_with_head
    except Exception:
        return  # exporter not available

    try:
        from cp_bg_bench_model._logging import logger as log
    except Exception:
        import logging

        logging.basicConfig(level=logging.INFO)
        log = logging.getLogger("export")

    # Enabled by default unless explicitly disabled in cfg.export.enable
    export_cfg = getattr(cfg, "export", None)
    if export_cfg is not None and hasattr(export_cfg, "enable") and not bool(export_cfg.enable):
        return
    if hasattr(trainer, "is_global_zero") and not bool(trainer.is_global_zero):
        return

    out_dir = None
    if export_cfg and getattr(export_cfg, "out_dir", None):
        out_dir = str(export_cfg.out_dir)
    if not out_dir:
        out_dir = f"{_hydra_output_dir(getattr(cfg.paths, 'output', '.'))}/exports"

    file_stem = "imgenc_finetuned"
    if export_cfg and getattr(export_cfg, "file_stem", None):
        file_stem = str(export_cfg.file_stem)

    merge_lora = bool(getattr(export_cfg, "merge_lora", False)) if export_cfg else False
    source = str(getattr(export_cfg, "source", "in_memory")) if export_cfg else "in_memory"

    lit_for_export = model

    if source == "best":
        best = _best_ckpt_path(trainer)
        if best:
            try:
                lit_for_export = type(model).load_from_checkpoint(best)  # may fail if constructor args mismatch
                log.info(f"Loaded best checkpoint for export: {best}")
            except Exception as e:
                log.info(f"Falling back to in-memory model for export. load_from_checkpoint failed: {e}")
        else:
            log.info("No best checkpoint found. Using in-memory model for export.")

    paths = export_image_encoder_with_head(
        lit_model=lit_for_export,
        out_dir=out_dir,
        file_stem=file_stem,
        merge_lora=merge_lora,
    )
    log.info(f"Exported image encoder artifacts={paths}")


# ---------- Training pipeline ----------

def train(cfg: DictConfig) -> None:
    import pytorch_lightning as pl
    import torch
    from pytorch_lightning import LightningDataModule, LightningModule, Trainer

    try:
        from cp_bg_bench_model._logging import logger as log
    except Exception:
        import logging

        logging.basicConfig(level=logging.INFO)
        log = logging.getLogger("train")

    torch.set_float32_matmul_precision("medium")

    seed = cfg.datamodule.seed if cfg.get("datamodule") and cfg.datamodule.get("seed") else RANDOM_SEED
    pl.seed_everything(seed, workers=True)

    data_dir = _stage_data_if_needed(cfg)
    log.info(f"Using dataset at '{data_dir}' (datamode={getattr(cfg.paths, 'data_stage_mode', 'none')})")

    log.info("Instantiating datamodule...")
    dm: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    # ---- infer precomputed dim if needed ----
    precomputed_in_dim = None
    try:
        mol_name = str(
            cfg.model.litmodule.get("perturbation_encoder_name")
            or cfg.model.litmodule.get("molecule_encoder_name")
            or ""
        ).lower()
        if mol_name == "precomputed":
            col = str(
                cfg.model.litmodule.get("perturbation_embedding_column")
                or cfg.model.litmodule.get("molecule_embedding_column")
                or ""
            )
            if not col:
                raise RuntimeError("perturbation_encoder_name=precomputed but no perturbation_embedding_column provided.")
            ds = getattr(dm, "dataset", None)
            if ds is None:
                raise RuntimeError("Datamodule does not expose `.dataset` for dim inference.")
            sample = ds[col][0]
            if isinstance(sample, torch.Tensor):
                precomputed_in_dim = int(sample.numel() if sample.ndim == 1 else sample.shape[-1])
            else:
                precomputed_in_dim = int(len(sample))
    except Exception as e:
        raise RuntimeError(
            f"Failed to infer precomputed fingerprint dimension for column "
            f"'{cfg.model.litmodule.get('perturbation_embedding_column') or cfg.model.litmodule.get('molecule_embedding_column') or '?'}'."
        ) from e

    log.info("Instantiating model...")
    model_cls_or_instance = hydra.utils.instantiate(
        cfg.model.litmodule,
        _convert_="none",
        precomputed_in_dim=precomputed_in_dim,
    )

    # The LR schedule is built inside `PretrainModule.configure_optimizers`
    # from `trainer.estimated_stepping_batches`, so no scheduler is passed in.
    if isinstance(model_cls_or_instance, LightningModule):
        model: LightningModule = model_cls_or_instance
        if hasattr(model, "set_cfg"):
            model.set_cfg(cfg)
    else:
        model: LightningModule = model_cls_or_instance(cfg=cfg)

    _apply_logging_env(cfg)
    callbacks = _build_callbacks(cfg)
    loggers = _build_loggers(cfg)
    _log_data_path_to_wandb(data_dir, loggers)
    default_root_dir = None if bool(cfg.logging.get("disable_local_disk", False)) else cfg.paths.output

    log.info("Instantiating trainer...")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.training.lightning.trainer,
        callbacks=callbacks,
        logger=loggers if loggers else False,
        default_root_dir=default_root_dir,
    )

    if cfg.training.get("train", True):
        log.info("Training...")
        trainer.fit(model, datamodule=dm)
        _export_if_enabled(trainer, model, cfg)

    if cfg.training.get("test", False):
        # Skip test phase cleanly if datamodule exposes no test data (e.g., test_frac=0).
        test_loaders = dm.test_dataloader() if hasattr(dm, "test_dataloader") else None
        has_tests = test_loaders not in (None, [], {})  # empty list/None means no tests
        if has_tests:
            log.info("Testing...")
            try:
                trainer.test(model, datamodule=dm, ckpt_path="best")
            except Exception:
                trainer.test(model, datamodule=dm)
        else:
            log.info("Skipping test phase (no test dataset).")

    if cfg.training.get("predict", False):
        log.info("Predicting...")
        try:
            trainer.predict(model, datamodule=dm, ckpt_path="best")
        except Exception:
            trainer.predict(model, datamodule=dm)


@hydra.main(config_path="../config", config_name="default.yaml", version_base="1.2")
def main(cfg: DictConfig) -> None:
    _ensure_tmpdirs(cfg)
    _set_ipc_strategy(cfg)
    try:
        sweep_params = HydraConfig.get().job.override_dirname or ""
    except Exception:
        sweep_params = ""
    try:
        if sweep_params:
            cfg.paths.output = f"{cfg.paths.output}/{sweep_params}"
    except Exception:
        pass
    train(cfg)


if __name__ == "__main__":
    main()
