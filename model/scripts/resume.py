from __future__ import annotations

import os

import hydra
from datasets import disable_caching
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

disable_caching()

OmegaConf.register_new_resolver("eval", eval)

# Default seed used when the Hydra config does not specify one. Pinned for
# reproducibility — matches ``model/config/datamodule/default.yaml``.
RANDOM_SEED: int = 42


def train(cfg: DictConfig) -> None:
    import pytorch_lightning as pl
    import torch
    from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
    from pytorch_lightning.loggers import Logger

    torch.set_float32_matmul_precision("medium")

    from cp_bg_bench_model._logging import logger as cp_bg_bench_model_logger

    seed = cfg.datamodule.seed if cfg.datamodule.get("seed") else RANDOM_SEED
    pl.seed_everything(seed, workers=True)
    
    cp_bg_bench_model_logger.info("Instantiating datamodule...")
    dm: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    cp_bg_bench_model_logger.info("Instantiating model...")
    lr_lambda = hydra.utils.instantiate(cfg.model.custom_scheduler)
    model: LightningModule = hydra.utils.instantiate(cfg.model.litmodule)
    model = model(lambda_scheduler=lr_lambda, cfg=cfg)

    # Check if resume_from_checkpoint is provided in config
    checkpoint_path = '${HOME}/projs/cp_bg_bench_model/experiments/results/cropped/openphenom-ecfp/checkpoints/last.ckpt'


    cp_bg_bench_model_logger.info("Instantiating callbacks and logger...")
    callbacks: list[Callback] = []
    for _, cb in cfg.training.callbacks.items():
        callbacks.append(hydra.utils.instantiate(cb))

    logger: list[Logger] = []
    for _, lg in cfg.training.logger.items():
        if lg._target_ == "pytorch_lightning.loggers.WandbLogger":
            cp_bg_bench_model_logger.info("Setting up WandB logger for resuming...")
            lg.id = cfg.training.logger.wandb.id  # Ensure WandB ID is provided
            lg.resume = "allow"  # Enable resuming
        logger.append(hydra.utils.instantiate(lg))

    cp_bg_bench_model_logger.info("Instantiating trainer for training...")
    trainer: Trainer = hydra.utils.instantiate(cfg.training.lightning.trainer)
    
    # Set resume_from_checkpoint to resume training from the checkpoint
    trainer = trainer(callbacks=callbacks, logger=logger,)

    if cfg.training.train:
        cp_bg_bench_model_logger.info("Resuming Training...")
        trainer.fit(model, datamodule=dm, ckpt_path=checkpoint_path)

    cp_bg_bench_model_logger.info("Training finished, moving to testing...")
    if cfg.training.test:
        trainer.test(model, datamodule=dm)

    # if cfg.training.predict:
    #     trainer.predict(model, datamodule=dm)


@hydra.main(config_path="./../config", config_name="default.yaml", version_base="1.2")
def main(cfg: DictConfig) -> None:
    import os
    tmp_dir = '/localscratch/ghaith/temp'
    os.makedirs(tmp_dir, exist_ok=True)
    os.environ["TMPDIR"] = tmp_dir

    sweep_params = HydraConfig.get().job.override_dirname
    cfg.paths.output = f"{cfg.paths.output}/{sweep_params}"
    cfg.training.logger.wandb.name = sweep_params
    os.makedirs(cfg.training.logger.wandb.save_dir, exist_ok=True)
    os.makedirs(cfg.training.logger.csv.save_dir, exist_ok=True)
    
    train(cfg)


if __name__ == "__main__":
    main()
