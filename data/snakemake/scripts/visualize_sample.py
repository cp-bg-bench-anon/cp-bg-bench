"""Rule N driver: sample 3 cells from each resharded dataset variant and plot."""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_from_disk
from lamin_utils import logger

from cp_bg_bench.config import JumpConfig, Rxrx1Config, Rxrx3CoreConfig
from cp_bg_bench.config import load as load_config
from cp_bg_bench.crops.extract import N_MASK_CHANNELS
from cp_bg_bench.runtime import dump as dump_runtime
from cp_bg_bench.runtime import probe

snakemake_obj = snakemake  # noqa: F821

allocation = probe()
dump_runtime(allocation, snakemake_obj.log.runtime)

cfg = load_config(snakemake_obj.input.global_config)

# NOTE: extend this block when adding a new data-source adapter.
if isinstance(cfg.data_source, JumpConfig):
    image_ch_labels = [k.removeprefix("s3_Orig") for k in cfg.data_source.channel_s3_keys]
elif isinstance(cfg.data_source, (Rxrx1Config, Rxrx3CoreConfig)):
    image_ch_labels = list(cfg.data_source.channel_names)
else:
    raise ValueError(f"Unknown data source type: {type(cfg.data_source)}")

n_image_channels = len(image_ch_labels)
ch_labels = ["NucMask", "CellMask"] + image_ch_labels
n_channels = N_MASK_CHANNELS + n_image_channels
image_size = cfg.global_.crops.image_size

_VARIANTS = ["crops", "seg", "crops_density", "seg_density"]
datasets = {
    v: load_from_disk(str(cfg.output_root / "datasets" / f"{v}_resharded"))
    for v in _VARIANTS
}

n_rows_total = len(datasets["crops"])
rng = np.random.default_rng(cfg.global_.selection.random_seed)
n_plots = min(3, n_rows_total)
indices = rng.choice(n_rows_total, size=n_plots, replace=False).tolist()

output_dir = cfg.output_root / "plots"
output_dir.mkdir(parents=True, exist_ok=True)

logger.info(f"visualize_sample: sampling indices {indices} from {n_rows_total} rows")


def _decode(row: dict) -> np.ndarray:
    """Reconstruct combined (mask + cell) view for display."""
    mask = np.frombuffer(row["mask"], dtype=np.uint8).reshape(N_MASK_CHANNELS, image_size, image_size)
    cell = np.frombuffer(row["cell"], dtype=np.uint8).reshape(n_image_channels, image_size, image_size)
    return np.concatenate([mask, cell], axis=0)


for plot_i, idx in enumerate(indices):
    row_key = datasets["crops"][idx]["row_key"]
    fig, axes = plt.subplots(
        len(_VARIANTS),
        n_channels,
        figsize=(2.0 * n_channels, 2.0 * len(_VARIANTS)),
        squeeze=False,
    )

    for row_i, variant in enumerate(_VARIANTS):
        arr = _decode(datasets[variant][idx])
        for ch in range(n_channels):
            ax = axes[row_i, ch]
            ax.imshow(arr[ch], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.axis("off")
            if row_i == 0:
                ax.set_title(ch_labels[ch], fontsize=7)
        axes[row_i, 0].set_ylabel(variant, fontsize=8, rotation=45, labelpad=40)

    fig.suptitle(f"idx={idx}  row_key={row_key}", fontsize=8)
    fig.tight_layout()
    out_path = output_dir / f"sample_{plot_i:02d}.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"visualize_sample: wrote {out_path}")

sentinel = Path(snakemake_obj.output.sentinel)
sentinel.write_text(f"n_plots={n_plots}\nindices={indices}\n")
logger.info(f"visualize_sample: done → {output_dir}")
