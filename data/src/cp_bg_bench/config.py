"""Pydantic v2 schema + loader for cp-bg-bench YAML configs.

Two YAMLs drive the pipeline: a global per-run file (``config/config.yml``)
and a data-source-specific file it points at (``config/jump.yml`` etc.). This
module parses both, resolves relative paths against the global file's parent,
and returns a :class:`PipelineConfig` with both halves attached.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

__all__ = [
    "CalibrateConfig",
    "ComputeConfig",
    "CropsConfig",
    "DaskConfig",
    "DataSourceConfig",
    "GlobalConfig",
    "GpuConfig",
    "JumpConfig",
    "JumpSample",
    "NormalizationConfig",
    "PathsConfig",
    "PatchingConfig",
    "PipelineConfig",
    "QualityFilterConfig",
    "Rxrx1Config",
    "Rxrx1Sample",
    "Rxrx3CoreConfig",
    "Rxrx3CoreSample",
    "SegmentationConfig",
    "SegmentationDiameters",
    "SelectionConfig",
    "ShardingConfig",
    "load",
]


class _Strict(BaseModel):
    """Base model: forbid unknown keys to surface typos loudly."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class PathsConfig(_Strict):
    root: Path


SelectionStrategy = Literal[
    "uniform_per_well",
    "all",
    "uniform_total",
    "uniform_per_compound_source",
]


class SelectionConfig(_Strict):
    strategy: SelectionStrategy
    cells_per_well: PositiveInt | None = None
    max_cells: PositiveInt | None = None
    target_cells: PositiveInt | None = None
    control_labels: list[str] | None = None
    random_seed: int

    @model_validator(mode="after")
    def _check_strategy_params(self) -> SelectionConfig:
        if self.strategy == "uniform_per_well" and self.cells_per_well is None:
            raise ValueError("cells_per_well is required when strategy=uniform_per_well")
        if self.strategy == "uniform_total" and self.max_cells is None:
            raise ValueError("max_cells is required when strategy=uniform_total")
        if self.strategy == "uniform_per_compound_source" and self.target_cells is None:
            raise ValueError("target_cells is required when strategy=uniform_per_compound_source")
        if self.control_labels is not None and self.strategy != "uniform_per_compound_source":
            raise ValueError(
                "control_labels is only valid when strategy=uniform_per_compound_source"
            )
        return self


Quantile = Annotated[float, Field(ge=0.0, le=1.0)]


class QualityFilterConfig(_Strict):
    enabled: bool
    quantiles: tuple[Quantile, Quantile]
    fields: list[str]

    @model_validator(mode="after")
    def _quantiles_ordered(self) -> QualityFilterConfig:
        lo, hi = self.quantiles
        if lo >= hi:
            raise ValueError(f"quality_filter.quantiles must satisfy low < high, got ({lo}, {hi})")
        if not self.fields:
            raise ValueError("quality_filter.fields must be non-empty")
        return self


NormalizationScheme = Literal["per_fov_percentile", "naive_u16_to_u8"]


class NormalizationConfig(_Strict):
    scheme: NormalizationScheme
    low: Quantile
    high: Quantile

    @model_validator(mode="after")
    def _bounds_ordered(self) -> NormalizationConfig:
        if self.low >= self.high:
            raise ValueError(
                f"crops.normalization.low must be < high, got ({self.low}, {self.high})"
            )
        return self


class CropsConfig(_Strict):
    patch_size: PositiveInt
    image_size: PositiveInt
    normalization: NormalizationConfig


class PatchingConfig(_Strict):
    patch_size_resized: PositiveInt
    pad_resized: PositiveInt


class ShardingConfig(_Strict):
    parquet_rows_per_shard: PositiveInt
    hf_rows_per_shard: PositiveInt
    hf_max_shards: PositiveInt
    snakemake_batch_size: PositiveInt


class CalibrateConfig(_Strict):
    fovs_per_source: PositiveInt = 12
    random_seed: int = 42
    min_success_fraction: Annotated[float, Field(ge=0.0, le=1.0)] = 0.7


class DaskConfig(_Strict):
    n_workers: PositiveInt
    threads_per_worker: PositiveInt


class GpuConfig(_Strict):
    cellpose_vram_per_sample_mb: PositiveInt
    resize_vram_mb_max: PositiveInt
    seg_fov_batch_size: PositiveInt = Field(default=50)


class ComputeConfig(_Strict):
    dask: DaskConfig
    gpu: GpuConfig


DataSourceName = Literal["jump", "rxrx1", "rxrx3_core"]


class GlobalConfig(_Strict):
    """Per-run configuration; mirrors ``config/config.yml``."""

    data_source: DataSourceName
    data_source_config: Path
    paths: PathsConfig
    selection: SelectionConfig
    quality_filter: QualityFilterConfig
    crops: CropsConfig
    patching: PatchingConfig
    sharding: ShardingConfig
    compute: ComputeConfig
    calibrate: CalibrateConfig = Field(default_factory=CalibrateConfig)
    max_fovs_per_plate: int | None = None
    max_fovs_per_well: int | None = None


class SegmentationDiameters(_Strict):
    nucleus: PositiveFloat
    cytosol: PositiveFloat


class SegmentationConfig(_Strict):
    model: Literal["cpsam"]
    channels_for_nucleus: list[int]
    channels_for_cell: list[int]
    default_diameters: SegmentationDiameters
    per_source_diameters: dict[str, SegmentationDiameters] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _channels_non_empty(self) -> SegmentationConfig:
        if not self.channels_for_nucleus:
            raise ValueError("segmentation.channels_for_nucleus must be non-empty")
        if not self.channels_for_cell:
            raise ValueError("segmentation.channels_for_cell must be non-empty")
        return self


class JumpSample(_Strict):
    """Whitelist entry for a JUMP source (optionally batch / plate / well / compound scoped)."""

    metadata_source: str
    metadata_batch: str | None = None
    metadata_plate: str | None = None
    metadata_well: str | None = None
    metadata_inchikey: str | None = None
    metadata_jcp2022: str | None = None


class JumpConfig(_Strict):
    """JUMP-CP (cpg0016) data-source config."""

    samples: list[JumpSample]
    metadata_tables: dict[str, str]
    channel_s3_keys: list[str]
    segmentation: SegmentationConfig

    @model_validator(mode="after")
    def _non_empty(self) -> JumpConfig:
        if not self.samples:
            raise ValueError("jump.samples must be non-empty")
        if not self.channel_s3_keys:
            raise ValueError("jump.channel_s3_keys must be non-empty")
        # plate/well/compound are load-bearing; orf/crispr are annotation
        # extras that are merged in when present but are not required so
        # users running compound-only workflows don't need to configure them.
        required_tables = {"plate", "well", "compound"}
        missing = required_tables - set(self.metadata_tables)
        if missing:
            raise ValueError(f"jump.metadata_tables missing required keys: {sorted(missing)}")
        return self


class Rxrx1Sample(_Strict):
    """Whitelist entry for an Rxrx1 experiment (optionally plate / well / siRNA scoped)."""

    experiment: str
    plate: str | None = None
    well: str | None = None
    sirna: str | None = None


class Rxrx1Config(_Strict):
    """Rxrx1 (Recursion Pharmaceuticals) data-source config.

    Images ship as a single bulk zip on GCS (``images_zip_url``). Rule B
    uses HTTP range requests (via ``remotezip``) to fetch individual PNGs
    without downloading the full archive. ``metadata_url`` points to a
    smaller metadata zip (~1 MB) that contains a single ``metadata.csv``.

    Image paths within the zip follow the fixed pattern:
    ``rxrx1/images/{experiment}/Plate{plate}/{well}_s{site}_w{channel}.png``
    where ``channel`` is 1-indexed over ``len(channel_names)``.

    Metadata CSV columns: site_id, well_id, cell_type, dataset, experiment,
    plate, well, site, well_type, sirna, sirna_id.
    """

    samples: list[Rxrx1Sample]
    metadata_url: str
    images_zip_url: str
    channel_names: list[str]
    segmentation: SegmentationConfig

    @model_validator(mode="after")
    def _non_empty(self) -> Rxrx1Config:
        if not self.samples:
            raise ValueError("rxrx1.samples must be non-empty")
        if not self.channel_names:
            raise ValueError("rxrx1.channel_names must be non-empty")
        return self

    @property
    def channel_zip_keys(self) -> list[str]:
        """Column names for per-channel zip-path columns in the metadata parquet."""
        return [f"zip_{name}" for name in self.channel_names]


class Rxrx3CoreSample(_Strict):
    """Whitelist entry for an RxRx3-core experiment (optionally plate / address / gene scoped)."""

    experiment: str
    plate: int | None = None
    address: str | None = None
    gene: str | None = None


class Rxrx3CoreConfig(_Strict):
    """RxRx3-core (Recursion Pharmaceuticals, 2024) HUVEC CRISPR KO data-source config.

    Images ship as 35 inline-bytes parquet shards on HuggingFace
    (``recursionpharma/rxrx3-core``). Shards 16–34 contain all CRISPR data;
    shards 0–15 contain compound data (excluded from this pipeline adapter).

    Parquet key format:
        ``{experiment_name}/Plate{plate}/{address}_s1_{channel_idx}``
    where ``channel_idx`` is 1-indexed over ``len(channel_names)``.

    Metadata CSV columns (``metadata_rxrx3_core.csv``):
        well_id, experiment_name, plate, address, gene, treatment,
        SMILES, concentration, perturbation_type, cell_type.

    Column mapping (metadata CSV → pipeline schema):
        experiment_name → Metadata_Source
        plate           → Metadata_Plate  (cast to str)
        address         → Metadata_Well   (used in id; not a standalone parquet column)
        gene            → Metadata_InChIKey  (gene symbol; "EMPTY_control" for controls)
        treatment       → Metadata_InChI     (guide-level e.g. "TP53_guide_1")
        perturbation_type → Metadata_PlateType
        (fixed)         → Metadata_Batch = "rxrx3_core"
        (fixed)         → Metadata_Site  = "1"
        id = "{experiment_name}__rxrx3_core__{plate}__{address}__1"
    """

    samples: list[Rxrx3CoreSample]
    hf_repo: str
    channel_names: list[str]
    segmentation: SegmentationConfig

    @model_validator(mode="after")
    def _non_empty(self) -> Rxrx3CoreConfig:
        if not self.samples:
            raise ValueError("rxrx3_core.samples must be non-empty")
        if not self.channel_names:
            raise ValueError("rxrx3_core.channel_names must be non-empty")
        return self


# Union alias grown as new data sources are added.
DataSourceConfig = JumpConfig | Rxrx1Config | Rxrx3CoreConfig


class PipelineConfig(_Strict):
    """Fully resolved pipeline config: global + data-source halves."""

    global_: GlobalConfig = Field(alias="global")
    data_source: DataSourceConfig
    source_path: Path
    data_source_path: Path

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    @property
    def paths(self) -> PathsConfig:
        return self.global_.paths

    @property
    def data_source_stem(self) -> str:
        """Filename stem of the data-source config — used as the output subdir.

        Running with ``data_source_config: jump.yml`` writes to
        ``<paths.root>/jump/``; running with ``smoke.yml`` writes to
        ``<paths.root>/smoke/`` so verification runs never clobber production
        artifacts.
        """
        return self.data_source_path.stem

    @property
    def output_root(self) -> Path:
        """Root path for all pipeline outputs: ``paths.root / data_source_stem``."""
        return self.paths.root / self.data_source_stem

    def fingerprint(self) -> str:
        """Stable SHA256 over the merged config. Used for cache / artifact keys."""
        payload = self.model_dump(mode="json", by_alias=True)
        blob = yaml.safe_dump(payload, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: YAML root must be a mapping, got {type(data).__name__}")
    return data


def _parse_data_source(name: DataSourceName, data: dict[str, Any]) -> DataSourceConfig:
    if name == "jump":
        return JumpConfig.model_validate(data)
    if name == "rxrx1":
        return Rxrx1Config.model_validate(data)
    if name == "rxrx3_core":
        return Rxrx3CoreConfig.model_validate(data)
    raise ValueError(f"unknown data_source: {name!r}")


def load(config_path: str | Path) -> PipelineConfig:
    """Load + validate global config and its referenced data-source config.

    ``data_source_config`` is resolved relative to the directory containing
    ``config_path`` (so configs can be checked in with stable relative paths
    regardless of where the pipeline is invoked from).
    """
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"config file not found: {config_path}")

    global_cfg = GlobalConfig.model_validate(_read_yaml(config_path))

    ds_path = global_cfg.data_source_config
    if not ds_path.is_absolute():
        ds_path = (config_path.parent / ds_path).resolve()
    if not ds_path.is_file():
        raise FileNotFoundError(
            f"data_source_config {global_cfg.data_source_config} resolved to "
            f"{ds_path}, which does not exist"
        )

    data_source_cfg = _parse_data_source(global_cfg.data_source, _read_yaml(ds_path))

    return PipelineConfig.model_validate(
        {
            "global": global_cfg,
            "data_source": data_source_cfg,
            "source_path": config_path,
            "data_source_path": ds_path,
        }
    )
