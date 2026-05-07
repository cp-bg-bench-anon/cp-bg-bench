"""Rules D + D2: per-batch FOV segmentation and plate-level collection.

Rule D  (segment_fov_batch): one Snakemake job per {plate_key}/{seg_batch_idx}.
  - Reads the FOV IDs for its batch from the metadata parquet.
  - Calls segment_plate(fov_ids=...) which runs two model.eval() calls for
    the batch and writes label arrays directly to the shared plate zarr.
  - Writes a per-batch parquet with per-cell stats.

Rule D2 (collect_plate_seg): one job per plate_key.
  - Concatenates all per-batch parquets into the plate-level parquet.
  - Writes the plate sentinel consumed by downstream rules.

Requires the seg-cpu or seg-gpu pixi environment:

    pixi run -e seg-cpu segment-smoke
    pixi run -e seg-gpu segment-gpu-smoke
"""


def _plate_seg_batch_sentinels(wildcards):
    """Return per-batch sentinels for one plate (used by collect_plate_seg)."""
    parquet = checkpoints.resolve_metadata.get().output.parquet
    df = pd.read_parquet(
        parquet,
        columns=["Metadata_Source", "Metadata_Batch", "Metadata_Plate"],
    )
    src, bat, plt = wildcards.plate_key.split("__", 2)
    n_fovs = int(
        ((df["Metadata_Source"] == src) & (df["Metadata_Batch"] == bat) & (df["Metadata_Plate"] == plt)).sum()
    )
    batch_size = CONFIG.global_.compute.gpu.seg_fov_batch_size
    n_batches = max(1, (n_fovs + batch_size - 1) // batch_size)
    return [
        f"{RESULTS}/segmentation/_batches/{wildcards.plate_key}/{i}.done"
        for i in range(n_batches)
    ]


rule segment_fov_batch:
    input:
        downloads=ancient(_plate_download_sentinels),
        metadata=f"{RESULTS}/meta/selected_metadata.parquet",
        global_config=ancient(str(CONFIG.source_path)),
    output:
        parquet=f"{RESULTS}/segmentation/_batches/{{plate_key}}/{{seg_batch_idx}}.parquet",
        sentinel=f"{RESULTS}/segmentation/_batches/{{plate_key}}/{{seg_batch_idx}}.done",
    wildcard_constraints:
        seg_batch_idx=r"\d+",
    threads: 4
    resources:
        gpu_mem_mb=lambda wc, config=CONFIG: (
            config.global_.compute.gpu.seg_fov_batch_size
            * config.global_.compute.gpu.cellpose_vram_per_sample_mb
        ),
    log:
        runtime=f"{LOGS}/runtime/segment_fov_batch_{{plate_key}}_{{seg_batch_idx}}.json",
    script:
        "../scripts/segment_fov_batch.py"


rule collect_plate_seg:
    input:
        batch_sentinels=_plate_seg_batch_sentinels,
        global_config=ancient(str(CONFIG.source_path)),
    output:
        parquet=f"{RESULTS}/segmentation/{{plate_key}}.parquet",
        sentinel=f"{RESULTS}/segmentation/_plates/{{plate_key}}.done",
    threads: 1
    log:
        runtime=f"{LOGS}/runtime/collect_plate_seg_{{plate_key}}.json",
    script:
        "../scripts/collect_plate_seg.py"
