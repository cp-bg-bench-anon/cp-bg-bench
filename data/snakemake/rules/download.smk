"""Rule B — download per-FOV TIFFs into per-plate Zarr v3 stores.

Wildcard is ``snakemake_batch`` (a batch id emitted by rule A). A
single batch may span multiple plates, so the rule writes into N
plate stores (under ``<RESULTS>/full_images/<source>__<batch>__
<plate>.zarr``) and marks completion with a sentinel file
``<RESULTS>/full_images/_batches/{batch}.done``. Snakemake tracks the
sentinel; the plate stores are the real (permanent, non-``temp()``)
artefacts.
"""


rule download_batch:
    input:
        parquet=ancient(f"{RESULTS}/meta/selected_metadata.parquet"),
        global_config=ancient(str(CONFIG.source_path)),
    output:
        sentinel=f"{RESULTS}/full_images/_batches/{{batch}}.done",
    log:
        log=f"{LOGS}/download/{{batch}}.log",
        runtime=f"{LOGS}/runtime/download_{{batch}}.json",
    threads: 4
    retries: 3
    script:
        "../scripts/download_batch.py"
