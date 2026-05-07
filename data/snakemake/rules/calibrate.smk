"""Rule C — calibrate Cellpose-SAM diameter estimates.

**Manual rule** — not a dependency of ``rule all``. Run explicitly with::

    pixi run -e seg-cpu calibrate-smoke   # smoke config
    pixi run -e seg-cpu calibrate         # full config

Or on a GPU node::

    pixi run -e seg-gpu calibrate-gpu-smoke

The rule reads rule A's parquet (for the FOV list + metadata) and requires
all rule-B download sentinels to be present (ensuring the Zarr stores are
populated). It writes a three-file report under
``<RESULTS>/calibration/{config_hash}.{yml,md,png}`` and touches a
``.done`` sentinel so Snakemake marks the job complete on rerun.

Review the generated ``.yml`` and hand-merge the suggested
``per_source_diameters`` into ``config/jump.yml`` before running rule D.
"""


rule calibrate:
    input:
        parquet=f"{RESULTS}/meta/selected_metadata.parquet",
        downloads=_download_sentinels,
        global_config=ancient(str(CONFIG.source_path)),
    output:
        sentinel=f"{RESULTS}/calibration/.done",
    threads: 1
    log:
        runtime=f"{LOGS}/runtime/calibrate.json",
    script:
        "../scripts/calibrate.py"
