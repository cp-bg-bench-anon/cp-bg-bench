"""Rule E: select_cells — one job reading all per-plate segmentation parquets."""


rule select_cells:
    input:
        sentinels=_segment_sentinels,
        global_config=ancient(str(CONFIG.source_path)),
        meta_parquet=f"{RESULTS}/meta/selected_metadata.parquet",
    output:
        parquet=f"{RESULTS}/selection/selected_cells.parquet",
        sentinel=f"{RESULTS}/selection/selected_cells.done",
    threads: 2
    log:
        runtime=f"{LOGS}/runtime/select_cells.json",
    script:
        "../scripts/select_cells.py"
