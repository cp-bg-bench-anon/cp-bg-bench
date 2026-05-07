"""Rule F: extract_crops_plate — per-plate crop extraction."""


rule extract_crops_plate:
    input:
        select_sentinel=ancient(f"{RESULTS}/selection/selected_cells.done"),
        seg_sentinel=ancient(f"{RESULTS}/segmentation/_plates/{{plate_key}}.done"),
        global_config=ancient(str(CONFIG.source_path)),
    output:
        parquet=f"{RESULTS}/crops_unfiltered/{{plate_key}}.parquet",
        sentinel=f"{RESULTS}/crops_unfiltered/_plates/{{plate_key}}.done",
    threads: 2
    resources:
        gpu_mem_mb=40000
    log:
        runtime=f"{LOGS}/runtime/extract_crops_plate_{{plate_key}}.json",
    script:
        "../scripts/extract_crops_plate.py"
