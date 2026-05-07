"""Rule I: quality_filter — filter cells by morphology quantile thresholds."""


rule quality_filter:
    input:
        hf_sentinel=f"{RESULTS}/datasets/crops_unfiltered_hf.done",
        global_config=ancient(str(CONFIG.source_path)),
    output:
        sentinel=f"{RESULTS}/datasets/crops_hf.done",
        report=f"{RESULTS}/quality_filter/report.yml",
    threads: 4
    log:
        runtime=f"{LOGS}/runtime/quality_filter.json",
    script:
        "../scripts/quality_filter.py"
