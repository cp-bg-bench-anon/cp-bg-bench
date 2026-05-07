"""Rule N: visualize_sample — plot 3 sampled cells across all four dataset variants."""


rule visualize_sample:
    input:
        expand(f"{RESULTS}/datasets/{{variant}}_resharded.done", variant=_VARIANTS),
        global_config=ancient(str(CONFIG.source_path)),
    output:
        sentinel=f"{RESULTS}/plots/visualize_sample.done",
    threads: 1
    log:
        runtime=f"{LOGS}/runtime/visualize_sample.json",
    script:
        "../scripts/visualize_sample.py"
