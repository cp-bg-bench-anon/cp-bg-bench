"""Rule A — resolve JUMP metadata into a per-FOV parquet.

Declared as a Snakemake ``checkpoint`` so that downstream rules (rule B
and beyond) can enumerate the unique ``snakemake_batch`` ids at DAG-
resume time by reading the parquet. Depends on :data:`CONFIG`,
:data:`RESULTS`, :data:`LOGS` from the top-level Snakefile.
"""


checkpoint resolve_metadata:
    input:
        global_config=str(CONFIG.source_path),
        data_source_config=str(CONFIG.data_source_path),
    output:
        parquet=f"{RESULTS}/meta/selected_metadata.parquet",
    log:
        log=f"{LOGS}/resolve_metadata.log",
        runtime=f"{LOGS}/runtime/resolve_metadata.json",
    threads: 4
    script:
        "../scripts/resolve_metadata.py"
