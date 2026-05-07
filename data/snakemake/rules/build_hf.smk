"""Rule H: build_unfiltered_hf — convert per-plate parquets to HuggingFace Dataset."""


def _perturbation_filter_inputs(wildcards=None):
    """Return siRNA allowlist path for rxrx1; empty list for other data sources."""
    if CONFIG.global_.data_source == "rxrx1":
        return [str(WORKFLOW_ROOT / "meta" / "rxrx1_valid_sirna_ids.txt")]
    return []


rule build_unfiltered_hf:
    input:
        extract_sentinels=_extract_sentinels,
        global_config=ancient(str(CONFIG.source_path)),
        perturbation_filter=_perturbation_filter_inputs,
    output:
        sentinel=f"{RESULTS}/datasets/crops_unfiltered_hf.done",
    threads: 2
    log:
        runtime=f"{LOGS}/runtime/build_unfiltered_hf.json",
    script:
        "../scripts/build_unfiltered_hf.py"
