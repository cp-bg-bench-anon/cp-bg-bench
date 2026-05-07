"""Rule M: reshard_variant — reshard each of the four dataset variants."""

_VARIANT_DEPS = {
    "crops":          f"{RESULTS}/datasets/crops_hf.done",
    "seg":            f"{RESULTS}/datasets/seg_hf.done",
    "crops_density":  f"{RESULTS}/datasets/crops_density_hf.done",
    "seg_density":    f"{RESULTS}/datasets/seg_density_hf.done",
}


def _variant_dep(wildcards):
    return _VARIANT_DEPS[wildcards.variant]


rule reshard_variant:
    input:
        hf_sentinel=_variant_dep,
        global_config=ancient(str(CONFIG.source_path)),
    output:
        sentinel=f"{RESULTS}/datasets/{{variant}}_resharded.done",
    wildcard_constraints:
        variant="crops|seg|crops_density|seg_density",
    threads: 2
    log:
        runtime=f"{LOGS}/runtime/reshard_{{variant}}.json",
    script:
        "../scripts/reshard_variant.py"
