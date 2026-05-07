"""Rules J, K, L: derive seg, crops_density, and seg_density variants."""


rule derive_seg:
    input:
        crops_sentinel=f"{RESULTS}/datasets/crops_hf.done",
        global_config=ancient(str(CONFIG.source_path)),
    output:
        sentinel=f"{RESULTS}/datasets/seg_hf.done",
    threads: 4
    log:
        runtime=f"{LOGS}/runtime/derive_seg.json",
    script:
        "../scripts/derive_seg.py"


rule derive_crops_density:
    input:
        crops_sentinel=f"{RESULTS}/datasets/crops_hf.done",
        global_config=ancient(str(CONFIG.source_path)),
    output:
        sentinel=f"{RESULTS}/datasets/crops_density_hf.done",
    threads: 4
    log:
        runtime=f"{LOGS}/runtime/derive_crops_density.json",
    script:
        "../scripts/derive_crops_density.py"


rule derive_seg_density:
    input:
        seg_sentinel=f"{RESULTS}/datasets/seg_hf.done",
        global_config=ancient(str(CONFIG.source_path)),
    output:
        sentinel=f"{RESULTS}/datasets/seg_density_hf.done",
    threads: 4
    log:
        runtime=f"{LOGS}/runtime/derive_seg_density.json",
    script:
        "../scripts/derive_seg_density.py"
