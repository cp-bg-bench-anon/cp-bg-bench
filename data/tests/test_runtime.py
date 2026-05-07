"""Tests for :mod:`cp_bg_bench.runtime`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cp_bg_bench import runtime
from cp_bg_bench.runtime import NodeAllocation, dump, probe

SMI_SAMPLE = "0, 81920, 76000\n1, 81920, 40000\n2, 81920, 78000\n"


@pytest.fixture
def fake_fs(tmp_path: Path) -> dict[str, Path]:
    """Filesystem stubs for cgroup + meminfo reads."""
    v2 = tmp_path / "memory.max"
    v2.write_text("17179869184\n")  # 16 GiB
    v1 = tmp_path / "memory.limit_in_bytes"  # intentionally absent from disk
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(
        "MemTotal:       65536000 kB\nMemFree:         1024000 kB\nMemAvailable:   32768000 kB\n"
    )
    return {"v2": v2, "v1": v1, "meminfo": meminfo}


def test_probe_parses_gpu_and_respects_visible(fake_fs: dict[str, Path]) -> None:
    env = {"CUDA_VISIBLE_DEVICES": "0,2", "SLURM_JOB_ID": "12345", "TMPDIR": "${SCRATCH}"}
    alloc = probe(
        env=env,
        nvidia_smi=lambda: SMI_SAMPLE,
        cgroup_v2_path=fake_fs["v2"],
        cgroup_v1_path=fake_fs["v1"],
        meminfo_path=fake_fs["meminfo"],
    )
    assert alloc.slurm_job_id == "12345"
    assert alloc.visible_gpu_ids == [0, 2]
    assert alloc.per_gpu_total_vram_mb == [81920, 81920]
    assert alloc.per_gpu_free_vram_mb == [76000, 78000]
    assert alloc.tmp_dir == Path("${SCRATCH}")
    assert alloc.cgroup_mem_limit_bytes == 17179869184
    assert alloc.available_mem_bytes == min(17179869184, 32768000 * 1024)
    assert alloc.allocated_cpus >= 1


def test_probe_without_gpu(fake_fs: dict[str, Path]) -> None:
    alloc = probe(
        env={},
        nvidia_smi=lambda: None,
        cgroup_v2_path=fake_fs["v2"],
        cgroup_v1_path=fake_fs["v1"],
        meminfo_path=fake_fs["meminfo"],
    )
    assert alloc.visible_gpu_ids == []
    assert alloc.per_gpu_total_vram_mb == []
    assert alloc.per_gpu_free_vram_mb == []
    assert alloc.slurm_job_id is None
    assert alloc.tmp_dir == Path("/tmp")


def test_probe_cgroup_v1_fallback(tmp_path: Path) -> None:
    missing_v2 = tmp_path / "memory.max"  # does not exist
    v1 = tmp_path / "memory.limit_in_bytes"
    v1.write_text("8589934592\n")  # 8 GiB
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemAvailable:   9000000 kB\n")

    alloc = probe(
        env={},
        nvidia_smi=lambda: None,
        cgroup_v2_path=missing_v2,
        cgroup_v1_path=v1,
        meminfo_path=meminfo,
    )
    assert alloc.cgroup_mem_limit_bytes == 8589934592
    assert alloc.available_mem_bytes == min(8589934592, 9000000 * 1024)


def test_probe_cgroup_unconstrained(tmp_path: Path) -> None:
    v2 = tmp_path / "memory.max"
    v2.write_text("max\n")
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemAvailable:   1000 kB\n")

    alloc = probe(
        env={},
        nvidia_smi=lambda: None,
        cgroup_v2_path=v2,
        cgroup_v1_path=tmp_path / "missing",
        meminfo_path=meminfo,
    )
    assert alloc.cgroup_mem_limit_bytes == 0
    assert alloc.available_mem_bytes == 1000 * 1024


def test_probe_falls_back_to_slurm_cpus_if_no_affinity(
    monkeypatch: pytest.MonkeyPatch, fake_fs: dict[str, Path]
) -> None:
    def _raise(_pid: int) -> set[int]:
        raise OSError("no affinity on this platform")

    monkeypatch.setattr(runtime.os, "sched_getaffinity", _raise, raising=False)
    alloc = probe(
        env={"SLURM_CPUS_ON_NODE": "7"},
        nvidia_smi=lambda: None,
        cgroup_v2_path=fake_fs["v2"],
        cgroup_v1_path=fake_fs["v1"],
        meminfo_path=fake_fs["meminfo"],
    )
    assert alloc.allocated_cpus == 7


def test_probe_with_mig_uuids(fake_fs: dict[str, Path]) -> None:
    """MIG UUIDs in CUDA_VISIBLE_DEVICES must not raise and must report GPU presence."""
    mig_uuid = "MIG-79ff4e19-2b18-5cd4-a2d3-f7315a4d936b"
    alloc = probe(
        env={"CUDA_VISIBLE_DEVICES": mig_uuid},
        nvidia_smi=lambda: None,
        mig_vram=lambda uuid: (20480, 18000),
        cgroup_v2_path=fake_fs["v2"],
        cgroup_v1_path=fake_fs["v1"],
        meminfo_path=fake_fs["meminfo"],
    )
    assert alloc.visible_gpu_ids == [0]
    assert alloc.per_gpu_total_vram_mb == [20480]
    assert alloc.per_gpu_free_vram_mb == [18000]


def test_probe_empty_visible_devices(fake_fs: dict[str, Path]) -> None:
    alloc = probe(
        env={"CUDA_VISIBLE_DEVICES": ""},
        nvidia_smi=lambda: SMI_SAMPLE,
        cgroup_v2_path=fake_fs["v2"],
        cgroup_v1_path=fake_fs["v1"],
        meminfo_path=fake_fs["meminfo"],
    )
    assert alloc.visible_gpu_ids == []
    assert alloc.per_gpu_total_vram_mb == []


def test_dump_writes_json(tmp_path: Path) -> None:
    alloc = NodeAllocation(
        slurm_job_id="1",
        visible_gpu_ids=[0],
        per_gpu_total_vram_mb=[81920],
        per_gpu_free_vram_mb=[40000],
        cgroup_mem_limit_bytes=1024,
        available_mem_bytes=1024,
        allocated_cpus=4,
        tmp_dir=Path("/tmp"),
    )
    out = dump(alloc, tmp_path / "nested" / "alloc.json")
    payload = json.loads(out.read_text())
    assert payload["slurm_job_id"] == "1"
    assert payload["per_gpu_free_vram_mb"] == [40000]
    assert payload["tmp_dir"] == "/tmp"
