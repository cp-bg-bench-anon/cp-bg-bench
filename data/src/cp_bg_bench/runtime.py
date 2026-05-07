"""Runtime resource discovery.

Every GPU/Dask rule calls :func:`probe` at job start to discover what SLURM
actually gave us: visible GPUs (and their free VRAM), cgroup memory ceiling,
pinned CPUs, temp dir. The result is dumped to ``logs/runtime/{rule}.json``
for audit and consumed to size Cellpose batches, Dask worker counts, etc.

All filesystem and subprocess dependencies are injected via keyword arguments
so the whole thing is trivially unit-testable without an actual SLURM
allocation.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

__all__ = ["NodeAllocation", "probe", "dump"]


CGROUP_V2_PATH = Path("/sys/fs/cgroup/memory.max")
CGROUP_V1_PATH = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
MEMINFO_PATH = Path("/proc/meminfo")


@dataclass(frozen=True)
class NodeAllocation:
    """Snapshot of what the current SLURM allocation actually provides."""

    slurm_job_id: str | None
    visible_gpu_ids: list[int]
    per_gpu_total_vram_mb: list[int]
    per_gpu_free_vram_mb: list[int]
    cgroup_mem_limit_bytes: int
    available_mem_bytes: int
    allocated_cpus: int
    tmp_dir: Path

    def to_json(self) -> dict[str, object]:
        payload = asdict(self)
        payload["tmp_dir"] = str(self.tmp_dir)
        return payload


class NvidiaSmiRunner(Protocol):
    def __call__(self) -> str | None: ...


class MigVramRunner(Protocol):
    def __call__(self, uuid: str) -> tuple[int, int]: ...


def _run_nvidia_smi() -> str | None:
    """Return raw CSV from ``nvidia-smi``, or ``None`` if the tool is absent."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout


def _run_mig_vram(uuid: str) -> tuple[int, int]:
    """Return ``(total_mb, free_mb)`` for the active MIG instance, or ``(0, 0)`` on failure.

    nvidia-smi reads ``CUDA_VISIBLE_DEVICES`` automatically when a MIG UUID is set
    there; ``--id=<MIG-UUID>`` is not supported and returns "No devices were found".
    """
    if shutil.which("nvidia-smi") is None:
        return 0, 0
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        line = result.stdout.strip().splitlines()[0].strip()
        total, free = (int(p.strip()) for p in line.split(","))
        return total, free
    except Exception:
        return 0, 0


def _parse_gpu_csv(csv: str) -> list[tuple[int, int, int]]:
    rows: list[tuple[int, int, int]] = []
    for line in csv.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            raise ValueError(f"unexpected nvidia-smi row: {line!r}")
        idx, total_mb, free_mb = (int(p) for p in parts)
        rows.append((idx, total_mb, free_mb))
    return rows


def _parse_visible(env_value: str | None, all_indices: list[int]) -> list[int]:
    if env_value is None:
        return list(all_indices)
    if env_value.strip() == "":
        return []
    return [int(x) for x in env_value.split(",") if x.strip() != ""]


def _read_cgroup_mem_limit(
    v2_path: Path = CGROUP_V2_PATH,
    v1_path: Path = CGROUP_V1_PATH,
) -> int:
    """Return the cgroup memory limit in bytes, or 0 if unconstrained / missing."""
    for path in (v2_path, v1_path):
        try:
            raw = path.read_text().strip()
        except (FileNotFoundError, PermissionError, IsADirectoryError):
            continue
        if raw in {"max", ""}:
            return 0
        try:
            return int(raw)
        except ValueError:
            continue
    return 0


def _read_meminfo_available(meminfo_path: Path = MEMINFO_PATH) -> int:
    """Return MemAvailable in bytes from ``/proc/meminfo``, or 0 on failure."""
    try:
        raw = meminfo_path.read_text()
    except (FileNotFoundError, PermissionError):
        return 0
    for line in raw.splitlines():
        if not line.startswith("MemAvailable:"):
            continue
        parts = line.split()
        # Format: "MemAvailable:  1234567 kB"
        if len(parts) >= 2 and parts[1].isdigit():
            return int(parts[1]) * 1024
    return 0


def _allocated_cpus(env: Mapping[str, str]) -> int:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    for key in ("SLURM_CPUS_ON_NODE", "SLURM_CPUS_PER_TASK"):
        value = env.get(key)
        if value and value.isdigit():
            return int(value)
    return os.cpu_count() or 1


def probe(
    *,
    env: Mapping[str, str] | None = None,
    nvidia_smi: NvidiaSmiRunner = _run_nvidia_smi,
    mig_vram: MigVramRunner = _run_mig_vram,
    cgroup_v2_path: Path = CGROUP_V2_PATH,
    cgroup_v1_path: Path = CGROUP_V1_PATH,
    meminfo_path: Path = MEMINFO_PATH,
) -> NodeAllocation:
    """Probe the current allocation.

    Keyword arguments exist purely so tests can inject mocks; call sites in
    rules should invoke ``probe()`` with no arguments.
    """
    env = os.environ if env is None else env

    cuda_visible = env.get("CUDA_VISIBLE_DEVICES", "")
    mig_tokens = [t.strip() for t in cuda_visible.split(",") if t.strip().startswith("MIG-")]

    if mig_tokens:
        # MIG is active: each token is an instance UUID, not an integer index.
        # Query VRAM per instance; fall back to (0, 0) if nvidia-smi is unavailable.
        visible_present = list(range(len(mig_tokens)))
        per_gpu_total = []
        per_gpu_free = []
        for uuid in mig_tokens:
            total, free = mig_vram(uuid)
            per_gpu_total.append(total)
            per_gpu_free.append(free)
    else:
        smi_csv = nvidia_smi()
        all_rows = _parse_gpu_csv(smi_csv) if smi_csv else []
        visible = _parse_visible(env.get("CUDA_VISIBLE_DEVICES"), [idx for idx, *_ in all_rows])
        row_by_idx = {idx: (total, free) for idx, total, free in all_rows}
        visible_present = [idx for idx in visible if idx in row_by_idx]
        per_gpu_total = [row_by_idx[idx][0] for idx in visible_present]
        per_gpu_free = [row_by_idx[idx][1] for idx in visible_present]

    cgroup_limit = _read_cgroup_mem_limit(cgroup_v2_path, cgroup_v1_path)
    meminfo_available = _read_meminfo_available(meminfo_path)

    # "min(..., stuff)" but treat 0 as unconstrained on each side.
    candidates = [x for x in (cgroup_limit, meminfo_available) if x > 0]
    available = min(candidates) if candidates else 0

    return NodeAllocation(
        slurm_job_id=env.get("SLURM_JOB_ID"),
        visible_gpu_ids=visible_present,
        per_gpu_total_vram_mb=per_gpu_total,
        per_gpu_free_vram_mb=per_gpu_free,
        cgroup_mem_limit_bytes=cgroup_limit,
        available_mem_bytes=available,
        allocated_cpus=_allocated_cpus(env),
        tmp_dir=Path(env.get("TMPDIR", "/tmp")),
    )


def dump(allocation: NodeAllocation, path: str | Path) -> Path:
    """Write ``allocation`` as pretty-printed JSON, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(allocation.to_json(), indent=2, sort_keys=True))
    tmp.replace(path)
    return path
