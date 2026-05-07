from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
from datasets import Dataset, load_from_disk
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from cp_bg_bench_model._constants import DatasetEnum
from cp_bg_bench_model.encoders.image_encoders import ImageEncoderRegistry

try:
    from cp_bg_bench_model._logging import logger as log
except Exception:
    import logging

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("datamodule")

# import os, torch, torch.multiprocessing as mp
# put IPC files on a fast local disk/scratch instead of /dev/shm
# os.environ.setdefault("TMPDIR", "${DATA_ROOT}/torch_ipc")
# os.makedirs(os.environ["TMPDIR"], exist_ok=True)
# mp.set_sharing_strategy("file_system")

import os

import torch.multiprocessing as mp

_MULTI_PROC_CONTEXT = mp.get_context("fork")


def _configure_torch_ipc() -> None:
    """
    Dataloader worker IPC can hang badly if you force file-based sharing onto a network filesystem.
    Prefer file_descriptor (default-like) which uses shared memory, and only use file_system on
    node-local scratch if you *must* avoid /dev/shm limits.
    """
    # Prefer node-local temp if provided by SLURM; otherwise leave TMPDIR alone.
    slurm_tmp = os.environ.get("SLURM_TMPDIR")
    if slurm_tmp:
        ipc_dir = Path(slurm_tmp) / "torch_ipc"
        ipc_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TMPDIR"] = str(ipc_dir)

    # Prefer shared-memory based transfer for speed and to avoid network-FS stalls.
    try:
        mp.set_sharing_strategy("file_descriptor")
    except Exception:
        pass


_configure_torch_ipc()


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.set_num_threads(1)


def _worker_init(worker_id: int) -> None:
    # keep each worker single-threaded for BLAS/OMP libs
    torch.set_num_threads(1)


class DecodeImage:
    def __init__(self, C: int = 5, H: int = 224, W: int = 224) -> None:
        self.C = int(C)
        self.H = int(H)
        self.W = int(W)
        self._total = self.C * self.H * self.W

    def __call__(self, img_bytes: bytes) -> torch.Tensor:
        arr = np.frombuffer(img_bytes, dtype=np.uint8).copy()
        t = torch.from_numpy(arr)
        if t.numel() != self._total:
            raise ValueError(f"Bad image buffer: got {t.numel()} bytes, expected {self._total}.")
        return t.view(self.C, self.H, self.W)


class TFOne:
    """
    Wrap torchvision v2 transforms so the callable is a top-level object.
    Holds both base and augment pipelines and applies the right one at call time.
    """

    def __init__(self, base_tf: T.Compose, augment: bool) -> None:
        self.base_tf = base_tf
        self.augment = bool(augment)
        self.transform = base_tf
        self.transform_augment = T.Compose([base_tf, AUGMENT_T]) if self.augment else base_tf

    def __call__(self, img_t: torch.Tensor, train: bool = False) -> torch.Tensor:
        tf = self.transform_augment if (train and self.augment) else self.transform
        out = tf(img_t)
        return out if isinstance(out, torch.Tensor) else torch.as_tensor(out)


def _to_bytes_one_image(v: object) -> bytes:
    # Accept bytes-like or a length-1 container of bytes-like.
    if isinstance(v, (bytes, bytearray)):
        return bytes(v)
    if isinstance(v, memoryview):
        return v.tobytes()
    if isinstance(v, (list, tuple)):
        if len(v) != 1:
            raise ValueError(f"Expected 1 image per row, got len={len(v)}")
        return _to_bytes_one_image(v[0])
    raise TypeError(f"Expected bytes-like or length-1 container, got {type(v)}")


class TrainRowTransform:
    def __init__(self, tf_one: TFOne, decoder: DecodeImage, pert_transform, pert_source_col: str) -> None:
        self.tf_one = tf_one
        self.decoder = decoder
        self.pert_transform = pert_transform
        self.pert_source_col = str(pert_source_col)

    def __call__(self, x: dict) -> dict:
        if DatasetEnum.IMG in x:
            imgs = x[DatasetEnum.IMG]
            if isinstance(imgs, (bytes, bytearray, memoryview)):
                imgs = [imgs]
            elif not isinstance(imgs, (list, tuple)):
                raise TypeError(f"Expected images as bytes-like or list/tuple, got {type(imgs)}")

            x[DatasetEnum.IMG] = torch.stack(
                [self.tf_one(self.decoder(_to_bytes_one_image(v)), train=True) for v in imgs],
                dim=0,
            )

        x[DatasetEnum.PERTURBATION] = self.pert_transform(x[self.pert_source_col])
        return x


class EvalRowTransform:
    def __init__(self, tf_one: TFOne, decoder: DecodeImage, pert_transform, pert_source_col: str) -> None:
        self.tf_one = tf_one
        self.decoder = decoder
        self.pert_transform = pert_transform
        self.pert_source_col = str(pert_source_col)

    def __call__(self, x: dict) -> dict:
        if DatasetEnum.IMG in x:
            imgs = x[DatasetEnum.IMG]
            if isinstance(imgs, (bytes, bytearray, memoryview)):
                imgs = [imgs]
            elif not isinstance(imgs, (list, tuple)):
                raise TypeError(f"Expected images as bytes-like or list/tuple, got {type(imgs)}")

            x[DatasetEnum.IMG] = torch.stack(
                [self.tf_one(self.decoder(_to_bytes_one_image(v)), train=False) for v in imgs],
                dim=0,
            )

        x[DatasetEnum.PERTURBATION] = self.pert_transform(x[self.pert_source_col])
        return x


class PertTransformList:
    """Keep perturbation identifiers as a plain Python list[str]. Picklable callable."""

    def __call__(self, perturbations: Sequence[str]) -> list[str]:
        return list(perturbations)


# Backward-compatible alias
MolTransformList = PertTransformList


class PertTransformPrecomputed:
    """Convert stored vectors into a torch.Tensor (single vector or batch). Picklable callable."""

    def __call__(self, rows: object) -> torch.Tensor:
        if isinstance(rows, torch.Tensor):
            return rows.to(dtype=torch.float32)
        arr = np.asarray(rows, dtype=np.float32)
        return torch.from_numpy(arr)


# Backward-compatible alias
MolTransformPrecomputed = PertTransformPrecomputed


class _PertTransformRandom:
    """Returns zero placeholders; RandomMoleculeEncoder ignores content."""

    def __call__(self, v: object) -> list[torch.Tensor]:
        if isinstance(v, (list, tuple)):
            n = len(v)
        elif isinstance(v, torch.Tensor):
            n = v.shape[0]
        else:
            n = 1
        return [torch.zeros(1) for _ in range(n)]


# Backward-compatible alias
_MolTransformRandom = _PertTransformRandom


def _build_pert_transform(perturbation_encoder_name: str) -> Callable[[object], object]:
    """Return a picklable callable for perturbation preprocessing.

    Name-based dispatch is intentional: the transform is built before encoder instantiation,
    so encoder.supports_strings is not accessible here.
    """
    name = perturbation_encoder_name.lower()
    if name in {"ecfp", "ecfp_cached", "gene_lookup", "whimf", "chemberta"}:
        return PertTransformList()
    if name == "precomputed":
        return PertTransformPrecomputed()
    if name == "random":
        return _PertTransformRandom()
    raise KeyError(f"Unknown perturbation encoder '{perturbation_encoder_name}'.")


# Backward-compatible alias
def _build_mol_transform(mol_encoder_name: str) -> Callable[[object], object]:
    return _build_pert_transform(mol_encoder_name)


class LightBCJitter(nn.Module):
    """Brightness/contrast jitter that supports any channel count (e.g., 5-channel images)."""

    def __init__(self, brightness: float = 0.1, contrast: float = 0.1):
        super().__init__()
        self.b = float(brightness)
        self.c = float(contrast)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure float for jitter math; keep on same device.
        if not torch.is_floating_point(x):
            x = x.float().div_(255.0)  # preserve original scale expectation (0-1 after ToImage)

        if self.b > 0:
            b_fac = 1.0 + (torch.rand(1, device=x.device, dtype=x.dtype) * 2 * self.b - self.b)
            x = x * b_fac
        if self.c > 0:
            mean = x.mean(dim=(-2, -1), keepdim=True)
            c_fac = 1.0 + (torch.rand(1, device=x.device, dtype=x.dtype) * 2 * self.c - self.c)
            x = (x - mean) * c_fac + mean
        # Keep downstream expectations: stay float32 to avoid later uint8 issues.
        return x.to(dtype=torch.float32)


AUGMENT_T = T.Compose(
    [
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=[0, 90], interpolation=T.InterpolationMode.BILINEAR),
        LightBCJitter(brightness=0.1, contrast=0.1),
    ]
)


class PerturbationBatchSampler:
    """
    Build batches with a controlled number of distinct perturbations.

    For each batch:
      - pick `perturbations_per_batch` perturbation-classes
      - sample `samples_per_perturbation` indices per chosen perturbation
      - distribute any remainder (+1) across random chosen perturbations

    perturbation_sampling:
      - "iid": each batch picks perturbations independently at random
      - "cycle": shuffle perturbations, then take consecutive blocks (better coverage)
    """

    def __init__(
        self,
        compound_ids: np.ndarray,
        *,
        batch_size: int,
        perturbations_per_batch: int,
        samples_per_perturbation: int | None = None,
        generator: torch.Generator | None = None,
        drop_last: bool = False,
        sample_perturbations_with_replacement: bool = False,
        sample_rows_with_replacement: bool = False,
        perturbation_sampling: str = "cycle",
        pinned_perturbation_ids: np.ndarray | None = None,
        row_strata: np.ndarray | None = None,
    ) -> None:
        self.compound_ids = np.asarray(compound_ids, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.perturbations_per_batch = int(perturbations_per_batch)
        self.drop_last = bool(drop_last)
        self.gen = generator or torch.Generator()

        self.sample_perturbations_with_replacement = bool(sample_perturbations_with_replacement)
        self.sample_rows_with_replacement = bool(sample_rows_with_replacement)

        if perturbation_sampling not in {"iid", "cycle"}:
            raise ValueError(f'perturbation_sampling must be one of {{"iid", "cycle"}}, got {perturbation_sampling}')
        self.perturbation_sampling = perturbation_sampling

        if self.compound_ids.ndim != 1:
            raise ValueError(f"compound_ids must be 1D, got {self.compound_ids.shape}")

        uniq, inv = np.unique(self.compound_ids, return_inverse=True)
        self._uniq_perturbations = uniq.astype(np.int64)
        self._perturbation_class = inv.astype(np.int64)
        self.n_perturbations = int(self._uniq_perturbations.size)

        # Resolve pinned perturbation IDs → internal class indices (0..n_perturbations-1).
        id_to_class = {int(cid): i for i, cid in enumerate(self._uniq_perturbations.tolist())}
        if pinned_perturbation_ids is not None and len(pinned_perturbation_ids) > 0:
            resolved = [id_to_class[int(pid)] for pid in pinned_perturbation_ids if int(pid) in id_to_class]
            missing = len(pinned_perturbation_ids) - len(resolved)
            if missing:
                log.warning(f"{missing} pinned perturbation(s) not in training set — skipped.")
            self._pinned_classes: list[int] = resolved
        else:
            self._pinned_classes = []

        if self.n_perturbations == 0:
            raise ValueError("No perturbations found.")
        n_random_slots = self.perturbations_per_batch - len(self._pinned_classes)
        if n_random_slots < 1:
            raise ValueError(
                f"perturbations_per_batch ({self.perturbations_per_batch}) must exceed "
                f"number of pinned perturbations ({len(self._pinned_classes)})."
            )
        if (not self.sample_perturbations_with_replacement) and n_random_slots > self.n_perturbations - len(self._pinned_classes):
            raise ValueError(
                f"perturbations_per_batch ({self.perturbations_per_batch}) > n_perturbations ({self.n_perturbations}). "
                "Lower perturbations_per_batch or enable sample_perturbations_with_replacement=True."
            )

        if samples_per_perturbation is None:
            base = self.batch_size // self.perturbations_per_batch
            if base < 1:
                raise ValueError(
                    f"batch_size ({self.batch_size}) too small for perturbations_per_batch ({self.perturbations_per_batch})."
                )
            self.samples_per_perturbation = int(base)
        else:
            self.samples_per_perturbation = int(samples_per_perturbation)
            if self.samples_per_perturbation < 1:
                raise ValueError("samples_per_perturbation must be >= 1.")

        self._base_quota = self.samples_per_perturbation
        self._remainder = self.batch_size - (self.perturbations_per_batch * self._base_quota)
        if self._remainder < 0:
            raise ValueError(
                f"perturbations_per_batch*samples_per_perturbation = {self.perturbations_per_batch * self._base_quota} "
                f"exceeds batch_size ({self.batch_size})."
            )

        self._pools: list[np.ndarray] = [np.nonzero(self._perturbation_class == c)[0] for c in range(self.n_perturbations)]
        if any(p.size == 0 for p in self._pools):
            raise RuntimeError("Internal error: found empty pool for a perturbation class.")

        # for "cycle" mode
        self._perm: torch.Tensor | None = None
        self._perm_pos: int = 0

        if row_strata is not None:
            rs = np.asarray(row_strata, dtype=np.int64)
            strata_pools: list[dict[int, np.ndarray]] = []
            for pool in self._pools:
                s = rs[pool]
                strata_pools.append({int(u): pool[s == u] for u in np.unique(s)})
            self._strata_pools: list[dict[int, np.ndarray]] | None = strata_pools
        else:
            self._strata_pools = None

    def __len__(self) -> int:
        n = int(self.compound_ids.shape[0])
        return n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size

    def _pick_perturbations(self) -> list[int]:
        n_pinned = len(self._pinned_classes)
        n_random = self.perturbations_per_batch - n_pinned
        non_pinned = [c for c in range(self.n_perturbations) if c not in self._pinned_classes]

        if self.perturbation_sampling == "iid":
            if self.sample_perturbations_with_replacement:
                random_picks = torch.randint(0, len(non_pinned), (n_random,), generator=self.gen).tolist()
            else:
                random_picks = torch.randperm(len(non_pinned), generator=self.gen)[:n_random].tolist()
            return self._pinned_classes + [non_pinned[i] for i in random_picks]

        # "cycle": shuffle once we run out, then take the next block
        if (self._perm is None) or (self._perm_pos + n_random > len(non_pinned)):
            self._perm = torch.randperm(len(non_pinned), generator=self.gen)
            self._perm_pos = 0
        block = self._perm[self._perm_pos : self._perm_pos + n_random]
        self._perm_pos += n_random
        return self._pinned_classes + [non_pinned[i] for i in block.tolist()]

    def _sample_from_pool(self, pool: np.ndarray, q: int) -> np.ndarray:
        if q <= 0:
            return np.empty((0,), dtype=np.int64)
        if (not self.sample_rows_with_replacement) and pool.size >= q:
            sel = torch.randperm(pool.size, generator=self.gen)[:q].numpy()
            return pool[sel]
        sel = torch.randint(0, pool.size, (q,), generator=self.gen).numpy()
        return pool[sel]

    def _sample_stratified(self, strata_dict: dict[int, np.ndarray], q: int) -> np.ndarray:
        strata = list(strata_dict.keys())
        n_s = len(strata)
        base, extra = divmod(q, n_s)
        order = torch.randperm(n_s, generator=self.gen).tolist()
        parts: list[np.ndarray] = []
        for rank, i in enumerate(order):
            quota = base + (1 if rank < extra else 0)
            if quota:
                parts.append(self._sample_from_pool(strata_dict[strata[i]], quota))
        return np.concatenate(parts) if parts else np.empty((0,), dtype=np.int64)

    def __iter__(self):
        for _ in range(len(self)):
            comp_choices = self._pick_perturbations()

            quotas = [self._base_quota] * self.perturbations_per_batch
            if self._remainder:
                order = torch.randperm(self.perturbations_per_batch, generator=self.gen).tolist()
                for i in order[: self._remainder]:
                    quotas[i] += 1

            batch_idx: list[int] = []
            for c, q in zip(comp_choices, quotas, strict=True):
                if self._strata_pools is not None:
                    picked = self._sample_stratified(self._strata_pools[int(c)], int(q))
                else:
                    picked = self._sample_from_pool(self._pools[int(c)], int(q))
                batch_idx.extend(picked.tolist())

            yield batch_idx


class EqualPerClassBatchSampler:
    """
    Per-batch equal class representation for a single label array.
    Uses all classes each batch, with replacement.
    """

    def __init__(
        self,
        labels: np.ndarray,
        batch_size: int,
        generator: torch.Generator | None = None,
        drop_last: bool = False,
        seed: int | None = None,
    ) -> None:
        self.labels = np.asarray(labels)
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        if generator is not None:
            self.gen = generator
        else:
            g = torch.Generator()
            if seed is not None:
                g.manual_seed(int(seed))
            self.gen = g

        classes, inv = np.unique(self.labels, return_inverse=True)
        self.class_ids = inv.astype(np.int64)  # 0..k-1 per sample
        self.k = int(classes.size)
        if self.k == 0:
            raise ValueError("No classes for stratification.")
        if self.batch_size < self.k:
            raise ValueError(f"batch_size ({self.batch_size}) < n_classes ({self.k}); cannot equalize per batch.")

        self.pools: list[np.ndarray] = [np.nonzero(self.class_ids == c)[0] for c in range(self.k)]
        base = self.batch_size // self.k
        r = self.batch_size - base * self.k
        self.base_quota = np.full(self.k, base, dtype=int)
        self.remainder = int(r)
        # self.gen.manual_seed(int(torch.seed()))

    def __iter__(self):
        n = int(self.labels.shape[0])
        num_batches = n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size
        for _ in range(num_batches):
            q = self.base_quota.copy()
            if self.remainder:
                order = torch.randperm(self.k, generator=self.gen).tolist()
                for i in order[: self.remainder]:
                    q[i] += 1

            batch_idx: list[int] = []
            for c in range(self.k):
                take = int(q[c])
                if take <= 0:
                    continue
                pool = self.pools[c]
                sel = torch.randint(0, pool.size, (take,), generator=self.gen).tolist()
                batch_idx.extend(pool[sel].tolist())
            yield batch_idx

    def __len__(self) -> int:
        n = int(self.labels.shape[0])
        return n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size


class ImageMoleculeDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_path: str | Path,
        image_encoder_name: str,
        perturbation_encoder_name: str,
        perturbation_embedding_column: str | None = None,
        perturbation_source_col: str | None = None,
        batch_size: int = 256,
        num_workers: int = 8,
        prefetch_factor: int = 1,
        seed: int = 42,
        augment: bool = True,
        split_by_column: str = "batch",
        test_frac: float = 0.15,
        val_frac: float = 0.15,
        *,
        in_channels: int = 5,
        image_size: int = 224,
        data_image_size: int | None = None,
        always_train_groups: Sequence[str] | None = None,
        stratify_by_column: str | None = "Metadata_Source",
        split_stratify_by: str | None = "maha_significant",
        perturbations_per_batch: int | None = 63,  # adjusted for batch size of 756
        samples_per_perturbation: int | None = None,
        sample_perturbations_with_replacement: bool = False,
        perturbation_exclude_values: Sequence[str] | None = None,
        valid_perturbation_keys_path: str = "",
        control_perturbation_keys: Sequence[str] | None = None,
        perturbation_stratify_by_column: str | None = None,
        # Hydra config-level metadata consumed by the encoder registry via interpolation.
        # Declared here so Hydra doesn't raise on unknown keys.
        channel_names: list[str] | None = None,  # noqa: ARG002
        # Legacy parameter names kept for backward compatibility
        mol_encoder_name: str | None = None,
        molecule_embedding_column: str | None = None,
        mol_source_col: str | None = None,
    ):
        super().__init__()

        # Support legacy parameter names (remove after all callers updated)
        if mol_encoder_name is not None:
            perturbation_encoder_name = mol_encoder_name
        if molecule_embedding_column is not None and perturbation_embedding_column is None:
            perturbation_embedding_column = molecule_embedding_column
        if mol_source_col is not None and perturbation_source_col is None:
            perturbation_source_col = mol_source_col

        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.prefetch_factor = int(prefetch_factor)
        self.dataset_path = str(dataset_path)
        self.seed = int(seed)
        self.augment = bool(augment)

        self.perturbations_per_batch = int(perturbations_per_batch) if perturbations_per_batch is not None else None
        self.samples_per_perturbation = int(samples_per_perturbation) if samples_per_perturbation is not None else None
        self.sample_perturbations_with_replacement = bool(sample_perturbations_with_replacement)
        self._perturbation_exclude: set[str] = set(perturbation_exclude_values) if perturbation_exclude_values else set()
        self._control_perturbation_keys: set[str] = set(control_perturbation_keys) if control_perturbation_keys else set()
        self._valid_perturbation_keys: set[str] | None = None
        if valid_perturbation_keys_path:
            p = Path(valid_perturbation_keys_path)
            if not p.exists():
                raise FileNotFoundError(f"valid_perturbation_keys_path not found: {p}")
            self._valid_perturbation_keys = set(p.read_text().splitlines())
        self._pinned_perturbation_ids: np.ndarray | None = None

        self._train_labels_for_strata: np.ndarray | None = None
        self._train_perturbation_ids: np.ndarray | None = None
        self._train_row_strata: np.ndarray | None = None
        self.perturbation_stratify_by_column = perturbation_stratify_by_column

        self.split_by_column = str(split_by_column)
        self.test_frac = float(test_frac)
        self.val_frac = float(val_frac)
        if not (0 <= self.test_frac < 1) or not (0 <= self.val_frac < 1):
            raise ValueError("test_frac and val_frac must be in [0, 1).")
        if self.test_frac + self.val_frac >= 1:
            raise ValueError("test_frac + val_frac must be < 1.")

        self.always_train_groups = list(always_train_groups) if always_train_groups else []
        self.stratify_by_column = stratify_by_column
        self.split_stratify_by = split_stratify_by

        self._in_channels = int(in_channels)
        self._image_size = int(image_size)
        self._data_image_size = int(data_image_size) if data_image_size is not None else self._image_size
        self.dataset: Dataset = load_from_disk(self.dataset_path)

        pert_name = perturbation_encoder_name.lower()
        self._is_random_pert = pert_name == "random"

        if perturbation_source_col is not None:
            # Explicit override: caller controls the source column (e.g. siRNA_ID for RxRx1).
            # precomputed validation is intentionally skipped when this is set.
            self._pert_source_col = str(perturbation_source_col)
        elif pert_name == "precomputed":
            if not perturbation_embedding_column:
                raise ValueError(
                    'perturbation_encoder_name="precomputed" requires perturbation_embedding_column (e.g. "ecfp4_2048").'
                )
            self._pert_source_col = str(perturbation_embedding_column)
        elif pert_name == "random":
            # compound_id is always B-sized and present after the add_column below;
            # RandomMoleculeEncoder ignores the values entirely.
            self._pert_source_col = "compound_id"
        else:
            self._pert_source_col = DatasetEnum.PERTURBATION.value  # "perturbation"

        self._pert_encoder_name = pert_name

        if pert_name == "random":
            core_cols: list[str] = [DatasetEnum.IMG]
            if self.stratify_by_column and self.stratify_by_column in self.dataset.column_names:
                core_cols.append(self.stratify_by_column)
        else:
            core_cols = [DatasetEnum.IMG, self._pert_source_col, self.split_by_column]
            if self.stratify_by_column:
                core_cols.append(self.stratify_by_column)
            # always_train_groups pins rows by perturbation value regardless of encoder type
            if self.always_train_groups and DatasetEnum.PERTURBATION not in core_cols:
                core_cols.append(DatasetEnum.PERTURBATION)

        base_keep_cols = sorted(set(core_cols))
        missing = [c for c in base_keep_cols if c not in self.dataset.column_names]
        if missing:
            raise KeyError(f"Missing required columns in dataset: {missing}. Available: {self.dataset.column_names}")

        to_drop = [c for c in self.dataset.column_names if c not in base_keep_cols]
        if to_drop:
            self.dataset = self.dataset.remove_columns(to_drop)

        if (self._perturbation_exclude or self._valid_perturbation_keys is not None) and pert_name != "random":
            # Vectorised filter: pull the column once, build a boolean mask, select by index.
            # Much faster than Dataset.filter(lambda row: ...) which calls Python per row.
            col = np.asarray(self.dataset[self._pert_source_col])
            mask = np.ones(len(col), dtype=bool)
            if self._perturbation_exclude:
                mask &= ~np.isin(col, list(self._perturbation_exclude))
            if self._valid_perturbation_keys is not None:
                allowed = self._valid_perturbation_keys | self._control_perturbation_keys
                mask &= np.isin(col, list(allowed))
            indices = np.where(mask)[0]
            dropped = len(mask) - len(indices)
            if dropped > 0:
                self.dataset = self.dataset.select(indices.tolist())
                log.info(f"Perturbation filter: dropped {dropped:,} rows ({len(self.dataset):,} remain)")
            else:
                log.info("Perturbation filter: no rows dropped (dataset already clean)")

        if "compound_id" not in self.dataset.column_names:
            if pert_name == "random":
                self.dataset = self.dataset.add_column("compound_id", range(len(self.dataset)))
            else:
                col = np.asarray(self.dataset[self._pert_source_col])
                _, inv = np.unique(col, return_inverse=True)
                self.dataset = self.dataset.add_column("compound_id", inv.astype(np.int64).tolist())

        if self._control_perturbation_keys and pert_name != "random":
            col = np.asarray(self.dataset[self._pert_source_col])
            cids = np.asarray(self.dataset["compound_id"], dtype=np.int64)
            pinned = []
            for key in self._control_perturbation_keys:
                mask = col == key
                if mask.any():
                    pinned.append(int(cids[np.where(mask)[0][0]]))
                else:
                    log.warning(f"control_perturbation_key {key!r} not found in dataset — skipped.")
            self._pinned_perturbation_ids = np.array(pinned, dtype=np.int64) if pinned else None

        keep_cols = sorted(set(base_keep_cols + ["compound_id"]))
        self.dataset.set_format(type=None, columns=keep_cols)

        log.info(f"Columns kept: {keep_cols} | dropped: {len(to_drop)} (of {len(to_drop) + len(keep_cols)})")

        try:
            base_tf = ImageEncoderRegistry.default_transform(image_encoder_name)
        except KeyError as e:
            raise KeyError(
                f'Unknown image encoder "{image_encoder_name}". Known: {ImageEncoderRegistry.list_names()}'
            ) from e

        self.pert_transform = _build_pert_transform(perturbation_encoder_name)

        decoder = DecodeImage(C=self._in_channels, H=self._data_image_size, W=self._data_image_size)
        if self._data_image_size != self._image_size:
            base_tf = T.Compose([T.Resize((self._image_size, self._image_size), antialias=True), base_tf])
        tf_one = TFOne(base_tf, self.augment)
        self.train_transform = TrainRowTransform(tf_one, decoder, self.pert_transform, self._pert_source_col)
        self.eval_transform = EvalRowTransform(tf_one, decoder, self.pert_transform, self._pert_source_col)

    def _cache_train_compound_fps_log1p(self) -> None:
        """Cache one fingerprint vector per unique training compound.

        Only meaningful when perturbation_encoder_name="precomputed".
        """
        self.train_compound_ids_unique: np.ndarray | None = None
        self.train_compound_fps_log1p: np.ndarray | None = None
        self.train_compound_fps_col: str | None = None

        if self._pert_encoder_name != "precomputed":
            return

        # Pull compound_id for all train rows (ints only; ok to materialize).
        comp_ids = np.asarray(self.train_dataset["compound_id"], dtype=np.int64)
        if comp_ids.size == 0:
            return

        # First occurrence per compound (fast, no full fingerprint materialization).
        uniq, first_pos = np.unique(comp_ids, return_index=True)

        # Fetch only those rows (small) and only the fingerprint column.
        subset = self.train_dataset.select(first_pos.tolist())
        fps_list = subset[self._pert_source_col]

        fps = np.stack([np.asarray(v, dtype=np.float32) for v in fps_list], axis=0)

        self.train_compound_ids_unique = uniq.astype(np.int64, copy=False)
        self.train_compound_fps_log1p = fps
        self.train_compound_fps_col = str(self._pert_source_col)

        log.info(
            f"Cached train perturbation fps: n_perturbations={fps.shape[0]}, dim={fps.shape[1]}, col={self.train_compound_fps_col}"
        )

    # ----- split helpers -----

    def _ensure_split_cache(self) -> None:
        if not hasattr(self, "_splits_ready"):
            self._splits_ready = False
            self._idx_train = self._idx_val = self._idx_test = None
        if not hasattr(self, "_split_values_raw"):
            if self._is_random_pert:
                self._split_values_raw = np.arange(len(self.dataset))
            else:
                self._split_values_raw = np.asarray(self.dataset[self.split_by_column])

    def _groupwise_split(self):
        self._ensure_split_cache()

        if self._is_random_pert:
            n = len(self.dataset)
            return np.arange(n), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

        # Rows with always_train_groups perturbation values go to train unconditionally.
        # This decouples control-well pinning from the batch-level split column.
        if self.always_train_groups and DatasetEnum.PERTURBATION in self.dataset.column_names:
            pert_col = np.asarray(self.dataset[DatasetEnum.PERTURBATION])
            always_train_mask = np.isin(pert_col, list(self.always_train_groups))
        else:
            always_train_mask = np.zeros(len(self.dataset), dtype=bool)

        forced_idx = np.where(always_train_mask)[0]
        splittable_idx = np.where(~always_train_mask)[0]

        col = self._split_values_raw[splittable_idx]
        uniq_groups = np.unique(col)
        n_groups = len(uniq_groups)
        if n_groups == 0 and len(splittable_idx) > 0:
            raise RuntimeError(f'No groups found in column "{self.split_by_column}".')

        n_test = int(round(self.test_frac * n_groups))
        n_val = int(round(self.val_frac * n_groups))
        pool = np.array(uniq_groups)
        rng = np.random.default_rng(self.seed)
        rng.shuffle(pool)
        test_groups = set(pool[:n_test])
        val_groups = set(pool[n_test : n_test + n_val])
        train_groups = set(pool[n_test + n_val :].tolist())

        is_test = np.isin(col, list(test_groups))
        is_val = np.isin(col, list(val_groups))
        is_train = ~(is_test | is_val)

        idx_train = np.concatenate([forced_idx, splittable_idx[np.nonzero(is_train)[0]]])
        idx_val = splittable_idx[np.nonzero(is_val)[0]]
        idx_test = splittable_idx[np.nonzero(is_test)[0]]

        def desc(name: str, rows: int, groups: int):
            log.info(f"{name:>5} | rows={rows:,} | groups={groups:,} (~{groups / max(n_groups, 1):.1%} of {n_groups:,})")

        log.info(
            f'Group-wise split by "{self.split_by_column}": '
            f"target test_frac={self.test_frac}, val_frac={self.val_frac}, seed={self.seed}"
        )
        if len(forced_idx):
            log.info(f"Forced into TRAIN by always_train_groups: {len(forced_idx):,} rows")
        desc("TEST", len(idx_test), len(test_groups))
        desc(" VAL", len(idx_val), len(val_groups))
        desc("TRAIN", len(idx_train), len(train_groups))

        if test_groups:
            log.info(f"TEST  groups: {sorted(test_groups)}")
        if val_groups:
            log.info(f"VAL   groups: {sorted(val_groups)}")
        log.info(f"TRAIN groups: {sorted(train_groups)}")

        return idx_train, idx_val, idx_test

    def setup(self, stage: str | None = None):
        self._ensure_split_cache()
        if not self._splits_ready:
            self._idx_train, self._idx_val, self._idx_test = self._groupwise_split()
            self._splits_ready = True

        self.train_dataset = self.dataset.select(self._idx_train)
        self.val_dataset = self.dataset.select(self._idx_val)
        self.test_dataset = self.dataset.select(self._idx_test)

        if self.stratify_by_column and self.stratify_by_column in self.train_dataset.column_names:
            self._train_labels_for_strata = np.asarray(self.train_dataset[self.stratify_by_column])
        else:
            self._train_labels_for_strata = None

        self._train_perturbation_ids = np.asarray(self.train_dataset["compound_id"], dtype=np.int64)

        if (
            self.perturbation_stratify_by_column
            and self.perturbation_stratify_by_column in self.train_dataset.column_names
        ):
            raw = np.asarray(self.train_dataset[self.perturbation_stratify_by_column])
            _, self._train_row_strata = np.unique(raw, return_inverse=True)
            log.info(
                f"Perturbation batch stratification by '{self.perturbation_stratify_by_column}': "
                f"{len(np.unique(self._train_row_strata))} strata"
            )
        else:
            self._train_row_strata = None

        # NEW: cache one fingerprint per unique training compound (only if precomputed fingerprints exist)
        self._cache_train_compound_fps_log1p()

        self.train_dataset.set_transform(self.train_transform)
        self.val_dataset.set_transform(self.eval_transform)
        self.test_dataset.set_transform(self.eval_transform)

        log.info("Transforms attached: train/val/test datasets are ready.")

    # ---------- loaders ----------

    # def _build_train_loader(self) -> DataLoader:
    #     if not self.stratify_by_column:
    #         return DataLoader(
    #             self.train_dataset,
    #             batch_size=self.batch_size,
    #             num_workers=self.num_workers,
    #             shuffle=True,
    #             drop_last=False,
    #             persistent_workers=(self.num_workers > 0),
    #             pin_memory=True,
    #             prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
    #             worker_init_fn=_worker_init,
    #         )

    #     labels = self._train_labels_for_strata
    #     assert labels is not None, "Internal: labels cache missing; setup() must run before dataloaders"

    #     # exact per-batch equal class representation over sources
    #     batch_sampler = EqualPerClassBatchSampler(
    #         labels=labels,
    #         batch_size=self.batch_size,
    #         generator=torch.Generator().manual_seed(self.seed),
    #         drop_last=False,
    #     )
    #     log.info(
    #         f'Using EqualPerClassBatchSampler over "{self.stratify_by_column}" '
    #         f"(classes={int(np.unique(labels).size)}; batch_size={self.batch_size})."
    #     )
    #     return DataLoader(
    #         self.train_dataset,
    #         batch_sampler=batch_sampler,
    #         num_workers=self.num_workers,
    #         drop_last=False,
    #         persistent_workers=(self.num_workers > 0),
    #         pin_memory=True,
    #         prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
    #         worker_init_fn=_worker_init,
    #     )

    # def train_dataloader(self):
    #     return self._build_train_loader()

    # def val_dataloader(self):
    #     return DataLoader(
    #         self.val_dataset,
    #         batch_size=self.batch_size,
    #         num_workers=self.num_workers,
    #         shuffle=False,
    #         drop_last=False,
    #         persistent_workers=(self.num_workers > 0),
    #         pin_memory=True,
    #         prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
    #         worker_init_fn=_worker_init,
    #     )

    # def test_dataloader(self):
    #     return DataLoader(
    #         self.test_dataset,
    #         batch_size=self.batch_size,
    #         num_workers=self.num_workers,
    #         shuffle=False,
    #         drop_last=False,
    #         persistent_workers=(self.num_workers > 0),
    #         pin_memory=True,
    #         prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
    #         worker_init_fn=_worker_init,
    #     )

    # def predict_dataloader(self):
    #     return DataLoader(
    #         self.test_dataset,
    #         batch_size=1,
    #         num_workers=self.num_workers,
    #         shuffle=False,
    #         drop_last=False,
    #     )
    def _build_train_loader(self) -> DataLoader:
        if self.perturbations_per_batch is not None:
            assert self._train_perturbation_ids is not None, "Internal: setup() must run before dataloaders"

            # Optional knobs (set as attributes or just edit these defaults)
            perturbation_sampling = getattr(self, "perturbation_sampling", "cycle")  # "cycle" or "iid"
            sample_rows_with_replacement = getattr(self, "sample_rows_with_replacement", False)

            batch_sampler = PerturbationBatchSampler(
                self._train_perturbation_ids,
                batch_size=self.batch_size,
                perturbations_per_batch=self.perturbations_per_batch,
                samples_per_perturbation=self.samples_per_perturbation,
                generator=torch.Generator().manual_seed(self.seed),
                drop_last=False,
                sample_perturbations_with_replacement=self.sample_perturbations_with_replacement,
                sample_rows_with_replacement=sample_rows_with_replacement,
                perturbation_sampling=perturbation_sampling,
                pinned_perturbation_ids=self._pinned_perturbation_ids,
                row_strata=self._train_row_strata,
            )

            log.info(
                f"Using PerturbationBatchSampler: batch_size={self.batch_size}, "
                f"perturbations_per_batch={self.perturbations_per_batch}, "
                f"samples_per_perturbation={batch_sampler.samples_per_perturbation}, "
                f"remainder={batch_sampler._remainder}, "
                f"perturbation_sampling={perturbation_sampling}, "
                f"sample_rows_with_replacement={sample_rows_with_replacement}, "
                f"row_strata={'enabled' if self._train_row_strata is not None else 'disabled'}."
            )

            return DataLoader(
                self.train_dataset,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                drop_last=False,
                persistent_workers=(self.num_workers > 0),
                pin_memory=True,
                prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
                worker_init_fn=_worker_init,
                multiprocessing_context=_MULTI_PROC_CONTEXT if self.num_workers > 0 else None,
            )

        # Otherwise fall back to your existing source-equalization behavior (can reduce positives).
        if not self.stratify_by_column:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=True,
                drop_last=False,
                persistent_workers=(self.num_workers > 0),
                pin_memory=True,
                prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
                worker_init_fn=_worker_init,
                multiprocessing_context=_MULTI_PROC_CONTEXT if self.num_workers > 0 else None,
            )

        labels = self._train_labels_for_strata
        assert labels is not None, "Internal: labels cache missing; setup() must run before dataloaders"

        batch_sampler = EqualPerClassBatchSampler(
            labels=labels,
            batch_size=self.batch_size,
            seed=self.seed,
            drop_last=False,
        )

        return DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
            drop_last=False,
            persistent_workers=(self.num_workers > 0),
            pin_memory=True,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            worker_init_fn=_worker_init,
            multiprocessing_context=_MULTI_PROC_CONTEXT if self.num_workers > 0 else None,
        )

    def train_dataloader(self):
        return self._build_train_loader()

    def val_dataloader(self):
        # If val split is disabled (val_frac=0), return an empty list to disable val loop cleanly.
        if not hasattr(self, "val_dataset") or len(self.val_dataset) == 0:
            return []

        nw = min(self.num_workers, 4)
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=nw,
            shuffle=False,
            drop_last=False,
            persistent_workers=(nw > 0),
            pin_memory=True,
            prefetch_factor=self.prefetch_factor if nw > 0 else None,
            worker_init_fn=_worker_init,
            multiprocessing_context=_MULTI_PROC_CONTEXT if nw > 0 else None,
        )

    # def val_dataloader(self):
    #     if len(self.val_dataset) == 0:
    #         return None
    #     # validation does not need 16 workers; reduce startup overhead
    #     nw = min(self.num_workers, 4)
    #     return DataLoader(
    #         self.val_dataset,
    #         batch_size=self.batch_size,
    #         num_workers=nw,
    #         shuffle=False,
    #         drop_last=False,
    #         persistent_workers=(nw > 0),
    #         pin_memory=True,
    #         prefetch_factor=self.prefetch_factor if nw > 0 else None,
    #         worker_init_fn=_worker_init,
    #         multiprocessing_context=_MULTI_PROC_CONTEXT,
    #     )

    def test_dataloader(self):
        if not hasattr(self, "test_dataset") or len(self.test_dataset) == 0:
            return []

        nw = min(self.num_workers, 4)
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=nw,
            shuffle=False,
            drop_last=False,
            persistent_workers=(nw > 0),
            pin_memory=True,
            prefetch_factor=self.prefetch_factor if nw > 0 else None,
            worker_init_fn=_worker_init,
            multiprocessing_context=_MULTI_PROC_CONTEXT if nw > 0 else None,
        )

    def predict_dataloader(self):
        nw = min(self.num_workers, 4)
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=nw,
            shuffle=False,
            drop_last=False,
            persistent_workers=(nw > 0),
            pin_memory=True,
            prefetch_factor=self.prefetch_factor if nw > 0 else None,
            worker_init_fn=_worker_init,
            multiprocessing_context=_MULTI_PROC_CONTEXT if nw > 0 else None,
        )
