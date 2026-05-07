"""Tests for :mod:`cp_bg_bench.io.jump` (rule A building blocks)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from cp_bg_bench import io as io_pkg  # noqa: F401  — ensures io package imports cleanly
from cp_bg_bench.config import JumpSample
from cp_bg_bench.io import jump as jump_mod
from cp_bg_bench.io.jump import (
    JumpS3Error,
    assign_snakemake_batches,
    download_metadata_tables,
    expand_to_fovs,
    fetch_load_data_per_plate,
    load_metadata,
    parse_s3_url,
    required_output_columns,
    resolve_metadata,
    select_samples,
)

# --------- synthetic fixtures -----------------------------------------------


def _make_meta() -> pd.DataFrame:
    """Merged plate+well+compound table, six wells across two sources."""
    return pd.DataFrame(
        {
            "Metadata_Source": ["source_2"] * 3 + ["source_4"] * 3,
            "Metadata_Batch": ["B1", "B1", "B2", "B1", "B1", "B1"],
            "Metadata_Plate": ["P1", "P1", "P2", "P3", "P3", "P3"],
            "Metadata_Well": ["A01", "A02", "B01", "A01", "A02", "A03"],
            "Metadata_PlateType": ["target1"] * 6,
            "Metadata_InChIKey": [f"K{i}" for i in range(6)],
            "Metadata_InChI": [f"InChI={i}" for i in range(6)],
            "Metadata_JCP2022": [f"JCP{i}" for i in range(6)],
            "Metadata_SMILES": [f"C{i}" for i in range(6)],
        }
    )


def _make_load_data(
    source: str, batch: str, plate: str, wells: list[str], sites: list[str]
) -> pd.DataFrame:
    """Synthetic JUMP ``load_data_with_illum`` CSV: one ``URL_Orig*`` per channel."""
    rows = []
    for well in wells:
        for site in sites:
            row: dict[str, Any] = {
                "Metadata_Source": source,
                "Metadata_Batch": batch,
                "Metadata_Plate": plate,
                "Metadata_Well": well,
                "Metadata_Site": site,
            }
            for channel in ("DNA", "AGP", "ER", "Mito", "RNA"):
                row[f"URL_Orig{channel}"] = f"s3://bucket/{plate}/{well}/{channel}_{site}.tiff"
            rows.append(row)
    return pd.DataFrame(rows)


CHANNEL_KEYS = ["s3_OrigDNA", "s3_OrigAGP", "s3_OrigER", "s3_OrigMito", "s3_OrigRNA"]


# --------- select_samples ---------------------------------------------------


def test_select_samples_and_within_entry() -> None:
    meta = _make_meta()
    result = select_samples(
        meta,
        [JumpSample(metadata_source="source_2", metadata_plate="P1")],
    )
    assert set(result["Metadata_Plate"]) == {"P1"}
    assert set(result["Metadata_Source"]) == {"source_2"}
    assert len(result) == 2  # A01, A02


def test_select_samples_or_across_entries() -> None:
    meta = _make_meta()
    result = select_samples(
        meta,
        [
            JumpSample(metadata_source="source_2", metadata_plate="P2"),
            JumpSample(metadata_source="source_4"),
        ],
    )
    # OR: all of source_4 plus P2 row from source_2.
    assert len(result) == 4
    assert set(result["Metadata_Source"]) == {"source_2", "source_4"}


def test_select_samples_deduplicates() -> None:
    meta = _make_meta()
    result = select_samples(
        meta,
        [
            JumpSample(metadata_source="source_2"),
            JumpSample(metadata_source="source_2", metadata_plate="P1"),
        ],
    )
    # Union of (all source_2) and (source_2 plate=P1) is just all source_2.
    assert len(result) == 3


def test_select_samples_dedup_on_well_key() -> None:
    """Deduplication uses the (source, batch, plate, well) key.

    Annotation columns like Metadata_InChIKey are allowed to drift without
    producing spurious FOV duplicates downstream. Here the same well appears
    twice with different annotations; the selector must collapse to one row.
    """
    meta = pd.DataFrame(
        {
            "Metadata_Source": ["source_2", "source_2"],
            "Metadata_Batch": ["B1", "B1"],
            "Metadata_Plate": ["P1", "P1"],
            "Metadata_Well": ["A01", "A01"],
            "Metadata_PlateType": ["COMPOUND", "COMPOUND"],
            "Metadata_InChIKey": ["K-a", "K-b"],
            "Metadata_InChI": ["I-a", "I-b"],
            "Metadata_JCP2022": ["JCPa", "JCPb"],
        }
    )
    result = select_samples(meta, [JumpSample(metadata_source="source_2")])
    assert len(result) == 1
    assert list(result["Metadata_Well"]) == ["A01"]


def test_select_samples_empty_raises() -> None:
    meta = _make_meta()
    with pytest.raises(ValueError, match="empty selection"):
        select_samples(
            meta,
            [JumpSample(metadata_source="source_does_not_exist")],
        )


def test_select_samples_field_not_in_columns_raises() -> None:
    meta = _make_meta().drop(columns=["Metadata_Plate"])
    with pytest.raises(KeyError, match="Metadata_Plate"):
        select_samples(
            meta,
            [JumpSample(metadata_source="source_2", metadata_plate="P1")],
        )


def test_select_samples_well_filter() -> None:
    meta = _make_meta()
    result = select_samples(
        meta,
        [JumpSample(metadata_source="source_4", metadata_plate="P3", metadata_well="A01")],
    )
    assert len(result) == 1
    assert result.iloc[0]["Metadata_Well"] == "A01"
    assert result.iloc[0]["Metadata_Plate"] == "P3"


def test_select_samples_well_or_across_entries() -> None:
    """Two entries selecting individual wells from different plates."""
    meta = _make_meta()
    result = select_samples(
        meta,
        [
            JumpSample(metadata_source="source_2", metadata_plate="P1", metadata_well="A01"),
            JumpSample(metadata_source="source_4", metadata_plate="P3", metadata_well="A03"),
        ],
    )
    assert len(result) == 2
    assert set(zip(result["Metadata_Plate"], result["Metadata_Well"], strict=True)) == {
        ("P1", "A01"),
        ("P3", "A03"),
    }


def test_select_samples_inchikey_filter() -> None:
    meta = _make_meta()
    # K2 appears in source_2 / P2 / B01 (row index 2 in _make_meta)
    result = select_samples(meta, [JumpSample(metadata_source="source_2", metadata_inchikey="K2")])
    assert len(result) == 1
    assert result.iloc[0]["Metadata_InChIKey"] == "K2"
    assert result.iloc[0]["Metadata_Plate"] == "P2"


def test_select_samples_jcp2022_filter() -> None:
    meta = _make_meta()
    # JCP3 appears in source_4 / P3 / A01
    result = select_samples(meta, [JumpSample(metadata_source="source_4", metadata_jcp2022="JCP3")])
    assert len(result) == 1
    assert result.iloc[0]["Metadata_JCP2022"] == "JCP3"


def test_select_samples_compound_and_well_filter() -> None:
    """Combining compound + well narrows to the exact intersection."""
    meta = _make_meta()
    # source_4 has K3/K4/K5 at wells A01/A02/A03; pick K3 which is A01.
    result = select_samples(
        meta,
        [JumpSample(metadata_source="source_4", metadata_inchikey="K3", metadata_well="A01")],
    )
    assert len(result) == 1
    assert result.iloc[0]["Metadata_InChIKey"] == "K3"
    assert result.iloc[0]["Metadata_Well"] == "A01"


# --------- expand_to_fovs ---------------------------------------------------


def test_expand_to_fovs_adds_id_and_s3_cols() -> None:
    meta = _make_meta()
    selected = select_samples(meta, [JumpSample(metadata_source="source_2")])
    load_data = _make_load_data("source_2", "B1", "P1", ["A01", "A02"], ["1", "2"])
    load_data = pd.concat(
        [load_data, _make_load_data("source_2", "B2", "P2", ["B01"], ["1", "2"])],
        ignore_index=True,
    )

    fov = expand_to_fovs(selected, load_data, CHANNEL_KEYS)

    # source_2 has 3 wells; P1 wells get 2 sites each (4 rows), P2 well gets 2 sites (2 rows).
    assert len(fov) == 6
    assert fov["id"].iloc[0] == "source_2__B1__P1__A01__1"
    assert (fov["s3_OrigDNA"] == fov["URL_OrigDNA"]).all()
    # Id uniqueness on the FOV level.
    assert fov["id"].is_unique


def test_expand_to_fovs_empty_raises() -> None:
    meta = _make_meta()
    selected = select_samples(meta, [JumpSample(metadata_source="source_2")])
    load_data = _make_load_data("source_4", "B1", "P3", ["A01"], ["1"])
    with pytest.raises(ValueError, match="0 rows"):
        expand_to_fovs(selected, load_data, CHANNEL_KEYS)


def test_expand_to_fovs_missing_site_raises() -> None:
    meta = _make_meta()
    selected = select_samples(meta, [JumpSample(metadata_source="source_2")])
    load_data = _make_load_data("source_2", "B1", "P1", ["A01"], ["1"]).drop(
        columns=["Metadata_Site"]
    )
    with pytest.raises(KeyError, match="Metadata_Site"):
        expand_to_fovs(selected, load_data, CHANNEL_KEYS)


def test_expand_to_fovs_rejects_bad_channel_key() -> None:
    meta = _make_meta()
    selected = select_samples(meta, [JumpSample(metadata_source="source_2")])
    load_data = _make_load_data("source_2", "B1", "P1", ["A01"], ["1"])
    with pytest.raises(ValueError, match="must start with 's3_'"):
        expand_to_fovs(selected, load_data, ["OrigDNA"])


@pytest.mark.parametrize(
    "column,bad_value",
    [
        ("Metadata_Source", "../evil"),
        ("Metadata_Batch", "foo/bar"),
        ("Metadata_Plate", "plate with space"),
        ("Metadata_Well", "A0\x00"),
        ("Metadata_Site", ""),
    ],
)
def test_expand_to_fovs_rejects_unsafe_metadata(column: str, bad_value: str) -> None:
    """Filesystem-unsafe identifier values must be rejected before they
    flow into per-plate Zarr directory names / FOV array names.

    Covers the path-traversal / id-separator-escape class of bugs: a
    ``..`` or ``/`` in Metadata_Batch would escape the ``full_images/``
    subtree; a NUL or whitespace in Metadata_Well would either collide
    with the ``__`` delimiter in composite ids or break downstream
    filename hygiene on some filesystems.
    """
    selected = pd.DataFrame(
        {
            "Metadata_Source": ["source_2"],
            "Metadata_Batch": ["B1"],
            "Metadata_Plate": ["P1"],
            "Metadata_Well": ["A01"],
            "Metadata_PlateType": ["t"],
            "Metadata_InChIKey": ["K"],
            "Metadata_InChI": ["I"],
        }
    )
    load_data = _make_load_data("source_2", "B1", "P1", ["A01"], ["1"])
    # Metadata_Site only lives on load_data; the other four are join keys
    # and must be poisoned on both sides or the merge would drop the row
    # before the safety check runs.
    if column == "Metadata_Site":
        load_data.loc[:, column] = bad_value
    else:
        selected.loc[:, column] = bad_value
        load_data.loc[:, column] = bad_value

    with pytest.raises(ValueError, match="unsafe"):
        expand_to_fovs(selected, load_data, CHANNEL_KEYS)


def test_expand_to_fovs_coerces_str_int_plate_dtypes() -> None:
    """Regression: merge succeeds when one side has str and the other int64.

    Real JUMP data hits this because `plate.csv.gz` carries plate IDs as
    strings (some are alphanumeric, e.g. ``BR00117035``) but pandas infers
    `int64` from the load_data CSV when every plate ID in that file is
    all-digit (e.g. ``1053601756``). Without explicit dtype coercion in
    `expand_to_fovs`, pandas raises
    ``ValueError: You are trying to merge on str and int64 columns``.
    """
    selected = pd.DataFrame(
        {
            "Metadata_Source": ["source_2", "source_2"],
            "Metadata_Batch": ["B1", "B1"],
            "Metadata_Plate": ["1053601756", "1053601756"],  # string
            "Metadata_Well": ["A01", "A02"],
            "Metadata_PlateType": ["COMPOUND", "COMPOUND"],
            "Metadata_InChIKey": ["K1", "K2"],
            "Metadata_InChI": ["I1", "I2"],
        }
    )
    load_data = _make_load_data("source_2", "B1", "1053601756", ["A01", "A02"], ["1"])
    # Simulate pd.read_csv inferring int64 for all-digit plate IDs.
    load_data["Metadata_Plate"] = load_data["Metadata_Plate"].astype("int64")

    fov = expand_to_fovs(selected, load_data, CHANNEL_KEYS)
    assert len(fov) == 2
    assert set(fov["Metadata_Well"]) == {"A01", "A02"}


# --------- assign_snakemake_batches ----------------------------------------


def test_assign_snakemake_batches_per_source_reset() -> None:
    fov = pd.DataFrame(
        {"Metadata_Source": ["source_2"] * 5 + ["source_4"] * 3, "Metadata_Site": list(range(8))}
    )
    bucketed = assign_snakemake_batches(fov, batch_size=2)
    assert list(bucketed["snakemake_batch"]) == [
        "source_2_snakemake_batch_0",
        "source_2_snakemake_batch_0",
        "source_2_snakemake_batch_1",
        "source_2_snakemake_batch_1",
        "source_2_snakemake_batch_2",
        "source_4_snakemake_batch_0",
        "source_4_snakemake_batch_0",
        "source_4_snakemake_batch_1",
    ]


def test_assign_snakemake_batches_rejects_zero() -> None:
    fov = pd.DataFrame({"Metadata_Source": ["s"]})
    with pytest.raises(ValueError):
        assign_snakemake_batches(fov, batch_size=0)


# --------- parse_s3_url -----------------------------------------------------


def test_parse_s3_url_ok() -> None:
    bucket, key = parse_s3_url("s3://cellpainting-gallery/cpg0016-jump/x/y.parquet")
    assert bucket == "cellpainting-gallery"
    assert key == "cpg0016-jump/x/y.parquet"


@pytest.mark.parametrize("bad", ["http://nope", "s3://only-bucket", "s3:///no-bucket"])
def test_parse_s3_url_errors(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_s3_url(bad)


# --------- S3 mock + fetch_load_data ---------------------------------------


class FakeBody:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def close(self) -> None:
        # s3_get_bytes wraps the body in contextlib.closing to match
        # boto3's streaming-body lifecycle; real StreamingBody has
        # close(), so our fake needs it too.
        return


class FakeS3:
    """Minimal drop-in for boto3's S3 client used by jump.fetch_load_data_per_plate."""

    def __init__(self, payloads: dict[tuple[str, str], bytes]) -> None:
        self._payloads = payloads
        self.calls: list[tuple[str, str]] = []

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        self.calls.append((Bucket, Key))
        try:
            payload = self._payloads[(Bucket, Key)]
        except KeyError as exc:
            raise RuntimeError(f"unexpected S3 fetch: {Bucket}/{Key}") from exc
        return {
            "ResponseMetadata": {"HTTPStatusCode": 200},
            "Body": FakeBody(payload),
        }


def _encode_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def test_fetch_load_data_uses_cache(tmp_path: Path) -> None:
    selected = pd.DataFrame(
        {
            "Metadata_Source": ["source_2"],
            "Metadata_Batch": ["B1"],
            "Metadata_Plate": ["P1"],
            "Metadata_Well": ["A01"],
        }
    )
    load = _make_load_data("source_2", "B1", "P1", ["A01"], ["1"])
    key = "cpg0016-jump/source_2/workspace/load_data_csv/B1/P1/load_data_with_illum.csv"
    fake = FakeS3({("cellpainting-gallery", key): _encode_csv(load)})

    df_first = fetch_load_data_per_plate(selected, tmp_path, s3_client=fake)
    df_second = fetch_load_data_per_plate(selected, tmp_path, s3_client=fake)

    assert len(df_first) == len(df_second) == 1
    assert len(fake.calls) == 1  # second call served from cache


def test_fetch_load_data_raises_on_non_200(tmp_path: Path) -> None:
    selected = pd.DataFrame(
        {
            "Metadata_Source": ["source_2"],
            "Metadata_Batch": ["B1"],
            "Metadata_Plate": ["P1"],
            "Metadata_Well": ["A01"],
        }
    )

    class BrokenS3:
        def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
            return {"ResponseMetadata": {"HTTPStatusCode": 500}, "Body": FakeBody(b"")}

    with pytest.raises(JumpS3Error, match="HTTP 500"):
        fetch_load_data_per_plate(selected, tmp_path, s3_client=BrokenS3())


def test_fetch_load_data_wraps_client_error(tmp_path: Path) -> None:
    """botocore ``ClientError`` / ``NoSuchKey`` surface as ``JumpS3Error``."""
    from botocore.exceptions import ClientError

    selected = pd.DataFrame(
        {
            "Metadata_Source": ["source_2"],
            "Metadata_Batch": ["B1"],
            "Metadata_Plate": ["P1"],
            "Metadata_Well": ["A01"],
        }
    )

    class NoSuchKeyS3:
        def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
            raise ClientError(
                error_response={
                    "Error": {"Code": "NoSuchKey", "Message": "The specified key does not exist."}
                },
                operation_name="GetObject",
            )

    with pytest.raises(JumpS3Error, match="failed to fetch"):
        fetch_load_data_per_plate(selected, tmp_path, s3_client=NoSuchKeyS3())


def test_fetch_load_data_does_not_swallow_programmer_errors(tmp_path: Path) -> None:
    """Non-botocore exceptions (e.g. TypeError) surface directly — not JumpS3Error.

    Guards against the "catch-Exception" anti-pattern where a misspelled kwarg
    on the real boto3 client would look like an S3 error.
    """
    selected = pd.DataFrame(
        {
            "Metadata_Source": ["source_2"],
            "Metadata_Batch": ["B1"],
            "Metadata_Plate": ["P1"],
            "Metadata_Well": ["A01"],
        }
    )

    class BadClient:
        def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
            raise TypeError("programmer error, not S3")

    with pytest.raises(TypeError, match="programmer error"):
        fetch_load_data_per_plate(selected, tmp_path, s3_client=BadClient())


# --------- download_metadata_tables + load_metadata ------------------------


def test_download_metadata_tables_uses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str]] = []

    def fake_urlretrieve(url: str, dst: str) -> None:
        calls.append((url, str(dst)))
        Path(dst).write_bytes(b"header\n1\n")

    monkeypatch.setattr(jump_mod.urllib.request, "urlretrieve", fake_urlretrieve)

    urls = {"plate": "https://example.com/plate.csv.gz"}
    first = download_metadata_tables(urls, tmp_path)
    second = download_metadata_tables(urls, tmp_path)

    assert first == second
    assert len(calls) == 1  # cache hit on second invocation


def _write_metadata_csvs(
    tmp_path: Path,
    *,
    plate: pd.DataFrame,
    well: pd.DataFrame,
    compound: pd.DataFrame,
    orf: pd.DataFrame | None = None,
    crispr: pd.DataFrame | None = None,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name, df in {"plate": plate, "well": well, "compound": compound}.items():
        p = tmp_path / f"{name}.csv"
        df.to_csv(p, index=False)
        paths[name] = p
    if orf is not None:
        p = tmp_path / "orf.csv"
        orf.to_csv(p, index=False)
        paths["orf"] = p
    if crispr is not None:
        p = tmp_path / "crispr.csv"
        crispr.to_csv(p, index=False)
        paths["crispr"] = p
    return paths


def test_load_metadata_merges_compound(tmp_path: Path) -> None:
    paths = _write_metadata_csvs(
        tmp_path,
        plate=pd.DataFrame(
            {
                "Metadata_Source": ["source_2"],
                "Metadata_Plate": ["P1"],
                "Metadata_PlateType": ["COMPOUND"],
            }
        ),
        well=pd.DataFrame(
            {
                "Metadata_Source": ["source_2", "source_2"],
                "Metadata_Plate": ["P1", "P1"],
                "Metadata_Well": ["A01", "A02"],
                "Metadata_Batch": ["B1", "B1"],
                "Metadata_JCP2022": ["JCP1", "JCP2"],
            }
        ),
        compound=pd.DataFrame(
            {
                "Metadata_JCP2022": ["JCP1", "JCP2"],
                "Metadata_InChIKey": ["K1", "K2"],
                "Metadata_InChI": ["InChI=1", "InChI=2"],
                "Metadata_SMILES": ["S1", "S2"],
            }
        ),
    )
    merged = load_metadata(paths)
    assert len(merged) == 2
    assert set(merged["Metadata_InChIKey"]) == {"K1", "K2"}


def test_load_metadata_keeps_orf_and_crispr_wells(tmp_path: Path) -> None:
    """ORF + CRISPR wells survive the annotation merge (no inner-join drop).

    Regression for the bug where ``load_metadata``'s inner-join on compound
    silently dropped every non-COMPOUND well. ORF wells' JCP2022 only lives
    in ``orf.csv.gz``; CRISPR wells' only in ``crispr.csv.gz``.
    """
    paths = _write_metadata_csvs(
        tmp_path,
        plate=pd.DataFrame(
            {
                "Metadata_Source": ["source_4", "source_13"],
                "Metadata_Plate": ["BR00117035", "CRISPR_PLATE_01"],
                "Metadata_PlateType": ["ORF", "CRISPR"],
            }
        ),
        well=pd.DataFrame(
            {
                "Metadata_Source": ["source_4", "source_13"],
                "Metadata_Plate": ["BR00117035", "CRISPR_PLATE_01"],
                "Metadata_Well": ["A01", "A01"],
                "Metadata_Batch": ["Batch1", "CRISPR_Batch1"],
                "Metadata_JCP2022": ["JCP_ORF_1", "JCP_CRISPR_1"],
            }
        ),
        compound=pd.DataFrame(
            {
                "Metadata_JCP2022": ["JCP_COMP_1"],
                "Metadata_InChIKey": ["K_compound"],
                "Metadata_InChI": ["InChI=c"],
                "Metadata_SMILES": ["S_compound"],
            }
        ),
        orf=pd.DataFrame({"Metadata_JCP2022": ["JCP_ORF_1"], "Metadata_Symbol": ["GENE_A"]}),
        crispr=pd.DataFrame({"Metadata_JCP2022": ["JCP_CRISPR_1"], "Metadata_Symbol": ["GENE_B"]}),
    )
    merged = load_metadata(paths)
    assert len(merged) == 2, "both ORF and CRISPR wells must survive"
    assert set(merged["Metadata_Plate"]) == {"BR00117035", "CRISPR_PLATE_01"}
    # Annotations picked up from the right tables; compound cols are NaN for
    # these wells (no matching JCP2022 in compound).
    assert set(merged["Metadata_Symbol"]) == {"GENE_A", "GENE_B"}
    assert merged["Metadata_InChIKey"].isna().all()


def test_load_metadata_optional_tables_ignored(tmp_path: Path) -> None:
    """Running without orf/crispr works — they're optional annotation extras."""
    paths = _write_metadata_csvs(
        tmp_path,
        plate=pd.DataFrame(
            {
                "Metadata_Source": ["source_2"],
                "Metadata_Plate": ["P1"],
                "Metadata_PlateType": ["COMPOUND"],
            }
        ),
        well=pd.DataFrame(
            {
                "Metadata_Source": ["source_2"],
                "Metadata_Plate": ["P1"],
                "Metadata_Well": ["A01"],
                "Metadata_Batch": ["B1"],
                "Metadata_JCP2022": ["JCP1"],
            }
        ),
        compound=pd.DataFrame(
            {
                "Metadata_JCP2022": ["JCP1"],
                "Metadata_InChIKey": ["K1"],
                "Metadata_InChI": ["InChI=1"],
                "Metadata_SMILES": ["S1"],
            }
        ),
    )
    merged = load_metadata(paths)
    assert len(merged) == 1
    assert merged["Metadata_InChIKey"].iloc[0] == "K1"


def test_load_metadata_requires_keys(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing metadata table"):
        load_metadata({"plate": tmp_path / "nope.csv"})


# --------- end-to-end resolve_metadata -------------------------------------


def test_resolve_metadata_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Fake urllib: write synthetic plate/well/compound/orf CSVs on "download".
    # Real JUMP URLs point at .csv.gz, so the payloads must actually be gzipped
    # (pandas.read_csv infers compression from the filename suffix).
    import gzip

    plate_df = pd.DataFrame(
        {
            "Metadata_Source": ["source_2"],
            "Metadata_Plate": ["P1"],
            "Metadata_PlateType": ["target1"],
        }
    )
    well_df = pd.DataFrame(
        {
            "Metadata_Source": ["source_2", "source_2"],
            "Metadata_Plate": ["P1", "P1"],
            "Metadata_Well": ["A01", "A02"],
            "Metadata_Batch": ["B1", "B1"],
            "Metadata_JCP2022": ["JCP1", "JCP2"],
        }
    )
    compound_df = pd.DataFrame(
        {
            "Metadata_JCP2022": ["JCP1", "JCP2"],
            "Metadata_InChIKey": ["K1", "K2"],
            "Metadata_InChI": ["I1", "I2"],
            "Metadata_SMILES": ["S1", "S2"],
        }
    )
    orf_df = pd.DataFrame({"Metadata_JCP2022": ["JCP_ORF"], "Metadata_Symbol": ["GENE_X"]})
    crispr_df = pd.DataFrame({"Metadata_JCP2022": ["JCP_CRISPR"], "Metadata_Symbol": ["GENE_Y"]})
    csv_payloads: dict[str, bytes] = {
        "plate.csv.gz": gzip.compress(plate_df.to_csv(index=False).encode()),
        "well.csv.gz": gzip.compress(well_df.to_csv(index=False).encode()),
        "compound.csv.gz": gzip.compress(compound_df.to_csv(index=False).encode()),
        "orf.csv.gz": gzip.compress(orf_df.to_csv(index=False).encode()),
        "crispr.csv.gz": gzip.compress(crispr_df.to_csv(index=False).encode()),
    }

    def fake_urlretrieve(url: str, dst: str) -> None:
        name = Path(url).name
        Path(dst).write_bytes(csv_payloads[name])

    monkeypatch.setattr(jump_mod.urllib.request, "urlretrieve", fake_urlretrieve)

    load = _make_load_data("source_2", "B1", "P1", ["A01", "A02"], ["1", "2"])
    key = "cpg0016-jump/source_2/workspace/load_data_csv/B1/P1/load_data_with_illum.csv"
    fake_s3 = FakeS3({("cellpainting-gallery", key): _encode_csv(load)})

    # Minimal JumpConfig stand-in — the resolver only reads fields, not
    # behaviour, so a typed object from pydantic keeps the interface honest.
    from cp_bg_bench.config import JumpConfig

    jump_cfg = JumpConfig.model_validate(
        {
            "samples": [{"metadata_source": "source_2"}],
            "metadata_tables": {
                "plate": "https://example.com/plate.csv.gz",
                "well": "https://example.com/well.csv.gz",
                "compound": "https://example.com/compound.csv.gz",
                "orf": "https://example.com/orf.csv.gz",
                "crispr": "https://example.com/crispr.csv.gz",
            },
            "channel_s3_keys": CHANNEL_KEYS,
            "segmentation": {
                "model": "cpsam",
                "channels_for_nucleus": [0],
                "channels_for_cell": [3, 0],
                "default_diameters": {"nucleus": 21, "cytosol": 53},
            },
        }
    )

    df = resolve_metadata(jump_cfg, cache_dir=tmp_path, batch_size=3, s3_client=fake_s3)

    expected_cols = required_output_columns(CHANNEL_KEYS)
    assert list(df.columns) == expected_cols
    assert len(df) == 4  # 2 wells × 2 sites
    assert df["id"].is_unique
    for col in ["id", "snakemake_batch", *CHANNEL_KEYS]:
        assert df[col].notna().all()
    # batch_size=3 → first three rows bucket 0, fourth row bucket 1 (same source).
    assert df["snakemake_batch"].tolist() == [
        "source_2_snakemake_batch_0",
        "source_2_snakemake_batch_0",
        "source_2_snakemake_batch_0",
        "source_2_snakemake_batch_1",
    ]
