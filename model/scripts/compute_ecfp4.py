"""
Compute ECFP4 fingerprints for the JUMP training compounds and write a lookup parquet.

The 1,467 training compounds are defined by the compound selection strategy:
  TARGET2 (302) ∪ RepHub-annotated compounds with ≥5-source coverage (1,250)
  minus overlap (85) = 1,467 unique compounds.

The authoritative compound list lives in the JUMP training config:
  data/config/jump_training.yml

The script reads unique JCP2022 IDs from that YAML, looks up their SMILES in
the cached compound metadata, and computes ECFP4 fingerprints.

Reads:
  <training-config>   jump_training.yml  (contains metadata_jcp2022: entries)
  <compound-csv>      compound.csv.gz    (JCP2022 → Metadata_SMILES + Metadata_InChI)

Writes:
  <output>   Parquet with columns [Metadata_SMILES, ecfp4_2048 (list[float32])]
             One row per unique SMILES.  ~12 MB for ~1,467 compounds.

Design: the fingerprint file is loaded once into a dict[str, np.ndarray] at
model init rather than stored per-row in the HF training dataset.  That keeps
the in-memory footprint at ~12 MB vs ~8 GB for per-row storage across 1M crops.

Fingerprint spec:
  RDKit MorganGenerator  radius=2, nBits=2048, includeChirality=True
  Count fingerprint → log1p(counts) → float32

Usage:
    pixi run python scripts/compute_ecfp4.py
    pixi run python scripts/compute_ecfp4.py \\
        --training-config /path/to/data/config/jump_training.yml \\
        --compound-csv /path/to/jump_training/meta/cache/jump_metadata/compound.csv.gz \\
        --output /path/to/ecfp4_fingerprints.parquet
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ── defaults (absolute paths, adjust if the sibling repos move) ───────────────

_DEFAULT_TRAINING_CONFIG = Path(
    "${WORKSPACE}-jump"
    "/data/config/jump_training.yml"
)
_DEFAULT_COMPOUND_CSV = Path(
    "${WORKSPACE}"
    "/meta/cache/jump_metadata/compound.csv.gz"
)
_DEFAULT_OUTPUT = Path(
    "${WORKSPACE}"
    "/meta/ecfp4_fingerprints.parquet"
)

# ── fingerprint helpers ───────────────────────────────────────────────────────

_JCP2022_RE = re.compile(r"metadata_jcp2022:\s*(JCP2022_\w+)")


def _parse_jcp2022_ids(training_config: Path) -> set[str]:
    """Extract unique JCP2022 IDs from the jump_training.yml YAML.

    The file has ~120k lines with entries like:
        metadata_jcp2022: JCP2022_000214
    A regex scan is faster and simpler than a full YAML parse.
    """
    ids: set[str] = set()
    with open(training_config) as fh:
        for line in fh:
            m = _JCP2022_RE.search(line)
            if m:
                ids.add(m.group(1))
    return ids


def _load_rdkit_gen():
    """Return an RDKit MorganGenerator (radius=2, 2048 bits, chirality)."""
    try:
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
    except ImportError as e:
        raise ImportError(
            "RDKit is required. It is included in the default pixi environment:\n"
            "  pixi run python scripts/compute_ecfp4.py"
        ) from e
    return GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)


def _fix_hypervalent_phosphorus(mol):
    """Demote one P=O double bond to P-O(-) for each P with ≥2 P=O bonds.

    The JUMP compound.csv.gz was generated with a non-standard P representation
    that RDKit's SMILES parser rejects.  Parsing via InChI yields a mol with
    the hypervalent P, which this fix demotes to a valid valence.
    """
    from rdkit import Chem
    from rdkit.Chem.rdchem import BondType

    rw = Chem.RWMol(mol)
    to_demote: list[tuple[int, int]] = []
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() != 15:  # only phosphorus
            continue
        dbl_oxygens = [
            (atom.GetIdx(), bond.GetOtherAtom(atom).GetIdx())
            for bond in atom.GetBonds()
            if bond.GetOtherAtom(atom).GetAtomicNum() == 8
            and bond.GetBondType() == BondType.DOUBLE
        ]
        if len(dbl_oxygens) >= 2:
            to_demote.append(dbl_oxygens[-1])  # demote exactly one

    for p_idx, o_idx in to_demote:
        rw.RemoveBond(p_idx, o_idx)
        rw.AddBond(p_idx, o_idx, BondType.SINGLE)
        o_atom = rw.GetAtomWithIdx(o_idx)
        o_atom.SetFormalCharge(-1)
        o_atom.SetNoImplicit(True)
        o_atom.SetNumExplicitHs(0)

    fixed = rw.GetMol()
    Chem.SanitizeMol(fixed)
    return fixed


def _mol_from_inchi_with_fix(inchi: str):
    """Parse mol from InChI, applying hypervalent-P fix if needed.

    InChI is more lenient than SMILES for phosphate-group representations.
    Falls back to None if both parsing and the fix fail.
    """
    from rdkit.Chem import inchi as rd_inchi

    if not inchi or not isinstance(inchi, str) or not inchi.startswith("InChI="):
        return None
    try:
        mol = rd_inchi.MolFromInchi(inchi, sanitize=True, treatWarningAsError=False)
    except Exception:
        return None
    if mol is not None:
        return mol
    # sanitize=True failed — try without, then apply the P fix
    try:
        mol = rd_inchi.MolFromInchi(inchi, sanitize=False, treatWarningAsError=False)
        if mol is None:
            return None
        return _fix_hypervalent_phosphorus(mol)
    except Exception:
        return None


def compute_ecfp4(smiles: str, gen, inchi: str | None = None) -> np.ndarray:
    """Morgan r=2, 2048-bit, chirality, log1p(counts) → float32.

    Primary path: parse from SMILES.
    Fallback: parse from InChI + apply hypervalent-phosphorus fix.
    Raises ValueError if both paths fail.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None and inchi:
        mol = _mol_from_inchi_with_fix(inchi)
    if mol is None:
        raise ValueError(
            f"Cannot parse compound — SMILES: {smiles!r}"
            + (f", InChI: {inchi!r}" if inchi else "")
        )
    fp = gen.GetCountFingerprintAsNumPy(mol).astype(np.float32)
    np.log1p(fp, out=fp)
    return fp


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute ECFP4 fingerprints for JUMP training compounds"
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=_DEFAULT_TRAINING_CONFIG,
        help="jump_training.yml from the data pipeline (default: %(default)s)",
    )
    parser.add_argument(
        "--compound-csv",
        type=Path,
        default=_DEFAULT_COMPOUND_CSV,
        help="JUMP compound.csv.gz (JCP2022 → SMILES, default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Output parquet path (default: %(default)s)",
    )
    args = parser.parse_args()

    for p in (args.training_config, args.compound_csv):
        if not p.exists():
            print(f"ERROR: required file not found: {p}")
            return 1

    # ── 1. JCP2022 IDs from the training config ───────────────────────────────
    print("[1/4] Parsing JCP2022 IDs from training config...")
    jcp_ids = _parse_jcp2022_ids(args.training_config)
    print(f"      {len(jcp_ids):,} unique JCP2022 IDs")

    # ── 2. JCP2022 → SMILES + InChI from compound metadata ──────────────────
    print("[2/4] Joining with compound metadata → SMILES + InChI...")
    compound = pd.read_csv(
        args.compound_csv,
        usecols=["Metadata_JCP2022", "Metadata_SMILES", "Metadata_InChI"],
    )
    compound = (
        compound[compound["Metadata_JCP2022"].isin(jcp_ids)]
        .dropna(subset=["Metadata_SMILES"])
        .drop_duplicates(subset=["Metadata_JCP2022"])
    )

    n_missing = len(jcp_ids) - len(compound)
    if n_missing > 0:
        print(f"      WARNING: {n_missing} JCP2022 IDs not found in compound.csv.gz")
        missing = jcp_ids - set(compound["Metadata_JCP2022"])
        for jcp in sorted(missing)[:5]:
            print(f"        {jcp}")
        if n_missing > 5:
            print(f"        ... and {n_missing - 5} more")

    print(f"      {len(compound):,} compounds with SMILES")

    # ── 3. ECFP4 fingerprints ─────────────────────────────────────────────────
    print("[3/4] Computing ECFP4 (radius=2, 2048-bit, chirality, log1p counts)...")
    print("      Fallback: InChI + hypervalent-P fix for unparseable SMILES")
    gen = _load_rdkit_gen()

    failures: list[str] = []
    records: list[dict] = []
    for row in compound.itertuples(index=False):
        smiles = row.Metadata_SMILES
        inchi = getattr(row, "Metadata_InChI", None)
        try:
            fp = compute_ecfp4(smiles, gen, inchi=inchi)
        except ValueError as exc:
            failures.append(str(exc))
            continue
        records.append({"Metadata_SMILES": smiles, "ecfp4_2048": fp.tolist()})

    n_ok = len(records)
    print(f"      {n_ok:,} computed, {len(failures)} failed")

    if failures:
        print(f"\nERROR: {len(failures)} compound(s) could not be fingerprinted:")
        for msg in failures[:20]:
            print(f"  {msg}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
        return 1

    # Defensive guard: unreachable via the ValueError path above (which already
    # exits) but catches any future silent-drop introduced in this loop.
    n_input = len(compound)
    if n_ok < n_input:
        print(f"ERROR: {n_input - n_ok} compound(s) missing from output")
        return 1

    # ── 4. write parquet ──────────────────────────────────────────────────────
    print(f"[4/4] Writing {args.output}...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(records)
    out_df.to_parquet(args.output, index=False)

    size_kb = args.output.stat().st_size / 1024
    print(f"      {n_ok:,} compounds  |  {size_kb:.0f} KB on disk")
    print()
    print("Load as lookup dict (model init):")
    print("  import pandas as pd, numpy as np")
    print(f"  df = pd.read_parquet('{args.output}')")
    print("  lookup = {")
    print("      row.Metadata_SMILES: np.array(row.ecfp4_2048, dtype='float32')")
    print("      for row in df.itertuples()")
    print("  }")
    return 0


if __name__ == "__main__":
    sys.exit(main())
