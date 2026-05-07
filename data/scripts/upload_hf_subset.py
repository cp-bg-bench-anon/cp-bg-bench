"""Upload the CP-BG-Bench JUMP subset to Hugging Face anonymously.

Pre-requisites:
    1. The anon HF user is created, profile is empty, and is a member of the
       target organisation with write access.
    2. The anon write token is generated from that user's session and exported:
           export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
    3. Do **not** run this in a shell where you have previously run
       ``huggingface-cli login`` with your real account; either use a fresh
       shell or run ``huggingface-cli logout`` first. The script reads
       ``HF_TOKEN`` directly so the cached login is never used.

Usage:
    python upload_hf_subset.py \\
        --repo_id cp-bg-bench-anon/cp-bg-bench \\
        --staging_dir ${DATA_ROOT}

The script is idempotent: re-running uploads only files whose blob hash
differs from what is already on the Hub.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi


VIEWS = ("jump_crops", "jump_seg", "jump_crops_density", "jump_seg_density")
EXTRA_FILES = ("README.md", "well_list.csv")


def _check_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit(
            "HF_TOKEN environment variable not set.\n"
            "Export the anon write token from the anonymous HF user account "
            "before running:  export HF_TOKEN=hf_xxx..."
        )
    return token


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo_id", required=True,
                    help="Target dataset repo, e.g. cp-bg-bench-anon/cp-bg-bench")
    ap.add_argument("--staging_dir", type=Path, required=True,
                    help="Local directory containing jump_crops/, jump_seg/, ...")
    ap.add_argument("--private", action="store_true",
                    help="Create the repo as private (default: public)")
    ap.add_argument("--commit_message", default="Add JUMP subset (anon review release)")
    args = ap.parse_args()

    token = _check_token()
    api = HfApi(token=token)

    # Identity sanity check — print the account this token belongs to so the
    # operator can verify they are NOT pushing under their real account.
    me = api.whoami()
    print(f"uploading as: {me['name']}  (type={me.get('type','user')})")
    print(f"repo: {args.repo_id} (private={args.private})")
    print(f"staging: {args.staging_dir}")
    print()

    # Validate staging layout up front so we fail before touching the Hub.
    for v in VIEWS:
        p = args.staging_dir / v
        if not p.is_dir():
            sys.exit(f"missing view directory: {p}")
    for f in EXTRA_FILES:
        p = args.staging_dir / f
        if not p.is_file():
            sys.exit(f"missing file: {p}")

    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        exist_ok=True,
        private=args.private,
    )

    # One commit per view directory keeps the commit log scoped and lets a
    # failed upload be retried without re-uploading the others.
    for v in VIEWS:
        local_dir = args.staging_dir / v
        print(f"uploading {v}/ ...")
        api.upload_folder(
            folder_path=str(local_dir),
            path_in_repo=v,
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=f"{args.commit_message}: {v}",
        )

    for f in EXTRA_FILES:
        print(f"uploading {f} ...")
        api.upload_file(
            path_or_fileobj=str(args.staging_dir / f),
            path_in_repo=f,
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message=f"{args.commit_message}: {f}",
        )

    print("\nverify the upload from a private browser window:")
    print(f"  https://huggingface.co/datasets/{args.repo_id}")
    print("specifically check that:")
    print("  - the 'uploaded by' link goes to an empty anon profile")
    print("  - no real-name avatars or bios appear anywhere on the page")
    print("  - the README and dataset card YAML show no identifying fields")


if __name__ == "__main__":
    main()
