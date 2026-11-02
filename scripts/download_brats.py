"""Download the BraTS 2023 Adult Glioma training archive from Synapse and verify it.

Requires access granted at https://www.synapse.org/Synapse:syn64952532 (CC-BY-NC 4.0) and a personal
access token with View and Download scopes:

    export SYNAPSE_AUTH_TOKEN=...
    python scripts/download_brats.py --output-dir /path/to/data
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

TRAINING_ZIP_ID = "syn51514132"
TRAINING_ZIP_NAME = "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip"
REQUEST_ACCESS_URL = "https://www.synapse.org/Synapse:syn64952532"
ATTRIBUTION = (
    'Data used in this publication were obtained as part of the Brain Tumor Segmentation (BraTS) '
    'Challenge project through Synapse ID: syn51156910.'
)


def fail(message: str) -> None:
    print(f"\nERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def login(token: str):
    try:
        import synapseclient
    except ImportError:
        fail('synapseclient is not installed. Run:  pip install -e ".[download]"')
    syn = synapseclient.Synapse(skip_checks=True, cache_root_dir=None)
    try:
        syn.login(authToken=token, silent=True)
    except Exception as exc:  # the client raises several exception types for bad tokens
        fail(f"Synapse login failed ({type(exc).__name__}: {exc}). Check the token and its scopes.")
    return syn


def fetch_archive(syn, output_dir: Path) -> Path:
    from synapseclient.core.exceptions import SynapseError

    try:
        entity = syn.get(TRAINING_ZIP_ID, downloadFile=False)
    except SynapseError as exc:
        fail(f"Cannot read {TRAINING_ZIP_ID}: {exc}\nHave you opened {REQUEST_ACCESS_URL} and accepted the terms?")
    if entity.name != TRAINING_ZIP_NAME:
        fail(f"{TRAINING_ZIP_ID} is named {entity.name!r}, expected {TRAINING_ZIP_NAME!r}; the dataset layout "
             "on Synapse may have changed. Stopping instead of downloading an unexpected file.")
    size = int(entity._file_handle["contentSize"])
    free = shutil.disk_usage(output_dir).free
    needed = int(size * 2.2)  # archive + unpacked copy
    print(f"{entity.name}: {size / 1e9:.1f} GB; free disk {free / 1e9:.1f} GB; need about {needed / 1e9:.1f} GB")
    if free < needed:
        fail("Not enough free disk space for the archive plus its unpacked copy.")
    try:
        entity = syn.get(TRAINING_ZIP_ID, downloadLocation=str(output_dir), ifcollision="overwrite.local")
    except SynapseError as exc:
        fail(f"Download refused: {exc}\nRequest access and accept the terms at {REQUEST_ACCESS_URL}, then retry.")
    return Path(entity.path)


def unpack(archive: Path, output_dir: Path) -> Path:
    with zipfile.ZipFile(archive) as zf:
        bad = zf.testzip()
        if bad is not None:
            fail(f"Corrupt archive: first bad member is {bad}. Delete {archive} and download again.")
        target = output_dir.resolve()
        for member in zf.namelist():  # reject path traversal before extracting anything
            if not (target / member).resolve().is_relative_to(target):
                fail(f"Unsafe path in archive: {member}")
        print(f"Unpacking {len(zf.namelist())} files into {output_dir} ...")
        zf.extractall(output_dir)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True, help="Directory to download into (on local SSD, not a network drive)")
    parser.add_argument("--keep-archive", action="store_true", help="Keep the .zip after unpacking")
    args = parser.parse_args()

    token = os.environ.get("SYNAPSE_AUTH_TOKEN")
    if not token:
        fail("SYNAPSE_AUTH_TOKEN is not set.\nSteps: request access at "
             f"{REQUEST_ACCESS_URL}, accept the terms, create a personal access token, export it, rerun.\n"
             "See the docstring of this script or docs/reference.md section 'Getting the data'.")
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    archive = fetch_archive(login(token), output_dir)
    unpack(archive, output_dir)
    if not args.keep_archive:
        archive.unlink()
    print(f"\nDone. Next:  python scripts/inspect_dataset.py --data-root {output_dir}")
    print(f"\nWhen you publish anything using this data, include: {ATTRIBUTION}")


if __name__ == "__main__":
    main()
