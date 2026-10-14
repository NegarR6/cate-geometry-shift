"""Download or verify external benchmark datasets (IHDP-100 replications)."""
from __future__ import annotations
import sys as _sys, pathlib as _pl
_ROOT = _pl.Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path: _sys.path.insert(0, str(_ROOT))
if str(_ROOT / 'src') not in _sys.path: _sys.path.insert(0, str(_ROOT / 'src'))


import argparse
import urllib.request
from pathlib import Path

IHDP_TRAIN_URL = "https://www.fredjo.com/files/ihdp_npci_1-100.train.npz"
IHDP_TEST_URL = "https://www.fredjo.com/files/ihdp_npci_1-100.test.npz"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    for fname, url in [
        ("ihdp_npci_1-100.train.npz", IHDP_TRAIN_URL),
        ("ihdp_npci_1-100.test.npz", IHDP_TEST_URL),
    ]:
        dest = data_dir / fname
        if dest.exists():
            print(f"[OK] Found {dest}")
            continue
        print(f"Downloading {url} -> {dest} ...")
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"[DONE] Saved {dest}")
        except Exception as exc:
            print(f"[WARN] Could not auto-download {fname} ({exc}). Please copy {fname} into {data_dir}/.")


if __name__ == "__main__":
    main()
