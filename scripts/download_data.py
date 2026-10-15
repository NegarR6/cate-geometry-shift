"""Download or verify external benchmark datasets.

IHDP: auto-download from fredjo.com
ACIC 2016: NOT auto-hosted here — place official files under data/acic2016/
         expected: x.csv and zymu_1.csv ... zymu_10.csv
"""
from __future__ import annotations
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
            print(f"[WARN] Could not auto-download {fname} ({exc}).")

    acic = data_dir / "acic2016"
    xcsv = acic / "x.csv"
    if xcsv.exists():
        n_zymu = len(list(acic.glob("zymu_*.csv")))
        print(f"[OK] ACIC 2016 found at {acic} ({n_zymu} zymu files)")
    else:
        print(
            f"[INFO] ACIC 2016 not found at {acic}.\n"
            "  Place official ACIC 2016 files there:\n"
            "    data/acic2016/x.csv\n"
            "    data/acic2016/zymu_1.csv ... zymu_10.csv\n"
            "  (Not redistributed by this script.)"
        )


if __name__ == "__main__":
    main()
