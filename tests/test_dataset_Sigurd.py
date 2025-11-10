"""Tiny HDF5 opener for MNIST files.

This script only opens the given HDF5 file and prints top-level dataset names,
their shapes and dtypes. It's intentionally minimal so you can extend it.
"""

from pathlib import Path
import sys
import h5py
import numpy as np


def main(argv=None):
    """Verify one or more MNIST HDF5 files.

    The script inspects top-level datasets and validates that `images` and
    `labels` are present. By default it expects labels to be in the set
    4..9 (the project's filtered subset) but you can pass a different set with
    --expect.
    """
    argv = argv or sys.argv[1:]
    import argparse

    parser = argparse.ArgumentParser(description="Verify MNIST HDF5 file contents")
    parser.add_argument("paths", nargs="+", help="One or more HDF5 files to inspect")
    parser.add_argument("--expect", default="4,5,6,7,8,9",
                        help="Comma-separated expected labels (default: 4,5,6,7,8,9). Use 0-9 for full set.")
    args = parser.parse_args(argv)

    try:
        expected = [int(x) for x in args.expect.split(",") if x.strip() != ""]
    except ValueError:
        print("--expect must be a comma-separated list of integers, e.g. 4,5,6,7,8,9")
        return 4

    overall_rc = 0
    for p in args.paths:
        path = Path(p)
        if not path.exists():
            print(f"File not found: {path}")
            overall_rc = 3
            continue

        print(f"\nOpened: {path}")
        try:
            with h5py.File(path, "r") as f:
                keys = list(f.keys())
                print("keys:", keys)
                if "images" not in f or "labels" not in f:
                    print("  ERROR: Expected datasets 'images' and 'labels' not found")
                    overall_rc = 5
                    continue

                imgs = f["images"]
                labels = f["labels"]

                try:
                    img_shape = tuple(imgs.shape)
                except Exception:
                    img_shape = None
                print("  images dtype, shape:", getattr(imgs, "dtype", None), img_shape)

                try:
                    lab_shape = tuple(labels.shape)
                except Exception:
                    lab_shape = None
                print("  labels dtype, shape:", getattr(labels, "dtype", None), lab_shape)

                # Load labels and check values
                try:
                    lab = np.array(labels[:])
                except Exception as e:
                    print(f"  ERROR reading labels: {e}")
                    overall_rc = 6
                    continue

                if lab.size == 0:
                    print("  WARNING: labels is empty")
                    overall_rc = max(overall_rc, 7)
                    continue

                unique, counts = np.unique(lab, return_counts=True)
                print("  unique labels:", unique.tolist())
                print("  counts per label:", dict(zip(unique.tolist(), counts.tolist())))
                print("  total samples:", lab.shape[0])

                # Validate expected labels
                unexpected = set(unique.tolist()) - set(expected)
                missing = set(expected) - set(unique.tolist())
                if unexpected:
                    print(f"  ERROR: found unexpected labels: {sorted(unexpected)}")
                    overall_rc = 8
                if missing:
                    print(f"  NOTE: expected labels missing from file: {sorted(missing)}")
                    # not necessarily fatal; leave as note

        except Exception as e:
            print(f"  ERROR opening file {path}: {e}")
            overall_rc = 9

    return overall_rc

main()
