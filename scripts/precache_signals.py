#!/usr/bin/env python3
"""Pre-generate raw signal cache files for all BIDS recordings.

This script writes {dataset_root}/.ecg_signal_cache/<uid>_ecg.f32 for every
recording that has an actual EDF file.  It must be run once before launching
DDP training so that the lazy dataset never needs to write to disk during the
multi-GPU run.

Usage:
    python scripts/precache_signals.py \
        --dataset-root /media/tnzr/HDD11/Datasets/ds005873 \
        [--max-recordings 0]
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Pre-cache raw ECG signal binaries.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--max-recordings", type=int, default=0,
                        help="Limit recordings (0 = all)")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    signal_cache_dir = dataset_root / ".ecg_signal_cache"
    signal_cache_dir.mkdir(parents=True, exist_ok=True)

    # Import project modules
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config.config import DEFAULT_CONFIG
    from src.data_loader import BIDSDataLoader

    config = DEFAULT_CONFIG
    config.data.dataset_root = str(dataset_root)
    loader = BIDSDataLoader(config.data)
    all_recordings = loader.list_all_recordings()

    # Filter to recordings with actual EDF files
    recordings = []
    for rec in all_recordings:
        try:
            path = loader.resolve_subject_edf_path(
                rec["subject_id"], rec["session_id"], "ecg", rec["run_id"]
            )
            if path is not None:
                recordings.append(rec)
        except Exception:
            pass

    logger.info("Found %d recordings with EDF files out of %d total",
                len(recordings), len(all_recordings))

    if args.max_recordings > 0:
        recordings = recordings[:args.max_recordings]
        logger.info("Limited to %d recordings", len(recordings))

    n_total = len(recordings)
    n_cached = 0
    n_skipped = 0
    n_errors = 0
    t_start = time.time()

    for i, recording in enumerate(recordings):
        uid = (
            f"sub-{recording['subject_id']}"
            f"_ses-{recording['session_id']}"
            f"_run-{recording['run_id']}"
        )
        bin_path = signal_cache_dir / f"{uid}_ecg.f32"
        meta_path = signal_cache_dir / f"{uid}_ecg.json"

        if bin_path.exists() and meta_path.exists():
            n_skipped += 1
            if (i + 1) % 50 == 0 or i < 5:
                logger.info("[%d/%d] Already cached: %s", i + 1, n_total, uid)
            continue

        if (i + 1) % 10 == 0 or i < 5:
            elapsed = time.time() - t_start
            eta = (elapsed / max(1, i)) * (n_total - i) / 60
            logger.info("[%d/%d] Caching %s | elapsed=%.0fs ETA=%.1fmin",
                        i + 1, n_total, uid, elapsed, eta)

        try:
            ecg_data, fs = loader.load_subject_edf(
                recording["subject_id"],
                recording["session_id"],
                "ecg",
                recording["run_id"],
            )
            signal_1d = (ecg_data[0] if ecg_data.ndim > 1 else ecg_data).astype(np.float32)
            n_samples = len(signal_1d)

            # Write atomically via .tmp -> rename
            tmp = bin_path.with_suffix(".tmp")
            try:
                signal_1d.tofile(str(tmp))
                tmp.rename(bin_path)
                with open(meta_path, "w") as fh:
                    json.dump({"fs": float(fs), "n_samples": n_samples}, fh)
                n_cached += 1
                logger.info("[%d/%d] Wrote %s (%.1f MB, %d samples, fs=%.0f)",
                            i + 1, n_total, uid,
                            bin_path.stat().st_size / 1e6, n_samples, fs)
            except Exception as write_err:
                logger.error("Write failed for %s: %s", uid, write_err)
                # Clean up partial file
                if tmp.exists():
                    tmp.unlink()
                if bin_path.exists():
                    bin_path.unlink()
                n_errors += 1

            # Free memory explicitly
            del signal_1d
            del ecg_data

        except Exception as e:
            logger.error("[%d/%d] Error loading %s: %s", i + 1, n_total, uid, e)
            n_errors += 1

    elapsed_total = time.time() - t_start
    logger.info(
        "Done: %d cached, %d already existed, %d errors | total=%.1f min",
        n_cached, n_skipped, n_errors, elapsed_total / 60
    )


if __name__ == "__main__":
    main()
