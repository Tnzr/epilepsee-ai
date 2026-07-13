import unittest

import numpy as np

from scripts.train import _split_indices_by_recording_timeline


class TestRecordingSplitNonEmpty(unittest.TestCase):
    def test_non_empty_splits_with_three_recordings(self):
        n_per = 12
        y = np.concatenate([
            np.array([8.0] * 2 + [-1.0] * (n_per - 2), dtype=np.float32),
            np.array([7.0] * 2 + [-1.0] * (n_per - 2), dtype=np.float32),
            np.array([6.0] * 2 + [-1.0] * (n_per - 2), dtype=np.float32),
        ])
        recording_ids = np.array(["rec_a"] * n_per + ["rec_b"] * n_per + ["rec_c"] * n_per)
        sample_end_times_s = np.concatenate([
            np.arange(n_per, dtype=np.float32),
            np.arange(n_per, dtype=np.float32),
            np.arange(n_per, dtype=np.float32),
        ])

        train_idx, val_idx, test_idx = _split_indices_by_recording_timeline(
            y=y,
            recording_ids=recording_ids,
            sample_end_times_s=sample_end_times_s,
            train_ratio=0.60,
            val_ratio=0.20,
            seed=42,
        )

        self.assertGreater(len(train_idx), 0)
        self.assertGreater(len(val_idx), 0)
        self.assertGreater(len(test_idx), 0)

        merged = np.concatenate([train_idx, val_idx, test_idx])
        self.assertEqual(len(np.unique(merged)), len(y))
        self.assertEqual(len(merged), len(y))


if __name__ == "__main__":
    unittest.main()
