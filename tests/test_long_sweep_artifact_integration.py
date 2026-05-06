import os
import tempfile
import unittest

import numpy as np

from config.config import DEFAULT_CONFIG
from src.evaluation import Evaluator


class TestLongSweepArtifactIntegration(unittest.TestCase):
    def setUp(self):
        self.config = DEFAULT_CONFIG
        self.evaluator = Evaluator(self.config)

    def test_bayes_payload_npz_roundtrip(self):
        n = 24
        pred_preictal = np.clip(np.sin(np.linspace(0, 3.14, n)) * 0.4 + 0.5, 0.0, 1.0).astype(np.float32)
        pred_countdown = np.linspace(-1.0, 9.5, n, dtype=np.float32)
        true_countdown = np.array(([-1.0] * 6) + list(np.linspace(9.0, 0.0, 18)), dtype=np.float32)
        sample_end_times_s = np.arange(n, dtype=np.float32)
        recording_ids = np.array(["rec_X"] * 12 + ["rec_Y"] * 12)

        payload = self.evaluator.simulate_bayesian_long_sweep(
            pred_preictal=pred_preictal,
            pred_countdown=pred_countdown,
            true_countdown=true_countdown,
            sample_end_times_s=sample_end_times_s,
            recording_ids=recording_ids,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = os.path.join(tmpdir, "test_predictions_long_sweep_bayes.npz")
            np.savez_compressed(
                npz_path,
                fused_preictal=payload["fused_preictal"],
                fused_preictal_smooth=payload["fused_preictal_smooth"],
                memory_risk=payload["memory_risk"],
                uncertainty=payload["uncertainty"],
                token_id=payload["token_id"],
                timeline_order_idx=payload["timeline_order_idx"],
                recording_ids=recording_ids,
                sample_end_times_s=sample_end_times_s,
            )

            loaded = np.load(npz_path)
            required_keys = {
                "fused_preictal",
                "fused_preictal_smooth",
                "memory_risk",
                "uncertainty",
                "token_id",
                "timeline_order_idx",
                "recording_ids",
                "sample_end_times_s",
            }
            self.assertTrue(required_keys.issubset(set(loaded.files)))

            self.assertEqual(loaded["fused_preictal"].shape[0], n)
            self.assertEqual(loaded["fused_preictal_smooth"].shape[0], n)
            self.assertEqual(loaded["memory_risk"].shape[0], n)
            self.assertEqual(loaded["uncertainty"].shape[0], n)
            self.assertEqual(loaded["token_id"].shape[0], n)
            self.assertEqual(loaded["timeline_order_idx"].shape[0], n)
            self.assertEqual(loaded["recording_ids"].shape[0], n)
            self.assertEqual(loaded["sample_end_times_s"].shape[0], n)

            self.assertTrue(np.all(loaded["fused_preictal"] >= 0.0))
            self.assertTrue(np.all(loaded["fused_preictal"] <= 1.0))
            self.assertTrue(np.all(loaded["fused_preictal_smooth"] >= 0.0))
            self.assertTrue(np.all(loaded["fused_preictal_smooth"] <= 1.0))


if __name__ == "__main__":
    unittest.main()
