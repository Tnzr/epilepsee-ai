import unittest

import numpy as np

from config.config import DEFAULT_CONFIG
from src.evaluation import Evaluator


class TestBayesianLongSweep(unittest.TestCase):
    def setUp(self):
        self.config = DEFAULT_CONFIG
        self.evaluator = Evaluator(self.config)

    def test_empty_inputs(self):
        payload = self.evaluator.simulate_bayesian_long_sweep(
            pred_preictal=np.array([], dtype=np.float32),
            pred_countdown=np.array([], dtype=np.float32),
            true_countdown=np.array([], dtype=np.float32),
            sample_end_times_s=np.array([], dtype=np.float32),
            recording_ids=np.array([], dtype=str),
        )

        self.assertEqual(payload["fused_preictal"].size, 0)
        self.assertEqual(payload["memory_risk"].size, 0)
        self.assertEqual(payload["uncertainty"].size, 0)
        self.assertEqual(payload["token_id"].size, 0)
        self.assertEqual(payload["timeline_order_idx"].size, 0)
        self.assertEqual(payload["metrics"], {})

    def test_shapes_ranges_and_metrics(self):
        n = 12
        pred_preictal = np.linspace(0.05, 0.95, n, dtype=np.float32)
        pred_countdown = np.linspace(-1.0, 9.5, n, dtype=np.float32)
        true_countdown = np.array([-1, -1, 8, 7, 6, 5, -1, 4, 3, 2, 1, 0], dtype=np.float32)
        sample_end_times_s = np.arange(n, dtype=np.float32)
        recording_ids = np.array(["rec_1"] * n)

        payload = self.evaluator.simulate_bayesian_long_sweep(
            pred_preictal=pred_preictal,
            pred_countdown=pred_countdown,
            true_countdown=true_countdown,
            sample_end_times_s=sample_end_times_s,
            recording_ids=recording_ids,
        )

        for key in ["fused_preictal", "fused_preictal_smooth", "memory_risk", "uncertainty", "token_id"]:
            self.assertEqual(len(payload[key]), n)

        self.assertEqual(len(payload["timeline_order_idx"]), n)

        self.assertTrue(np.all(payload["fused_preictal"] >= 0.0))
        self.assertTrue(np.all(payload["fused_preictal"] <= 1.0))
        self.assertTrue(np.all(payload["fused_preictal_smooth"] >= 0.0))
        self.assertTrue(np.all(payload["fused_preictal_smooth"] <= 1.0))
        self.assertTrue(np.all(payload["memory_risk"] >= 0.0))
        self.assertTrue(np.all(payload["memory_risk"] <= 1.0))
        self.assertTrue(np.all(payload["uncertainty"] >= 0.0))

        metrics = payload["metrics"]
        required_metric_keys = [
            "bayes_accuracy",
            "bayes_auroc",
            "bayes_f1_score",
            "bayes_sensitivity",
            "bayes_specificity",
            "bayes_precision",
            "bayes_ece",
            "bayes_brier",
        ]
        for metric_key in required_metric_keys:
            self.assertIn(metric_key, metrics)

    def test_recording_state_reset(self):
        # Two recordings with identical streams should produce identical
        # fused trajectories when memory state resets per recording.
        rec_seq_pred = np.array([0.1, 0.4, 0.7, 0.8, 0.6, 0.2], dtype=np.float32)
        rec_seq_countdown = np.array([-1.0, 9.0, 7.0, 5.0, 3.0, 1.0], dtype=np.float32)
        rec_seq_true = np.array([-1.0, 8.5, 7.0, 4.5, 2.5, 0.5], dtype=np.float32)
        rec_seq_times = np.arange(6, dtype=np.float32)

        pred_preictal = np.concatenate([rec_seq_pred, rec_seq_pred], axis=0)
        pred_countdown = np.concatenate([rec_seq_countdown, rec_seq_countdown], axis=0)
        true_countdown = np.concatenate([rec_seq_true, rec_seq_true], axis=0)
        sample_end_times_s = np.concatenate([rec_seq_times, rec_seq_times], axis=0)
        recording_ids = np.array(["rec_A"] * 6 + ["rec_B"] * 6)

        payload = self.evaluator.simulate_bayesian_long_sweep(
            pred_preictal=pred_preictal,
            pred_countdown=pred_countdown,
            true_countdown=true_countdown,
            sample_end_times_s=sample_end_times_s,
            recording_ids=recording_ids,
        )

        fused = payload["fused_preictal"]
        fused_smooth = payload["fused_preictal_smooth"]

        np.testing.assert_allclose(fused[:6], fused[6:], rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(fused_smooth[:6], fused_smooth[6:], rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
