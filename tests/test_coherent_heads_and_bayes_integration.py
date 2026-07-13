import unittest
from copy import deepcopy

import numpy as np
import torch

from config.config import DEFAULT_CONFIG, ModelConfig
from src.data_loader import SeizureDataset
from src.evaluation import Evaluator
from src.models import ModelFactory


class _TinyEvalModel(torch.nn.Module):
    """Small deterministic model for evaluator integration tests."""

    def __init__(self, in_dim: int, max_countdown: float):
        super().__init__()
        self.pre = torch.nn.Linear(in_dim, 1)
        self.reg = torch.nn.Linear(in_dim, 1)
        self.max_countdown = float(max_countdown)

    def forward(self, x: torch.Tensor):
        pooled = x.mean(dim=1)
        pre = torch.sigmoid(self.pre(pooled)).squeeze(-1)
        countdown = torch.sigmoid(self.reg(pooled)).squeeze(-1) * self.max_countdown
        return pre, countdown


class TestCoherentHeadsMultiScale(unittest.TestCase):
    def _make_cfg(self, model_type: str, coherent_heads: bool) -> ModelConfig:
        cfg = ModelConfig()
        cfg.model_type = model_type
        cfg.coherent_heads = coherent_heads
        cfg.dropout = 0.0
        return cfg

    def _assert_delta_one(self, model_type: str):
        base_cfg = self._make_cfg(model_type, coherent_heads=False)
        coh_cfg = self._make_cfg(model_type, coherent_heads=True)

        torch.manual_seed(7)
        base_model = ModelFactory.create_model(base_cfg)
        torch.manual_seed(7)
        coh_model = ModelFactory.create_model(coh_cfg)

        self.assertEqual(coh_model.fc_regress[0].in_features, base_model.fc_regress[0].in_features + 1)

        x = torch.randn(2, 96, base_cfg.ecg_feature_dim)
        p_base, c_base = base_model(x)
        p_coh, c_coh = coh_model(x)

        self.assertEqual(tuple(p_base.shape), (2,))
        self.assertEqual(tuple(c_base.shape), (2,))
        self.assertEqual(tuple(p_coh.shape), (2,))
        self.assertEqual(tuple(c_coh.shape), (2,))

    def test_tcn_coherent_head_wiring(self):
        self._assert_delta_one("tcn")

    def test_inception_coherent_head_wiring(self):
        self._assert_delta_one("inception_1d")

    def test_temporal_transformer_coherent_head_wiring(self):
        self._assert_delta_one("temporal_transformer")


class TestEvaluatorDefaultBayesIntegration(unittest.TestCase):
    def test_evaluate_includes_bayes_metrics_when_metadata_exists(self):
        config = deepcopy(DEFAULT_CONFIG)
        config.model.model_type = "ecg_lstm"

        n = 30
        t = 64
        f = int(config.model.ecg_feature_dim)

        rng = np.random.default_rng(123)
        features = rng.normal(size=(n, t, f)).astype(np.float32)
        labels = np.array(([-1.0] * 10) + list(np.linspace(9.0, 0.0, 20)), dtype=np.float32)

        sample_end_times_s = np.arange(n, dtype=np.float32)
        recording_ids = np.array(["rec_A"] * 15 + ["rec_B"] * 15)

        dataset = SeizureDataset(
            features=features,
            labels=labels,
            sample_end_times_s=sample_end_times_s,
            recording_ids=recording_ids,
        )

        model = _TinyEvalModel(in_dim=f, max_countdown=config.model.output_countdown_max)
        evaluator = Evaluator(config)

        metrics = evaluator.evaluate(model, dataset, torch.device("cpu"))

        self.assertIn("bayes_accuracy", metrics)
        self.assertIn("bayes_auroc", metrics)
        self.assertIn("bayes_ece", metrics)
        self.assertIn("bayes_brier", metrics)

    def test_evaluate_omits_bayes_metrics_when_disabled(self):
        config = deepcopy(DEFAULT_CONFIG)
        config.model.model_type = "ecg_lstm"
        config.evaluation.enable_bayesian_memory_eval = False

        n = 24
        t = 48
        f = int(config.model.ecg_feature_dim)
        rng = np.random.default_rng(99)
        features = rng.normal(size=(n, t, f)).astype(np.float32)
        labels = np.array(([-1.0] * 8) + list(np.linspace(8.0, 0.0, 16)), dtype=np.float32)
        sample_end_times_s = np.arange(n, dtype=np.float32)
        recording_ids = np.array(["rec_A"] * 12 + ["rec_B"] * 12)

        dataset = SeizureDataset(
            features=features,
            labels=labels,
            sample_end_times_s=sample_end_times_s,
            recording_ids=recording_ids,
        )

        model = _TinyEvalModel(in_dim=f, max_countdown=config.model.output_countdown_max)
        evaluator = Evaluator(config)
        metrics = evaluator.evaluate(model, dataset, torch.device("cpu"))

        self.assertNotIn("bayes_accuracy", metrics)
        self.assertNotIn("bayes_auroc", metrics)
        self.assertNotIn("bayes_ece", metrics)
        self.assertNotIn("bayes_brier", metrics)


if __name__ == "__main__":
    unittest.main()
