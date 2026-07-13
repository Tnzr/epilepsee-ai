import numpy as np

from scripts.visualize_token_explainability import _rolling_mean_2d


def test_rolling_mean_handles_window_larger_than_sequence():
    matrix = np.arange(20, dtype=np.float32).reshape(5, 4)
    rolled = _rolling_mean_2d(matrix, window=25)

    assert rolled.shape == matrix.shape
    assert np.all(np.isfinite(rolled))


def test_rolling_mean_empty_matrix():
    matrix = np.zeros((0, 3), dtype=np.float32)
    rolled = _rolling_mean_2d(matrix, window=25)

    assert rolled.shape == matrix.shape
