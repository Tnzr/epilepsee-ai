#!/usr/bin/env python
"""Token and embedding explainability visualizer.

Creates:
- Token timeline
- Token histogram with semantic decoding
- Rolling token heatmap waterfall
- Rolling embedding/channel heatmap waterfall

It supports two artifact styles:
1) Standard predictions: models/test_predictions.npz
2) Long-sweep Bayesian output: models/test_predictions_long_sweep_bayes.npz

If token ids are not present in artifacts, tokens are reconstructed from
pred_preictal and pred_countdown using the same discretization scheme used by
Bayesian long-sweep simulation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def _safe_1d(arr: Optional[np.ndarray], n: int, fill: float = 0.0) -> np.ndarray:
    if arr is None:
        return np.full(n, fill, dtype=np.float32)
    out = np.asarray(arr).reshape(-1).astype(np.float32)
    if out.size == n:
        return out
    if out.size == 0:
        return np.full(n, fill, dtype=np.float32)
    if out.size > n:
        return out[:n]
    padded = np.full(n, fill, dtype=np.float32)
    padded[:out.size] = out
    return padded


def _infer_n(pred: Dict[str, np.ndarray], bayes: Dict[str, np.ndarray]) -> int:
    for key in [
        "pred_preictal",
        "pred_countdown",
        "true_preictal",
        "true_countdown",
        "token_id",
        "fused_preictal_smooth",
    ]:
        if key in bayes:
            return int(np.asarray(bayes[key]).reshape(-1).shape[0])
        if key in pred:
            return int(np.asarray(pred[key]).reshape(-1).shape[0])
    raise ValueError("Could not infer sample count from artifacts")


def _reconstruct_tokens(
    pred_preictal: np.ndarray,
    pred_countdown: np.ndarray,
    n_countdown_bins: int,
    n_prob_bins: int,
    max_countdown: float,
) -> Tuple[np.ndarray, int]:
    n = pred_preictal.shape[0]
    token_count = (n_countdown_bins + 1) * n_prob_bins
    token_id = np.zeros(n, dtype=np.int32)

    for i in range(n):
        p_raw = float(np.clip(pred_preictal[i], 0.0, 1.0))
        cd_raw = float(pred_countdown[i])

        if cd_raw < 0:
            countdown_bin = 0
        else:
            frac = min(max(cd_raw / max(max_countdown, 1e-6), 0.0), 0.999999)
            countdown_bin = 1 + int(frac * n_countdown_bins)

        prob_bin = min(int(p_raw * n_prob_bins), n_prob_bins - 1)
        token_id[i] = countdown_bin * n_prob_bins + prob_bin

    return token_id, token_count


def _rank_bin(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Assign bins by rank so ties/collapsed values still distribute across bins."""
    n = values.shape[0]
    if n == 0:
        return np.array([], dtype=np.int32)
    order = np.argsort(values, kind="mergesort")
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n, dtype=np.int64)
    frac = (rank + 0.5) / float(n)
    bins = np.minimum((frac * n_bins).astype(np.int32), n_bins - 1)
    return bins


def _adaptive_tokens_from_rank(
    pred_preictal: np.ndarray,
    pred_countdown: np.ndarray,
    n_countdown_bins: int,
    n_prob_bins: int,
) -> Tuple[np.ndarray, int]:
    """Build adaptive tokens using rank-based bins over observed outputs."""
    pbin = _rank_bin(pred_preictal.astype(np.float32), n_prob_bins)
    cbin_pos = _rank_bin(pred_countdown.astype(np.float32), n_countdown_bins)

    countdown_bin = np.where(pred_countdown < 0, 0, 1 + cbin_pos).astype(np.int32)
    token_id = (countdown_bin * n_prob_bins + pbin).astype(np.int32)
    token_count = (n_countdown_bins + 1) * n_prob_bins
    return token_id, token_count


def _token_entropy(counts: np.ndarray) -> float:
    total = float(np.sum(counts))
    if total <= 0:
        return 0.0
    p = counts[counts > 0].astype(np.float64) / total
    return float(-np.sum(p * np.log2(p)))


def _decode_token(token: int, n_countdown_bins: int, n_prob_bins: int, max_countdown: float) -> Dict[str, object]:
    countdown_bin = token // n_prob_bins
    prob_bin = token % n_prob_bins

    prob_lo = prob_bin / float(n_prob_bins)
    prob_hi = (prob_bin + 1) / float(n_prob_bins)

    if countdown_bin == 0:
        cd_text = "interictal/unknown"
    else:
        cd_lo = (countdown_bin - 1) / float(n_countdown_bins) * max_countdown
        cd_hi = countdown_bin / float(n_countdown_bins) * max_countdown
        cd_text = f"{cd_lo:.2f}-{cd_hi:.2f} min to event"

    meaning = f"P(preictal) in [{prob_lo:.2f}, {prob_hi:.2f}), countdown {cd_text}"

    return {
        "token_id": int(token),
        "countdown_bin": int(countdown_bin),
        "prob_bin": int(prob_bin),
        "prob_range": f"[{prob_lo:.2f}, {prob_hi:.2f})",
        "meaning": meaning,
    }


def _rolling_mean_2d(matrix: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window == 1:
        return matrix.astype(np.float32)

    kernel = np.ones(window, dtype=np.float32) / float(window)
    out = np.zeros_like(matrix, dtype=np.float32)
    for col in range(matrix.shape[1]):
        out[:, col] = np.convolve(matrix[:, col], kernel, mode="same")
    return out


def _zscore_rows(matrix: np.ndarray) -> np.ndarray:
    out = np.zeros_like(matrix, dtype=np.float32)
    for i in range(matrix.shape[0]):
        row = matrix[i].astype(np.float32)
        mu = float(np.mean(row))
        sigma = float(np.std(row))
        if sigma < 1e-8:
            out[i] = 0.0
        else:
            out[i] = (row - mu) / sigma
    return out


def _build_embedding_matrix(pred: Dict[str, np.ndarray], bayes: Dict[str, np.ndarray], n: int) -> Tuple[np.ndarray, List[str]]:
    # Prefer explicit embedding-like arrays if available.
    for key in ["embeddings", "embedding", "latent", "z", "token_embedding"]:
        if key in bayes:
            arr = np.asarray(bayes[key])
            if arr.ndim == 2 and arr.shape[0] == n:
                return arr.astype(np.float32), [f"emb_{i}" for i in range(arr.shape[1])]
        if key in pred:
            arr = np.asarray(pred[key])
            if arr.ndim == 2 and arr.shape[0] == n:
                return arr.astype(np.float32), [f"emb_{i}" for i in range(arr.shape[1])]

    channels: List[np.ndarray] = []
    names: List[str] = []

    def add_channel(name: str, source: Dict[str, np.ndarray], key: str, normalize: bool = False) -> None:
        if key not in source:
            return
        v = _safe_1d(source[key], n)
        if normalize:
            vmax = float(np.max(np.abs(v)))
            if vmax > 1e-8:
                v = v / vmax
        channels.append(v)
        names.append(name)

    add_channel("pred_preictal", pred, "pred_preictal")
    add_channel("pred_countdown", pred, "pred_countdown", normalize=True)
    add_channel("true_preictal", pred, "true_preictal")
    add_channel("true_countdown", pred, "true_countdown", normalize=True)

    add_channel("fused_preictal", bayes, "fused_preictal")
    add_channel("fused_preictal_smooth", bayes, "fused_preictal_smooth")
    add_channel("memory_risk", bayes, "memory_risk")
    add_channel("uncertainty", bayes, "uncertainty")

    if not channels:
        # Worst case fallback keeps script robust even with minimal artifacts.
        channels = [np.zeros(n, dtype=np.float32)]
        names = ["empty_channel"]

    matrix = np.stack(channels, axis=1).astype(np.float32)
    return matrix, names


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize token explainability and rolling heatmaps")
    parser.add_argument("--predictions-npz", type=str, default="models/test_predictions.npz")
    parser.add_argument("--bayes-npz", type=str, default="models/test_predictions_long_sweep_bayes.npz")
    parser.add_argument("--output-dir", type=str, default="visualizations/explainability")
    parser.add_argument("--n-countdown-bins", type=int, default=8)
    parser.add_argument("--n-prob-bins", type=int, default=4)
    parser.add_argument("--max-countdown-min", type=float, default=10.0)
    parser.add_argument("--rolling-window", type=int, default=25)
    parser.add_argument("--max-hist-tokens", type=int, default=24)
    parser.add_argument(
        "--tokenization-scheme",
        type=str,
        default="auto",
        choices=["auto", "fixed", "adaptive"],
        help="Tokenization scheme: fixed Bayesian bins, adaptive rank bins, or auto fallback",
    )
    parser.add_argument(
        "--min-unique-tokens",
        type=int,
        default=4,
        help="Auto mode fallback threshold when fixed tokens collapse",
    )
    args = parser.parse_args()

    pred_path = Path(args.predictions_npz)
    bayes_path = Path(args.bayes_npz)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred = _load_npz(pred_path)
    bayes = _load_npz(bayes_path) if bayes_path.exists() else {}

    n = _infer_n(pred, bayes)

    pred_preictal = _safe_1d(pred.get("pred_preictal"), n)
    pred_countdown = _safe_1d(pred.get("pred_countdown"), n)

    token_count = int((args.n_countdown_bins + 1) * args.n_prob_bins)

    has_bayes_token = "token_id" in bayes
    if has_bayes_token:
        token_id_fixed = _safe_1d(bayes["token_id"], n).astype(np.int32)
    else:
        token_id_fixed, _ = _reconstruct_tokens(
            pred_preictal=pred_preictal,
            pred_countdown=pred_countdown,
            n_countdown_bins=int(args.n_countdown_bins),
            n_prob_bins=int(args.n_prob_bins),
            max_countdown=float(args.max_countdown_min),
        )

    token_id_adaptive, _ = _adaptive_tokens_from_rank(
        pred_preictal=pred_preictal,
        pred_countdown=pred_countdown,
        n_countdown_bins=int(args.n_countdown_bins),
        n_prob_bins=int(args.n_prob_bins),
    )

    fixed_unique = int(np.unique(token_id_fixed).size)
    adaptive_unique = int(np.unique(token_id_adaptive).size)

    if args.tokenization_scheme == "fixed":
        token_id = token_id_fixed
        scheme_used = "fixed"
    elif args.tokenization_scheme == "adaptive":
        token_id = token_id_adaptive
        scheme_used = "adaptive"
    else:
        if fixed_unique < int(args.min_unique_tokens):
            token_id = token_id_adaptive
            scheme_used = "adaptive(auto-fallback)"
        else:
            token_id = token_id_fixed
            scheme_used = "fixed(auto)"

    if "sample_end_times_s" in bayes:
        t = _safe_1d(bayes["sample_end_times_s"], n)
        t = t - float(np.min(t))
        x = t / 60.0
        x_label = "Time since start (minutes)"
    else:
        x = np.arange(n, dtype=np.float32)
        x_label = "Sample index"

    one_hot = np.zeros((n, token_count), dtype=np.float32)
    valid_mask = (token_id >= 0) & (token_id < token_count)
    one_hot[np.arange(n)[valid_mask], token_id[valid_mask]] = 1.0
    token_rolling = _rolling_mean_2d(one_hot, window=int(args.rolling_window))

    emb_matrix, emb_names = _build_embedding_matrix(pred, bayes, n)
    emb_roll = _rolling_mean_2d(emb_matrix, window=int(args.rolling_window))
    emb_roll_norm = _zscore_rows(emb_roll.T)

    counts = np.bincount(token_id.clip(0, token_count - 1), minlength=token_count)
    top_tokens = np.argsort(counts)[::-1]
    top_tokens = top_tokens[counts[top_tokens] > 0]
    top_tokens = top_tokens[: max(1, int(args.max_hist_tokens))]

    fig = plt.figure(figsize=(18, 13))
    gs = GridSpec(5, 1, figure=fig, hspace=0.35)

    # 1) Token timeline + preictal traces
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(x, pred_preictal, color="#1f77b4", linewidth=1.2, alpha=0.9, label="pred_preictal")
    if "fused_preictal_smooth" in bayes:
        ax1.plot(x, _safe_1d(bayes["fused_preictal_smooth"], n), color="#d62728", linewidth=1.2, alpha=0.8, label="fused_preictal_smooth")
    ax1.scatter(x, token_id, c=token_id, cmap="tab20", s=8, alpha=0.5, label="token_id")
    ax1.set_ylabel("Prob / Token")
    ax1.set_title("Token Creation Timeline")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper right", fontsize=9)

    # 2) Histogram
    ax2 = fig.add_subplot(gs[1])
    ax2.bar(np.arange(len(top_tokens)), counts[top_tokens], color="#2ca02c", alpha=0.85)
    ax2.set_xticks(np.arange(len(top_tokens)))
    ax2.set_xticklabels([str(int(tk)) for tk in top_tokens], rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Count")
    ax2.set_title("Token Histogram (Most Frequent Tokens)")
    ax2.grid(True, alpha=0.2, axis="y")

    # 3) Rolling token heatmap waterfall
    ax3 = fig.add_subplot(gs[2])
    im3 = ax3.imshow(
        token_rolling.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="magma",
        extent=[float(x[0]), float(x[-1]), 0, token_count - 1],
    )
    ax3.set_ylabel("Token ID")
    ax3.set_title(f"Rolling Token Heatmap Waterfall (window={int(args.rolling_window)})")
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.018, pad=0.01)
    cbar3.set_label("Rolling token occupancy")

    # 4) Token correlation matrix
    ax_corr = fig.add_subplot(gs[3])
    # Compute correlation between token time series
    if token_rolling.shape[1] > 1:
        corr_matrix = np.corrcoef(token_rolling.T)
        im_corr = ax_corr.imshow(
            corr_matrix,
            aspect="equal",
            origin="upper",
            interpolation="nearest",
            cmap="coolwarm",
            vmin=-1, vmax=1,
        )
        ax_corr.set_xticks(np.arange(token_count))
        ax_corr.set_yticks(np.arange(token_count))
        ax_corr.set_xticklabels([str(i) for i in range(token_count)], fontsize=6)
        ax_corr.set_yticklabels([str(i) for i in range(token_count)], fontsize=6)
        ax_corr.set_title("Token Activation Correlation Matrix")
        cbar_corr = plt.colorbar(im_corr, ax=ax_corr, fraction=0.046, pad=0.04)
        cbar_corr.set_label("Correlation")
    else:
        ax_corr.text(0.5, 0.5, "Not enough tokens for correlation", ha="center", va="center", transform=ax_corr.transAxes)
        ax_corr.set_title("Token Correlation Matrix (insufficient data)")

    # 5) Rolling embedding/channel heatmap waterfall
    ax4 = fig.add_subplot(gs[4])
    im4 = ax4.imshow(
        emb_roll_norm,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="coolwarm",
        extent=[float(x[0]), float(x[-1]), 0, len(emb_names) - 1],
    )
    ax4.set_yticks(np.arange(len(emb_names)))
    ax4.set_yticklabels(emb_names, fontsize=8)
    ax4.set_ylabel("Embedding / channel")
    ax4.set_xlabel(x_label)
    ax4.set_title(f"Rolling Embedding Heatmap Waterfall (window={int(args.rolling_window)}, z-score by channel)")
    cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.018, pad=0.01)
    cbar4.set_label("Normalized activation")

    fig.suptitle(
        f"Explainability: Token Semantics, Histogram, and Rolling Waterfalls | scheme={scheme_used} | unique_tokens={int(np.unique(token_id).size)}",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    fig.subplots_adjust(top=0.96, bottom=0.05)

    fig_path = output_dir / "token_embedding_explainability.png"
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Save decoded token table for semantic inspection
    csv_path = output_dir / "token_meaning_histogram.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["token_id", "count", "countdown_bin", "prob_bin", "prob_range", "meaning"],
        )
        writer.writeheader()
        for tk in top_tokens:
            decoded = _decode_token(
                int(tk),
                n_countdown_bins=int(args.n_countdown_bins),
                n_prob_bins=int(args.n_prob_bins),
                max_countdown=float(args.max_countdown_min),
            )
            writer.writerow({
                "token_id": int(tk),
                "count": int(counts[tk]),
                "countdown_bin": decoded["countdown_bin"],
                "prob_bin": decoded["prob_bin"],
                "prob_range": decoded["prob_range"],
                "meaning": decoded["meaning"],
            })

    diag = {
        "n_samples": int(n),
        "scheme_used": scheme_used,
        "has_bayes_token": bool(has_bayes_token),
        "fixed_unique_tokens": fixed_unique,
        "adaptive_unique_tokens": adaptive_unique,
        "selected_unique_tokens": int(np.unique(token_id).size),
        "fixed_entropy_bits": _token_entropy(np.bincount(token_id_fixed.clip(0, token_count - 1), minlength=token_count)),
        "adaptive_entropy_bits": _token_entropy(np.bincount(token_id_adaptive.clip(0, token_count - 1), minlength=token_count)),
        "selected_entropy_bits": _token_entropy(counts),
        "pred_preictal_min": float(np.min(pred_preictal)),
        "pred_preictal_max": float(np.max(pred_preictal)),
        "pred_countdown_min": float(np.min(pred_countdown)),
        "pred_countdown_max": float(np.max(pred_countdown)),
    }
    diag_path = output_dir / "token_diagnostics.json"
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2)

    print(f"Saved explainability figure: {fig_path}")
    print(f"Saved token semantics table: {csv_path}")
    print(f"Saved token diagnostics: {diag_path}")


if __name__ == "__main__":
    main()
