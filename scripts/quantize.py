#!/usr/bin/env python
"""
Post-training quantization and export utility.

Supports:
- dynamic_int8: Dynamic quantization for nn.Linear / nn.LSTM / nn.GRU
- float16:      FP16 weight conversion (useful for mobile/server validation)

Outputs:
- Quantized checkpoint (.pt)
- Optional TorchScript export (.ts)
- Optional ONNX export (.onnx)
- JSON report with size + latency comparison

Examples:
    python scripts/quantize.py --model-type eegnet
    python scripts/quantize.py --model-type temporal_transformer --mode float16
    python scripts/quantize.py --config models/config.yaml --checkpoint models/best_model.pt
    python scripts/quantize.py --model-type tcn --no-torchscript
"""

import os
import sys
import json
import time
import argparse
import tempfile
from pathlib import Path
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config, DEFAULT_CONFIG
from src.models import ModelFactory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantize and export trained models")
    parser.add_argument("--config", type=str, default="models/config.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default="models/best_model.pt", help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="models/quantized", help="Output directory")
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        choices=[
            "ecg_lstm", "cnn_lstm", "multimodal",
            "eegnet", "mobilenet_1d", "tcn", "inception_1d",
            "temporal_transformer", "multimodal_transformer",
        ],
        help="Override model type from config",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="dynamic_int8",
        choices=["dynamic_int8", "float16"],
        help="Quantization/export mode",
    )
    parser.add_argument("--sequence-length", type=int, default=600, help="Input sequence length")
    parser.add_argument("--batch-size", type=int, default=1, help="Benchmark batch size")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations for latency")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--no-torchscript", action="store_true", help="Skip TorchScript export")
    parser.add_argument("--no-onnx", action="store_false", dest="export_onnx", help="Skip ONNX export")
    parser.add_argument("--onnx-opset", type=int, default=17, help="ONNX opset version")
    return parser.parse_args()


def _load_config(config_path: Path) -> Config:
    if config_path.exists():
        return Config.from_yaml(str(config_path))
    return DEFAULT_CONFIG


def _make_example_inputs(config: Config, batch_size: int, sequence_length: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    model_type = config.model.model_type
    if ModelFactory.is_multimodal(model_type):
        ecg = torch.randn(batch_size, sequence_length, config.model.ecg_feature_dim, dtype=torch.float32)
        eeg = torch.randn(batch_size, sequence_length, config.model.eeg_feature_dim, dtype=torch.float32)
        motion = torch.randn(batch_size, sequence_length, config.model.motion_feature_dim, dtype=torch.float32)
        return ecg, eeg, motion
    return torch.randn(batch_size, sequence_length, config.model.ecg_feature_dim, dtype=torch.float32)


def _measure_state_dict_size_mb(model: nn.Module) -> float:
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as handle:
        tmp_path = Path(handle.name)
    try:
        torch.save(model.state_dict(), tmp_path)
        size_mb = tmp_path.stat().st_size / (1024 * 1024)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return float(size_mb)


def _benchmark_latency_ms(model: nn.Module, example_inputs, warmup: int, iters: int) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(max(0, warmup)):
            if isinstance(example_inputs, tuple):
                _ = model(*example_inputs)
            else:
                _ = model(example_inputs)

        t0 = time.perf_counter()
        for _ in range(max(1, iters)):
            if isinstance(example_inputs, tuple):
                _ = model(*example_inputs)
            else:
                _ = model(example_inputs)
        elapsed_s = time.perf_counter() - t0

    return float((elapsed_s / max(1, iters)) * 1000.0)


def _load_model(config: Config, checkpoint_path: Path) -> nn.Module:
    model = ModelFactory.create_model(config.model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state = checkpoint["model_state"]
    elif isinstance(checkpoint, dict):
        state = checkpoint
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(checkpoint)}")

    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint/model mismatch while loading weights. "
            f"Requested model_type='{config.model.model_type}', checkpoint='{checkpoint_path}'. "
            "Use matching --model-type and --checkpoint (or --config) from the same training run."
        ) from exc
    model.eval()
    return model


def _apply_quantization(model: nn.Module, mode: str, model_type: str) -> nn.Module:
    if mode == "dynamic_int8":
        if "transformer" in model_type:
            raise ValueError(
                "dynamic_int8 is not stable for this Transformer-based model in current PyTorch flow. "
                "Use --mode float16 for temporal_transformer/multimodal_transformer."
            )
        qmodel = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.LSTM, nn.GRU},
            dtype=torch.qint8,
        )
        return qmodel

    if mode == "float16":
        return model.half()

    raise ValueError(f"Unknown quantization mode: {mode}")


def _export_torchscript(model: nn.Module, example_inputs, output_path: Path) -> str:
    try:
        with torch.no_grad():
            if isinstance(example_inputs, tuple):
                traced = torch.jit.trace(model, example_inputs, strict=False)
            else:
                traced = torch.jit.trace(model, example_inputs, strict=False)
            traced = torch.jit.optimize_for_inference(traced)
            traced.save(str(output_path))
        return "ok"
    except Exception as exc:  # noqa: BLE001
        return f"failed: {exc}"


def _export_onnx(model: nn.Module, example_inputs, output_path: Path, opset: int = 17) -> str:
    try:
        model.eval()
        with torch.no_grad():
            if isinstance(example_inputs, tuple):
                input_names = [f"input_{i}" for i in range(len(example_inputs))]
                dynamic_axes = {
                    name: {0: "batch_size", 1: "sequence_length"}
                    for name in input_names
                }
                dynamic_axes["output"] = {0: "batch_size", 1: "sequence_length"}
            else:
                input_names = ["input"]
                dynamic_axes = {
                    "input": {0: "batch_size", 1: "sequence_length"},
                    "output": {0: "batch_size", 1: "sequence_length"},
                }
            torch.onnx.export(
                model,
                example_inputs,
                str(output_path),
                export_params=True,
                opset_version=opset,
                do_constant_folding=True,
                input_names=input_names,
                output_names=["output"],
                dynamic_axes=dynamic_axes,
                training=torch.onnx.TrainingMode.EVAL,
            )
        return "ok"
    except Exception as exc:  # noqa: BLE001
        return f"failed: {exc}"


def main() -> int:
    args = parse_args()

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Train first or pass --checkpoint."
        )

    config = _load_config(config_path)
    if args.model_type:
        config.model.model_type = args.model_type

    model_type = config.model.model_type
    print(f"Model type: {model_type}")
    print(f"Mode: {args.mode}")

    baseline_model = _load_model(config, checkpoint_path).cpu().float().eval()
    example_inputs = _make_example_inputs(config, args.batch_size, args.sequence_length)

    if args.mode == "float16":
        if isinstance(example_inputs, tuple):
            example_inputs_q = tuple(inp.half() for inp in example_inputs)
        else:
            example_inputs_q = example_inputs.half()
    else:
        example_inputs_q = example_inputs

    baseline_size_mb = _measure_state_dict_size_mb(baseline_model)
    baseline_latency_ms = _benchmark_latency_ms(
        baseline_model, example_inputs, args.warmup, args.iters
    )

    qmodel = _apply_quantization(baseline_model, args.mode, model_type).cpu().eval()
    quant_size_mb = _measure_state_dict_size_mb(qmodel)
    quant_latency_ms = _benchmark_latency_ms(
        qmodel, example_inputs_q, args.warmup, args.iters
    )

    stem = f"{model_type}_{args.mode}"
    quant_ckpt_path = output_dir / f"{stem}.pt"
    torch.save({"model_state": qmodel.state_dict(), "mode": args.mode, "model_type": model_type}, quant_ckpt_path)

    torchscript_status = "skipped"
    torchscript_path = output_dir / f"{stem}.ts"
    if not args.no_torchscript:
        torchscript_status = _export_torchscript(qmodel, example_inputs_q, torchscript_path)

    onnx_status = "skipped"
    onnx_path = output_dir / f"{stem}.onnx"
    if args.export_onnx:
        onnx_inputs = example_inputs_q if args.mode != "dynamic_int8" else example_inputs
        if args.mode == "dynamic_int8":
            print("NOTE: dynamic int8 ONNX export is unstable; exporting baseline float model for ONNX instead.")
        onnx_status = _export_onnx(qmodel if args.mode != "dynamic_int8" else baseline_model, onnx_inputs, onnx_path, args.onnx_opset)

    report: Dict[str, object] = {
        "model_type": model_type,
        "mode": args.mode,
        "checkpoint": str(checkpoint_path),
        "quant_checkpoint": str(quant_ckpt_path),
        "torchscript": str(torchscript_path),
        "torchscript_status": torchscript_status,
        "onnx": str(onnx_path),
        "onnx_status": onnx_status,
        "baseline_size_mb": baseline_size_mb,
        "quantized_size_mb": quant_size_mb,
        "size_reduction_pct": (1.0 - quant_size_mb / max(1e-12, baseline_size_mb)) * 100.0,
        "baseline_latency_ms": baseline_latency_ms,
        "quantized_latency_ms": quant_latency_ms,
        "latency_speedup_x": baseline_latency_ms / max(1e-12, quant_latency_ms),
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
    }

    report_path = output_dir / f"{stem}_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 64)
    print("QUANTIZATION SUMMARY")
    print("=" * 64)
    print(f"Baseline size:  {baseline_size_mb:.4f} MB")
    print(f"Quantized size: {quant_size_mb:.4f} MB")
    print(f"Reduction:      {report['size_reduction_pct']:.2f}%")
    print(f"Baseline lat:   {baseline_latency_ms:.4f} ms")
    print(f"Quantized lat:  {quant_latency_ms:.4f} ms")
    print(f"Speedup:        {report['latency_speedup_x']:.3f}x")
    print(f"Saved checkpoint: {quant_ckpt_path}")
    print(f"TorchScript:      {torchscript_status}")
    print(f"ONNX export:      {onnx_status}")
    print(f"Report:           {report_path}")
    print("=" * 64)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
