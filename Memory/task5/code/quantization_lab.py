#!/usr/bin/env python3
"""A CPU-only, pedagogical lab for LLM quantization concepts.

This program deliberately separates *mechanism* from *production implementation*.
It is not a replacement for CUDA low-bit kernels, real E4M3/E5M2 codecs, GPTQ,
or AWQ reference implementations.  It is an inspectable model of the invariants
behind W8A16, group-wise W4, calibration-aware weight quantization, and KV-cache
quantization.

Run from Memory/task5:
    python3 code/quantization_lab.py

Artifacts are written to ../assets and ../results relative to this file.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "assets"
RESULT_DIR = ROOT / "results"
EPS = 1e-12


@dataclass(frozen=True)
class TensorQuantization:
    """Integer codes and enough metadata to reconstruct a quantized tensor."""

    q: np.ndarray
    scale: np.ndarray
    bits: int
    original_shape: tuple[int, ...]
    group_size: int | None = None


def signed_qmax(bits: int) -> int:
    """Return the positive endpoint of a symmetric signed integer grid."""
    if bits < 2 or bits > 8:
        raise ValueError("This teaching lab accepts signed integer bit-widths in [2, 8].")
    return (1 << (bits - 1)) - 1


def symmetric_quantize(x: np.ndarray, bits: int) -> TensorQuantization:
    """Per-tensor absmax symmetric quantization with an all-zero safe path.

    The stored scale is `absmax / qmax`; dequantization therefore multiplies an
    integer code by scale.  For an all-zero tensor, scale is set to 1 rather than
    zero so the representation remains finite and exactly reconstructs zeros.
    """
    x32 = np.asarray(x, dtype=np.float32)
    qmax = signed_qmax(bits)
    absmax = float(np.max(np.abs(x32))) if x32.size else 0.0
    scale = np.float32(absmax / qmax) if absmax > 0.0 else np.float32(1.0)
    q = np.clip(np.rint(x32 / scale), -qmax, qmax).astype(np.int8)
    return TensorQuantization(q=q, scale=np.asarray(scale), bits=bits, original_shape=x32.shape)


def symmetric_dequantize(state: TensorQuantization) -> np.ndarray:
    """Recover the floating-point approximation stored by `symmetric_quantize`."""
    return state.q.astype(np.float32) * state.scale.astype(np.float32)


def groupwise_quantize_weights(
    weight: np.ndarray, bits: int = 4, group_size: int = 16
) -> TensorQuantization:
    """Quantize a 2-D [out_features, in_features] matrix per row and input group.

    This common teaching granularity gives each output-row / input-group slice an
    independent scale. Codes are kept in int8 for transparency; the storage
    accounting function separately reports packed logical 4-bit bytes.
    """
    w = np.asarray(weight, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError("groupwise_quantize_weights expects a 2-D linear weight matrix.")
    if group_size <= 0:
        raise ValueError("group_size must be positive.")

    out_features, in_features = w.shape
    n_groups = (in_features + group_size - 1) // group_size
    qmax = signed_qmax(bits)
    q = np.empty_like(w, dtype=np.int8)
    scales = np.empty((out_features, n_groups), dtype=np.float32)

    for row in range(out_features):
        for group in range(n_groups):
            start = group * group_size
            end = min(start + group_size, in_features)
            chunk = w[row, start:end]
            absmax = float(np.max(np.abs(chunk)))
            scale = np.float32(absmax / qmax) if absmax > 0.0 else np.float32(1.0)
            q[row, start:end] = np.clip(np.rint(chunk / scale), -qmax, qmax).astype(np.int8)
            scales[row, group] = scale

    return TensorQuantization(
        q=q, scale=scales, bits=bits, original_shape=w.shape, group_size=group_size
    )


def groupwise_dequantize_weights(state: TensorQuantization) -> np.ndarray:
    """Recover a 2-D groupwise weight matrix from codes and per-group scales."""
    if state.group_size is None or len(state.original_shape) != 2:
        raise ValueError("Expected a 2-D groupwise TensorQuantization state.")
    out_features, in_features = state.original_shape
    result = np.empty((out_features, in_features), dtype=np.float32)
    for row in range(out_features):
        for group in range(state.scale.shape[1]):
            start = group * state.group_size
            end = min(start + state.group_size, in_features)
            result[row, start:end] = state.q[row, start:end].astype(np.float32) * state.scale[row, group]
    return result


def output_mse(x: np.ndarray, original_weight: np.ndarray, restored_weight: np.ndarray) -> float:
    """Measure the reconstruction loss of a linear layer on a calibration batch."""
    reference = x @ original_weight.T
    restored = x @ restored_weight.T
    return float(np.mean((reference - restored) ** 2))


def matrix_mse(original: np.ndarray, restored: np.ndarray) -> float:
    return float(np.mean((np.asarray(original) - np.asarray(restored)) ** 2))


def logical_weight_bytes(state: TensorQuantization) -> float:
    """Return ideal packed storage bytes for codes plus FP32 scales.

    It intentionally excludes alignment, bias, zero-points, runtime workspace,
    and kernel-specific packing headers.  It therefore is a storage *accounting
    aid*, not a deployment-memory benchmark.
    """
    code_bytes = state.q.size * state.bits / 8.0
    scale_bytes = state.scale.size * 4.0
    return code_bytes + scale_bytes


class W8A16Linear:
    """Weight INT8 + activation FP16/FP32 teaching layer with float matmul.

    The implementation exposes the storage / numerical contract of W8A16. It
    does not call a native INT8 GEMM, so it must not be interpreted as a speed
    benchmark. In a real deployment, the target backend determines whether codes
    stay quantized throughout matrix multiplication.
    """

    def __init__(self, float_weight: np.ndarray, bias: np.ndarray | None = None) -> None:
        if np.asarray(float_weight).ndim != 2:
            raise ValueError("W8A16Linear expects a 2-D weight matrix.")
        self.quantized_weight = symmetric_quantize(np.asarray(float_weight, dtype=np.float32), bits=8)
        self.bias = None if bias is None else np.asarray(bias, dtype=np.float32)

    @property
    def restored_weight(self) -> np.ndarray:
        return symmetric_dequantize(self.quantized_weight)

    def forward(self, activation: np.ndarray) -> np.ndarray:
        """Use a floating-point activation with the restored INT8 weight approximation."""
        activation32 = np.asarray(activation, dtype=np.float32)
        result = activation32 @ self.restored_weight.T
        return result if self.bias is None else result + self.bias


def gptq_like_quantize(
    weight: np.ndarray,
    calibration_activations: np.ndarray,
    bits: int = 4,
    group_size: int = 16,
    damping: float = 1e-3,
) -> TensorQuantization:
    """A small, explicit *GPTQ-like* sequential error-compensation demonstration.

    GPTQ uses approximate second-order information for one-shot weight
    quantization. This teaching version forms a damped inverse Gram matrix from a
    calibration batch. It visits input columns from left to right, quantizes a
    current column, then propagates that column's residual into later unquantized
    columns using inverse-Gram couplings. It deliberately omits production GPTQ
    features such as activation ordering, Cholesky-based robust factorization,
    batching, optimized group handling, and bit packing.
    """
    original = np.asarray(weight, dtype=np.float32)
    x = np.asarray(calibration_activations, dtype=np.float32)
    if original.ndim != 2 or x.ndim != 2 or original.shape[1] != x.shape[1]:
        raise ValueError("Expected weight [out, in] and calibration [batch, in].")

    out_features, in_features = original.shape
    n_groups = (in_features + group_size - 1) // group_size
    qmax = signed_qmax(bits)
    gram = (x.T @ x) / max(x.shape[0], 1)
    h_inv = np.linalg.inv(gram + damping * np.eye(in_features, dtype=np.float32)).astype(np.float32)
    working = original.copy()
    q = np.zeros_like(original, dtype=np.int8)
    scales = np.zeros((out_features, n_groups), dtype=np.float32)

    for group in range(n_groups):
        start = group * group_size
        end = min(start + group_size, in_features)
        # Freeze a per-row scale for this group so all columns in the group share
        # one quantization grid, then make residual compensation observable.
        absmax = np.max(np.abs(working[:, start:end]), axis=1)
        group_scale = np.where(absmax > 0.0, absmax / qmax, 1.0).astype(np.float32)
        scales[:, group] = group_scale
        for column in range(start, end):
            codes = np.clip(np.rint(working[:, column] / group_scale), -qmax, qmax).astype(np.int8)
            restored_column = codes.astype(np.float32) * group_scale
            residual = working[:, column] - restored_column
            q[:, column] = codes
            # The residual is compensated only into columns not yet quantized.
            if column + 1 < in_features:
                denominator = float(h_inv[column, column])
                coupling = h_inv[column, column + 1 :] / max(denominator, EPS)
                working[:, column + 1 :] -= residual[:, None] * coupling[None, :]

    return TensorQuantization(q=q, scale=scales, bits=bits, original_shape=original.shape, group_size=group_size)


def awq_like_scale_search(
    weight: np.ndarray,
    calibration_activations: np.ndarray,
    bits: int = 4,
    group_size: int = 16,
    candidate_alphas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> tuple[TensorQuantization, np.ndarray, float]:
    """A small *AWQ-like* activation-aware equivalent-scaling demonstration.

    Let `channel_scale` be positive.  Replacing X with X / channel_scale and W
    with W * channel_scale preserves X @ W.T in exact arithmetic. The search
    uses activation RMS to create candidate channel scales, quantizes the scaled
    weight groupwise, maps it back, and retains the alpha with minimum calibration
    output MSE. This mirrors AWQ's activation-aware equivalent-transformation
    intuition but is not the full AWQ method or its official scale search.
    """
    original = np.asarray(weight, dtype=np.float32)
    x = np.asarray(calibration_activations, dtype=np.float32)
    if original.ndim != 2 or x.ndim != 2 or original.shape[1] != x.shape[1]:
        raise ValueError("Expected weight [out, in] and calibration [batch, in].")

    rms = np.sqrt(np.mean(x**2, axis=0)).astype(np.float32)
    normalized_rms = rms / max(float(np.mean(rms)), EPS)
    best_error = float("inf")
    best_state: TensorQuantization | None = None
    best_scale: np.ndarray | None = None
    best_alpha = float("nan")

    for alpha in candidate_alphas:
        channel_scale = np.clip(normalized_rms**alpha, 0.25, 4.0).astype(np.float32)
        scaled_weight = original * channel_scale[None, :]
        state = groupwise_quantize_weights(scaled_weight, bits=bits, group_size=group_size)
        restored_weight = groupwise_dequantize_weights(state) / channel_scale[None, :]
        error = output_mse(x, original, restored_weight)
        if error < best_error:
            best_error = error
            best_state = state
            best_scale = channel_scale
            best_alpha = alpha

    assert best_state is not None and best_scale is not None
    return best_state, best_scale, best_alpha


def dequantize_awq_like(state: TensorQuantization, channel_scale: np.ndarray) -> np.ndarray:
    """Undo AWQ-like equivalent scaling after reconstructing grouped codes."""
    return groupwise_dequantize_weights(state) / np.asarray(channel_scale, dtype=np.float32)[None, :]


def quantize_last_axis(x: np.ndarray, bits: int = 8, group_size: int | None = None) -> TensorQuantization:
    """Quantize any tensor independently along groups of its last dimension.

    `group_size=None` means one scale for each prefix location. A positive group
    size yields one scale per group for every prefix location, analogous to a
    pedagogical per-vector / per-group KV cache scheme.
    """
    value = np.asarray(x, dtype=np.float32)
    if value.ndim < 1:
        raise ValueError("Expected a tensor with a last dimension.")
    last_dim = value.shape[-1]
    actual_group_size = last_dim if group_size is None else group_size
    if actual_group_size <= 0:
        raise ValueError("group_size must be positive.")
    n_groups = (last_dim + actual_group_size - 1) // actual_group_size
    qmax = signed_qmax(bits)
    q = np.empty_like(value, dtype=np.int8)
    scales = np.empty(value.shape[:-1] + (n_groups,), dtype=np.float32)
    flat_value = value.reshape(-1, last_dim)
    flat_q = q.reshape(-1, last_dim)
    flat_scales = scales.reshape(-1, n_groups)

    for row in range(flat_value.shape[0]):
        for group in range(n_groups):
            start = group * actual_group_size
            end = min(start + actual_group_size, last_dim)
            chunk = flat_value[row, start:end]
            absmax = float(np.max(np.abs(chunk)))
            scale = np.float32(absmax / qmax) if absmax > 0.0 else np.float32(1.0)
            flat_q[row, start:end] = np.clip(np.rint(chunk / scale), -qmax, qmax).astype(np.int8)
            flat_scales[row, group] = scale

    return TensorQuantization(q=q, scale=scales, bits=bits, original_shape=value.shape, group_size=actual_group_size)


def dequantize_last_axis(state: TensorQuantization) -> np.ndarray:
    """Restore a tensor quantized by `quantize_last_axis`."""
    if state.group_size is None:
        raise ValueError("Expected a last-axis group quantization state.")
    last_dim = state.original_shape[-1]
    flat_q = state.q.reshape(-1, last_dim)
    flat_scale = state.scale.reshape(-1, state.scale.shape[-1])
    restored = np.empty_like(flat_q, dtype=np.float32)
    for row in range(flat_q.shape[0]):
        for group in range(flat_scale.shape[1]):
            start = group * state.group_size
            end = min(start + state.group_size, last_dim)
            restored[row, start:end] = flat_q[row, start:end].astype(np.float32) * flat_scale[row, group]
    return restored.reshape(state.original_shape)


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Stable softmax used only for the tiny attention sensitivity demonstration."""
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def single_head_attention(query: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Compute a minimal [heads, dim] x [heads, tokens, dim] attention output."""
    q = np.asarray(query, dtype=np.float32)
    k = np.asarray(key, dtype=np.float32)
    v = np.asarray(value, dtype=np.float32)
    scores = np.einsum("hd,htd->ht", q, k) / np.sqrt(q.shape[-1])
    probabilities = softmax(scores, axis=-1)
    return np.einsum("ht,htd->hd", probabilities, v)


def make_teaching_problem(seed: int = 2026) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create deterministic *synthetic* tensors for a numerical teaching lab.

    The values are not a model or a benchmark dataset. They intentionally combine
    normal values, a few weight outliers, and high-RMS activation channels so the
    impact of scale granularity is visible in a reproducible small example.
    """
    rng = np.random.default_rng(seed)
    weight = rng.normal(0.0, 0.45, size=(24, 48)).astype(np.float32)
    weight[:, [5, 18, 41]] *= np.array([4.0, 3.0, 5.0], dtype=np.float32)
    calibration = rng.normal(0.0, 1.0, size=(512, 48)).astype(np.float32)
    calibration[:, [3, 17, 31]] *= np.array([4.0, 3.0, 5.0], dtype=np.float32)
    evaluation = rng.normal(0.0, 1.0, size=(256, 48)).astype(np.float32)
    evaluation[:, [3, 17, 31]] *= np.array([4.0, 3.0, 5.0], dtype=np.float32)
    bias = rng.normal(0.0, 0.05, size=(24,)).astype(np.float32)
    return weight, calibration, evaluation, bias


def evaluate_weight_methods() -> list[dict[str, Any]]:
    """Run W8A16 and three W4 teaching variants on the same matrices."""
    weight, calibration, evaluation, bias = make_teaching_problem()
    reference_calibration = calibration @ weight.T + bias
    reference_evaluation = evaluation @ weight.T + bias
    fp16_bytes = weight.size * 2.0

    # W8A16 uses one scale for the complete matrix in this simplest teaching path.
    w8_layer = W8A16Linear(weight, bias)
    w8_restored = w8_layer.restored_weight

    # Baseline W4: a single global scale; group W4 adds per-row group scales.
    w4_tensor = symmetric_quantize(weight, bits=4)
    w4_tensor_restored = symmetric_dequantize(w4_tensor)
    w4_group = groupwise_quantize_weights(weight, bits=4, group_size=16)
    w4_group_restored = groupwise_dequantize_weights(w4_group)

    # The two named research methods are intentionally conservative teaching
    # approximations; their names below preserve that distinction in all outputs.
    gptq_like = gptq_like_quantize(weight, calibration, bits=4, group_size=16)
    gptq_like_restored = groupwise_dequantize_weights(gptq_like)
    awq_like, awq_channel_scale, best_alpha = awq_like_scale_search(
        weight, calibration, bits=4, group_size=16
    )
    awq_like_restored = dequantize_awq_like(awq_like, awq_channel_scale)

    methods = [
        ("FP16 reference", None, weight),
        ("W8A16 per-tensor", w8_layer.quantized_weight, w8_restored),
        ("W4 per-tensor", w4_tensor, w4_tensor_restored),
        ("W4 group-wise", w4_group, w4_group_restored),
        ("GPTQ-like W4", gptq_like, gptq_like_restored),
        ("AWQ-like W4", awq_like, awq_like_restored),
    ]
    results: list[dict[str, Any]] = []
    for name, state, restored in methods:
        prediction_calibration = calibration @ restored.T + bias
        prediction_evaluation = evaluation @ restored.T + bias
        logical_bytes = fp16_bytes if state is None else logical_weight_bytes(state)
        results.append(
            {
                "method": name,
                "weight_mse": matrix_mse(weight, restored),
                "calibration_output_mse": matrix_mse(reference_calibration, prediction_calibration),
                "evaluation_output_mse": matrix_mse(reference_evaluation, prediction_evaluation),
                "logical_weight_bytes": logical_bytes,
                "relative_to_fp16_percent": 100.0 * logical_bytes / fp16_bytes,
                "notes": (
                    "Reference (logical FP16 storage)."
                    if state is None
                    else "Codes plus FP32 scale metadata; excludes alignment and packing headers."
                ),
            }
        )

    group_result = next(item for item in results if item["method"] == "W4 group-wise")
    awq_result = next(item for item in results if item["method"] == "AWQ-like W4")
    # alpha=0 is included in the search and equals the groupwise baseline, so the
    # selected candidate must not be worse on the calibration objective.
    assert awq_result["calibration_output_mse"] <= group_result["calibration_output_mse"] + 1e-7
    assert np.all(np.isfinite(w8_layer.forward(evaluation)))
    for item in results:
        assert np.isfinite(item["evaluation_output_mse"])
        assert item["logical_weight_bytes"] > 0.0

    # Store a method note only in the console/JSON so plots stay compact.
    results[-1]["selected_activation_scale_alpha"] = best_alpha
    return results


def evaluate_kv_methods(seed: int = 2027) -> list[dict[str, Any]]:
    """Compare one-scale-per-vector and group scales for a toy K/V attention read."""
    rng = np.random.default_rng(seed)
    # [K_or_V, heads, tokens, head_dim] — intentionally small and deterministic.
    kv_cache = rng.normal(0.0, 0.7, size=(2, 4, 48, 64)).astype(np.float32)
    kv_cache[..., [7, 23, 47]] *= np.array([5.0, 3.0, 4.0], dtype=np.float32)
    query = rng.normal(0.0, 0.8, size=(4, 64)).astype(np.float32)
    reference = single_head_attention(query, kv_cache[0], kv_cache[1])
    fp16_bytes = kv_cache.size * 2.0

    methods = [
        ("FP16 reference", None, kv_cache),
        ("FP8 per-vector scale", quantize_last_axis(kv_cache, bits=8, group_size=None), None),
        ("FP8 group scale (16)", quantize_last_axis(kv_cache, bits=8, group_size=16), None),
    ]
    results: list[dict[str, Any]] = []
    for name, state, restored in methods:
        reconstructed = kv_cache if state is None else dequantize_last_axis(state)
        attention_output = single_head_attention(query, reconstructed[0], reconstructed[1])
        logical_bytes = fp16_bytes if state is None else logical_weight_bytes(state)
        results.append(
            {
                "method": name,
                "kv_mse": matrix_mse(kv_cache, reconstructed),
                "attention_output_mse": matrix_mse(reference, attention_output),
                "logical_kv_bytes": logical_bytes,
                "relative_to_fp16_percent": 100.0 * logical_bytes / fp16_bytes,
                "scale_shape": "reference" if state is None else list(state.scale.shape),
                "notes": (
                    "Reference (logical FP16 storage)."
                    if state is None
                    else "1-byte teaching low-precision codes plus FP32 scale metadata."
                ),
            }
        )

    for item in results:
        assert np.isfinite(item["kv_mse"])
        assert np.isfinite(item["attention_output_mse"])
        assert item["logical_kv_bytes"] > 0.0
    return results


def plot_weight_tradeoff(results: list[dict[str, Any]], output_path: Path) -> None:
    """Create a deterministic chart from the actual weight-lab calculation."""
    methods = [item["method"] for item in results]
    storage = [item["relative_to_fp16_percent"] for item in results]
    error = [max(item["evaluation_output_mse"], 1e-12) for item in results]
    colors = ["#7dd3fc", "#38bdf8", "#c084fc", "#a78bfa", "#fbbf24", "#34d399"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=170)
    fig.suptitle("Teaching experiment: quantization storage and linear-output error", fontsize=14, weight="bold")
    axes[0].bar(methods, storage, color=colors)
    axes[0].axhline(100.0, color="#334155", linewidth=1.0, linestyle="--", label="FP16 logical storage")
    axes[0].set_ylabel("Logical weight storage (% of FP16)")
    axes[0].set_ylim(0, max(storage) * 1.18)
    axes[0].tick_params(axis="x", rotation=25, labelsize=8)
    axes[0].legend(frameon=True, fontsize=8)

    axes[1].bar(methods, error, color=colors)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Evaluation linear-output MSE (log scale)")
    axes[1].tick_params(axis="x", rotation=25, labelsize=8)
    axes[1].text(
        0.02,
        0.98,
        "Fixed synthetic tensors; not a model benchmark.",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#475569",
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_kv_tradeoff(results: list[dict[str, Any]], output_path: Path) -> None:
    """Create a deterministic chart from the actual KV-cache calculation."""
    methods = [item["method"] for item in results]
    storage = [item["relative_to_fp16_percent"] for item in results]
    tensor_error = [max(item["kv_mse"], 1e-12) for item in results]
    attention_error = [max(item["attention_output_mse"], 1e-12) for item in results]
    colors = ["#7dd3fc", "#a78bfa", "#34d399"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=170)
    fig.suptitle("Teaching experiment: FP8-style KV cache quantization", fontsize=14, weight="bold")
    axes[0].bar(methods, storage, color=colors)
    axes[0].axhline(100.0, color="#334155", linewidth=1.0, linestyle="--", label="FP16 logical storage")
    axes[0].set_ylabel("Logical KV storage (% of FP16)")
    axes[0].set_ylim(0, max(storage) * 1.18)
    axes[0].tick_params(axis="x", rotation=20, labelsize=8)
    axes[0].legend(frameon=True, fontsize=8)

    x = np.arange(len(methods))
    width = 0.36
    axes[1].bar(x - width / 2, tensor_error, width, label="KV reconstruction MSE", color="#a78bfa")
    axes[1].bar(x + width / 2, attention_error, width, label="Attention-output MSE", color="#34d399")
    axes[1].set_yscale("log")
    axes[1].set_xticks(x, methods, rotation=20, ha="right", fontsize=8)
    axes[1].set_ylabel("Error (log scale)")
    axes[1].legend(frameon=True, fontsize=8)
    axes[1].text(
        0.02,
        0.98,
        "A tiny deterministic attention read, not a serving benchmark.",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#475569",
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def compact_trace(weight_results: list[dict[str, Any]], kv_results: list[dict[str, Any]]) -> str:
    """Format a trace whose measurements can be inspected without opening JSON."""
    lines = ["All invariants passed. This is a structural CPU quantization lab, not a GPU benchmark.", ""]
    lines.append("[1] Weight quantization methods")
    lines.append("  method                 storage%  eval_output_mse")
    for item in weight_results:
        lines.append(
            f"  {item['method']:<21} {item['relative_to_fp16_percent']:>7.2f}  {item['evaluation_output_mse']:.8f}"
        )
    lines.append("")
    lines.append("[2] FP8-style KV cache methods")
    lines.append("  method                 storage%  kv_mse       attention_output_mse")
    for item in kv_results:
        lines.append(
            f"  {item['method']:<21} {item['relative_to_fp16_percent']:>7.2f}  "
            f"{item['kv_mse']:.8f}  {item['attention_output_mse']:.8f}"
        )
    lines.append("")
    lines.append("Interpretation: lower-bit codes reduce ideal logical storage; scale granularity and calibration-aware choices redistribute quantization error.")
    return "\n".join(lines)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    weight_results = evaluate_weight_methods()
    kv_results = evaluate_kv_methods()
    plot_weight_tradeoff(weight_results, ASSET_DIR / "04_weight_quantization_tradeoffs.png")
    plot_kv_tradeoff(kv_results, ASSET_DIR / "05_kv_quantization_tradeoffs.png")
    payload = {"weight_methods": weight_results, "kv_cache_methods": kv_results}
    (RESULT_DIR / "quantization_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(compact_trace(weight_results, kv_results))


if __name__ == "__main__":
    main()
