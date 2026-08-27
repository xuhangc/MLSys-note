"""Task5 Quantization Lab — original CPU-first teaching implementation.

This file intentionally separates three different things that are often conflated:
1) a numerical quantizer, 2) storage packing / low-bit kernel execution, and
3) end-to-end production inference benchmarking.

It is a self-contained educational lab, not a drop-in replacement for GPTQ,
AWQ, a vendor FP8 kernel, KIVI, or an inference engine quantization backend.
Run: python3 quantization_lab.py
Outputs: assets/05_weight_memory_budget.png, assets/06_weight_error_granularity.png,
         assets/07_kv_cache_quantization_error.png, results/quantization_lab_results.json
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "assets"
RESULT_DIR = ROOT / "results"
EPS = 1e-8


def weight_storage_gb(parameter_count: int, bits_per_weight: int) -> float:
    """Return decimal GB needed by weights only; scales and packing metadata are excluded."""
    return parameter_count * bits_per_weight / 8 / 1_000_000_000


def qmax_for_bits(bits: int) -> int:
    """Return the positive integer endpoint of a signed symmetric quantizer."""
    if not 2 <= bits <= 8:
        raise ValueError("This teaching lab accepts integer bit widths from 2 through 8.")
    return (1 << (bits - 1)) - 1


def quantize_symmetric(x: torch.Tensor, bits: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor signed symmetric quantization returning an int8 container and a scale.

    A 4-bit quantizer still uses an int8 container here. That preserves the numerical
    values -7 through 7 for clarity; a production implementation additionally packs
    two 4-bit values into one byte and dispatches a matching kernel.
    """
    qmax = qmax_for_bits(bits)
    x32 = x.detach().float()
    scale = x32.abs().amax().clamp_min(EPS) / qmax
    q = torch.clamp(torch.round(x32 / scale), -qmax, qmax).to(torch.int8)
    return q, scale


def dequantize_symmetric(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Restore an approximate float32 tensor from signed integers and one scale."""
    return q.float() * scale.float()


def quantize_groupwise_weights(
    weight: torch.Tensor, bits: int = 4, group_size: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize an [out_features, in_features] matrix with one symmetric scale per group."""
    if weight.ndim != 2:
        raise ValueError("Expected a 2D linear weight matrix shaped [out_features, in_features].")
    if group_size <= 0:
        raise ValueError("group_size must be positive.")

    out_features, in_features = weight.shape
    groups = math.ceil(in_features / group_size)
    qweight = torch.zeros_like(weight, dtype=torch.int8)
    scales = torch.empty((out_features, groups), dtype=torch.float32, device=weight.device)

    for row in range(out_features):
        for group in range(groups):
            start = group * group_size
            end = min(start + group_size, in_features)
            q_group, scale = quantize_symmetric(weight[row, start:end], bits=bits)
            qweight[row, start:end] = q_group
            scales[row, group] = scale
    return qweight, scales


def dequantize_groupwise_weights(qweight: torch.Tensor, scales: torch.Tensor, group_size: int) -> torch.Tensor:
    """Restore a groupwise-quantized weight matrix to float32."""
    out_features, in_features = qweight.shape
    expected_groups = math.ceil(in_features / group_size)
    if tuple(scales.shape) != (out_features, expected_groups):
        raise ValueError("Scale shape does not match quantized weight shape and group_size.")

    restored = torch.empty_like(qweight, dtype=torch.float32)
    for row in range(out_features):
        for group in range(expected_groups):
            start = group * group_size
            end = min(start + group_size, in_features)
            restored[row, start:end] = qweight[row, start:end].float() * scales[row, group]
    return restored


@dataclass
class W8A16Linear:
    """A transparent weight-only layer: W8 storage, A16/FP32-style compute after dequantization.

    It deliberately uses floating-point F.linear after reconstructing the weight. This
    teaches the W8A16 storage and numerical pathway but does not claim a low-bit GEMM
    speedup. The latter requires packing plus a supported kernel.
    """

    qweight: torch.Tensor
    scales: torch.Tensor
    bias: torch.Tensor | None
    group_size: int

    @classmethod
    def from_float(cls, weight: torch.Tensor, bias: torch.Tensor | None = None, group_size: int = 32) -> "W8A16Linear":
        qweight, scales = quantize_groupwise_weights(weight, bits=8, group_size=group_size)
        stored_bias = None if bias is None else bias.detach().float().clone()
        return cls(qweight=qweight, scales=scales, bias=stored_bias, group_size=group_size)

    def dequantized_weight(self) -> torch.Tensor:
        return dequantize_groupwise_weights(self.qweight, self.scales, self.group_size)

    def __call__(self, activations: torch.Tensor) -> torch.Tensor:
        return F.linear(activations.float(), self.dequantized_weight(), self.bias)


def channel_rms(calibration_activations: torch.Tensor) -> torch.Tensor:
    """Calculate a diagonal second-order proxy: RMS activation magnitude per input channel."""
    if calibration_activations.ndim < 2:
        raise ValueError("Calibration activations need a final feature dimension and at least one sample axis.")
    reduce_dims = tuple(range(calibration_activations.ndim - 1))
    return calibration_activations.float().square().mean(dim=reduce_dims).sqrt().clamp_min(EPS)


def gptq_like_groupwise_quantize(
    weight: torch.Tensor,
    calibration_activations: torch.Tensor,
    bits: int = 4,
    group_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Teaching-only calibration-weighted scale search inspired by GPTQ's objective.

    GPTQ proper uses approximate second-order information and sequential error
    compensation. This small routine does neither. It searches a handful of scales
    for every group and minimizes diagonal, activation-weighted reconstruction error.
    The name contains '-like' to prevent equating it with the published algorithm.
    """
    if weight.ndim != 2:
        raise ValueError("Expected a 2D linear weight matrix.")
    importance = channel_rms(calibration_activations)
    if importance.numel() != weight.size(1):
        raise ValueError("Calibration feature dimension must match the number of input features.")

    out_features, in_features = weight.shape
    groups = math.ceil(in_features / group_size)
    qmax = qmax_for_bits(bits)
    candidates = (0.70, 0.82, 0.94, 1.00, 1.10, 1.24, 1.40)
    qweight = torch.zeros_like(weight, dtype=torch.int8)
    scales = torch.empty((out_features, groups), dtype=torch.float32)

    for row in range(out_features):
        for group in range(groups):
            start = group * group_size
            end = min(start + group_size, in_features)
            values = weight[row, start:end].float()
            weights = importance[start:end].square()
            base_scale = values.abs().amax().clamp_min(EPS) / qmax
            best_error = float("inf")
            best_q: torch.Tensor | None = None
            best_scale: torch.Tensor | None = None
            for multiplier in candidates:
                scale = base_scale * multiplier
                proposal_q = torch.clamp(torch.round(values / scale), -qmax, qmax).to(torch.int8)
                proposal = proposal_q.float() * scale
                weighted_error = ((proposal - values).square() * weights).mean().item()
                if weighted_error < best_error:
                    best_error = weighted_error
                    best_q, best_scale = proposal_q, scale
            assert best_q is not None and best_scale is not None
            qweight[row, start:end] = best_q
            scales[row, group] = best_scale
    return qweight, scales


def awq_inspired_quantize(
    weight: torch.Tensor,
    calibration_activations: torch.Tensor,
    bits: int = 4,
    group_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Teaching-only activation-aware equivalent transform followed by groupwise quantization.

    For output y = x W^T, scale each input channel by an inverse factor in x and the
    matching direct factor in W. The full-precision output is unchanged before
    quantization. Larger calibration RMS produces a larger channel factor, which can
    reduce relative quantization damage for salient channels. Production AWQ searches
    its scales more carefully and uses packed kernels; this is only its core intuition.
    """
    importance = channel_rms(calibration_activations)
    if importance.numel() != weight.size(1):
        raise ValueError("Calibration feature dimension must match the number of input features.")
    channel_scale = (importance / importance.mean()).clamp(0.5, 2.0).sqrt()
    transformed_weight = weight.float() * channel_scale.unsqueeze(0)
    qweight, scales = quantize_groupwise_weights(transformed_weight, bits=bits, group_size=group_size)
    return qweight, scales, channel_scale


def awq_inspired_linear(
    activations: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    channel_scale: torch.Tensor,
    group_size: int,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the inverse activation scale and reconstructed transformed weight through F.linear."""
    restored_weight = dequantize_groupwise_weights(qweight, scales, group_size)
    return F.linear(activations.float() / channel_scale, restored_weight, bias)


@dataclass
class FP8Tensor:
    """Scaled real E4M3 float8 storage plus the factor required for restoration."""

    encoded: torch.Tensor
    scale: torch.Tensor

    def dequantize(self) -> torch.Tensor:
        return self.encoded.float() * self.scale.float()


def quantize_fp8_e4m3(x: torch.Tensor) -> FP8Tensor:
    """Use PyTorch's real E4M3 storage dtype with an amax scale, when available.

    This is materially different from putting integer quantization values into int8.
    It is still a numerical experiment, not a claim that the current CPU dispatches
    an optimized FP8 GEMM.
    """
    dtype = getattr(torch, "float8_e4m3fn", None)
    if dtype is None:
        raise RuntimeError("This PyTorch build does not expose torch.float8_e4m3fn.")
    x32 = x.detach().float()
    finite_max = torch.finfo(dtype).max
    scale = x32.abs().amax().clamp_min(EPS) / finite_max
    encoded = (x32 / scale).to(dtype)
    return FP8Tensor(encoded=encoded, scale=scale)


def quantize_key_per_channel(key: torch.Tensor, bits: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize key cache along the token axis, yielding one scale per channel.

    Expected shape is [..., tokens, head_dim]. The returned scale has shape
    [..., 1, head_dim], so each channel shares a scale across positions.
    """
    if key.ndim < 2:
        raise ValueError("Key cache needs at least [tokens, head_dim].")
    qmax = qmax_for_bits(bits)
    scale = key.float().abs().amax(dim=-2, keepdim=True).clamp_min(EPS) / qmax
    qkey = torch.clamp(torch.round(key.float() / scale), -qmax, qmax).to(torch.int8)
    return qkey, scale


def quantize_value_per_token(value: torch.Tensor, bits: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize value cache along head_dim, yielding one scale per token vector."""
    if value.ndim < 1:
        raise ValueError("Value cache cannot be scalar.")
    qmax = qmax_for_bits(bits)
    scale = value.float().abs().amax(dim=-1, keepdim=True).clamp_min(EPS) / qmax
    qvalue = torch.clamp(torch.round(value.float() / scale), -qmax, qmax).to(torch.int8)
    return qvalue, scale


def relative_mse(reference: torch.Tensor, approximation: torch.Tensor) -> float:
    """Return MSE divided by reference energy, which is comparable across tensors."""
    numerator = (reference.float() - approximation.float()).square().mean()
    denominator = reference.float().square().mean().clamp_min(EPS)
    return float((numerator / denominator).item())


@dataclass
class DeploymentMetrics:
    name: str
    latency_ms: float
    throughput_tokens_s: float
    vram_mb: float
    quality_error: float
    quality_budget: float
    kernel_supported: bool


def deployment_decision(metrics: DeploymentMetrics, baseline: DeploymentMetrics) -> tuple[str, str]:
    """Make a transparent accept/tune/reject decision from equal-workload metrics."""
    if metrics.quality_error > metrics.quality_budget:
        return "reject", "质量误差超过预先声明的预算。"
    if not metrics.kernel_supported:
        return "tune", "质量与容量可接受，但应先验证匹配的量化 kernel 和打包格式。"

    lower_latency = metrics.latency_ms < baseline.latency_ms
    higher_throughput = metrics.throughput_tokens_s > baseline.throughput_tokens_s
    lower_vram = metrics.vram_mb < baseline.vram_mb
    if lower_vram and (lower_latency or higher_throughput):
        return "accept", "容量收益成立，且同一工作负载下至少一个性能指标改善。"
    if lower_vram:
        return "tune", "容量收益成立，但端到端速度未改善；检查反量化和 kernel 路径。"
    return "reject", "容量或性能收益不足以覆盖部署切换成本。"


def benchmark_ms(function: Callable[[], torch.Tensor], warmup: int = 10, iterations: int = 100) -> float:
    """Measure a CPU wall-clock mean for a demonstration only, not a production benchmark."""
    for _ in range(warmup):
        function()
    start = time.perf_counter()
    for _ in range(iterations):
        function()
    return (time.perf_counter() - start) * 1000 / iterations


def deterministic_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create deterministic, non-random tensors with smooth structure and known outliers."""
    index = torch.arange(8 * 128, dtype=torch.float32).reshape(8, 128)
    weight = 0.45 * torch.sin(index / 13.0) + 0.18 * torch.cos(index / 7.0)
    weight[1, 17] = 2.8
    weight[6, 96] = -2.4

    calibration_index = torch.arange(64 * 128, dtype=torch.float32).reshape(64, 128)
    calibration = 0.7 * torch.sin(calibration_index / 19.0) + 0.15 * torch.cos(calibration_index / 5.0)
    calibration[:, 17] *= 4.0
    calibration[:, 96] *= 3.0

    evaluation_index = torch.arange(12 * 128, dtype=torch.float32).reshape(12, 128)
    evaluation = 0.65 * torch.cos(evaluation_index / 17.0) + 0.20 * torch.sin(evaluation_index / 9.0)
    evaluation[:, 17] *= 3.5
    evaluation[:, 96] *= 2.5
    return weight, calibration, evaluation


def build_charts(results: dict[str, float]) -> None:
    """Render precise charts from deterministic lab values; all figures are labelled as toy-lab outputs."""
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    formats = ["FP16", "INT8", "INT4"]
    bits = [16, 8, 4]
    memory = [weight_storage_gb(7_000_000_000, item) for item in bits]
    colors = ["#1c7ed6", "#7048e8", "#d9480f"]
    fig, axis = plt.subplots(figsize=(10, 5.5), dpi=180)
    bars = axis.bar(formats, memory, color=colors, width=0.62)
    axis.set_title("7B Model: Theoretical Weight Storage Only")
    axis.set_ylabel("Decimal GB")
    axis.set_ylim(0, 16)
    for bar, value in zip(bars, memory):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.28, f"{value:.1f} GB", ha="center", fontweight="bold")
    axis.text(0.5, -0.18, "Excludes scales, packing metadata, activations, and KV cache.", transform=axis.transAxes, ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "05_weight_memory_budget.png", bbox_inches="tight")
    plt.close(fig)

    labels = ["Per-tensor", "Group 32", "Group 8", "GPTQ-like", "AWQ-inspired"]
    values = [results["per_tensor_output_mse"], results["group32_output_mse"], results["group8_output_mse"], results["gptq_like_output_mse"], results["awq_inspired_output_mse"]]
    fig, axis = plt.subplots(figsize=(10, 5.5), dpi=180)
    bars = axis.bar(labels, values, color=["#d9480f", "#7048e8", "#4263eb", "#12b886", "#1c7ed6"])
    axis.set_title("Deterministic Toy Linear Layer: 4-bit Relative Output MSE")
    axis.set_ylabel("Relative MSE (lower is better)")
    axis.tick_params(axis="x", rotation=15)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2e}", ha="center", va="bottom", fontsize=8)
    axis.text(0.5, -0.21, "Illustrative numerical lab; not a benchmark of published GPTQ or AWQ implementations.", transform=axis.transAxes, ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "06_weight_error_granularity.png", bbox_inches="tight")
    plt.close(fig)

    labels = ["Key: per-channel", "Value: per-token"]
    values = [results["key_4bit_relative_mse"], results["value_4bit_relative_mse"]]
    fig, axis = plt.subplots(figsize=(9, 5.2), dpi=180)
    bars = axis.bar(labels, values, color=["#7048e8", "#1c7ed6"], width=0.55)
    axis.set_title("Deterministic Toy KV Cache: 4-bit Reconstruction Error")
    axis.set_ylabel("Relative MSE (lower is better)")
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2e}", ha="center", va="bottom", fontsize=9)
    axis.text(0.5, -0.19, "Different cache statistics can favor different scale axes; validate on the target model.", transform=axis.transAxes, ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "07_kv_cache_quantization_error.png", bbox_inches="tight")
    plt.close(fig)


def run_lab() -> dict[str, float]:
    """Run all numerical examples, assertions, CPU timing, and figure generation."""
    torch.set_num_threads(1)
    weight, calibration, evaluation = deterministic_tensors()
    reference_output = F.linear(evaluation, weight)

    per_tensor_q, per_tensor_s = quantize_symmetric(weight, bits=4)
    per_tensor_output = F.linear(evaluation, dequantize_symmetric(per_tensor_q, per_tensor_s))

    group32_q, group32_s = quantize_groupwise_weights(weight, bits=4, group_size=32)
    group32_output = F.linear(evaluation, dequantize_groupwise_weights(group32_q, group32_s, group_size=32))

    group8_q, group8_s = quantize_groupwise_weights(weight, bits=4, group_size=8)
    group8_output = F.linear(evaluation, dequantize_groupwise_weights(group8_q, group8_s, group_size=8))

    gptq_q, gptq_s = gptq_like_groupwise_quantize(weight, calibration, bits=4, group_size=32)
    gptq_output = F.linear(evaluation, dequantize_groupwise_weights(gptq_q, gptq_s, group_size=32))

    awq_q, awq_s, channel_scale = awq_inspired_quantize(weight, calibration, bits=4, group_size=32)
    awq_output = awq_inspired_linear(evaluation, awq_q, awq_s, channel_scale, group_size=32)

    w8a16 = W8A16Linear.from_float(weight, group_size=32)
    w8a16_output = w8a16(evaluation)

    fp8_state = quantize_fp8_e4m3(evaluation)
    fp8_restored = fp8_state.dequantize()

    token_index = torch.arange(1 * 2 * 16 * 32, dtype=torch.float32).reshape(1, 2, 16, 32)
    key = 0.8 * torch.sin(token_index / 11.0)
    value = 0.6 * torch.cos(token_index / 13.0)
    qkey, key_scale = quantize_key_per_channel(key, bits=4)
    qvalue, value_scale = quantize_value_per_token(value, bits=4)
    restored_key = dequantize_symmetric(qkey, key_scale)
    restored_value = dequantize_symmetric(qvalue, value_scale)

    # Verify numerical states rather than pretending an int8 container is physically packed int4.
    assert per_tensor_q.dtype == torch.int8
    assert tuple(group32_s.shape) == (8, 4)
    assert w8a16.qweight.dtype == torch.int8
    assert fp8_state.encoded.dtype == torch.float8_e4m3fn
    assert restored_key.shape == key.shape and restored_value.shape == value.shape
    assert torch.isfinite(awq_output).all()

    fp16_cpu_ms = benchmark_ms(lambda: F.linear(evaluation, weight))
    fake_w8a16_cpu_ms = benchmark_ms(lambda: w8a16(evaluation))

    results: dict[str, float] = {
        "per_tensor_output_mse": relative_mse(reference_output, per_tensor_output),
        "group32_output_mse": relative_mse(reference_output, group32_output),
        "group8_output_mse": relative_mse(reference_output, group8_output),
        "gptq_like_output_mse": relative_mse(reference_output, gptq_output),
        "awq_inspired_output_mse": relative_mse(reference_output, awq_output),
        "w8a16_output_mse": relative_mse(reference_output, w8a16_output),
        "fp8_e4m3_relative_mse": relative_mse(evaluation, fp8_restored),
        "key_4bit_relative_mse": relative_mse(key, restored_key),
        "value_4bit_relative_mse": relative_mse(value, restored_value),
        "fp16_cpu_ms": fp16_cpu_ms,
        "fake_w8a16_cpu_ms": fake_w8a16_cpu_ms,
        "fp16_7b_weight_gb": weight_storage_gb(7_000_000_000, 16),
        "int8_7b_weight_gb": weight_storage_gb(7_000_000_000, 8),
        "int4_7b_weight_gb": weight_storage_gb(7_000_000_000, 4),
    }

    baseline = DeploymentMetrics("FP16 baseline", 42.0, 100.0, 14_000.0, 0.0, 0.02, True)
    candidate = DeploymentMetrics("INT4 candidate", 39.0, 118.0, 4_200.0, results["awq_inspired_output_mse"], 0.02, True)
    decision, reason = deployment_decision(candidate, baseline)
    assert decision in {"accept", "tune", "reject"}

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    full_result = {
        "lab_results": {key: round(value, 10) for key, value in results.items()},
        "deployment_example": {
            "baseline": asdict(baseline),
            "candidate": asdict(candidate),
            "decision": decision,
            "reason": reason,
            "warning": "These deployment numbers are deliberately illustrative policy inputs, not backend measurements.",
        },
    }
    (RESULT_DIR / "quantization_lab_results.json").write_text(json.dumps(full_result, indent=2, ensure_ascii=False), encoding="utf-8")
    build_charts(results)

    print("Quantization lab checks passed.")
    print(json.dumps(full_result, indent=2, ensure_ascii=False))
    return results


if __name__ == "__main__":
    run_lab()
