"""从 GPU 存储层级到 FlashAttention：可运行的教学实验。

本文件实现一个数值等价的、按块流式处理的 Attention 前向模拟。
它用于解释 Online Softmax 的状态更新，并不替代 CUDA/Triton 的生产级 FlashAttention kernel。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch


@dataclass(frozen=True)
class AttentionMemoryReport:
    """以字节记录一个 attention head-group 的主要激活张量规模。"""

    qkv_bytes: int
    score_bytes: int
    probability_bytes: int
    output_bytes: int
    total_dense_bytes: int
    linear_streaming_state_bytes: int


def bytes_to_mib(byte_count: int) -> float:
    """使用二进制 MiB 便于阅读张量大小。"""
    return byte_count / (1024**2)


def attention_activation_report(
    sequence_length: int,
    num_heads: int,
    head_dim: int,
    element_bytes: int = 2,
    accumulator_bytes: int = 4,
) -> AttentionMemoryReport:
    """估算单层自注意力前向中易成为瓶颈的激活规模。

    这里显式计入每个 head 的 score 与概率矩阵。训练时还会有反向传播保存
    的状态，真实峰值还取决于框架、融合策略和重计算策略；因此该函数用于
    展示 N² 中间矩阵的增长趋势，而不是给出端到端显存预算。
    """
    token_vectors = sequence_length * num_heads * head_dim
    qkv_bytes = 3 * token_vectors * element_bytes
    score_bytes = num_heads * sequence_length * sequence_length * element_bytes
    probability_bytes = score_bytes
    output_bytes = token_vectors * element_bytes

    # 流式路径保存每一行的稳定 softmax 状态 m、ℓ，以及未归一化向量分子 u。
    running_max_bytes = sequence_length * num_heads * accumulator_bytes
    running_mass_bytes = sequence_length * num_heads * accumulator_bytes
    numerator_bytes = sequence_length * num_heads * head_dim * accumulator_bytes

    return AttentionMemoryReport(
        qkv_bytes=qkv_bytes,
        score_bytes=score_bytes,
        probability_bytes=probability_bytes,
        output_bytes=output_bytes,
        total_dense_bytes=qkv_bytes + score_bytes + probability_bytes + output_bytes,
        linear_streaming_state_bytes=running_max_bytes + running_mass_bytes + numerator_bytes,
    )


def dense_attention_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """显式物化完整分数矩阵的标准 scaled dot-product attention 基线。"""
    if q.ndim != 2 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q、k、v 必须是同形状的二维张量 [sequence_length, head_dim]。")
    scores = (q @ k.transpose(0, 1)) / math.sqrt(q.shape[1])
    probabilities = torch.softmax(scores, dim=-1)
    return probabilities @ v


def streaming_attention_exact(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tile_size: int,
) -> torch.Tensor:
    """用 Online Softmax 实现精确的分块 attention 前向模拟。

    对每一个 query tile，函数流式扫描全部 key/value tile。函数只持有当前
    score tile 与三项行级状态：运行最大值 m、指数质量 ℓ、未归一化分子 u。
    最终输出为 u / ℓ。这样将每一步的向量归一化推迟到扫描结束，仍与标准
    softmax 数学等价。
    """
    if tile_size <= 0:
        raise ValueError("tile_size 必须为正整数。")
    if q.ndim != 2 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q、k、v 必须是同形状的二维张量 [sequence_length, head_dim]。")

    sequence_length, head_dim = q.shape
    scale = 1.0 / math.sqrt(head_dim)
    output = torch.empty_like(q)

    for q_start in range(0, sequence_length, tile_size):
        q_stop = min(q_start + tile_size, sequence_length)
        q_tile = q[q_start:q_stop]
        row_count = q_tile.shape[0]

        running_max = torch.full(
            (row_count, 1), -torch.inf, dtype=q.dtype, device=q.device
        )
        running_mass = torch.zeros((row_count, 1), dtype=q.dtype, device=q.device)
        numerator = torch.zeros((row_count, head_dim), dtype=q.dtype, device=q.device)

        for kv_start in range(0, sequence_length, tile_size):
            kv_stop = min(kv_start + tile_size, sequence_length)
            k_tile = k[kv_start:kv_stop]
            v_tile = v[kv_start:kv_stop]

            score_tile = (q_tile @ k_tile.transpose(0, 1)) * scale
            tile_max = score_tile.max(dim=-1, keepdim=True).values
            next_max = torch.maximum(running_max, tile_max)

            # 旧状态的指数基准从 running_max 变为 next_max。
            old_rescale = torch.exp(running_max - next_max)
            tile_weights = torch.exp(score_tile - next_max)
            next_mass = old_rescale * running_mass + tile_weights.sum(dim=-1, keepdim=True)
            next_numerator = old_rescale * numerator + tile_weights @ v_tile

            running_max = next_max
            running_mass = next_mass
            numerator = next_numerator

        output[q_start:q_stop] = numerator / running_mass

    return output


def trace_one_query_row(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    row_index: int,
    tile_size: int,
) -> List[Dict[str, float]]:
    """追踪一个 query 行在扫描不同 K/V tile 后的 Online Softmax 状态。"""
    if not 0 <= row_index < q.shape[0]:
        raise IndexError("row_index 超出 query 序列范围。")

    q_row = q[row_index : row_index + 1]
    scale = 1.0 / math.sqrt(q.shape[1])
    running_max = torch.full((1, 1), -torch.inf, dtype=q.dtype, device=q.device)
    running_mass = torch.zeros((1, 1), dtype=q.dtype, device=q.device)
    numerator = torch.zeros((1, q.shape[1]), dtype=q.dtype, device=q.device)
    trace: List[Dict[str, float]] = []

    for kv_start in range(0, k.shape[0], tile_size):
        kv_stop = min(kv_start + tile_size, k.shape[0])
        score_tile = (q_row @ k[kv_start:kv_stop].transpose(0, 1)) * scale
        next_max = torch.maximum(running_max, score_tile.max(dim=-1, keepdim=True).values)
        old_rescale = torch.exp(running_max - next_max)
        tile_weights = torch.exp(score_tile - next_max)
        running_mass = old_rescale * running_mass + tile_weights.sum(dim=-1, keepdim=True)
        numerator = old_rescale * numerator + tile_weights @ v[kv_start:kv_stop]
        running_max = next_max

        trace.append(
            {
                "kv_range_start": float(kv_start),
                "kv_range_stop": float(kv_stop),
                "running_max": float(running_max.item()),
                "running_mass": float(running_mass.item()),
                "partial_output_norm": float((numerator / running_mass).norm().item()),
            }
        )

    return trace


def print_memory_growth(sequence_lengths: Iterable[int]) -> None:
    """打印 N² 中间矩阵随序列长度增长的直观报告。"""
    print("\n=== Dense Attention 的中间激活增长（32 heads, head_dim=128, BF16/FP16）===")
    header = f"{'N':>8} | {'score':>12} | {'score+P':>12} | {'dense total':>12} | {'stream state':>12}"
    print(header)
    print("-" * len(header))
    for n in sequence_lengths:
        report = attention_activation_report(n, num_heads=32, head_dim=128)
        print(
            f"{n:8d} | {bytes_to_mib(report.score_bytes):10.1f} MiB | "
            f"{bytes_to_mib(report.score_bytes + report.probability_bytes):10.1f} MiB | "
            f"{bytes_to_mib(report.total_dense_bytes):10.1f} MiB | "
            f"{bytes_to_mib(report.linear_streaming_state_bytes):10.1f} MiB"
        )


def verify_exactness() -> None:
    """在包含非整除 tile 的场景中验证分块流式计算与基线完全一致。"""
    cases = [(7, 5, 3, 11), (13, 8, 4, 29), (17, 6, 5, 47)]
    print("\n=== 数值等价性验证 ===")
    for sequence_length, head_dim, tile_size, seed in cases:
        generator = torch.Generator().manual_seed(seed)
        q = torch.randn(sequence_length, head_dim, dtype=torch.float64, generator=generator)
        k = torch.randn(sequence_length, head_dim, dtype=torch.float64, generator=generator)
        v = torch.randn(sequence_length, head_dim, dtype=torch.float64, generator=generator)
        reference = dense_attention_reference(q, k, v)
        streamed = streaming_attention_exact(q, k, v, tile_size=tile_size)
        max_error = (reference - streamed).abs().max().item()
        print(
            f"N={sequence_length:2d}, d={head_dim:2d}, tile={tile_size}: "
            f"max_abs_error={max_error:.3e}"
        )
        torch.testing.assert_close(reference, streamed, rtol=1e-12, atol=1e-12)

    q = torch.randn(9, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(101))
    k = torch.randn(9, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(102))
    v = torch.randn(9, 4, dtype=torch.float64, generator=torch.Generator().manual_seed(103))
    print("\n=== 单行状态追踪（query row 2, tile=3）===")
    for state in trace_one_query_row(q, k, v, row_index=2, tile_size=3):
        print(
            f"K/V[{int(state['kv_range_start'])}:{int(state['kv_range_stop'])}]  "
            f"m={state['running_max']:+.4f}, ℓ={state['running_mass']:.4f}, "
            f"||u/ℓ||={state['partial_output_norm']:.4f}"
        )


if __name__ == "__main__":
    print_memory_growth([512, 4096, 16384, 131072])
    verify_exactness()
    print("\n通过：流式实现与密集 attention 基线在测试用例中数值等价。")
