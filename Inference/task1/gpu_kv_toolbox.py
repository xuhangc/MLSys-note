"""GPU 与 KV Cache 学习工具箱。

此文件专门用于教学估算，不读取真实 GPU 状态。所有容量预算均不包括模型权重、
激活、临时工作区、CUDA 运行时及通信缓冲区；部署时应预留安全余量。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil


BYTES_PER_GIB = 1024**3


def gib(num_bytes: float) -> float:
    """将字节转换为 GiB，适合与常见 GPU 标称显存（如 80 GiB）对照。"""
    return num_bytes / BYTES_PER_GIB


@dataclass(frozen=True)
class ModelKVShape:
    """影响 KV Cache 体积的最小模型形状描述。"""

    layers: int
    kv_heads: int
    head_dim: int
    dtype_bytes: int = 2  # FP16/BF16 的每个元素占 2 bytes


def kv_cache_bytes(
    *,
    sequence_length: int,
    batch_size: int,
    shape: ModelKVShape,
) -> int:
    """估算完整 KV Cache：K 与 V 各一份，且每层、每请求都需要保存。"""
    return (
        2
        * sequence_length
        * batch_size
        * shape.layers
        * shape.kv_heads
        * shape.head_dim
        * shape.dtype_bytes
    )


def kv_bytes_per_token_per_request(shape: ModelKVShape) -> int:
    """计算一个新 token 给单个请求带来的 KV Cache 增量。"""
    return 2 * shape.layers * shape.kv_heads * shape.head_dim * shape.dtype_bytes


def max_concurrent_sequences(
    *,
    cache_budget_gib: float,
    max_sequence_length: int,
    shape: ModelKVShape,
    safety_factor: float = 0.90,
) -> int:
    """在给定缓存预算内，估算能容纳的等长并发请求上限。

    safety_factor 用于避免把显存预算卡得过满；该函数向下取整以保证不超出预算。
    """
    usable_bytes = cache_budget_gib * BYTES_PER_GIB * safety_factor
    one_request_bytes = kv_cache_bytes(
        sequence_length=max_sequence_length,
        batch_size=1,
        shape=shape,
    )
    return int(usable_bytes // one_request_bytes)


def naive_attention_score_bytes(
    *,
    batch_size: int,
    heads: int,
    sequence_length: int,
    dtype_bytes: int = 2,
) -> int:
    """估算显式物化 attention score/probability 张量的单份大小。

    形状是 [B, H, S, S]，因此它对 S 是二次增长。为保守起见，训练反向传播
    还可能需要额外保存张量；本函数只计算单个显式矩阵，不是训练总显存。
    """
    return batch_size * heads * sequence_length**2 * dtype_bytes


def arithmetic_intensity(flops: float, bytes_moved: float) -> float:
    """返回算术强度 FLOPs/Byte；数值较低时通常更接近带宽受限。"""
    if bytes_moved <= 0:
        raise ValueError("bytes_moved 必须为正数")
    return flops / bytes_moved


def roofline_upper_bound_tflops(
    *,
    arithmetic_intensity_flop_per_byte: float,
    memory_bandwidth_tb_per_s: float,
    peak_compute_tflops: float,
) -> float:
    """用简化 Roofline 模型返回性能上界（TFLOP/s）。"""
    bandwidth_limited = arithmetic_intensity_flop_per_byte * memory_bandwidth_tb_per_s
    return min(bandwidth_limited, peak_compute_tflops)


def paged_allocation(
    *,
    sequence_length: int,
    tokens_per_page: int,
) -> tuple[int, int]:
    """计算分页缓存所需页数和最后一页的未使用 token 槽位。"""
    if tokens_per_page <= 0:
        raise ValueError("tokens_per_page 必须为正数")
    page_count = ceil(sequence_length / tokens_per_page)
    unused_slots = page_count * tokens_per_page - sequence_length
    return page_count, unused_slots


def report() -> None:
    """打印一个统一示例，便于观察二次 Attention 与线性 KV Cache 的差别。"""
    # 仅作教学示例：32 层、32 个 KV head、每个 head 的维度是 128、BF16/FP16。
    mha = ModelKVShape(layers=32, kv_heads=32, head_dim=128)
    gqa = ModelKVShape(layers=32, kv_heads=8, head_dim=128)
    mqa = ModelKVShape(layers=32, kv_heads=1, head_dim=128)

    print("=== 每请求 KV Cache（GiB） ===")
    for tokens in (2_048, 8_192, 32_768):
        print(
            f"S={tokens:>6,}: "
            f"MHA={gib(kv_cache_bytes(sequence_length=tokens, batch_size=1, shape=mha)):.3f}, "
            f"GQA={gib(kv_cache_bytes(sequence_length=tokens, batch_size=1, shape=gqa)):.3f}, "
            f"MQA={gib(kv_cache_bytes(sequence_length=tokens, batch_size=1, shape=mqa)):.3f}"
        )

    print("\n=== 随着上下文长度增长的两本账（GiB） ===")
    for tokens in (2_048, 8_192, 32_768):
        score = naive_attention_score_bytes(
            batch_size=1, heads=32, sequence_length=tokens
        )
        cache = kv_cache_bytes(sequence_length=tokens, batch_size=1, shape=mha)
        print(
            f"S={tokens:>6,}: 显式 score 矩阵={gib(score):.3f}; "
            f"MHA KV Cache={gib(cache):.3f}"
        )

    print("\n=== 并发预算示例 ===")
    cache_budget_gib = 40
    concurrent = max_concurrent_sequences(
        cache_budget_gib=cache_budget_gib,
        max_sequence_length=32_768,
        shape=gqa,
        safety_factor=0.90,
    )
    print(
        f"在 {cache_budget_gib} GiB 纯缓存预算、90% 安全系数、GQA、32K 上下文下，"
        f"估算可容纳 {concurrent} 条等长请求。"
    )

    print("\n=== 分页分配示例 ===")
    pages, unused = paged_allocation(sequence_length=8_321, tokens_per_page=128)
    print(f"8,321 tokens，页长 128：需要 {pages} 页，末页闲置 {unused} 个 token 槽位。")

    print("\n=== 简化 Roofline 示例 ===")
    ai = arithmetic_intensity(flops=2e12, bytes_moved=1e12)
    ceiling = roofline_upper_bound_tflops(
        arithmetic_intensity_flop_per_byte=ai,
        memory_bandwidth_tb_per_s=3.0,
        peak_compute_tflops=1_000.0,
    )
    print(f"算术强度={ai:.1f} FLOP/Byte，性能上界={ceiling:.1f} TFLOP/s。")


if __name__ == "__main__":
    report()
