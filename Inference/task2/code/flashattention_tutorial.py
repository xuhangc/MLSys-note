"""An executable, CPU-friendly FlashAttention teaching implementation.

This file intentionally simulates the *algorithmic contract* of FlashAttention:
blocks of scores are consumed immediately and the full N x N score matrix is
never materialized. It is not intended to replace a CUDA/Triton kernel.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple, Union

import torch

Tensor = torch.Tensor


@dataclass
class TiledAttentionStats:
    """Shape-level diagnostics rather than allocator-level GPU measurements."""

    seq_len: int
    block_size: int
    q_tiles: int
    kv_tiles: int
    peak_score_elements: int
    dense_score_elements: int
    score_working_set_ratio: float


def _check_qkv(q: Tensor, k: Tensor, v: Tensor) -> Tuple[int, int, int]:
    """Validate the simple single-head [sequence, feature] teaching interface."""
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError("q, k, v must all have shape [sequence_length, feature_dim].")
    if q.shape[1] != k.shape[1]:
        raise ValueError("q and k must have the same key/query feature dimension.")
    if k.shape[0] != v.shape[0]:
        raise ValueError("k and v must have the same sequence length.")
    if q.shape[0] != k.shape[0]:
        raise ValueError("This self-attention demo expects q, k, v with equal sequence length.")
    if not (q.device == k.device == v.device):
        raise ValueError("q, k, v must be on the same device.")
    return q.shape[0], q.shape[1], v.shape[1]


def _causal_mask(q_start: int, q_count: int, k_start: int, k_count: int, device: torch.device) -> Tensor:
    """Return [q_count, k_count] validity for a causal self-attention block."""
    q_positions = torch.arange(q_start, q_start + q_count, device=device)[:, None]
    k_positions = torch.arange(k_start, k_start + k_count, device=device)[None, :]
    return k_positions <= q_positions


def reference_attention(q: Tensor, k: Tensor, v: Tensor, *, causal: bool = False) -> Tensor:
    """Compute the usual dense scaled-dot-product attention for comparison.

    The tensor ``scores`` has N^2 elements, which is exactly the intermediate
    allocation that FlashAttention avoids persisting in high-bandwidth memory.
    """
    seq_len, d_key, _ = _check_qkv(q, k, v)
    scores = (q @ k.T) / math.sqrt(d_key)
    if causal:
        valid = torch.arange(seq_len, device=q.device)[:, None] >= torch.arange(seq_len, device=q.device)[None, :]
        scores = scores.masked_fill(~valid, -torch.inf)
    probabilities = torch.softmax(scores, dim=-1)
    return probabilities @ v


def flash_attention_tiled(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    block_size: int = 128,
    causal: bool = False,
    return_stats: bool = False,
) -> Union[Tensor, Tuple[Tensor, TiledAttentionStats]]:
    """Compute exact attention through tiles and online softmax statistics.

    For each query row we maintain three *linear-size* states:

    * ``m``: largest score observed so far;
    * ``l``: sum of exponentials expressed relative to ``m``;
    * ``n``: unnormalised value-weighted numerator expressed relative to ``m``.

    The final result is ``n / l``. Carrying ``n`` rather than repeatedly
    normalising ``out`` is a useful teaching form: it makes the change of
    softmax reference point explicit and delays one division until the end of
    a query tile.
    """
    seq_len, d_key, d_value = _check_qkv(q, k, v)
    if block_size <= 0:
        raise ValueError("block_size must be a positive integer.")

    scale = 1.0 / math.sqrt(d_key)
    output = torch.empty((seq_len, d_value), dtype=v.dtype, device=v.device)
    n_q_tiles = math.ceil(seq_len / block_size)
    n_kv_tiles = math.ceil(seq_len / block_size)

    for q_start in range(0, seq_len, block_size):
        q_end = min(q_start + block_size, seq_len)
        q_block = q[q_start:q_end] * scale
        q_count = q_end - q_start

        # m=-inf and l=n=0 encode "no score has been consumed yet".
        m = torch.full((q_count, 1), -torch.inf, dtype=q.dtype, device=q.device)
        l = torch.zeros((q_count, 1), dtype=q.dtype, device=q.device)
        n = torch.zeros((q_count, d_value), dtype=v.dtype, device=v.device)

        for kv_start in range(0, seq_len, block_size):
            kv_end = min(kv_start + block_size, seq_len)
            k_block = k[kv_start:kv_end]
            v_block = v[kv_start:kv_end]

            # This score tile exists only for the duration of this loop body.
            scores = q_block @ k_block.T
            if causal:
                valid = _causal_mask(q_start, q_count, kv_start, kv_end - kv_start, q.device)
                scores = scores.masked_fill(~valid, -torch.inf)

            block_max = scores.max(dim=-1, keepdim=True).values
            m_new = torch.maximum(m, block_max)

            # Re-express earlier statistics using the newly chosen reference m_new.
            # torch.where prevents the initial -inf - -inf case from producing NaN.
            has_old_mass = torch.isfinite(m)
            old_scale = torch.where(has_old_mass, torch.exp(m - m_new), torch.zeros_like(m))

            # Invalid causal entries have zero probability mass. Replacing their
            # exponent input with -inf avoids undefined (-inf)-(-inf).
            valid_score = torch.isfinite(scores) & torch.isfinite(m_new)
            exponent_input = torch.where(valid_score, scores - m_new, torch.full_like(scores, -torch.inf))
            p_unnormalised = torch.exp(exponent_input)

            l_new = old_scale * l + p_unnormalised.sum(dim=-1, keepdim=True)
            n_new = old_scale * n + p_unnormalised @ v_block
            m, l, n = m_new, l_new, n_new

        output[q_start:q_end] = n / l

    if not return_stats:
        return output

    peak_score_elements = min(block_size, seq_len) ** 2
    dense_score_elements = seq_len ** 2
    stats = TiledAttentionStats(
        seq_len=seq_len,
        block_size=block_size,
        q_tiles=n_q_tiles,
        kv_tiles=n_kv_tiles,
        peak_score_elements=peak_score_elements,
        dense_score_elements=dense_score_elements,
        score_working_set_ratio=dense_score_elements / peak_score_elements,
    )
    return output, stats


def flash_attention_tiled_bh(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    block_size: int = 128,
    causal: bool = False,
) -> Tensor:
    """Apply the teaching kernel independently to [batch, heads, sequence, dim].

    Production implementations fuse and parallelise these axes; this wrapper
    only makes the relationship to ordinary multi-head attention explicit.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Expected q, k, v with shape [batch, heads, sequence, feature].")
    if q.shape[:3] != k.shape[:3] or q.shape[:3] != v.shape[:3]:
        raise ValueError("q, k, v must agree on batch, head, and sequence axes.")

    batch, heads, seq_len, d_value = v.shape
    result = torch.empty((batch, heads, seq_len, d_value), dtype=v.dtype, device=v.device)
    for batch_index in range(batch):
        for head_index in range(heads):
            result[batch_index, head_index] = flash_attention_tiled(
                q[batch_index, head_index],
                k[batch_index, head_index],
                v[batch_index, head_index],
                block_size=block_size,
                causal=causal,
            )
    return result


def verify_forward_equivalence() -> None:
    """Verify ordinary and causal outputs for uneven final tiles."""
    cases = [
        # (sequence length, key dimension, value dimension, block size, causal)
        (8, 4, 5, 2, False),
        (13, 7, 3, 5, False),
        (11, 6, 4, 4, True),
        (1, 3, 2, 8, True),
    ]
    for seed, (seq_len, d_key, d_value, block_size, causal) in enumerate(cases, start=11):
        torch.manual_seed(seed)
        q = torch.randn(seq_len, d_key, dtype=torch.float64)
        k = torch.randn(seq_len, d_key, dtype=torch.float64)
        v = torch.randn(seq_len, d_value, dtype=torch.float64)
        expected = reference_attention(q, k, v, causal=causal)
        actual, stats = flash_attention_tiled(q, k, v, block_size=block_size, causal=causal, return_stats=True)
        max_error = (expected - actual).abs().max().item()
        print(
            f"N={seq_len:2d}, d_k={d_key:2d}, B={block_size:2d}, causal={str(causal):5s} "
            f"| max_abs_error={max_error:.3e} | score working-set ratio={stats.score_working_set_ratio:.1f}x"
        )
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)


def verify_gradients() -> None:
    """Compare gradients of a scalar loss; use float64 to make the test strict."""
    torch.manual_seed(2026)
    q = torch.randn(7, 4, dtype=torch.float64, requires_grad=True)
    k = torch.randn(7, 4, dtype=torch.float64, requires_grad=True)
    v = torch.randn(7, 3, dtype=torch.float64, requires_grad=True)
    probe = torch.randn(7, 3, dtype=torch.float64)

    ref_loss = (reference_attention(q, k, v, causal=True) * probe).sum()
    ref_grads = torch.autograd.grad(ref_loss, (q, k, v), retain_graph=False)

    tiled_loss = (flash_attention_tiled(q, k, v, block_size=3, causal=True) * probe).sum()
    tiled_grads = torch.autograd.grad(tiled_loss, (q, k, v), retain_graph=False)

    for name, expected, actual in zip(("dQ", "dK", "dV"), ref_grads, tiled_grads):
        max_error = (expected - actual).abs().max().item()
        print(f"gradient {name}: max_abs_error={max_error:.3e}")
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    print("Forward-equivalence checks")
    verify_forward_equivalence()
    print("\nGradient-equivalence check")
    verify_gradients()
    print("\nAll checks passed.")
