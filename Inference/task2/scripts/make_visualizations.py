#!/usr/bin/env python3
"""Generate deterministic explanatory plots for the FlashAttention study note."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/ubuntu/flashattention_notes")
OUT = ROOT / "assets" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

# Prefer a CJK-capable font where the environment provides one; fall back safely.
plt.rcParams.update(
    {
        "font.sans-serif": ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 11,
        "axes.titleweight": "bold",
        "axes.labelcolor": "#1f2937",
        "xtick.color": "#374151",
        "ytick.color": "#374151",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

NAVY = "#0f172a"
BLUE = "#2563eb"
CYAN = "#0891b2"
MAGENTA = "#db2777"
GREEN = "#16a34a"
AMBER = "#d97706"
GRID = "#d1d5db"


def _save(fig: plt.Figure, filename: str) -> None:
    fig.savefig(OUT / filename, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_quadratic_score_memory() -> None:
    """Plot score-only storage at batch=1, heads=32, FP16.

    This is a formula-driven capacity illustration, not a measured allocator trace.
    The formula is heads * N^2 * bytes_per_element / 2^30.
    """
    lengths = np.array([1024, 2048, 4096, 8192, 16384, 32768])
    heads, dtype_bytes = 32, 2
    gib = heads * lengths.astype(np.float64) ** 2 * dtype_bytes / (1024 ** 3)

    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    ax.plot(lengths, gib, color=MAGENTA, marker="o", linewidth=2.8, markersize=7)
    ax.fill_between(lengths, gib, color=MAGENTA, alpha=0.10)
    for n, value in zip(lengths, gib):
        ax.annotate(f"{value:g} GiB", (n, value), xytext=(0, 10), textcoords="offset points", ha="center", color=MAGENTA, fontsize=9, weight="bold")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(lengths, [f"{n // 1024}K" for n in lengths])
    ax.set_yticks([1 / 16, 1 / 4, 1, 4, 16, 64], ["0.0625", "0.25", "1", "4", "16", "64"])
    ax.grid(True, which="both", color=GRID, linewidth=0.8, alpha=0.7)
    ax.set_title("Dense attention score storage grows quadratically")
    ax.set_xlabel("Sequence length N (log2 scale)")
    ax.set_ylabel("Score tensor only (GiB, B=1, H=32, FP16)")
    ax.text(
        0.02,
        0.96,
        r"memory = H × N² × 2 bytes",
        transform=ax.transAxes,
        va="top",
        color=NAVY,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"},
    )
    _save(fig, "quadratic_score_memory.png")


def plot_working_set_heatmap() -> None:
    """Visualise dense-score / largest-score-tile element ratio = (N / B)^2."""
    lengths = np.array([1024, 2048, 4096, 8192, 16384, 32768])
    blocks = np.array([64, 128, 256, 512])
    ratio = (lengths[None, :] / blocks[:, None]) ** 2

    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    image = ax.imshow(np.log2(ratio), cmap="magma", aspect="auto")
    cbar = fig.colorbar(image, ax=ax, pad=0.02)
    cbar.set_label("log₂(dense score elements / one score tile)")
    ax.set_xticks(np.arange(len(lengths)), [f"{n // 1024}K" for n in lengths])
    ax.set_yticks(np.arange(len(blocks)), [str(b) for b in blocks])
    ax.set_xlabel("Sequence length N")
    ax.set_ylabel("Block size B")
    ax.set_title("Peak score-tile working set relative to dense score matrix")
    for row in range(len(blocks)):
        for col in range(len(lengths)):
            color = "white" if np.log2(ratio[row, col]) > 9 else NAVY
            ax.text(col, row, f"{ratio[row, col]:,.0f}×", ha="center", va="center", color=color, fontsize=9, weight="bold")
    ax.text(
        0.01,
        -0.22,
        "Each cell is (N/B)². It compares score elements only; real kernels also carry Q/K/V tiles and row-wise state.",
        transform=ax.transAxes,
        fontsize=9,
        color="#4b5563",
    )
    _save(fig, "score_working_set_heatmap.png")


def _online_state(scores: np.ndarray, values: np.ndarray, block_size: int) -> list[dict[str, float]]:
    """Return exact online-softmax state after each deterministic score block."""
    m, l, numerator = -math.inf, 0.0, 0.0
    trace: list[dict[str, float]] = []
    for start in range(0, len(scores), block_size):
        score_block = scores[start : start + block_size]
        value_block = values[start : start + block_size]
        block_max = float(score_block.max())
        m_new = max(m, block_max)
        old_scale = 0.0 if not math.isfinite(m) else math.exp(m - m_new)
        weights = np.exp(score_block - m_new)
        l = old_scale * l + float(weights.sum())
        numerator = old_scale * numerator + float(weights @ value_block)
        m = m_new
        trace.append({"block": len(trace) + 1, "m": m, "l": l, "numerator": numerator, "output": numerator / l})
    return trace


def plot_online_softmax_trace() -> None:
    """Show a hand-picked, non-random six-score example processed in three tiles."""
    scores = np.array([-1.1, 0.2, 2.4, -0.3, 3.0, 1.2])
    values = np.array([0.4, -0.8, 1.0, 0.6, -0.2, 0.9])
    trace = _online_state(scores, values, block_size=2)
    blocks = np.array([item["block"] for item in trace])
    m = np.array([item["m"] for item in trace])
    l = np.array([item["l"] for item in trace])
    numerator = np.array([item["numerator"] for item in trace])
    output = np.array([item["output"] for item in trace])
    exact = float(np.exp(scores - scores.max()) @ values / np.exp(scores - scores.max()).sum())

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.5), gridspec_kw={"width_ratios": [1.05, 1]})
    ax = axes[0]
    ax.plot(blocks, m, "o-", color=MAGENTA, linewidth=2.4, label="running max m")
    ax.plot(blocks, l, "o-", color=BLUE, linewidth=2.4, label="rescaled denominator l")
    ax.set_xticks(blocks, ["scores 1–2", "scores 3–4", "scores 5–6"])
    ax.set_title("Online softmax state after each tile")
    ax.set_xlabel("Processed score tile")
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1]
    ax.plot(blocks, numerator, "o-", color=AMBER, linewidth=2.4, label="unnormalized numerator N")
    ax.plot(blocks, output, "o-", color=GREEN, linewidth=2.4, label="current output N / l")
    ax.axhline(exact, color=NAVY, linestyle="--", linewidth=1.4, label=f"dense reference = {exact:.4f}")
    ax.set_xticks(blocks, ["tile 1", "tile 2", "tile 3"])
    ax.set_title("Value-weighted accumulation reaches the dense answer")
    ax.set_xlabel("Processed score tile")
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, loc="best")
    fig.suptitle("A deterministic six-element example: no full probability vector is stored", y=1.02, weight="bold", color=NAVY)
    _save(fig, "online_softmax_trace.png")


def main() -> None:
    plot_quadratic_score_memory()
    plot_working_set_heatmap()
    plot_online_softmax_trace()
    print(f"Wrote plots to {OUT}")


if __name__ == "__main__":
    main()
