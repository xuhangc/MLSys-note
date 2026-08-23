"""Plot the example GPU measurements reported in the cited Datawhale notebooks.

These values are a documented *single-machine teaching example*, not universal
benchmarks. The charts are deliberately labelled as source example measurements.
"""
from pathlib import Path

import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Source: Datawhale notebook 76, RTX 5070 Ti Laptop GPU, Qwen2.5-0.5B,
# FP32 + AdamW, batch=1, seq_len=768, warmup=2, iters=5.
STRATEGIES = ["baseline", "checkpoint", "offload", "hybrid"]
PEAK_MEMORY_MIB = [9782.74, 9450.76, 9448.33, 9454.64]
STEP_TIME_MS = [478.113, 558.504, 1875.037, 768.882]
THROUGHPUT = [2.092, 1.790, 0.533, 1.301]
COLORS = ["#5B8FF9", "#8B5CF6", "#F59E0B", "#10B981"]


def add_labels(axis, values, fmt):
    for index, value in enumerate(values):
        axis.text(index, value, fmt.format(value), ha="center", va="bottom", fontsize=9)


def plot_strategy_benchmark():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    plots = [
        (PEAK_MEMORY_MIB, "Peak allocated memory", "MiB", "{:.0f}"),
        (STEP_TIME_MS, "Training step time", "ms / step", "{:.0f}"),
        (THROUGHPUT, "Training throughput", "samples / second", "{:.3f}"),
    ]
    for axis, (values, title, ylabel, fmt) in zip(axes, plots):
        axis.bar(STRATEGIES, values, color=COLORS)
        axis.set_title(title, fontweight="bold")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        add_labels(axis, values, fmt)
    fig.suptitle(
        "Source example: activation-memory strategies on one GPU workload", fontweight="bold", fontsize=14
    )
    fig.savefig(OUTPUT_DIR / "datawhale_strategy_example.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_amp_example():
    # Source: Datawhale notebook 73, RTX 5070 Ti Laptop GPU, Qwen2.5-0.5B,
    # batch=1, seq_len=256, warmup=3, iters=10. Metrics are not comparable
    # with the notebook-76 workload because its sequence length differs.
    modes = ["FP32", "AMP (BF16)"]
    step_time = [244.490, 191.331]
    throughput = [4.090, 5.227]
    peak_memory = [9474.12, 9466.15]
    colors = ["#64748B", "#22C55E"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    plots = [
        (step_time, "Training step time", "ms / step", "{:.1f}"),
        (throughput, "Training throughput", "samples / second", "{:.3f}"),
        (peak_memory, "Peak allocated memory", "MiB", "{:.1f}"),
    ]
    for axis, (values, title, ylabel, fmt) in zip(axes, plots):
        axis.bar(modes, values, color=colors)
        axis.set_title(title, fontweight="bold")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        add_labels(axis, values, fmt)
    fig.suptitle(
        "Source example: FP32 versus AMP on one GPU workload", fontweight="bold", fontsize=14
    )
    fig.savefig(OUTPUT_DIR / "datawhale_amp_example.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    plot_strategy_benchmark()
    plot_amp_example()
    print(f"Saved figures under {OUTPUT_DIR}")
