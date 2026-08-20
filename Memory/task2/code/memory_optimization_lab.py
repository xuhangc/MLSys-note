"""A self-contained PyTorch lab for training-memory optimization.

Run on CPU:
    python memory_optimization_lab.py

Run the real CUDA benchmark (when a CUDA device is available):
    python memory_optimization_lab.py --benchmark

The benchmark reports environment-specific peak allocated memory and elapsed time.
"""
from __future__ import annotations

import argparse
import copy
import time
from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# -----------------------------------------------------------------------------
# Part 1. Gradient accumulation
# -----------------------------------------------------------------------------
class TinyClassifier(nn.Module):
    def __init__(self, in_features: int = 8, classes: int = 4) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.GELU(),
            nn.Linear(32, classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def iter_micro_batches(batch: dict[str, torch.Tensor], micro_batch_size: int):
    """Yield synchronized slices for every tensor in a batch dictionary."""
    if micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be positive")
    sizes = {value.size(0) for value in batch.values()}
    if len(sizes) != 1:
        raise ValueError("all tensors in batch must share dimension 0")
    batch_size = sizes.pop()
    for start in range(0, batch_size, micro_batch_size):
        stop = min(start + micro_batch_size, batch_size)
        yield {name: value[start:stop] for name, value in batch.items()}


def full_batch_step(model: nn.Module, optimizer: torch.optim.Optimizer, batch: dict[str, torch.Tensor]) -> float:
    """Reference implementation: one logical batch, one backward, one update."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(batch["features"])
    loss = F.cross_entropy(logits, batch["labels"], reduction="mean")
    loss.backward()
    optimizer.step()
    return loss.detach().item()


def accumulated_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    micro_batch_size: int,
    max_grad_norm: float | None = None,
) -> float:
    """One logical update using exact sample-weighted gradient accumulation.

    Cross entropy is summed on each micro-batch. The summed gradients are divided
    by the logical-batch sample count exactly once before ``optimizer.step()``.
    This remains correct even when the final micro-batch is smaller.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    num_samples = batch["labels"].size(0)
    total_loss_sum = 0.0

    for micro_batch in iter_micro_batches(batch, micro_batch_size):
        logits = model(micro_batch["features"])
        micro_loss_sum = F.cross_entropy(logits, micro_batch["labels"], reduction="sum")
        micro_loss_sum.backward()
        total_loss_sum += micro_loss_sum.detach().item()

    # Convert the accumulated sum-gradient into the full-batch mean-gradient.
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(num_samples)

    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    return total_loss_sum / num_samples


def demo_gradient_accumulation() -> None:
    torch.manual_seed(7)
    batch = {
        "features": torch.randn(13, 8),
        "labels": torch.randint(0, 4, (13,)),
    }
    initial_model = TinyClassifier()
    full_model = copy.deepcopy(initial_model)
    accum_model = copy.deepcopy(initial_model)

    full_optimizer = torch.optim.SGD(full_model.parameters(), lr=0.05)
    accum_optimizer = torch.optim.SGD(accum_model.parameters(), lr=0.05)
    full_loss = full_batch_step(full_model, full_optimizer, batch)
    accum_loss = accumulated_step(accum_model, accum_optimizer, batch, micro_batch_size=3)

    largest_difference = max(
        (left - right).abs().max().item()
        for left, right in zip(full_model.parameters(), accum_model.parameters())
    )
    assert largest_difference < 1e-6, f"unexpected parameter difference: {largest_difference}"
    print(f"[accumulation] full_loss={full_loss:.6f}, accumulated_loss={accum_loss:.6f}, max_diff={largest_difference:.2e}")


# -----------------------------------------------------------------------------
# Part 2. Activation checkpointing
# -----------------------------------------------------------------------------
class ResidualMLPBlock(nn.Module):
    """A small Transformer-like residual MLP block with nontrivial activations."""
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))


class BlockStack(nn.Module):
    def __init__(self, dim: int, depth: int, mode: str = "baseline") -> None:
        super().__init__()
        self.blocks = nn.ModuleList(ResidualMLPBlock(dim) for _ in range(depth))
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            if self.mode == "checkpoint":
                # Non-reentrant is the recommended modern API path.
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x


def demo_checkpoint_correctness() -> None:
    torch.manual_seed(11)
    baseline = BlockStack(dim=16, depth=3, mode="baseline")
    checkpointed = copy.deepcopy(baseline)
    checkpointed.mode = "checkpoint"
    x_normal = torch.randn(2, 5, 16, requires_grad=True)
    x_checkpoint = x_normal.detach().clone().requires_grad_(True)
    target = torch.randn(2, 5, 16)

    baseline_loss = F.mse_loss(baseline(x_normal), target)
    checkpoint_loss = F.mse_loss(checkpointed(x_checkpoint), target)
    baseline_loss.backward()
    checkpoint_loss.backward()

    assert torch.allclose(baseline_loss, checkpoint_loss, atol=1e-6)
    assert torch.allclose(x_normal.grad, x_checkpoint.grad, atol=1e-6)
    for left, right in zip(baseline.parameters(), checkpointed.parameters()):
        assert torch.allclose(left.grad, right.grad, atol=1e-6)
    print("[checkpointing] output loss and all checked gradients agree.")


# -----------------------------------------------------------------------------
# Part 3. Activation offload
# -----------------------------------------------------------------------------
class SaveActivationsOnCPU(nn.Module):
    """Wrap one module so its saved-for-backward tensors are parked on CPU.

    This changes only storage placement for autograd's saved tensors; it does not
    move model parameters or the current forward output to CPU. On a CUDA run,
    pinning host memory can support more efficient asynchronous transfers.
    """
    def __init__(self, module: nn.Module, pin_memory: bool = True) -> None:
        super().__init__()
        self.module = module
        self.pin_memory = pin_memory

    def forward(self, *args, **kwargs):
        with torch.autograd.graph.save_on_cpu(
            pin_memory=self.pin_memory and torch.cuda.is_available(),
            device_type="cuda",
        ):
            return self.module(*args, **kwargs)


@dataclass(frozen=True)
class ActivationChunk:
    name: str
    bytes_: int
    keep_score: float
    offloadable: bool = True


@dataclass(frozen=True)
class OffloadPlan:
    total_bytes: int
    kept_bytes: int
    offloaded_bytes: int
    gpu_budget_bytes: int
    transfer_ms_one_way: float
    saved_ratio: float
    offloaded_names: tuple[str, ...]


def plan_activation_offload(
    chunks: Iterable[ActivationChunk], gpu_budget_bytes: int, bandwidth_gbps: float
) -> OffloadPlan:
    """Greedily offload low-priority chunks until the GPU activation budget fits.

    ``bandwidth_gbps`` is treated as GiB/s and reports a one-way ideal transfer.
    A training step must normally include both GPU-to-CPU and CPU-to-GPU movement.
    """
    if gpu_budget_bytes <= 0 or bandwidth_gbps <= 0:
        raise ValueError("budget and bandwidth must both be positive")
    chunks = tuple(chunks)
    total = sum(chunk.bytes_ for chunk in chunks)
    kept = total
    offloaded: list[str] = []
    for chunk in sorted((c for c in chunks if c.offloadable), key=lambda c: (c.keep_score, c.bytes_, c.name)):
        if kept <= gpu_budget_bytes:
            break
        kept -= chunk.bytes_
        offloaded.append(chunk.name)
    offloaded_bytes = total - kept
    one_way_ms = 1000.0 * offloaded_bytes / (bandwidth_gbps * 1024**3)
    return OffloadPlan(
        total_bytes=total,
        kept_bytes=kept,
        offloaded_bytes=offloaded_bytes,
        gpu_budget_bytes=gpu_budget_bytes,
        transfer_ms_one_way=one_way_ms,
        saved_ratio=(offloaded_bytes / total) if total else 0.0,
        offloaded_names=tuple(offloaded),
    )


def demo_offload_planner() -> None:
    mib = 1024**2
    chunks = [
        ActivationChunk("embedding", 256 * mib, keep_score=0.95),
        ActivationChunk("block_01", 192 * mib, keep_score=0.25),
        ActivationChunk("block_02", 160 * mib, keep_score=0.10),
        ActivationChunk("logits", 128 * mib, keep_score=0.80),
    ]
    plan = plan_activation_offload(chunks, gpu_budget_bytes=384 * mib, bandwidth_gbps=8.0)
    assert plan.kept_bytes == 384 * mib
    assert plan.offloaded_names == ("block_02", "block_01")
    print(
        "[offload planner] "
        f"saved={plan.saved_ratio:.1%}, move={plan.offloaded_names}, "
        f"ideal one-way={plan.transfer_ms_one_way:.2f} ms, "
        f"round-trip≈{2 * plan.transfer_ms_one_way:.2f} ms"
    )


# -----------------------------------------------------------------------------
# Part 4. Optional CUDA measurement and an environment-specific tradeoff plot
# -----------------------------------------------------------------------------
def _measure_cuda_step(model: nn.Module, x: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requires a CUDA-capable PyTorch installation")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = model(x)
    F.mse_loss(output, target).backward()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000
    peak_mib = torch.cuda.max_memory_allocated() / 1024**2
    return peak_mib, elapsed_ms


def benchmark_and_plot(output_path: str = "memory_time_tradeoff.png") -> None:
    if not torch.cuda.is_available():
        print("[benchmark] CUDA unavailable; skip actual peak-memory measurement.")
        return
    device = torch.device("cuda")
    torch.manual_seed(123)
    dim, depth, batch_size, seq_len = 1024, 12, 2, 256
    initial = BlockStack(dim, depth, mode="baseline").to(device)
    baseline = copy.deepcopy(initial).to(device)
    checkpointed = copy.deepcopy(initial).to(device)
    checkpointed.mode = "checkpoint"

    # Offload all wrapped blocks for demonstration. In production, profile first
    # and wrap only selected regions because transfer cost can dominate.
    offloaded = nn.Sequential(*[SaveActivationsOnCPU(copy.deepcopy(block)) for block in initial.blocks]).to(device)
    x = torch.randn(batch_size, seq_len, dim, device=device, requires_grad=True)
    target = torch.randn_like(x)

    results: list[tuple[str, float, float]] = []
    for name, model in [("baseline", baseline), ("checkpoint", checkpointed), ("offload", offloaded)]:
        model.zero_grad(set_to_none=True)
        x.grad = None
        peak_mib, elapsed_ms = _measure_cuda_step(model, x, target)
        results.append((name, peak_mib, elapsed_ms))
        print(f"[benchmark] {name:10s}: peak={peak_mib:8.1f} MiB, time={elapsed_ms:8.1f} ms")

    labels, memory, elapsed = zip(*results)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    colors = {"baseline": "#254e70", "checkpoint": "#f28e2b", "offload": "#2a9d8f"}
    for name, peak_mib, elapsed_ms in results:
        ax.scatter(peak_mib, elapsed_ms, s=130, color=colors[name], zorder=3)
        ax.annotate(name, (peak_mib, elapsed_ms), xytext=(7, 7), textcoords="offset points")
    ax.set_xlabel("Peak allocated GPU memory (MiB)")
    ax.set_ylabel("Forward + backward time (ms)")
    ax.set_title("Measured memory–time tradeoff on this CUDA environment")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    print(f"[benchmark] wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true", help="run real CUDA benchmark and create a plot")
    parser.add_argument("--plot", default="memory_time_tradeoff.png")
    args = parser.parse_args()
    demo_gradient_accumulation()
    demo_checkpoint_correctness()
    demo_offload_planner()
    if args.benchmark:
        benchmark_and_plot(args.plot)


if __name__ == "__main__":
    main()
