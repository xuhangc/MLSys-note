"""Reproducible benchmark for activation checkpointing and CPU activation offload.

Run on a CUDA machine, for example:
    python benchmark_memory_strategies.py --seq-len 512 --batch-size 2 --depth 8

The script intentionally rebuilds the model for every strategy, keeps the same
seed/input/optimizer hyperparameters, warms up before timing, synchronizes CUDA
before and after the timed region, and writes machine-readable JSON plus a PNG.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class Workload:
    """All variables that must stay fixed across a fair strategy comparison."""

    batch_size: int = 2
    seq_len: int = 256
    vocab_size: int = 2048
    d_model: int = 256
    n_heads: int = 8
    depth: int = 6
    ff_mult: int = 4
    warmup_steps: int = 2
    measured_steps: int = 6
    learning_rate: float = 1e-3
    seed: int = 2026


@dataclass(frozen=True)
class Budget:
    """The acceptance boundary; change these values to fit a real deployment."""

    memory_cap_mb: float = 10_000.0
    min_samples_per_s: float = 1.0
    max_relative_eval_loss_increase: float = 0.02
    min_meaningful_memory_saving_mb: float = 256.0
    min_throughput_ratio: float = 0.70


class TransformerBlock(nn.Module):
    """A small pre-norm transformer block with activation-heavy intermediate states."""

    def __init__(self, d_model: int, n_heads: int, ff_mult: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Linear(d_model * ff_mult, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attended, _ = self.attn(h, h, h, need_weights=False)
        x = x + attended
        return x + self.mlp(self.norm2(x))


class TinyTransformerLM(nn.Module):
    """A compact language model that can optionally checkpoint every block."""

    def __init__(self, cfg: Workload) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position_embedding = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg.d_model, cfg.n_heads, cfg.ff_mult) for _ in range(cfg.depth)]
        )
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor, use_checkpoint: bool = False) -> torch.Tensor:
        positions = torch.arange(token_ids.size(1), device=token_ids.device).unsqueeze(0)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)
        for block in self.blocks:
            if use_checkpoint:
                # non-reentrant checkpointing records the graph and does not require
                # the input activation itself to have requires_grad=True.
                hidden = checkpoint(block, hidden, use_reentrant=False)
            else:
                hidden = block(hidden)
        return self.lm_head(self.final_norm(hidden))


def synchronize_if_needed(device: torch.device) -> None:
    """Turn asynchronous CUDA execution into a reliable timing boundary."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def cuda_peak_metrics(device: torch.device) -> tuple[float | None, float | None]:
    """Return allocated and reserved peaks in MiB for the current measurement window."""
    if device.type != "cuda":
        return None, None
    allocated = torch.cuda.max_memory_allocated(device) / (1024**2)
    reserved = torch.cuda.max_memory_reserved(device) / (1024**2)
    return round(allocated, 2), round(reserved, 2)


def context_for_strategy(strategy: str, device: torch.device):
    """Choose where autograd stores tensors needed later by backward.

    save_on_cpu moves saved tensors away from GPU during forward and restores them
    when backward needs them. It is deliberately only used on CUDA: on CPU there
    is no device-residency trade-off to benchmark.
    """
    if strategy in {"offload", "hybrid"} and device.type == "cuda":
        return torch.autograd.graph.save_on_cpu(pin_memory=True, device_type="cuda")
    return nullcontext()


def strategy_uses_checkpoint(strategy: str) -> bool:
    return strategy in {"checkpoint", "hybrid"}


def fresh_batch(cfg: Workload, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Create one fixed supervised next-token batch shared by all strategies."""
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed)
    tokens_cpu = torch.randint(
        low=0,
        high=cfg.vocab_size,
        size=(cfg.batch_size, cfg.seq_len),
        generator=generator,
    )
    # Predict the next token: token[t] -> target[t + 1].
    return tokens_cpu[:, :-1].to(device), tokens_cpu[:, 1:].to(device)


def one_training_step(
    model: TinyTransformerLM,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    strategy: str,
    device: torch.device,
) -> float:
    optimizer.zero_grad(set_to_none=True)
    with context_for_strategy(strategy, device):
        logits = model(inputs, use_checkpoint=strategy_uses_checkpoint(strategy))
        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
        loss.backward()
    optimizer.step()
    return float(loss.detach())


@torch.no_grad()
def evaluate_loss(
    model: TinyTransformerLM,
    inputs: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    was_training = model.training
    model.eval()
    logits = model(inputs, use_checkpoint=False)
    loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
    model.train(was_training)
    return float(loss)


def run_one_strategy(strategy: str, cfg: Workload, device: torch.device) -> dict[str, Any]:
    """Measure one strategy under an otherwise identical workload."""
    # Re-seed before construction so all strategies begin from identical weights.
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
        torch.cuda.empty_cache()

    model = TinyTransformerLM(cfg).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    inputs, labels = fresh_batch(cfg, device)

    for _ in range(cfg.warmup_steps):
        one_training_step(model, optimizer, inputs, labels, strategy, device)
    synchronize_if_needed(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    losses = [
        one_training_step(model, optimizer, inputs, labels, strategy, device)
        for _ in range(cfg.measured_steps)
    ]
    synchronize_if_needed(device)
    elapsed_s = time.perf_counter() - start
    eval_loss = evaluate_loss(model, inputs, labels)
    peak_allocated_mb, peak_reserved_mb = cuda_peak_metrics(device)

    return {
        "name": strategy,
        "status": "ok",
        "step_time_ms": round(elapsed_s * 1000 / cfg.measured_steps, 3),
        "samples_per_s": round(cfg.batch_size * cfg.measured_steps / elapsed_s, 3),
        "train_loss_last": round(losses[-1], 6),
        "eval_loss": round(eval_loss, 6),
        "peak_allocated_mb": peak_allocated_mb,
        "peak_reserved_mb": peak_reserved_mb,
    }


def safe_run(strategy: str, cfg: Workload, device: torch.device) -> dict[str, Any]:
    """Keep an OOM result in the report instead of aborting the comparison."""
    try:
        return run_one_strategy(strategy, cfg, device)
    except torch.OutOfMemoryError as exc:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return {"name": strategy, "status": "oom", "error": str(exc).splitlines()[0]}


def judge_candidates(results: list[dict[str, Any]], budget: Budget) -> dict[str, Any]:
    """Filter against hard constraints, then state whether the best option is decisive."""
    baseline = next((row for row in results if row["name"] == "baseline" and row["status"] == "ok"), None)
    if baseline is None or baseline["peak_allocated_mb"] is None:
        return {
            "decision": "not_applicable",
            "reason": "CUDA peak-memory metrics are required for a GPU-memory decision.",
            "feasible_names": [],
            "best_candidate": None,
        }

    loss_cap = baseline["eval_loss"] * (1 + budget.max_relative_eval_loss_increase)
    feasible: list[dict[str, Any]] = []
    annotated: list[dict[str, Any]] = []
    for row in results:
        item = dict(row)
        if row["status"] != "ok":
            item["feasible"] = False
            annotated.append(item)
            continue
        item["memory_ok"] = row["peak_allocated_mb"] <= budget.memory_cap_mb
        item["speed_ok"] = row["samples_per_s"] >= budget.min_samples_per_s
        item["quality_ok"] = row["eval_loss"] <= loss_cap
        item["feasible"] = item["memory_ok"] and item["speed_ok"] and item["quality_ok"]
        annotated.append(item)
        if item["feasible"]:
            feasible.append(item)

    feasible.sort(key=lambda row: (row["peak_allocated_mb"], -row["samples_per_s"], row["eval_loss"]))
    if not feasible:
        return {
            "decision": "reject",
            "reason": "No tested strategy simultaneously meets memory, throughput, and quality constraints.",
            "loss_cap": round(loss_cap, 6),
            "feasible_names": [],
            "best_candidate": None,
            "annotated_results": annotated,
        }

    best = feasible[0]
    saving_mb = baseline["peak_allocated_mb"] - best["peak_allocated_mb"]
    throughput_ratio = best["samples_per_s"] / baseline["samples_per_s"]
    decisive = (
        best["name"] != "baseline"
        and saving_mb >= budget.min_meaningful_memory_saving_mb
        and throughput_ratio >= budget.min_throughput_ratio
    )
    return {
        "decision": "accept" if decisive else "tune",
        "reason": (
            "Best feasible strategy clears the explicit memory-saving and throughput bars."
            if decisive
            else "A feasible strategy exists, but its net gain is not yet decisive; retest under higher pressure or tune its scope."
        ),
        "loss_cap": round(loss_cap, 6),
        "feasible_names": [row["name"] for row in feasible],
        "best_candidate": best["name"],
        "memory_saving_mb_vs_baseline": round(saving_mb, 2),
        "throughput_ratio_vs_baseline": round(throughput_ratio, 4),
        "annotated_results": annotated,
    }


def plot_results(results: list[dict[str, Any]], output_path: Path) -> None:
    """Create a chart only from measured successful CUDA runs."""
    rows = [row for row in results if row["status"] == "ok" and row["peak_allocated_mb"] is not None]
    if not rows:
        return
    names = [row["name"] for row in rows]
    memory = [row["peak_allocated_mb"] for row in rows]
    speed = [row["samples_per_s"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    colors = ["#5B8FF9", "#8B5CF6", "#F59E0B", "#10B981"][: len(rows)]
    axes[0].bar(names, memory, color=colors)
    axes[0].set_title("Peak allocated GPU memory")
    axes[0].set_ylabel("MiB")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(names, speed, color=colors)
    axes[1].set_title("Training throughput")
    axes[1].set_ylabel("samples / second")
    axes[1].grid(axis="y", alpha=0.25)
    for axis, values in zip(axes, (memory, speed)):
        for index, value in enumerate(values):
            axis.text(index, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Activation-memory strategy benchmark", fontsize=14, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="benchmark_output")
    parser.add_argument("--batch-size", type=int, default=Workload.batch_size)
    parser.add_argument("--seq-len", type=int, default=Workload.seq_len)
    parser.add_argument("--depth", type=int, default=Workload.depth)
    parser.add_argument("--warmup", type=int, default=Workload.warmup_steps)
    parser.add_argument("--iters", type=int, default=Workload.measured_steps)
    parser.add_argument("--memory-cap-mb", type=float, default=Budget.memory_cap_mb)
    parser.add_argument("--min-samples-per-s", type=float, default=Budget.min_samples_per_s)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Workload(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        depth=args.depth,
        warmup_steps=args.warmup,
        measured_steps=args.iters,
    )
    budget = Budget(memory_cap_mb=args.memory_cap_mb, min_samples_per_s=args.min_samples_per_s)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)

    if device.type != "cuda":
        print("CUDA is unavailable. The script exits without a GPU-memory verdict; run it on a CUDA machine.")
        return

    strategies = ["baseline", "checkpoint", "offload", "hybrid"]
    results = [safe_run(strategy, cfg, device) for strategy in strategies]
    decision = judge_candidates(results, budget)
    report = {
        "device": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "workload": asdict(cfg),
        "budget": asdict(budget),
        "results": results,
        "decision": decision,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot_results(results, output_dir / "benchmark_summary.png")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSaved: {output_dir / 'benchmark_report.json'}")
    print(f"Saved: {output_dir / 'benchmark_summary.png'}")


if __name__ == "__main__":
    main()
