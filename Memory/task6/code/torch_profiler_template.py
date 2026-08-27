#!/usr/bin/env python3
"""Minimal template for collecting real PyTorch profiling evidence.

Use this file only in an environment with PyTorch and, for CUDA evidence, a
CUDA-capable build and device. It intentionally does not claim that a trace was
collected until `collect_torch_profile` returns successfully. The tiny model is
an instrumentation example, not an LLM benchmark.

Example:
    python3 code/torch_profiler_template.py --output-dir results/torch_trace
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable


def require_torch() -> Any:
    """Import PyTorch lazily so this source file remains inspectable everywhere."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to collect a real profiler trace.") from exc
    return torch


def synchronize_if_cuda(torch: Any) -> None:
    """Fence asynchronous CUDA work when a wall-clock timing boundary is needed."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark_step(step_fn: Callable[[], None], torch: Any, warmup: int = 5, iters: int = 30) -> dict[str, float]:
    """Warm up first, then measure mean, P50 and P95 of exactly the same step."""
    if warmup < 0 or iters <= 0:
        raise ValueError("warmup must be non-negative and iters must be positive.")
    for _ in range(warmup):
        step_fn()
    synchronize_if_cuda(torch)

    samples_ms: list[float] = []
    for _ in range(iters):
        synchronize_if_cuda(torch)
        start = time.perf_counter()
        step_fn()
        synchronize_if_cuda(torch)
        samples_ms.append((time.perf_counter() - start) * 1000.0)

    samples = torch.tensor(samples_ms, dtype=torch.float64)
    return {
        "mean_step_ms": float(samples.mean().item()),
        "p50_step_ms": float(samples.quantile(0.50).item()),
        "p95_step_ms": float(samples.quantile(0.95).item()),
        "iters": float(iters),
        "warmup": float(warmup),
    }


def collect_torch_profile(
    step_fn: Callable[[], None],
    torch: Any,
    output_dir: Path,
    warmup: int = 3,
    iters: int = 8,
) -> dict[str, Any]:
    """Collect a short scheduled CPU/CUDA trace and export the trace directory.

    `profile_memory=True` reports profiler-tracked tensor allocation/release
    information. It is not a substitute for a full allocator snapshot. The
    active recording window excludes one profiler warm-up step when possible.
    """
    if warmup < 0 or iters <= 1:
        raise ValueError("warmup must be non-negative and iters must exceed one.")
    output_dir.mkdir(parents=True, exist_ok=True)
    for _ in range(warmup):
        step_fn()
    synchronize_if_cuda(torch)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    active_steps = max(1, iters - 1)
    schedule = torch.profiler.schedule(wait=0, warmup=1, active=active_steps, repeat=1)
    trace_handler = torch.profiler.tensorboard_trace_handler(str(output_dir))

    with torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=trace_handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(iters):
            step_fn()
            prof.step()
    synchronize_if_cuda(torch)

    sort_key = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
    top_operators = prof.key_averages().table(sort_by=sort_key, row_limit=12)
    (output_dir / "top_operators.txt").write_text(top_operators + "\n", encoding="utf-8")
    return {
        "status": "collected",
        "tool": "torch.profiler",
        "activities": [activity.name for activity in activities],
        "schedule": {"wait": 0, "warmup": 1, "active": active_steps, "repeat": 1},
        "record_shapes": True,
        "profile_memory": True,
        "trace_dir": str(output_dir),
        "top_operators_path": str(output_dir / "top_operators.txt"),
        "sort_key": sort_key,
    }


def build_tiny_step(torch: Any, device: str) -> Callable[[], None]:
    """Build a repeatable demonstration step; replace it with the real workload."""
    torch.manual_seed(2026)
    model = torch.nn.Sequential(
        torch.nn.Linear(512, 2048),
        torch.nn.GELU(),
        torch.nn.Linear(2048, 512),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = torch.randn(32, 512, device=device)
    target = torch.randn(32, 512, device=device)

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        with torch.profiler.record_function("forward_and_loss"):
            output = model(batch)
            loss = torch.nn.functional.mse_loss(output, target)
        with torch.profiler.record_function("backward"):
            loss.backward()
        with torch.profiler.record_function("optimizer_step"):
            optimizer.step()

    return step


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect a short real PyTorch profiler trace.")
    parser.add_argument("--output-dir", default="results/torch_trace", help="Directory for TensorBoard trace and summary.")
    parser.add_argument("--device", default=None, choices=("cpu", "cuda"), help="Default is CUDA when available, otherwise CPU.")
    args = parser.parse_args()

    torch = require_torch()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable.")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    step = build_tiny_step(torch, device)
    timing = benchmark_step(step, torch)
    profile = collect_torch_profile(step, torch, Path(args.output_dir))
    payload = {
        "provenance": "real torch.profiler run of the template tiny model; replace with the target workload before drawing conclusions",
        "device": device,
        "timing": timing,
        "profile": profile,
    }
    summary_path = Path(args.output_dir) / "profile_summary.json"
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
