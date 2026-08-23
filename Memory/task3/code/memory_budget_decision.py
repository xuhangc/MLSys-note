"""Turn measured memory-strategy results into an explicit engineering decision.

This module has no PyTorch dependency so its selection logic can be unit-tested
on any machine. Feed it the JSON-like rows emitted by the benchmark script.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class MemoryBudget:
    """Hard feasibility constraints plus two 'worth keeping' thresholds."""

    memory_cap_mb: float
    min_samples_per_s: float
    max_relative_loss_increase: float = 0.02
    min_meaningful_saving_mb: float = 512.0
    min_throughput_ratio: float = 0.70


def validate_budget(budget: MemoryBudget) -> None:
    """Fail fast when a decision boundary is nonsensical or incomplete."""
    if budget.memory_cap_mb <= 0:
        raise ValueError("memory_cap_mb must be positive")
    if budget.min_samples_per_s <= 0:
        raise ValueError("min_samples_per_s must be positive")
    if not 0 <= budget.max_relative_loss_increase < 1:
        raise ValueError("max_relative_loss_increase must lie in [0, 1)")
    if budget.min_meaningful_saving_mb < 0:
        raise ValueError("min_meaningful_saving_mb must be non-negative")
    if not 0 < budget.min_throughput_ratio <= 1:
        raise ValueError("min_throughput_ratio must lie in (0, 1]")


def annotate_feasibility(
    candidates: Iterable[dict[str, Any]], budget: MemoryBudget
) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
    """Add pass/fail flags without mutating the caller's records.

    The baseline supplies the loss reference. A candidate is feasible only if it
    satisfies *all* hard limits: memory, throughput, and quality.
    """
    validate_budget(budget)
    rows = [dict(row) for row in candidates]
    baseline = next(
        (row for row in rows if row.get("name") == "baseline" and row.get("status", "ok") == "ok"),
        None,
    )
    if baseline is None:
        raise ValueError("A successful baseline row is required")
    if "eval_loss" not in baseline:
        raise ValueError("The baseline needs eval_loss to establish a quality guardrail")

    quality_cap = float(baseline["eval_loss"]) * (1 + budget.max_relative_loss_increase)
    for row in rows:
        successful = row.get("status", "ok") == "ok"
        metric_keys = {"peak_allocated_mb", "samples_per_s", "eval_loss"}
        has_metrics = metric_keys.issubset(row)
        if not successful or not has_metrics:
            row.update({"memory_ok": False, "speed_ok": False, "quality_ok": False, "feasible": False})
            continue
        row["memory_ok"] = float(row["peak_allocated_mb"]) <= budget.memory_cap_mb
        row["speed_ok"] = float(row["samples_per_s"]) >= budget.min_samples_per_s
        row["quality_ok"] = float(row["eval_loss"]) <= quality_cap
        row["feasible"] = row["memory_ok"] and row["speed_ok"] and row["quality_ok"]
    return rows, baseline, quality_cap


def decide_memory_plan(candidates: Iterable[dict[str, Any]], budget: MemoryBudget) -> dict[str, Any]:
    """Return accept, tune, or reject with auditable quantitative evidence.

    Ranking is intentionally lexicographic: after hard constraints, minimise peak
    memory first, preserve throughput second, then choose lower evaluation loss.
    That ranking should be changed if the team's objective differs.
    """
    rows, baseline, quality_cap = annotate_feasibility(candidates, budget)
    feasible = [row for row in rows if row["feasible"]]
    if not feasible:
        return {
            "decision": "reject",
            "reason": "No candidate satisfies memory, throughput, and quality constraints together.",
            "quality_cap": round(quality_cap, 6),
            "feasible_names": [],
            "annotated_candidates": rows,
        }

    feasible.sort(key=lambda row: (row["peak_allocated_mb"], -row["samples_per_s"], row["eval_loss"]))
    best = feasible[0]
    memory_saving = float(baseline["peak_allocated_mb"]) - float(best["peak_allocated_mb"])
    throughput_ratio = float(best["samples_per_s"]) / float(baseline["samples_per_s"])
    decisive = (
        best["name"] != "baseline"
        and memory_saving >= budget.min_meaningful_saving_mb
        and throughput_ratio >= budget.min_throughput_ratio
    )
    if decisive:
        decision, reason = (
            "accept",
            "The best feasible non-baseline strategy clears both practical-gain thresholds.",
        )
    else:
        decision, reason = (
            "tune",
            "A strategy is feasible, but its memory saving or throughput retention is not yet decisive.",
        )
    return {
        "decision": decision,
        "reason": reason,
        "quality_cap": round(quality_cap, 6),
        "best_candidate": best["name"],
        "feasible_names": [row["name"] for row in feasible],
        "memory_saving_mb_vs_baseline": round(memory_saving, 2),
        "throughput_ratio_vs_baseline": round(throughput_ratio, 4),
        "annotated_candidates": rows,
    }


def _self_test() -> None:
    budget = MemoryBudget(memory_cap_mb=12_000, min_samples_per_s=6)
    candidates = [
        {"name": "baseline", "peak_allocated_mb": 18_000, "samples_per_s": 8.0, "eval_loss": 1.06},
        {"name": "checkpoint", "peak_allocated_mb": 11_700, "samples_per_s": 6.4, "eval_loss": 1.07},
        {"name": "offload", "peak_allocated_mb": 9_500, "samples_per_s": 4.2, "eval_loss": 1.06},
    ]
    decision = decide_memory_plan(candidates, budget)
    assert decision["decision"] == "accept"
    assert decision["best_candidate"] == "checkpoint"
    assert decision["feasible_names"] == ["checkpoint"]

    impossible = decide_memory_plan(
        [
            {"name": "baseline", "peak_allocated_mb": 18_000, "samples_per_s": 8.0, "eval_loss": 1.06},
            {"name": "checkpoint", "peak_allocated_mb": 12_100, "samples_per_s": 6.5, "eval_loss": 1.07},
        ],
        budget,
    )
    assert impossible["decision"] == "reject"
    print("Self-test passed.")


if __name__ == "__main__":
    _self_test()
