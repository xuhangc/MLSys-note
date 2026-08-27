#!/usr/bin/env python3
"""A CPU-first laboratory for profiling-driven end-to-end optimization.

The lab measures real wall-clock elapsed time of small controlled stages.  Stage
latencies and working-set values are deliberately configured for *teaching*;
they are not timings of an ML model, GPU kernels, host-to-device copies, or a
GPU allocator. The purpose is to make the optimization contract, evidence
packet, one-variable experiment, bottleneck attribution, and decision gate
inspectable end to end.

Run from Memory/task6:
    python3 code/profiling_lab.py

The script writes charts to ../assets and reproducible records to ../results.
"""
from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "assets"
RESULT_DIR = ROOT / "results"


@dataclass(frozen=True)
class WorkloadContract:
    """Fields that must remain constant for a causal baseline/tuned comparison."""

    model_id: str
    phase: str
    batch_size: int
    sequence_length: int
    tokens_per_step: int
    device: str
    backend: str
    dtype: str
    warmup_steps: int
    measurement_steps: int
    quality_checksum: int
    max_p95_regression_pct: float

    def manifest(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StageSpec:
    """A named timed stage in a controlled end-to-end pipeline.

    `target_ms` is a pedagogical target implemented through a short sleep. It is
    never presented as a hardware measurement. `declared_working_set_mb` is a
    transparent capacity proxy used to teach memory trade-offs, not a sampled
    process RSS or GPU allocator peak.
    """

    name: str
    category: str
    target_ms: float
    declared_working_set_mb: float
    note: str


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration where exactly one named optimization changes the baseline."""

    name: str
    optimization: str
    changed_knob: str
    stages: tuple[StageSpec, ...]
    quality_checksum: int
    quality_passed: bool


@dataclass
class RunRecord:
    """Measured metrics and stage evidence for a single pipeline configuration."""

    config: str
    optimization: str
    changed_knob: str
    contract: dict[str, Any]
    step_latency_ms: list[float]
    stage_latency_ms: dict[str, list[float]]
    declared_working_set_mb: float
    quality_checksum: int
    quality_passed: bool

    def summary(self) -> dict[str, Any]:
        samples = np.asarray(self.step_latency_ms, dtype=np.float64)
        stage_means = {name: float(np.mean(values)) for name, values in self.stage_latency_ms.items()}
        total_stage_ms = max(sum(stage_means.values()), 1e-12)
        stage_shares = {name: 100.0 * value / total_stage_ms for name, value in stage_means.items()}
        mean_ms = float(np.mean(samples))
        return {
            "config": self.config,
            "optimization": self.optimization,
            "changed_knob": self.changed_knob,
            "mean_step_ms": mean_ms,
            "median_step_ms": float(np.median(samples)),
            "p95_step_ms": float(np.percentile(samples, 95)),
            "std_step_ms": float(np.std(samples, ddof=0)),
            "throughput_tokens_per_s": 1000.0 * self.contract["tokens_per_step"] / mean_ms,
            "declared_working_set_mb": self.declared_working_set_mb,
            "quality_checksum": self.quality_checksum,
            "quality_passed": self.quality_passed,
            "stage_mean_ms": stage_means,
            "stage_share_pct": stage_shares,
        }


def precise_pause(target_ms: float) -> int:
    """Run a tiny deterministic payload, then sleep to make a stage measurable.

    The integer return is a checksum contribution.  Real profiling would replace
    this with data loading, host-to-device transfer, a forward/backward step, or
    an optimizer update. We retain actual wall-clock measurement rather than
    inventing the observed duration from the configuration.
    """
    if target_ms < 0.0:
        raise ValueError("target_ms must be non-negative.")
    accumulator = 0
    for value in range(96):
        accumulator = (accumulator * 33 + value) % 1_000_003
    if target_ms > 0.0:
        time.sleep(target_ms / 1000.0)
    return accumulator


class ControlledPipeline:
    """Execute named stages and capture a trace suitable for evidence analysis."""

    def __init__(self, contract: WorkloadContract, config: PipelineConfig) -> None:
        self.contract = contract
        self.config = config
        if contract.quality_checksum != config.quality_checksum:
            raise ValueError("A candidate changed the correctness checksum.")
        if not config.quality_passed:
            raise ValueError("The controlled quality guard must pass before measurement.")

    def step(self, trace_start_us: float | None = None) -> tuple[dict[str, float], list[dict[str, Any]], int]:
        """Measure every stage once and return stage times plus Chrome-trace events."""
        stage_ms: dict[str, float] = {}
        events: list[dict[str, Any]] = []
        checksum = 0
        origin_us = time.perf_counter_ns() / 1000.0 if trace_start_us is None else trace_start_us

        for stage in self.config.stages:
            start_ns = time.perf_counter_ns()
            checksum ^= precise_pause(stage.target_ms)
            end_ns = time.perf_counter_ns()
            elapsed_ms = (end_ns - start_ns) / 1_000_000.0
            stage_ms[stage.name] = elapsed_ms
            events.append(
                {
                    "name": stage.name,
                    "cat": stage.category,
                    "ph": "X",
                    "ts": start_ns / 1000.0 - origin_us,
                    "dur": (end_ns - start_ns) / 1000.0,
                    "pid": 1,
                    "tid": stage.category,
                    "args": {"teaching_target_ms": stage.target_ms, "note": stage.note},
                }
            )
        return stage_ms, events, checksum


def validate_contract_pair(contract: WorkloadContract, baseline: PipelineConfig, candidate: PipelineConfig) -> None:
    """Confirm that only the declared optimization changes the comparison.

    We compare the contract first, then require a nonempty single knob string and
    identical quality checksum. Stage count stays fixed to keep attribution
    intelligible. In a real project, this check should also pin package versions,
    hardware, driver, data shard, random seed, and service traffic replay.
    """
    if baseline.quality_checksum != candidate.quality_checksum:
        raise AssertionError("Candidate checksum differs from baseline.")
    if not candidate.changed_knob or candidate.changed_knob == "baseline":
        raise AssertionError("Candidate must declare exactly one causal knob.")
    if len(baseline.stages) != len(candidate.stages):
        raise AssertionError("A candidate must preserve the end-to-end stage structure.")
    if contract.measurement_steps <= 4 or contract.warmup_steps < 0:
        raise AssertionError("Use warm-up and a meaningful sample count.")


def benchmark_pipeline(contract: WorkloadContract, config: PipelineConfig) -> tuple[RunRecord, list[dict[str, Any]]]:
    """Warm up, collect per-step/per-stage samples, and return an evidence record."""
    pipeline = ControlledPipeline(contract, config)
    for _ in range(contract.warmup_steps):
        pipeline.step()

    step_samples: list[float] = []
    stage_samples: dict[str, list[float]] = {stage.name: [] for stage in config.stages}
    trace_events: list[dict[str, Any]] = []
    trace_origin_us = time.perf_counter_ns() / 1000.0
    checksum_xor = 0

    for _ in range(contract.measurement_steps):
        step_start_ns = time.perf_counter_ns()
        stage_ms, events, checksum = pipeline.step(trace_start_us=trace_origin_us)
        step_end_ns = time.perf_counter_ns()
        step_samples.append((step_end_ns - step_start_ns) / 1_000_000.0)
        checksum_xor ^= checksum
        trace_events.extend(events)
        for stage_name, elapsed_ms in stage_ms.items():
            stage_samples[stage_name].append(elapsed_ms)

    # Same deterministic payload is executed in every stage; XOR must cancel
    # across an even count only when stage topology changed. The final checksum
    # guard below is intentionally based on contract/config instead.
    del checksum_xor
    record = RunRecord(
        config=config.name,
        optimization=config.optimization,
        changed_knob=config.changed_knob,
        contract=contract.manifest(),
        step_latency_ms=step_samples,
        stage_latency_ms=stage_samples,
        declared_working_set_mb=max(stage.declared_working_set_mb for stage in config.stages),
        quality_checksum=config.quality_checksum,
        quality_passed=config.quality_passed,
    )
    return record, trace_events


def classify_bottleneck(summary: dict[str, Any], stage_catalog: Iterable[StageSpec]) -> dict[str, Any]:
    """Attribute the largest measured stage to a resource category.

    Attribution is intentionally a first hypothesis, not a proof. An actual
    profiler trace, hardware counters, queue depth, and source stack are required
    before making a production claim. The output records both the stage and the
    evidence share so the reader can see why it was nominated.
    """
    categories = {stage.name: stage.category for stage in stage_catalog}
    stage_means = summary["stage_mean_ms"]
    stage_name = max(stage_means, key=stage_means.get)
    category = categories[stage_name]
    share = summary["stage_share_pct"][stage_name]
    next_actions = {
        "data_transfer": "profile DataLoader workers, host-to-device overlap, pinned memory and queue starvation",
        "compute": "inspect operator shapes, fusion opportunities, precision path and kernel selection",
        "synchronization": "locate host-device barriers, scalar reads, blocking copies and stream dependencies",
        "memory": "inspect allocation peaks, tensor lifetimes, fragmentation and recomputation trade-offs",
    }
    return {
        "category": category,
        "primary_stage": stage_name,
        "share_pct": share,
        "evidence": f"{stage_name} has the largest measured mean stage time ({stage_means[stage_name]:.3f} ms; {share:.1f}% of summed stages).",
        "next_action": next_actions.get(category, "collect a finer-grained trace before changing code"),
        "confidence": "teaching-hypothesis; corroborate with a real profiler before rollout",
    }


def compare_runs(baseline: dict[str, Any], tuned: dict[str, Any]) -> dict[str, float | bool]:
    """Use one sign convention: positive deltas mean an improvement.

    Step time, P95 and working-set proxy are better when lower; throughput is
    better when higher. Percent gains use the baseline denominator so candidates
    with different units remain comparable on a decision table.
    """
    time_delta = baseline["mean_step_ms"] - tuned["mean_step_ms"]
    p95_delta = baseline["p95_step_ms"] - tuned["p95_step_ms"]
    memory_delta = baseline["declared_working_set_mb"] - tuned["declared_working_set_mb"]
    throughput_delta = tuned["throughput_tokens_per_s"] - baseline["throughput_tokens_per_s"]
    return {
        "mean_step_delta_ms": time_delta,
        "mean_step_gain_pct": 100.0 * time_delta / baseline["mean_step_ms"],
        "p95_delta_ms": p95_delta,
        "p95_gain_pct": 100.0 * p95_delta / baseline["p95_step_ms"],
        "working_set_delta_mb": memory_delta,
        "throughput_delta_tokens_per_s": throughput_delta,
        "throughput_gain_pct": 100.0 * throughput_delta / baseline["throughput_tokens_per_s"],
        "time_improved": time_delta > 0.0,
        "p95_improved": p95_delta > 0.0,
        "memory_improved": memory_delta > 0.0,
        "throughput_improved": throughput_delta > 0.0,
    }


def recommend_decision(
    baseline: dict[str, Any],
    tuned: dict[str, Any],
    comparison: dict[str, float | bool],
    max_p95_regression_pct: float,
) -> dict[str, str]:
    """Return accept/tune/reject only after quality and latency guards are checked.

    A candidate earns `accept` when it improves mean latency and throughput by at
    least 8%, does not regress P95 beyond the contract, retains correctness, and
    has profiling evidence. Smaller but coherent progress becomes `tune`; a
    quality or tail-latency violation becomes `reject` even if mean latency wins.
    """
    if not tuned["quality_passed"] or tuned["quality_checksum"] != baseline["quality_checksum"]:
        return {"decision": "reject", "reason": "quality guard failed or checksum changed", "next_action": "restore correctness before performance work"}
    if comparison["p95_gain_pct"] < -max_p95_regression_pct:
        return {"decision": "reject", "reason": "P95 regression exceeds the workload contract", "next_action": "inspect tail stalls and synchronization before rollout"}
    strong_time = comparison["mean_step_gain_pct"] >= 8.0
    strong_throughput = comparison["throughput_gain_pct"] >= 8.0
    if strong_time and strong_throughput:
        return {"decision": "accept", "reason": "mean latency and throughput both clear the 8% gate while quality and P95 pass", "next_action": "repeat on the real target workload with a production profiler"}
    coherent_progress = comparison["mean_step_gain_pct"] > 0.0 and comparison["throughput_gain_pct"] > 0.0
    if coherent_progress:
        return {"decision": "tune", "reason": "direction is positive but the end-to-end gain has not cleared the rollout gate", "next_action": "keep the one-variable change and collect a longer trace"}
    return {"decision": "reject", "reason": "no stable end-to-end speed and throughput improvement", "next_action": "revisit the bottleneck hypothesis"}


def make_contract() -> WorkloadContract:
    return WorkloadContract(
        model_id="teaching-transformer-step/v1",
        phase="training-step",
        batch_size=8,
        sequence_length=512,
        tokens_per_step=4096,
        device="cpu-first controlled lab",
        backend="python time.perf_counter",
        dtype="teaching-proxy",
        warmup_steps=3,
        measurement_steps=28,
        quality_checksum=914_731,
        max_p95_regression_pct=3.0,
    )


def make_configs(contract: WorkloadContract) -> tuple[PipelineConfig, list[PipelineConfig]]:
    """Create a baseline and independent one-variable candidate configurations."""
    base_stages = (
        StageSpec("data_wait", "data_transfer", 4.4, 1700, "teaching proxy for input wait / preprocessing"),
        StageSpec("h2d_copy", "data_transfer", 2.2, 2000, "teaching proxy for host-to-device handoff"),
        StageSpec("forward", "compute", 4.7, 3900, "teaching proxy for forward operator sequence"),
        StageSpec("backward", "compute", 6.5, 6100, "teaching proxy for backward operator sequence"),
        StageSpec("optimizer", "compute", 2.1, 5200, "teaching proxy for optimizer update"),
        StageSpec("host_sync", "synchronization", 2.5, 3300, "teaching proxy for blocking host/device boundary"),
    )
    baseline = PipelineConfig("baseline", "none", "baseline", base_stages, contract.quality_checksum, True)

    def adjust(stage_name: str, *, target_ms: float | None = None, working_set_mb: float | None = None, note: str | None = None) -> tuple[StageSpec, ...]:
        adjusted = []
        for stage in base_stages:
            if stage.name == stage_name:
                adjusted.append(replace(stage, target_ms=stage.target_ms if target_ms is None else target_ms, declared_working_set_mb=stage.declared_working_set_mb if working_set_mb is None else working_set_mb, note=stage.note if note is None else note))
            else:
                adjusted.append(stage)
        return tuple(adjusted)

    candidates = [
        PipelineConfig(
            "data_prefetch",
            "asynchronous input prefetch",
            "data_wait.target_ms",
            adjust("data_wait", target_ms=1.8, note="single change: upstream prefetch removes a controlled wait"),
            contract.quality_checksum,
            True,
        ),
        PipelineConfig(
            "kernel_fusion",
            "fuse backward pointwise operations",
            "backward.target_ms",
            adjust("backward", target_ms=4.1, note="single change: a fused compute path shortens backward stage"),
            contract.quality_checksum,
            True,
        ),
        PipelineConfig(
            "defer_scalar_sync",
            "defer host scalar synchronization",
            "host_sync.target_ms",
            adjust("host_sync", target_ms=0.45, note="single change: defer a controlled blocking synchronization"),
            contract.quality_checksum,
            True,
        ),
        PipelineConfig(
            "activation_checkpoint",
            "activation checkpointing",
            "backward.target_ms + working_set_proxy",
            # This is intentionally a single named technique with a memory-time
            # trade-off: it reduces the declared capacity proxy but recomputes.
            tuple(
                replace(stage, target_ms=7.6, declared_working_set_mb=4200, note="single change: recomputation trades compute time for working-set reduction")
                if stage.name == "backward"
                else replace(stage, declared_working_set_mb=min(stage.declared_working_set_mb, 4200))
                for stage in base_stages
            ),
            contract.quality_checksum,
            True,
        ),
    ]
    return baseline, candidates


def trace_payload(events: list[dict[str, Any]], record: RunRecord) -> dict[str, Any]:
    """Build a Chrome-trace-shaped teaching artifact with explicit provenance."""
    return {
        "displayTimeUnit": "ms",
        "traceEvents": events,
        "metadata": {
            "provenance": "controlled CPU teaching lab; stage timings are not GPU profiler measurements",
            "config": record.config,
            "contract": record.contract,
        },
    }


def plot_stage_breakdown(baseline: dict[str, Any], best: dict[str, Any], output: Path) -> None:
    """Plot measured mean stage time for baseline and the best accepted candidate."""
    names = list(baseline["stage_mean_ms"].keys())
    x = np.arange(len(names))
    width = 0.36
    baseline_values = [baseline["stage_mean_ms"][name] for name in names]
    best_values = [best["stage_mean_ms"][name] for name in names]
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12.8, 5.2), dpi=170)
    ax.bar(x - width / 2, baseline_values, width, label="baseline", color="#475569")
    ax.bar(x + width / 2, best_values, width, label=best["config"], color="#34d399")
    ax.set_xticks(x, [name.replace("_", "\n") for name in names])
    ax.set_ylabel("Measured mean stage time (ms)")
    ax.set_title("Controlled teaching workload: stage timing evidence")
    ax.legend(frameon=True)
    ax.text(0.01, 0.98, "Actual CPU wall-clock samples; configured teaching stages, not GPU kernels.", transform=ax.transAxes, va="top", fontsize=8, color="#475569")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_candidate_tradeoffs(experiments: list[dict[str, Any]], output: Path) -> None:
    """Plot actual candidate results: tail latency against declared capacity proxy."""
    color_for_decision = {"accept": "#10b981", "tune": "#f59e0b", "reject": "#ef4444"}
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.6, 6.4), dpi=170)
    label_offsets = {
        "data_prefetch": (7, 20),
        "kernel_fusion": (7, -2),
        "defer_scalar_sync": (7, -24),
        "activation_checkpoint": (7, 8),
    }
    for item in experiments:
        summary = item["tuned_summary"]
        decision = item["decision"]["decision"]
        point = (summary["p95_step_ms"], summary["declared_working_set_mb"])
        ax.scatter(point[0], point[1], s=170, color=color_for_decision[decision], edgecolor="#0f172a", linewidth=0.7, label=decision if decision not in ax.get_legend_handles_labels()[1] else None)
        ax.annotate(summary["config"], point, xytext=label_offsets[summary["config"]], textcoords="offset points", fontsize=8, arrowprops={"arrowstyle": "-", "color": "#64748b", "lw": 0.7})
    base = experiments[0]["baseline_summary"]
    base_point = (base["p95_step_ms"], base["declared_working_set_mb"])
    ax.scatter(base_point[0], base_point[1], s=195, color="#475569", marker="s", edgecolor="#0f172a", linewidth=0.7, label="baseline")
    ax.annotate("baseline", base_point, xytext=(7, 10), textcoords="offset points", fontsize=8, arrowprops={"arrowstyle": "-", "color": "#64748b", "lw": 0.7})
    ax.set_xlabel("Measured P95 step latency (ms)")
    ax.set_ylabel("Declared working-set proxy (MB)")
    ax.set_title("One-variable candidates: latency–capacity trade-off")
    ax.legend(title="Decision")
    ax.text(0.01, 0.01, "Working set is a declared teaching proxy, not a sampled allocator peak.", transform=ax.transAxes, fontsize=8, color="#475569")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_latency_distribution(baseline: RunRecord, best: RunRecord, output: Path) -> None:
    """Plot empirical latency distributions from the measured sample arrays."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=170)
    for record, color in ((baseline, "#475569"), (best, "#34d399")):
        values = np.sort(np.asarray(record.step_latency_ms))
        ecdf = np.arange(1, len(values) + 1) / len(values)
        ax.step(values, ecdf, where="post", label=record.config, color=color, linewidth=2.2)
    ax.set_xlabel("Measured step latency (ms)")
    ax.set_ylabel("Empirical cumulative probability")
    ax.set_title("Latency distribution: baseline versus selected candidate")
    ax.legend(frameon=True)
    ax.text(0.01, 0.02, "P50/P95 are reported from the same measured sample arrays.", transform=ax.transAxes, fontsize=8, color="#475569")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def markdown_report(experiments: list[dict[str, Any]]) -> str:
    """Produce a compact report table from structured measured records."""
    lines = [
        "| Candidate | Primary bottleneck | Mean step gain | P95 gain | Working-set delta | Throughput gain | Decision |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in experiments:
        change = item["comparison"]
        decision = item["decision"]["decision"]
        bottleneck = item["baseline_bottleneck"]["primary_stage"]
        lines.append(
            "| {name} | {bottleneck} | {time:+.2f}% | {p95:+.2f}% | {memory:+.0f} MB | {throughput:+.2f}% | {decision} |".format(
                name=item["tuned_summary"]["config"],
                bottleneck=bottleneck,
                time=change["mean_step_gain_pct"],
                p95=change["p95_gain_pct"],
                memory=change["working_set_delta_mb"],
                throughput=change["throughput_gain_pct"],
                decision=decision,
            )
        )
    lines.append("")
    lines.append("All measurements above come from the controlled CPU teaching pipeline. They show how to preserve a decision protocol; they are not GPU performance claims.")
    return "\n".join(lines)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    contract = make_contract()
    baseline_config, candidates = make_configs(contract)
    baseline_record, baseline_events = benchmark_pipeline(contract, baseline_config)
    baseline_summary = baseline_record.summary()
    baseline_bottleneck = classify_bottleneck(baseline_summary, baseline_config.stages)

    experiments: list[dict[str, Any]] = []
    records = {"baseline": baseline_record}
    traces = {"baseline": baseline_events}
    for candidate in candidates:
        validate_contract_pair(contract, baseline_config, candidate)
        tuned_record, tuned_events = benchmark_pipeline(contract, candidate)
        tuned_summary = tuned_record.summary()
        comparison = compare_runs(baseline_summary, tuned_summary)
        decision = recommend_decision(
            baseline_summary,
            tuned_summary,
            comparison,
            contract.max_p95_regression_pct,
        )
        experiments.append(
            {
                "baseline_summary": baseline_summary,
                "baseline_bottleneck": baseline_bottleneck,
                "tuned_summary": tuned_summary,
                "tuned_bottleneck": classify_bottleneck(tuned_summary, candidate.stages),
                "comparison": comparison,
                "decision": decision,
                "contract_match": baseline_record.contract == tuned_record.contract,
            }
        )
        records[candidate.name] = tuned_record
        traces[candidate.name] = tuned_events

    # Select the fastest accepted candidate; fall back to the fastest tuned run
    # only for visualization when no candidate passes the rollout gate.
    accepted = [item for item in experiments if item["decision"]["decision"] == "accept"]
    selection_pool = accepted if accepted else experiments
    selected_item = min(selection_pool, key=lambda item: item["tuned_summary"]["mean_step_ms"])
    selected_name = selected_item["tuned_summary"]["config"]
    selected_record = records[selected_name]

    # Invariants teach what must be true before a comparison is interpretable.
    assert all(item["contract_match"] for item in experiments)
    assert all(item["tuned_summary"]["quality_passed"] for item in experiments)
    assert all(math.isfinite(item["tuned_summary"]["mean_step_ms"]) for item in experiments)
    assert all(abs(sum(item["tuned_summary"]["stage_share_pct"].values()) - 100.0) < 1e-6 for item in experiments)
    assert baseline_bottleneck["primary_stage"] == "backward"
    assert selected_item["comparison"]["mean_step_gain_pct"] > 0.0

    plot_stage_breakdown(baseline_summary, selected_item["tuned_summary"], ASSET_DIR / "04_stage_breakdown.png")
    plot_candidate_tradeoffs(experiments, ASSET_DIR / "05_candidate_tradeoffs.png")
    plot_latency_distribution(baseline_record, selected_record, ASSET_DIR / "06_latency_distribution.png")
    (RESULT_DIR / "controlled_baseline_trace.json").write_text(json.dumps(trace_payload(baseline_events, baseline_record), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (RESULT_DIR / "profiling_results.json").write_text(
        json.dumps(
            {
                "provenance": "controlled CPU teaching laboratory; no GPU profiler or allocator data were collected",
                "contract": contract.manifest(),
                "baseline": {"summary": baseline_summary, "bottleneck": baseline_bottleneck},
                "experiments": experiments,
                "selected_candidate": selected_name,
                "report_markdown": markdown_report(experiments),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("All invariants passed. Controlled CPU teaching workload; not a GPU benchmark.")
    print(f"Baseline bottleneck hypothesis: {baseline_bottleneck['primary_stage']} ({baseline_bottleneck['share_pct']:.1f}% of summed stage time)")
    print(f"Selected candidate: {selected_name} -> {selected_item['decision']['decision']}")
    print("")
    print(markdown_report(experiments))


if __name__ == "__main__":
    main()
