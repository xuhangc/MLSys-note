"""Task6 Inference Benchmark Lab — original deterministic teaching implementation.

The lab models the accounting and decision logic of three benchmark designs:
- general inference comparison (TTFT, TPOT, throughput, memory),
- speculative decoding (acceptance, draft cost, verify cost), and
- prefix caching (reuse, saved prefill work, maintenance cost).

It deliberately does NOT execute an LLM server or claim real vLLM/SGLang results.
All profiles are fixed analytical scenarios so the calculations are reproducible.
Run: python3 inference_benchmark_lab.py
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "assets"
RESULT_DIR = ROOT / "results"


@dataclass(frozen=True)
class Workload:
    """All conditions that must remain fixed in one fair benchmark comparison."""

    name: str
    model: str
    backend: str
    batch_size: int
    prompt_tokens: int
    generated_tokens: int
    dtype: str
    cache_policy: str
    warmup_runs: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.generated_tokens


@dataclass(frozen=True)
class RunMetrics:
    """A measured or scenario-provided result under one explicit workload contract."""

    label: str
    prefill_ms: float
    decode_ms: float
    peak_memory_mb: float
    quality_passed: bool

    def summary(self, workload: Workload) -> dict[str, float | bool | str]:
        """Calculate user-facing latency and throughput metrics from phase timing."""
        if self.prefill_ms < 0 or self.decode_ms < 0:
            raise ValueError("Timing must be non-negative.")
        total_ms = self.prefill_ms + self.decode_ms
        output_tokens = workload.batch_size * workload.generated_tokens
        tpot_ms = self.decode_ms / workload.generated_tokens if workload.generated_tokens else 0.0
        throughput = output_tokens * 1000.0 / total_ms if total_ms else 0.0
        return {
            "label": self.label,
            "ttft_ms": round(self.prefill_ms, 3),
            "tpot_ms": round(tpot_ms, 3),
            "total_ms": round(total_ms, 3),
            "throughput_tokens_s": round(throughput, 3),
            "prefill_share": round(self.prefill_ms / total_ms, 4) if total_ms else 0.0,
            "decode_share": round(self.decode_ms / total_ms, 4) if total_ms else 0.0,
            "peak_memory_mb": round(self.peak_memory_mb, 3),
            "quality_passed": self.quality_passed,
        }


def diagnose_bottleneck(summary: dict[str, float | bool | str], memory_budget_mb: float) -> dict[str, str]:
    """Classify the dominant constraint with memory as the hard constraint."""
    if memory_budget_mb <= 0:
        raise ValueError("memory_budget_mb must be positive.")
    memory_pressure = float(summary["peak_memory_mb"]) / memory_budget_mb
    if memory_pressure >= 0.90:
        return {
            "bottleneck": "memory-bound",
            "reason": "峰值显存已达到预算的 90% 以上；先处理 KV Cache、并发或容量问题。",
        }
    if float(summary["prefill_share"]) >= 0.60:
        return {
            "bottleneck": "prefill-bound",
            "reason": "Prefill 占总时延的 60% 以上；优先检查 prompt 复用、attention 和预填充批处理。",
        }
    if float(summary["decode_share"]) >= 0.60:
        return {
            "bottleneck": "decode-bound",
            "reason": "Decode 占总时延的 60% 以上；优先检查每 token 调度、KV 读取和投机解码。",
        }
    return {
        "bottleneck": "balanced",
        "reason": "三类压力均未显著主导；应以更细粒度 profiling 而非直接更换技术栈推进。",
    }


def compare_runs(baseline: dict[str, float | bool | str], candidate: dict[str, float | bool | str]) -> dict[str, float | bool]:
    """Use one sign convention: positive deltas mean the candidate improves that metric."""
    baseline_throughput = float(baseline["throughput_tokens_s"])
    candidate_throughput = float(candidate["throughput_tokens_s"])
    return {
        "ttft_improvement_ms": round(float(baseline["ttft_ms"]) - float(candidate["ttft_ms"]), 3),
        "tpot_improvement_ms": round(float(baseline["tpot_ms"]) - float(candidate["tpot_ms"]), 3),
        "total_latency_improvement_ms": round(float(baseline["total_ms"]) - float(candidate["total_ms"]), 3),
        "memory_saved_mb": round(float(baseline["peak_memory_mb"]) - float(candidate["peak_memory_mb"]), 3),
        "throughput_gain": round(candidate_throughput / baseline_throughput - 1.0, 4) if baseline_throughput else 0.0,
        "quality_passed": bool(candidate["quality_passed"]),
    }


def benchmark_decision(
    comparison: dict[str, float | bool],
    diagnosis: dict[str, str],
    min_throughput_gain: float = 0.10,
    max_ttft_regression_ms: float = 20.0,
) -> dict[str, str]:
    """Return accept/tune/reject after enforcing the quality guardrail first."""
    if not bool(comparison["quality_passed"]):
        return {"decision": "reject", "reason": "质量约束未通过；性能收益不能抵消质量回归。"}

    ttft_regression = -float(comparison["ttft_improvement_ms"])
    if float(comparison["throughput_gain"]) >= min_throughput_gain and ttft_regression <= max_ttft_regression_ms:
        return {"decision": "accept", "reason": "吞吐提升达标，且 TTFT 回归仍在预先声明的交互预算内。"}
    if float(comparison["throughput_gain"]) > 0 or float(comparison["memory_saved_mb"]) > 0:
        target = diagnosis["bottleneck"]
        return {"decision": "tune", "reason": f"候选有局部收益，但仍应围绕 {target} 做下一轮调优。"}
    return {"decision": "reject", "reason": "候选没有可证明的容量或性能收益。"}


def expected_tokens_per_target_call(acceptance_rate: float, draft_tokens: int) -> float:
    """Compute an independent-acceptance teaching model for speculative progress.

    A target verification call emits at least one token. If the first proposed token is
    accepted, progress grows by another token; this continues through `draft_tokens`.
    Thus the expectation is 1 + a + a² + ... + a^draft_tokens. It is an analytical
    model, not an estimate of a particular engine's sampling path.
    """
    if not 0.0 <= acceptance_rate <= 1.0:
        raise ValueError("acceptance_rate must lie in [0, 1].")
    if draft_tokens < 1:
        raise ValueError("draft_tokens must be at least one.")
    return sum(acceptance_rate**power for power in range(draft_tokens + 1))


def speculative_cost_model(
    output_tokens: int,
    acceptance_rate: float,
    draft_tokens: int,
    target_step_ms: float,
    draft_step_ms: float,
    verify_call_ms: float,
) -> dict[str, float]:
    """Compare serial target decoding with a transparent speculative cost model.

    `verify_call_ms` represents one batched target verification cost. A production
    system must measure it, because it changes with model, batch, sequence length,
    hardware and kernel selection.
    """
    if min(output_tokens, target_step_ms, draft_step_ms, verify_call_ms) <= 0:
        raise ValueError("All costs and output_tokens must be positive.")
    progress = expected_tokens_per_target_call(acceptance_rate, draft_tokens)
    target_calls = math.ceil(output_tokens / progress)
    baseline_ms = output_tokens * target_step_ms
    speculative_ms = target_calls * (draft_tokens * draft_step_ms + verify_call_ms)
    return {
        "acceptance_rate": round(acceptance_rate, 4),
        "draft_tokens": float(draft_tokens),
        "expected_tokens_per_target_call": round(progress, 4),
        "estimated_target_calls": float(target_calls),
        "baseline_ms": round(baseline_ms, 4),
        "speculative_ms": round(speculative_ms, 4),
        "estimated_speedup": round(baseline_ms / speculative_ms, 4),
        "draft_cost_share": round(draft_tokens * draft_step_ms / (draft_tokens * draft_step_ms + verify_call_ms), 4),
        "verify_cost_share": round(verify_call_ms / (draft_tokens * draft_step_ms + verify_call_ms), 4),
    }


def speculative_decision(
    result: dict[str, float],
    min_acceptance_rate: float = 0.60,
    min_speedup: float = 1.05,
) -> dict[str, str]:
    """Screen a speculative setup without hiding either acceptance or verification cost."""
    if float(result["acceptance_rate"]) < min_acceptance_rate:
        return {"decision": "reject", "reason": "接受率未达门槛；继续支付 draft 与 verify 开销缺乏依据。"}
    if float(result["estimated_speedup"]) >= min_speedup:
        return {"decision": "accept", "reason": "在该成本模型中，接受率与端到端速度模型均达到阈值。"}
    return {"decision": "tune", "reason": "接受率可用但预估速度收益不足；调整 draft 大小、proposal 长度或 verify 路径。"}


@dataclass(frozen=True)
class PrefixRequest:
    """A deterministic request representation for a prefix reuse exercise."""

    request_id: str
    prefix_key: str
    prefix_tokens: int
    suffix_tokens: int


def simulate_prefix_cache(requests: Iterable[PrefixRequest]) -> list[dict[str, float | str | bool]]:
    """Model first-use miss and later reuse hit with no eviction; record per-request savings."""
    seen: set[str] = set()
    events: list[dict[str, float | str | bool]] = []
    for request in requests:
        hit = request.prefix_key in seen
        saved_tokens = request.prefix_tokens if hit else 0
        events.append(
            {
                "request_id": request.request_id,
                "prefix_key": request.prefix_key,
                "cache_hit": hit,
                "prompt_tokens": request.prefix_tokens + request.suffix_tokens,
                "saved_prefill_tokens": saved_tokens,
            }
        )
        seen.add(request.prefix_key)
    return events


def prefix_cache_economics(
    events: Iterable[dict[str, float | str | bool]],
    prefill_ms_per_token: float,
    maintenance_ms_per_request: float,
) -> dict[str, float]:
    """Turn prefix hits into saved prefill time and subtract cache maintenance cost."""
    items = list(events)
    if not items:
        raise ValueError("At least one request is required.")
    if min(prefill_ms_per_token, maintenance_ms_per_request) < 0:
        raise ValueError("Cost inputs cannot be negative.")
    hit_count = sum(bool(item["cache_hit"]) for item in items)
    saved_tokens = sum(float(item["saved_prefill_tokens"]) for item in items)
    gross_saved_ms = saved_tokens * prefill_ms_per_token
    maintenance_ms = len(items) * maintenance_ms_per_request
    return {
        "request_count": float(len(items)),
        "cache_hits": float(hit_count),
        "hit_rate": round(hit_count / len(items), 4),
        "saved_prefill_tokens": float(saved_tokens),
        "gross_saved_ms": round(gross_saved_ms, 4),
        "maintenance_ms": round(maintenance_ms, 4),
        "net_saved_ms": round(gross_saved_ms - maintenance_ms, 4),
    }


def prefix_cache_decision(result: dict[str, float], min_hit_rate: float = 0.50) -> dict[str, str]:
    """Decide whether a cache helps the observed reuse distribution after maintenance."""
    if float(result["net_saved_ms"]) <= 0:
        return {"decision": "reject", "reason": "缓存维护成本已超过重复 prefill 的节省。"}
    if float(result["hit_rate"]) >= min_hit_rate:
        return {"decision": "accept", "reason": "命中率与净 prefill 节省均为正，值得进入真实服务评估。"}
    return {"decision": "tune", "reason": "已有净节省但命中率偏低；优先优化路由、chunk 粒度或缓存失效策略。"}


def make_figures(general: dict[str, object], speculative_rows: list[dict[str, float]], prefix: dict[str, float]) -> None:
    """Create exact visualizations from analytical models; figures are not backend measurements."""
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    baseline = general["baseline"]
    candidate = general["candidate"]
    labels = [str(baseline["label"]), str(candidate["label"])]
    prefill = [float(baseline["ttft_ms"]), float(candidate["ttft_ms"])]
    decode = [float(baseline["total_ms"]) - float(baseline["ttft_ms"]), float(candidate["total_ms"]) - float(candidate["ttft_ms"])]
    fig, axis = plt.subplots(figsize=(10.5, 5.8), dpi=180)
    axis.bar(labels, prefill, label="Prefill / TTFT", color="#1c7ed6")
    axis.bar(labels, decode, bottom=prefill, label="Decode", color="#7048e8")
    axis.set_ylabel("Analytical scenario latency (ms)")
    axis.set_title("Phase Accounting: Compare Prefill and Decode Before Selecting an Optimizer")
    axis.legend(loc="upper right")
    for index, (first, second) in enumerate(zip(prefill, decode)):
        axis.text(index, first / 2, f"Prefill\n{first:.0f}", ha="center", va="center", color="white", fontweight="bold")
        axis.text(index, first + second / 2, f"Decode\n{second:.0f}", ha="center", va="center", color="white", fontweight="bold")
    axis.text(0.5, -0.19, "Fixed teaching scenario — not a measurement from an LLM backend.", transform=axis.transAxes, ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "05_phase_accounting_scenario.png", bbox_inches="tight")
    plt.close(fig)

    acceptance = [row["acceptance_rate"] for row in speculative_rows]
    progress = [row["expected_tokens_per_target_call"] for row in speculative_rows]
    speedup = [row["estimated_speedup"] for row in speculative_rows]
    fig, left_axis = plt.subplots(figsize=(10.5, 5.8), dpi=180)
    left_axis.plot(acceptance, progress, marker="o", color="#1c7ed6", linewidth=2.5, label="Expected tokens / target call")
    left_axis.set_xlabel("Acceptance rate")
    left_axis.set_ylabel("Expected progress", color="#1c7ed6")
    left_axis.tick_params(axis="y", labelcolor="#1c7ed6")
    left_axis.set_title("Speculative Decoding: Acceptance Must Offset Draft and Verify Cost")
    right_axis = left_axis.twinx()
    right_axis.plot(acceptance, speedup, marker="s", color="#7048e8", linewidth=2.5, label="Estimated speedup")
    right_axis.axhline(1.0, color="#d9480f", linestyle="--", linewidth=1.5, label="Baseline")
    right_axis.set_ylabel("Analytical speedup vs serial target", color="#7048e8")
    right_axis.tick_params(axis="y", labelcolor="#7048e8")
    lines, names = left_axis.get_legend_handles_labels()
    lines2, names2 = right_axis.get_legend_handles_labels()
    left_axis.legend(lines + lines2, names + names2, loc="upper left")
    left_axis.text(0.5, -0.20, "Fixed cost model only; actual speed requires same-workload backend measurement.", transform=left_axis.transAxes, ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "06_speculative_acceptance_frontier.png", bbox_inches="tight")
    plt.close(fig)

    hit_rates = [round(step / 10, 1) for step in range(11)]
    request_count = float(prefix["request_count"])
    potential_saved_ms = float(prefix["saved_prefill_tokens"]) / max(float(prefix["hit_rate"]), 1e-9) * 0.35
    maintenance = float(prefix["maintenance_ms"])
    net_savings = [rate * potential_saved_ms - maintenance for rate in hit_rates]
    fig, axis = plt.subplots(figsize=(10.5, 5.8), dpi=180)
    axis.plot(hit_rates, net_savings, marker="o", color="#12b886", linewidth=2.5)
    axis.axhline(0.0, color="#d9480f", linestyle="--", linewidth=1.5)
    axis.axvline(float(prefix["hit_rate"]), color="#7048e8", linestyle=":", linewidth=2, label="Observed toy-workload hit rate")
    axis.set_xlabel("Prefix-cache hit rate")
    axis.set_ylabel("Analytical net prefill saving (ms)")
    axis.set_title("Prefix Cache: Hit Rate Matters Only After Maintenance Cost")
    axis.legend(loc="upper left")
    axis.text(0.5, -0.19, f"Scenario uses {int(request_count)} fixed requests and a fixed per-request maintenance cost; not backend data.", transform=axis.transAxes, ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / "07_prefix_cache_payoff_scenario.png", bbox_inches="tight")
    plt.close(fig)


def run_lab() -> dict[str, object]:
    """Run the complete deterministic lab, assertions, JSON record and visualizations."""
    workload = Workload(
        name="long-context interactive scenario",
        model="illustrative-7b",
        backend="same-backend-for-comparison",
        batch_size=2,
        prompt_tokens=2048,
        generated_tokens=128,
        dtype="bf16",
        cache_policy="baseline-none",
        warmup_runs=5,
    )
    baseline = RunMetrics("baseline", prefill_ms=260.0, decode_ms=180.0, peak_memory_mb=10_800.0, quality_passed=True).summary(workload)
    candidate = RunMetrics("candidate", prefill_ms=265.0, decode_ms=126.0, peak_memory_mb=9_900.0, quality_passed=True).summary(workload)
    comparison = compare_runs(baseline, candidate)
    diagnosis = diagnose_bottleneck(candidate, memory_budget_mb=16_000.0)
    general_decision = benchmark_decision(comparison, diagnosis)

    speculative_rows = [
        speculative_cost_model(128, acceptance, draft_tokens=4, target_step_ms=1.0, draft_step_ms=0.08, verify_call_ms=2.5)
        for acceptance in (0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0)
    ]
    speculative_selected = speculative_cost_model(128, acceptance_rate=0.8, draft_tokens=4, target_step_ms=1.0, draft_step_ms=0.08, verify_call_ms=2.5)
    speculative_verdict = speculative_decision(speculative_selected)

    requests = [
        PrefixRequest("r1", "policy-v1", 900, 120),
        PrefixRequest("r2", "policy-v1", 900, 140),
        PrefixRequest("r3", "product-doc-a", 700, 160),
        PrefixRequest("r4", "policy-v1", 900, 100),
        PrefixRequest("r5", "product-doc-a", 700, 130),
        PrefixRequest("r6", "one-off", 500, 110),
    ]
    events = simulate_prefix_cache(requests)
    prefix_result = prefix_cache_economics(events, prefill_ms_per_token=0.35, maintenance_ms_per_request=12.0)
    prefix_verdict = prefix_cache_decision(prefix_result)

    # Assertions encode mechanics, not a claim about production performance.
    assert workload.total_tokens == 2176
    assert baseline["ttft_ms"] == 260.0
    assert candidate["tpot_ms"] == round(126.0 / 128.0, 3)
    assert diagnosis["bottleneck"] == "prefill-bound"
    assert comparison["throughput_gain"] > 0
    assert expected_tokens_per_target_call(0.0, 4) == 1.0
    assert expected_tokens_per_target_call(1.0, 4) == 5.0
    assert speculative_selected["estimated_speedup"] > 1.0
    assert sum(bool(item["cache_hit"]) for item in events) == 3
    assert prefix_result["hit_rate"] == 0.5
    assert prefix_result["net_saved_ms"] > 0

    report: dict[str, object] = {
        "notice": "All values are deterministic teaching scenarios, not LLM or serving-backend measurements.",
        "workload_contract": asdict(workload),
        "general_benchmark": {
            "baseline": baseline,
            "candidate": candidate,
            "comparison": comparison,
            "candidate_diagnosis": diagnosis,
            "decision": general_decision,
        },
        "speculative_scenario": {
            "selected": speculative_selected,
            "decision": speculative_verdict,
            "acceptance_frontier": speculative_rows,
        },
        "prefix_cache_scenario": {
            "events": events,
            "summary": prefix_result,
            "decision": prefix_verdict,
        },
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "inference_benchmark_lab_results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    make_figures(report["general_benchmark"], speculative_rows, prefix_result)
    print("Inference benchmark lab checks passed.")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    run_lab()
