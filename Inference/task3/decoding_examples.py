"""CPU-first, original examples for an LLM decoding learning note.

Run:
    python3 decoding_examples.py

Outputs:
    assets/05_sampling_probability_comparison.png
    assets/06_scheduler_timeline.png

The file intentionally uses toy distributions and a scheduler simulation.
It demonstrates algorithmic control flow, not model-quality benchmarking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal, Sequence

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

ASSET_DIR = "/home/ubuntu/llm-decoding-notes/assets"


# ---------------------------------------------------------------------------
# 1. Sampling: temperature, top-k, top-p, and categorical sampling
# ---------------------------------------------------------------------------

def temperature_scale(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Scale a logits tensor by positive temperature without mutating input.

    Args:
        logits: Shape ``[..., vocab_size]``.
        temperature: A strictly positive scalar. Lower values sharpen the
            distribution and higher values flatten it.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return logits / temperature


def filter_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep exactly the k largest logits in each row; mask the rest with -inf."""
    vocab_size = logits.size(-1)
    if k <= 0 or k >= vocab_size:
        return logits.clone()

    kept_indices = logits.topk(k, dim=-1).indices
    remove_mask = torch.ones_like(logits, dtype=torch.bool)
    remove_mask.scatter_(-1, kept_indices, False)
    return logits.masked_fill(remove_mask, -torch.inf)


def filter_top_p(
    logits: torch.Tensor,
    p: float,
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    """Nucleus filtering with a dynamic candidate set.

    Tokens are sorted by probability. The first token that makes cumulative
    probability exceed p is kept; subsequent tokens are removed. The
    ``min_tokens_to_keep`` guard prevents an empty support.
    """
    if not 0 < p <= 1:
        raise ValueError("p must lie in (0, 1]")
    if min_tokens_to_keep < 1:
        raise ValueError("min_tokens_to_keep must be at least 1")
    if p == 1:
        return logits.clone()

    sorted_logits, sorted_indices = logits.sort(dim=-1, descending=True)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = sorted_probs.cumsum(dim=-1)

    sorted_remove = cumulative_probs > p
    # Shift right: retain the boundary token whose addition crosses p.
    sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
    sorted_remove[..., 0] = False
    sorted_remove[..., :min_tokens_to_keep] = False

    remove_mask = torch.zeros_like(sorted_remove).scatter(
        dim=-1, index=sorted_indices, src=sorted_remove
    )
    return logits.masked_fill(remove_mask, -torch.inf)


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply sampling controls and sample one index from each batch row.

    Returns:
        ``(token_ids, final_probs)`` with shapes ``[batch, 1]`` and
        ``[batch, vocab_size]`` respectively.
    """
    work = temperature_scale(logits, temperature)
    if top_k is not None:
        work = filter_top_k(work, top_k)
    if top_p is not None:
        work = filter_top_p(work, top_p)

    final_probs = F.softmax(work, dim=-1)
    if not torch.isfinite(final_probs).all() or not torch.allclose(
        final_probs.sum(dim=-1), torch.ones_like(final_probs.sum(dim=-1))
    ):
        raise RuntimeError("filtering produced an invalid categorical distribution")
    token_ids = torch.multinomial(final_probs, num_samples=1, generator=generator)
    return token_ids, final_probs


def sampling_demo_and_plot() -> None:
    """Make a reproducible probability comparison, using one fixed logits row."""
    logits = torch.tensor([[2.4, 1.8, 1.2, 0.9, 0.5, 0.1, -0.2, -0.7, -1.1, -1.6]])
    token_names = [f"t{i}" for i in range(logits.size(-1))]

    variants: list[tuple[str, torch.Tensor]] = [
        ("base", logits),
        ("T = 0.5", temperature_scale(logits, 0.5)),
        ("T = 1.5", temperature_scale(logits, 1.5)),
        ("top-k = 3", filter_top_k(logits, 3)),
        ("top-p = 0.80", filter_top_p(logits, 0.80)),
    ]

    fig, axes = plt.subplots(1, len(variants), figsize=(18, 4.2), sharey=True)
    colors = ["#77bdfb", "#8b5cf6", "#22d3ee", "#f59e0b", "#34d399"]
    for axis, (label, transformed), color in zip(axes, variants, colors):
        probs = F.softmax(transformed, dim=-1).squeeze(0)
        bars = axis.bar(token_names, probs.numpy(), color=color, edgecolor="#172554")
        for bar, probability in zip(bars, probs):
            if probability > 0:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    float(probability) + 0.015,
                    f"{float(probability):.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        axis.set_title(label, weight="bold")
        axis.set_xlabel("candidate token")
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("normalized probability")
    axes[0].set_ylim(0, 1.05)
    fig.suptitle("How decoding controls reshape one next-token distribution", fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(f"{ASSET_DIR}/05_sampling_probability_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Exact speculative decoding: acceptance-rejection plus residual sampling
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpeculativeResult:
    """One exact speculative round, represented with human-readable events."""

    output_tokens: list[int]
    accepted_prefix_length: int
    rejection_index: int | None
    repair_token: int
    events: list[str]


def normalise_distribution(distribution: torch.Tensor) -> torch.Tensor:
    """Return a 1-D categorical distribution, rejecting invalid inputs early."""
    distribution = distribution.float().clamp_min(0)
    total = distribution.sum()
    if total <= 0:
        raise ValueError("a categorical distribution needs positive mass")
    return distribution / total


def residual_distribution(target: torch.Tensor, draft: torch.Tensor) -> torch.Tensor:
    """Build the correction distribution max(target - draft, 0), then normalize."""
    target = normalise_distribution(target)
    draft = normalise_distribution(draft)
    residual = (target - draft).clamp_min(0)
    # In exact arithmetic a rejection implies positive residual mass. The
    # fallback handles finite-precision edge cases without returning NaNs.
    return normalise_distribution(residual) if residual.sum() > 1e-12 else target


def categorical_draw(probabilities: torch.Tensor, generator: torch.Generator) -> int:
    """Sample one token id from a 1-D probability vector."""
    return int(torch.multinomial(normalise_distribution(probabilities), 1, generator=generator).item())


def speculative_round(
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    proposed_tokens: Sequence[int],
    *,
    generator: torch.Generator,
) -> SpeculativeResult:
    """Execute one *exact* speculative decoding round on precomputed distributions.

    ``draft_probs`` has shape ``[K, V]``. ``target_probs`` has shape ``[K+1, V]``:
    each of the first K rows evaluates the matching draft-token position, and
    the last row supplies the extra target token when all K proposals pass.

    This function models the probability-correction rule; a production system
    also has to evaluate the target model on the proposed suffix efficiently.
    """
    k = len(proposed_tokens)
    if draft_probs.shape[0] != k or target_probs.shape[0] != k + 1:
        raise ValueError("expected draft [K,V] and target [K+1,V] distributions")

    output: list[int] = []
    events: list[str] = []
    for index, token_id in enumerate(proposed_tokens):
        q = normalise_distribution(draft_probs[index])
        p = normalise_distribution(target_probs[index])
        acceptance_probability = min(1.0, float(p[token_id] / q[token_id]))
        coin = float(torch.rand((), generator=generator))
        if coin <= acceptance_probability:
            output.append(int(token_id))
            events.append(
                f"position {index}: accepted token {token_id} (u={coin:.3f}, alpha={acceptance_probability:.3f})"
            )
            continue

        replacement = categorical_draw(residual_distribution(p, q), generator)
        output.append(replacement)
        events.append(
            f"position {index}: rejected token {token_id}; residual sample -> {replacement}"
        )
        return SpeculativeResult(
            output_tokens=output,
            accepted_prefix_length=index,
            rejection_index=index,
            repair_token=replacement,
            events=events,
        )

    # All K draft tokens were accepted. The target model contributes one extra
    # token, which preserves progress comparable to a K+1-token target pass.
    extra = categorical_draw(target_probs[k], generator)
    output.append(extra)
    events.append(f"all {k} proposed tokens accepted; target extra sample -> {extra}")
    return SpeculativeResult(
        output_tokens=output,
        accepted_prefix_length=k,
        rejection_index=None,
        repair_token=extra,
        events=events,
    )


def speculative_demo() -> SpeculativeResult:
    """Create a deterministic toy round that contains a rejection and repair."""
    draft = torch.tensor(
        [
            [0.05, 0.55, 0.20, 0.20],
            [0.15, 0.20, 0.50, 0.15],
            [0.10, 0.10, 0.70, 0.10],
        ]
    )
    target = torch.tensor(
        [
            [0.10, 0.60, 0.15, 0.15],  # candidate 1 is very likely accepted
            [0.35, 0.15, 0.20, 0.30],  # candidate 2 is often rejected
            [0.15, 0.15, 0.55, 0.15],
        ]
    )
    return speculative_round(
        draft_probs=draft[:2],
        target_probs=target,
        proposed_tokens=[1, 2],
        generator=torch.Generator().manual_seed(42),
    )


# ---------------------------------------------------------------------------
# 3. Multi-token lookahead: a deliberately approximate prefix verifier
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LookaheadRound:
    proposed: list[int]
    accepted_prefix: list[int]
    first_rejected_at: int | None
    rollback_suffix: list[int]


class LookaheadVerifier:
    """Teaching simulator for proposal -> prefix verification -> rollback.

    This ratio rule is intentionally *not* an exact replacement for the
    residual correction used in speculative decoding. It is useful for seeing
    why a proposal chain has a prefix dependency and where rollback begins.
    """

    def __init__(self, max_lookahead: int = 4, min_probability_ratio: float = 0.70):
        if max_lookahead < 1:
            raise ValueError("max_lookahead must be positive")
        if not 0 < min_probability_ratio <= 1:
            raise ValueError("min_probability_ratio must lie in (0, 1]")
        self.max_lookahead = max_lookahead
        self.min_probability_ratio = min_probability_ratio

    def propose(self, candidates: Sequence[int]) -> list[int]:
        return list(candidates[: self.max_lookahead])

    def verify_prefix(
        self,
        draft_probs: torch.Tensor,
        target_probs: torch.Tensor,
        candidates: Sequence[int],
    ) -> LookaheadRound:
        proposed = self.propose(candidates)
        accepted: list[int] = []
        for position, token_id in enumerate(proposed):
            q = float(normalise_distribution(draft_probs[position])[token_id])
            p = float(normalise_distribution(target_probs[position])[token_id])
            if q == 0 or p / q >= self.min_probability_ratio:
                accepted.append(int(token_id))
            else:
                return LookaheadRound(
                    proposed=proposed,
                    accepted_prefix=accepted,
                    first_rejected_at=position,
                    rollback_suffix=proposed[position:],
                )
        return LookaheadRound(proposed, accepted, None, [])


def lookahead_demo() -> LookaheadRound:
    """Verify three proposed tokens and reject the third by a configurable ratio."""
    verifier = LookaheadVerifier(max_lookahead=3, min_probability_ratio=0.75)
    draft = torch.tensor(
        [[0.1, 0.5, 0.2, 0.2], [0.1, 0.1, 0.6, 0.2], [0.1, 0.2, 0.2, 0.5]]
    )
    target = torch.tensor(
        [[0.1, 0.45, 0.25, 0.2], [0.1, 0.1, 0.55, 0.25], [0.4, 0.2, 0.2, 0.2]]
    )
    return verifier.verify_prefix(draft, target, candidates=[1, 2, 3, 0])


# ---------------------------------------------------------------------------
# 4. Decode scheduling: continuous batching under two simple budgets
# ---------------------------------------------------------------------------

Phase = Literal["prefill", "decode", "finished"]


@dataclass
class Request:
    request_id: str
    prompt_tokens: int
    max_new_tokens: int
    priority: int = 0
    cache_hit: bool = False
    arrival_tick: int = 0
    phase: Phase = "prefill"
    generated_tokens: int = 0
    first_scheduled_tick: int | None = None

    @property
    def done(self) -> bool:
        return self.generated_tokens >= self.max_new_tokens

    @property
    def waiting_ticks(self) -> int:
        # Updated by the scheduler immediately before sorting.
        return getattr(self, "_waiting_ticks", 0)


@dataclass(frozen=True)
class ScheduleEvent:
    tick: int
    request_id: str
    phase: str
    action: str
    generated_tokens: int


class ContinuousBatchScheduler:
    """A small, inspectable model of a continuous-batching scheduler.

    Policy: reserve up to ``max_decode_batch`` slots for active decode requests,
    then use the remaining slots for prefill. Cache hits receive a ranking boost;
    priority and waiting time avoid a pure FIFO queue. This is educational logic,
    not a substitute for a production KV-cache allocator.
    """

    def __init__(self, max_batch_size: int = 3, prefill_token_budget: int = 12):
        self.max_batch_size = max_batch_size
        self.prefill_token_budget = prefill_token_budget
        self.requests: list[Request] = []
        self.timeline: list[ScheduleEvent] = []
        self.tick = 0

    def submit(
        self,
        request_id: str,
        prompt_tokens: int,
        max_new_tokens: int,
        *,
        priority: int = 0,
        cache_hit: bool = False,
    ) -> None:
        if prompt_tokens < 1 or max_new_tokens < 1:
            raise ValueError("prompt_tokens and max_new_tokens must both be positive")
        self.requests.append(
            Request(
                request_id=request_id,
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_new_tokens,
                priority=priority,
                cache_hit=cache_hit,
                arrival_tick=self.tick,
            )
        )

    def _rank(self, request: Request) -> tuple[int, int, int, int, str]:
        """Lower tuple sorts first; decode gets an explicit latency preference."""
        phase_rank = 0 if request.phase == "decode" else 1
        cache_rank = 0 if request.cache_hit else 1
        aging_bonus = request.waiting_ticks // 3
        return (
            phase_rank,
            cache_rank,
            -(request.priority + aging_bonus),
            request.prompt_tokens,
            request.request_id,
        )

    def pick_batch(self) -> list[Request]:
        active = [request for request in self.requests if request.phase != "finished"]
        for request in active:
            request._waiting_ticks = self.tick - request.arrival_tick  # type: ignore[attr-defined]

        decode_candidates = sorted(
            (request for request in active if request.phase == "decode"), key=self._rank
        )
        selected = decode_candidates[: self.max_batch_size]

        remaining_slots = self.max_batch_size - len(selected)
        used_prefill_tokens = 0
        for request in sorted(
            (request for request in active if request.phase == "prefill"), key=self._rank
        ):
            if remaining_slots == 0:
                break
            effective_prefill_cost = 1 if request.cache_hit else request.prompt_tokens
            if used_prefill_tokens + effective_prefill_cost > self.prefill_token_budget:
                continue
            selected.append(request)
            used_prefill_tokens += effective_prefill_cost
            remaining_slots -= 1
        return selected

    def execute_tick(self) -> list[ScheduleEvent]:
        batch = self.pick_batch()
        if not batch:
            return []

        events: list[ScheduleEvent] = []
        for request in batch:
            if request.first_scheduled_tick is None:
                request.first_scheduled_tick = self.tick

            if request.phase == "prefill":
                request.phase = "decode"
                action = "prefill_complete"
            else:
                request.generated_tokens += 1
                if request.done:
                    request.phase = "finished"
                    action = "finish"
                else:
                    action = "decode_one_token"

            events.append(
                ScheduleEvent(
                    tick=self.tick,
                    request_id=request.request_id,
                    phase=request.phase,
                    action=action,
                    generated_tokens=request.generated_tokens,
                )
            )
        self.timeline.extend(events)
        self.tick += 1
        return events

    def run(self, max_ticks: int = 50) -> list[ScheduleEvent]:
        while self.tick < max_ticks and any(not request.done for request in self.requests):
            self.execute_tick()
        return self.timeline


def scheduler_demo_and_plot() -> list[ScheduleEvent]:
    """Run a fixed workload and draw one event per request/tick in a timeline."""
    scheduler = ContinuousBatchScheduler(max_batch_size=3, prefill_token_budget=10)
    scheduler.submit("A", prompt_tokens=7, max_new_tokens=3, priority=2, cache_hit=False)
    scheduler.submit("B", prompt_tokens=3, max_new_tokens=4, priority=1, cache_hit=True)
    scheduler.submit("C", prompt_tokens=5, max_new_tokens=2, priority=3, cache_hit=False)
    events = scheduler.run()

    colors = {
        "prefill_complete": "#3b82f6",
        "decode_one_token": "#8b5cf6",
        "finish": "#22c55e",
    }
    request_ids = ["A", "B", "C"]
    y_positions = {request_id: index for index, request_id in enumerate(request_ids)}

    fig, axis = plt.subplots(figsize=(12, 3.8))
    for event in events:
        axis.barh(
            y_positions[event.request_id],
            width=0.82,
            left=event.tick + 0.09,
            color=colors[event.action],
            edgecolor="white",
            height=0.62,
        )
        short_label = {"prefill_complete": "P", "decode_one_token": "D", "finish": "F"}[event.action]
        axis.text(
            event.tick + 0.50,
            y_positions[event.request_id],
            short_label,
            ha="center",
            va="center",
            color="white",
            weight="bold",
            fontsize=9,
        )
    axis.set_yticks(list(y_positions.values()), [f"request {item}" for item in request_ids])
    axis.set_xlabel("scheduler tick")
    axis.set_title("Continuous batching simulation: P=prefill, D=decode one token, F=finish", weight="bold")
    axis.set_xlim(0, max(event.tick for event in events) + 1)
    axis.set_xticks(range(max(event.tick for event in events) + 1))
    axis.grid(axis="x", alpha=0.20)
    axis.spines[["top", "right"]].set_visible(False)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=label.replace("_", " "))
        for label, color in colors.items()
    ]
    axis.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(f"{ASSET_DIR}/06_scheduler_timeline.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return events


def main() -> None:
    sampling_demo_and_plot()
    sampling_generator = torch.Generator().manual_seed(42)
    chosen, probabilities = sample_next_token(
        torch.tensor([[2.4, 1.8, 1.2, 0.9, 0.5, 0.1, -0.2, -0.7, -1.1, -1.6]]),
        temperature=0.7,
        top_k=5,
        top_p=0.9,
        generator=sampling_generator,
    )
    print("sampling demo token:", int(chosen.item()))
    print("sampling support:", int((probabilities > 0).sum().item()))

    speculative = speculative_demo()
    print("speculative result:", speculative)

    lookahead = lookahead_demo()
    print("lookahead result:", lookahead)

    events = scheduler_demo_and_plot()
    print("scheduler events:", len(events))
    assert all(request_id in {"A", "B", "C"} for request_id in (event.request_id for event in events))
    print("generated plots in:", ASSET_DIR)


if __name__ == "__main__":
    main()
