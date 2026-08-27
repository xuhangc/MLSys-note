"""KV Cache serving mechanisms: executable teaching lab.

This file is original educational code for Inference/task4.  It intentionally
models control-plane decisions rather than replacing vLLM or SGLang kernels.
Run `python3 kv_cache_lab.py` to execute assertions and write plots to assets/.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import ceil, log1p
from pathlib import Path
from typing import Iterable
import heapq

import matplotlib.pyplot as plt


ASSET_DIR = Path(__file__).resolve().parent / "assets"


# -----------------------------------------------------------------------------
# 1. KV cache capacity: a byte-level model
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelShape:
    """The dimensions relevant to conventional K/V cache storage."""

    layers: int
    kv_heads: int
    head_dim: int
    dtype_bytes: int = 2  # FP16/BF16


def kv_cache_bytes(shape: ModelShape, sequence_tokens: int, batch_size: int = 1) -> int:
    """Return bytes for independent K and V tensors of a decoder-only model.

    The factor 2 stores one Key and one Value element.  `kv_heads` rather than
    query-head count is intentional: MQA/GQA reduce this dimension directly.
    """
    if sequence_tokens < 0 or batch_size < 1:
        raise ValueError("sequence_tokens must be non-negative and batch_size positive")
    return (
        2
        * shape.layers
        * batch_size
        * shape.kv_heads
        * shape.head_dim
        * sequence_tokens
        * shape.dtype_bytes
    )


def gibibytes(byte_count: int) -> float:
    return byte_count / (1024**3)


def capacity_rows(shape: ModelShape, tokens: Iterable[int], batch_size: int) -> list[dict[str, float]]:
    """Create a small data table that makes linear scaling easy to inspect."""
    return [
        {
            "tokens": token_count,
            "kv_gib": gibibytes(kv_cache_bytes(shape, token_count, batch_size)),
        }
        for token_count in tokens
    ]


# -----------------------------------------------------------------------------
# 2. Paged KV cache: logical sequence positions map to physical blocks
# -----------------------------------------------------------------------------

@dataclass
class PagedRequest:
    request_id: str
    token_count: int
    block_table: list[int] = field(default_factory=list)


class PagedKVPool:
    """A small allocator that exposes the essential PagedAttention invariant.

    `block_table[i]` is the physical block that contains logical block `i` for
    one request.  The class does not implement the CUDA attention kernel; its
    purpose is to make allocation boundaries and release behavior inspectable.
    """

    def __init__(self, total_blocks: int, block_size: int):
        if total_blocks < 1 or block_size < 1:
            raise ValueError("total_blocks and block_size must be positive")
        self.total_blocks = total_blocks
        self.block_size = block_size
        self._free: deque[int] = deque(range(total_blocks))
        self._live: dict[str, PagedRequest] = {}

    @property
    def free_block_count(self) -> int:
        return len(self._free)

    @property
    def used_block_count(self) -> int:
        return self.total_blocks - self.free_block_count

    def _take_blocks(self, count: int) -> list[int]:
        if count > self.free_block_count:
            raise MemoryError(
                f"need {count} blocks but only {self.free_block_count} are free"
            )
        return [self._free.popleft() for _ in range(count)]

    def admit_prompt(self, request_id: str, prompt_tokens: int) -> PagedRequest:
        """Allocate only the blocks that the current prompt needs."""
        if request_id in self._live:
            raise ValueError(f"duplicate request id: {request_id}")
        if prompt_tokens < 1:
            raise ValueError("a prompt must contain at least one token")
        block_count = ceil(prompt_tokens / self.block_size)
        request = PagedRequest(request_id, prompt_tokens, self._take_blocks(block_count))
        self._live[request_id] = request
        return request

    def append_one_token(self, request_id: str) -> None:
        """Append a decode token and allocate exactly when it opens a new block."""
        request = self._live[request_id]
        if request.token_count % self.block_size == 0:
            request.block_table.extend(self._take_blocks(1))
        request.token_count += 1

    def physical_location(self, request_id: str, logical_token_index: int) -> tuple[int, int]:
        """Translate a logical token position into (physical block id, offset)."""
        request = self._live[request_id]
        if not 0 <= logical_token_index < request.token_count:
            raise IndexError("logical token index lies outside the request")
        logical_block, offset = divmod(logical_token_index, self.block_size)
        return request.block_table[logical_block], offset

    def release(self, request_id: str) -> None:
        """Return a finished request's physical blocks to the shared pool."""
        request = self._live.pop(request_id)
        self._free.extend(request.block_table)


def static_vs_paged_slots(lengths: list[int], max_tokens: int, block_size: int) -> tuple[int, int]:
    """Compare reserved token slots, not bytes, for a controlled toy workload."""
    if any(length < 1 or length > max_tokens for length in lengths):
        raise ValueError("each length must be in [1, max_tokens]")
    static_slots = len(lengths) * max_tokens
    paged_slots = sum(ceil(length / block_size) * block_size for length in lengths)
    return static_slots, paged_slots


# -----------------------------------------------------------------------------
# 3. Radix prefix cache: compressed edges make shared prefixes explicit
# -----------------------------------------------------------------------------

@dataclass
class RadixNode:
    edge_tokens: tuple[int, ...]
    parent: "RadixNode | None" = None
    children: list["RadixNode"] = field(default_factory=list)
    last_used: int = 0


@dataclass(frozen=True)
class PrefixResolution:
    hit_tokens: tuple[int, ...]
    miss_tokens: tuple[int, ...]

    @property
    def hit_length(self) -> int:
        return len(self.hit_tokens)


class RadixPrefixCache:
    """A compressed radix tree keyed by token ids.

    This model stores token routes rather than GPU tensors.  In a serving engine,
    the fully matched path would also carry references to K/V blocks.  The
    compressed-edge splitting here is the important structural operation.
    """

    def __init__(self) -> None:
        self.root = RadixNode(())
        self.clock = 0

    @staticmethod
    def _lcp(left: tuple[int, ...], right: tuple[int, ...]) -> int:
        matched = 0
        for a, b in zip(left, right):
            if a != b:
                break
            matched += 1
        return matched

    def match(self, prompt_tokens: Iterable[int]) -> PrefixResolution:
        """Return only a contiguous prefix; matching never jumps through prompt."""
        prompt = tuple(prompt_tokens)
        node, offset = self.root, 0
        while offset < len(prompt):
            remainder = prompt[offset:]
            child, matched = max(
                ((candidate, self._lcp(candidate.edge_tokens, remainder)) for candidate in node.children),
                key=lambda pair: pair[1],
                default=(None, 0),
            )
            if child is None or matched != len(child.edge_tokens):
                offset += matched
                break
            self.clock += 1
            child.last_used = self.clock
            node = child
            offset += matched
        return PrefixResolution(prompt[:offset], prompt[offset:])

    def insert(self, prompt_tokens: Iterable[int]) -> None:
        """Insert a complete token route, splitting an existing compressed edge if needed."""
        prompt = tuple(prompt_tokens)
        if not prompt:
            return
        node, offset = self.root, 0
        while offset < len(prompt):
            remainder = prompt[offset:]
            child, overlap = max(
                ((candidate, self._lcp(candidate.edge_tokens, remainder)) for candidate in node.children),
                key=lambda pair: pair[1],
                default=(None, 0),
            )
            if child is None or overlap == 0:
                node.children.append(RadixNode(remainder, parent=node, last_used=self.clock))
                return
            if overlap == len(child.edge_tokens):
                node = child
                offset += overlap
                continue

            # Existing edge [common | old_tail] becomes a new shared node with
            # two children: old tail and the incoming new tail.
            common = child.edge_tokens[:overlap]
            old_tail = child.edge_tokens[overlap:]
            split = RadixNode(common, parent=node, last_used=self.clock)
            child.edge_tokens = old_tail
            child.parent = split
            index = node.children.index(child)
            node.children[index] = split
            split.children.append(child)
            incoming_tail = remainder[overlap:]
            if incoming_tail:
                split.children.append(RadixNode(incoming_tail, parent=split, last_used=self.clock))
            return

    def resolve_and_store(self, prompt_tokens: Iterable[int]) -> PrefixResolution:
        """Match first, then store the whole request so later calls may reuse it."""
        resolution = self.match(prompt_tokens)
        self.insert(tuple(prompt_tokens))
        return resolution

    def routes(self) -> list[tuple[int, ...]]:
        """Return terminal paths for debugging and tests."""
        output: list[tuple[int, ...]] = []

        def walk(node: RadixNode, prefix: tuple[int, ...]) -> None:
            full = prefix + node.edge_tokens
            if not node.children and node is not self.root:
                output.append(full)
            for child in node.children:
                walk(child, full)

        walk(self.root, ())
        return sorted(output)


def chunk_tokens(tokens: Iterable[int], chunk_size: int) -> list[tuple[int, ...]]:
    """Break a non-reused prefill suffix into scheduler-sized units."""
    normalized = tuple(tokens)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    return [normalized[i : i + chunk_size] for i in range(0, len(normalized), chunk_size)]


def prefill_plan(cache: RadixPrefixCache, prompt_tokens: Iterable[int], chunk_size: int) -> dict[str, object]:
    """Expose the relationship between prefix hit and chunked prefill work."""
    resolution = cache.resolve_and_store(prompt_tokens)
    return {
        "hit_length": resolution.hit_length,
        "reused_prefix": resolution.hit_tokens,
        "prefill_chunks": chunk_tokens(resolution.miss_tokens, chunk_size),
    }


# -----------------------------------------------------------------------------
# 4. KV cache scheduling: a value score plus lazy heap deletion
# -----------------------------------------------------------------------------

@dataclass
class CacheEntry:
    key: str
    blocks: int
    hits: int
    last_used: int
    version: int = 0
    active_references: int = 0


class CacheValueScheduler:
    """A bounded prefix-cache policy, intentionally simple but stateful.

    The heap may contain historical priorities.  `version` gives each refresh a
    new generation number, so a popped stale entry cannot evict a recently used
    cache item.  Production engines add policies for tenants, deadlines, swap,
    and reference counts to actual K/V blocks; this class isolates the core idea.
    """

    def __init__(self, capacity_blocks: int):
        if capacity_blocks < 1:
            raise ValueError("capacity_blocks must be positive")
        self.capacity_blocks = capacity_blocks
        self.used_blocks = 0
        self.clock = 0
        self.entries: dict[str, CacheEntry] = {}
        self._eviction_heap: list[tuple[float, int, str]] = []
        self.events: list[str] = []

    def _value(self, entry: CacheEntry) -> float:
        age = self.clock - entry.last_used
        recency = 1.0 / (1.0 + age)
        reuse_density = log1p(entry.hits) / entry.blocks
        # Higher value means "retain longer".  The constants are explanatory,
        # not a universal serving policy.
        return 0.70 * reuse_density + 0.25 * recency + 0.05 * entry.active_references

    def _refresh(self, entry: CacheEntry) -> None:
        entry.version += 1
        heapq.heappush(self._eviction_heap, (self._value(entry), entry.version, entry.key))

    def _evict_one(self) -> str:
        deferred: list[tuple[float, int, str]] = []
        while self._eviction_heap:
            _, version, key = heapq.heappop(self._eviction_heap)
            entry = self.entries.get(key)
            if entry is None or version != entry.version:
                continue  # removed or superseded: lazy deletion
            if entry.active_references:
                deferred.append((self._value(entry), entry.version, key))
                continue
            self.used_blocks -= entry.blocks
            self.entries.pop(key)
            self.events.append(f"evict:{key}")
            for item in deferred:
                heapq.heappush(self._eviction_heap, item)
            return key
        for item in deferred:
            heapq.heappush(self._eviction_heap, item)
        raise MemoryError("all cached entries are actively referenced")

    def touch(self, key: str, blocks: int, active: bool = False) -> bool:
        """Access or admit an entry and return whether this call was a cache hit."""
        if not 1 <= blocks <= self.capacity_blocks:
            raise ValueError("entry blocks must be within cache capacity")
        self.clock += 1
        if key in self.entries:
            entry = self.entries[key]
            entry.hits += 1
            entry.last_used = self.clock
            entry.active_references += int(active)
            self._refresh(entry)
            self.events.append(f"hit:{key}")
            return True

        while self.used_blocks + blocks > self.capacity_blocks:
            self._evict_one()
        entry = CacheEntry(key=key, blocks=blocks, hits=1, last_used=self.clock, active_references=int(active))
        self.entries[key] = entry
        self.used_blocks += blocks
        self._refresh(entry)
        self.events.append(f"admit:{key}")
        return False

    def release_reference(self, key: str) -> None:
        entry = self.entries[key]
        entry.active_references = max(0, entry.active_references - 1)
        self._refresh(entry)

    def snapshot(self) -> list[tuple[str, int, int, float]]:
        """Show cache state sorted from highest to lowest retention value."""
        return sorted(
            ((e.key, e.blocks, e.hits, round(self._value(e), 4)) for e in self.entries.values()),
            key=lambda row: (-row[3], row[0]),
        )


# -----------------------------------------------------------------------------
# 5. A deterministic toy benchmark: measure -> compare -> decide
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class PromptRequest:
    prefix_key: str
    shared_prefix_tokens: int
    unique_suffix_tokens: int


@dataclass(frozen=True)
class BenchmarkSummary:
    name: str
    request_count: int
    hit_rate: float
    reused_tokens: int
    mean_ttft_ms: float
    mean_maintenance_ms: float


def simulate_prefix_workload(name: str, requests: list[PromptRequest], cache_enabled: bool) -> BenchmarkSummary:
    """Generate a deterministic, explicitly synthetic TTFT accounting model.

    These numbers are not vLLM/SGLang performance claims.  They simply ensure
    that every plotted difference is traceable to a reused-token quantity.
    """
    seen_prefixes: set[str] = set()
    ttfts: list[float] = []
    maintenance: list[float] = []
    hit_count = 0
    reused_tokens = 0
    fixed_dispatch_ms = 12.0
    prefill_cost_ms_per_token = 0.080
    lookup_ms = 1.2
    insertion_ms = 0.4

    for request in requests:
        is_hit = cache_enabled and request.prefix_key in seen_prefixes
        hit_tokens = request.shared_prefix_tokens if is_hit else 0
        recompute_tokens = request.shared_prefix_tokens + request.unique_suffix_tokens - hit_tokens
        ttfts.append(fixed_dispatch_ms + prefill_cost_ms_per_token * recompute_tokens + (lookup_ms if cache_enabled else 0.0))
        maintenance.append((lookup_ms + (0.0 if is_hit else insertion_ms)) if cache_enabled else 0.0)
        hit_count += int(is_hit)
        reused_tokens += hit_tokens
        if cache_enabled:
            seen_prefixes.add(request.prefix_key)

    return BenchmarkSummary(
        name=name,
        request_count=len(requests),
        hit_rate=hit_count / len(requests) if requests else 0.0,
        reused_tokens=reused_tokens,
        mean_ttft_ms=sum(ttfts) / len(ttfts) if ttfts else 0.0,
        mean_maintenance_ms=sum(maintenance) / len(maintenance) if maintenance else 0.0,
    )


def compare_to_baseline(baseline: BenchmarkSummary, candidate: BenchmarkSummary) -> dict[str, float]:
    """Keep metric directions explicit: negative TTFT delta is an improvement."""
    return {
        "hit_rate_gain": candidate.hit_rate - baseline.hit_rate,
        "ttft_delta_ms": candidate.mean_ttft_ms - baseline.mean_ttft_ms,
        "maintenance_delta_ms": candidate.mean_maintenance_ms - baseline.mean_maintenance_ms,
        "reused_tokens_gain": candidate.reused_tokens - baseline.reused_tokens,
    }


def decide_prefix_cache(baseline: BenchmarkSummary, candidate: BenchmarkSummary) -> tuple[str, str]:
    """Return a conservative accept/tune/reject teaching decision."""
    delta = compare_to_baseline(baseline, candidate)
    if candidate.hit_rate >= 0.50 and delta["ttft_delta_ms"] < -5.0 and candidate.mean_maintenance_ms <= 2.0:
        return "accept", "在固定工作负载下，命中、TTFT 改善与维护成本同时达标。"
    if delta["ttft_delta_ms"] < 0.0 and candidate.hit_rate > 0.0:
        return "tune", "存在收益，但应先调整前缀聚类、分块粒度或淘汰策略。"
    return "reject", "当前复用模式不足以抵消缓存维护成本。"


# -----------------------------------------------------------------------------
# 6. Deterministic visualizations
# -----------------------------------------------------------------------------

def _configure_matplotlib() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold"})


def plot_memory_growth() -> Path:
    _configure_matplotlib()
    shape = ModelShape(layers=32, kv_heads=8, head_dim=128, dtype_bytes=2)
    rows = capacity_rows(shape, [1024, 2048, 4096, 8192, 16384], batch_size=8)
    x = [row["tokens"] for row in rows]
    y = [row["kv_gib"] for row in rows]
    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=160)
    ax.plot(x, y, marker="o", linewidth=3, color="#12B8C4", label="32 layers, 8 KV heads, batch=8")
    ax.fill_between(x, y, color="#12B8C4", alpha=0.14)
    for token_count, size in zip(x, y):
        ax.annotate(f"{size:.2f} GiB", (token_count, size), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9)
    ax.set_title("KV Cache Memory Grows Linearly with Context Length")
    ax.set_xlabel("sequence length (tokens)")
    ax.set_ylabel("K/V cache capacity (GiB)")
    ax.legend(frameon=True)
    fig.tight_layout()
    path = ASSET_DIR / "05_kv_memory_scaling.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_allocation_waste() -> Path:
    _configure_matplotlib()
    lengths, maximum, block_size = [3, 6, 13, 7, 16], 16, 4
    static_slots, paged_slots = static_vs_paged_slots(lengths, maximum, block_size)
    actual_slots = sum(lengths)
    fig, ax = plt.subplots(figsize=(8.8, 5.0), dpi=160)
    labels = ["actual\ntokens", "static\nreservation", "paged\nblocks"]
    values = [actual_slots, static_slots, paged_slots]
    colors = ["#22A699", "#F06A6A", "#4A78C2"]
    bars = ax.bar(labels, values, color=colors, width=0.6)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.4, f"{value} slots", ha="center", fontweight="bold")
    ax.axhline(actual_slots, color="#22A699", linestyle="--", linewidth=1.5)
    ax.set_title("Toy Workload: Reservation Waste vs. Fixed-Size Paging")
    ax.set_ylabel("token-equivalent slots")
    ax.text(0.5, -0.20, "Five requests with actual lengths 3, 6, 13, 7, 16; page size = 4.", ha="center", transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    path = ASSET_DIR / "06_paging_capacity_toy_model.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def example_workloads() -> dict[str, list[PromptRequest]]:
    """Three deterministic request distributions, not measurements from a backend."""
    return {
        "no reuse": [PromptRequest(f"u{i}", 256, 96) for i in range(8)],
        "shared system": [PromptRequest("system", 256, 96 + 8 * (i % 3)) for i in range(8)],
        "two families": [PromptRequest("A" if i % 2 == 0 else "B", 192, 112 + 8 * (i % 2)) for i in range(8)],
    }


def plot_prefix_cache_benchmark() -> tuple[Path, list[tuple[BenchmarkSummary, BenchmarkSummary]]]:
    _configure_matplotlib()
    results: list[tuple[BenchmarkSummary, BenchmarkSummary]] = []
    for name, requests in example_workloads().items():
        results.append((simulate_prefix_workload(name, requests, False), simulate_prefix_workload(name, requests, True)))

    names = [baseline.name for baseline, _ in results]
    baseline_ttft = [baseline.mean_ttft_ms for baseline, _ in results]
    cached_ttft = [candidate.mean_ttft_ms for _, candidate in results]
    hit_rates = [candidate.hit_rate * 100 for _, candidate in results]
    x = list(range(len(names)))
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.2, 4.8), dpi=160)
    width = 0.35
    left.bar([i - width / 2 for i in x], baseline_ttft, width, label="cache off", color="#A0AEC0")
    left.bar([i + width / 2 for i in x], cached_ttft, width, label="cache on", color="#5B5BD6")
    left.set_xticks(x, names, rotation=10)
    left.set_ylabel("synthetic mean TTFT (ms)")
    left.set_title("TTFT accounting model")
    left.legend()
    right.bar(names, hit_rates, color=["#CAD2D9", "#18A999", "#E6A23C"])
    right.set_ylim(0, 105)
    right.set_ylabel("request cache-hit rate (%)")
    right.set_title("Prefix reuse observed")
    for i, rate in enumerate(hit_rates):
        right.text(i, rate + 3, f"{rate:.0f}%", ha="center", fontweight="bold")
    fig.suptitle("Deterministic Toy Benchmark — Mechanism Check, Not Backend Measurement", fontweight="bold", y=1.02)
    fig.tight_layout()
    path = ASSET_DIR / "07_prefix_cache_toy_benchmark.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path, results


# -----------------------------------------------------------------------------
# 7. Tests and reproducible entry point
# -----------------------------------------------------------------------------

def run_tests() -> None:
    shape = ModelShape(layers=2, kv_heads=4, head_dim=8, dtype_bytes=2)
    assert kv_cache_bytes(shape, sequence_tokens=10, batch_size=3) == 2 * 2 * 4 * 8 * 10 * 3 * 2

    pool = PagedKVPool(total_blocks=8, block_size=4)
    request = pool.admit_prompt("r1", prompt_tokens=6)
    assert len(request.block_table) == 2
    pool.append_one_token("r1")  # token 7 stays in the existing tail block
    pool.append_one_token("r1")  # token 8 stays in the existing tail block
    pool.append_one_token("r1")  # token 9 opens a new block
    assert len(request.block_table) == 3
    assert pool.physical_location("r1", 8)[0] == request.block_table[2]
    pool.release("r1")
    assert pool.free_block_count == 8

    tree = RadixPrefixCache()
    tree.insert([1, 2, 3, 4, 5])
    tree.insert([1, 2, 3, 9])
    resolution = tree.match([1, 2, 3, 9, 10])
    assert resolution.hit_length == 4 and resolution.miss_tokens == (10,)
    assert (1, 2, 3, 4, 5) in tree.routes() and (1, 2, 3, 9) in tree.routes()
    plan = prefill_plan(tree, [1, 2, 3, 9, 10, 11, 12], chunk_size=2)
    assert plan["hit_length"] == 4 and plan["prefill_chunks"] == [(10, 11), (12,)]

    scheduler = CacheValueScheduler(capacity_blocks=5)
    assert scheduler.touch("system", 2) is False
    assert scheduler.touch("tools", 2) is False
    assert scheduler.touch("system", 2) is True
    scheduler.touch("retrieval", 3)  # must evict the lower-value entry
    assert scheduler.used_blocks <= 5 and "system" in scheduler.entries
    assert any(event.startswith("evict:") for event in scheduler.events)

    requests = example_workloads()["shared system"]
    baseline = simulate_prefix_workload("baseline", requests, False)
    candidate = simulate_prefix_workload("cache", requests, True)
    assert candidate.hit_rate == 7 / 8
    assert candidate.mean_ttft_ms < baseline.mean_ttft_ms
    assert decide_prefix_cache(baseline, candidate)[0] == "accept"


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    run_tests()
    plot_memory_growth()
    plot_allocation_waste()
    _, benchmark_results = plot_prefix_cache_benchmark()

    print("All teaching-lab assertions passed.")
    for baseline, candidate in benchmark_results:
        delta = compare_to_baseline(baseline, candidate)
        print(
            f"{candidate.name:13s} hit_rate={candidate.hit_rate:5.1%} "
            f"TTFT_delta={delta['ttft_delta_ms']:6.2f} ms "
            f"reused_tokens={candidate.reused_tokens}"
        )


if __name__ == "__main__":
    main()
