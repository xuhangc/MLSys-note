#!/usr/bin/env python3
"""A CPU-only learning lab for KV-cache-aware LLM serving.

This file is an original educational reconstruction, not production vLLM/SGLang
code. It models four separable concerns:

1. Paged KV allocation: a logical sequence owns a block table, while its token
   slots can reside in non-contiguous physical pages.
2. Radix prefix lookup: a compressed trie finds reusable leading tokens.
3. Chunked prefill: only the unmatched suffix is split into schedulable chunks.
4. Value-aware eviction: finite cache capacity is allocated using reuse,
   recency, and size rather than FIFO arrival order.

Run:
    python3 code/serving_memory_lab.py

The lab has no PyTorch or GPU dependency. It validates data-structure
invariants with assertions and prints an explanatory trace.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import heapq
import math
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

Token = int
TokenSeq = Tuple[Token, ...]


# ---------------------------------------------------------------------------
# Part 1. A logical request and a paged physical KV pool
# ---------------------------------------------------------------------------


@dataclass
class SequenceState:
    """One request's logical token sequence and its physical block references."""

    request_id: str
    tokens: List[Token]
    block_table: List[int] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.tokens)


class PagedKVPool:
    """CPU model of a fixed-size KV-page allocator.

    In a real engine, a page contains K and V tensors for all layers/heads, not
    token IDs. Token IDs make the mapping observable without requiring a GPU.
    """

    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks <= 0 or block_size <= 0:
            raise ValueError("num_blocks and block_size must both be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.pages: List[List[Optional[Token]]] = [
            [None] * block_size for _ in range(num_blocks)
        ]
        # A stack intentionally makes later allocations visibly non-contiguous.
        self.free_blocks: Deque[int] = deque(range(num_blocks))

    def blocks_for(self, token_count: int) -> int:
        """Return ceil(token_count / block_size), including a partial final page."""
        return math.ceil(token_count / self.block_size) if token_count else 0

    def _allocate(self, count: int) -> List[int]:
        if len(self.free_blocks) < count:
            raise MemoryError(
                f"KV cache OOM: need {count} free blocks, have {len(self.free_blocks)}"
            )
        return [self.free_blocks.pop() for _ in range(count)]

    def prefill(self, request_id: str, prompt: Sequence[Token]) -> SequenceState:
        """Allocate exactly enough pages for a prompt and write each logical token."""
        state = SequenceState(request_id=request_id, tokens=list(prompt))
        state.block_table = self._allocate(self.blocks_for(state.length))
        for logical_position, token in enumerate(state.tokens):
            page_index = logical_position // self.block_size
            offset = logical_position % self.block_size
            physical_block = state.block_table[page_index]
            self.pages[physical_block][offset] = token
        return state

    def append_decode_token(self, state: SequenceState, token: Token) -> None:
        """Append one decoded token, allocating one page only at a page boundary."""
        logical_position = state.length
        if logical_position == len(state.block_table) * self.block_size:
            state.block_table.extend(self._allocate(1))
        page_index = logical_position // self.block_size
        offset = logical_position % self.block_size
        physical_block = state.block_table[page_index]
        self.pages[physical_block][offset] = token
        state.tokens.append(token)

    def materialize(self, state: SequenceState) -> List[Token]:
        """Reconstruct logical order by following the block table, then trim tail slack."""
        gathered: List[Token] = []
        for physical_block in state.block_table:
            gathered.extend(self.pages[physical_block])
        logical = gathered[: state.length]
        if any(token is None for token in logical):
            raise RuntimeError("logical cache contains an unwritten slot")
        return [token for token in logical if token is not None]

    def release(self, state: SequenceState) -> None:
        """Return all pages to the pool. A production engine must also honour refcounts."""
        for physical_block in state.block_table:
            self.pages[physical_block] = [None] * self.block_size
            self.free_blocks.appendleft(physical_block)
        state.block_table.clear()

    @property
    def used_blocks(self) -> int:
        return self.num_blocks - len(self.free_blocks)


# ---------------------------------------------------------------------------
# Part 2. Compressed radix tree for longest-prefix reuse
# ---------------------------------------------------------------------------


def lcp_length(left: Sequence[Token], right: Sequence[Token]) -> int:
    """Length of the longest common *leading* subsequence of two token sequences."""
    matched = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        matched += 1
    return matched


@dataclass
class RadixNode:
    """A compressed edge plus children; one edge can carry multiple tokens."""

    fragment: TokenSeq
    children: List["RadixNode"] = field(default_factory=list)
    cache_key: Optional[str] = None


@dataclass(frozen=True)
class PrefixMatch:
    """Result of matching a new request against the cached radix tree."""

    hit_length: int
    reusable_prefix: TokenSeq
    miss_suffix: TokenSeq
    deepest_cache_key: Optional[str]


class RadixPrefixIndex:
    """Compressed prefix index with split-on-insert and longest-prefix matching.

    The index stores tokens as keys and a symbolic cache key as the value. The
    actual KV pages would be kept by a page allocator such as ``PagedKVPool``.
    """

    def __init__(self) -> None:
        self.root = RadixNode(fragment=())

    def insert(self, tokens: Sequence[Token], cache_key: str) -> None:
        """Insert a cached sequence, splitting an existing edge when paths diverge."""
        tail = tuple(tokens)
        if not tail:
            raise ValueError("an empty prefix is not useful to cache")

        parent = self.root
        while tail:
            matching_child: Optional[RadixNode] = None
            shared = 0
            for child in parent.children:
                candidate = lcp_length(child.fragment, tail)
                if candidate:
                    matching_child, shared = child, candidate
                    break

            if matching_child is None:
                parent.children.append(RadixNode(fragment=tail, cache_key=cache_key))
                return

            child = matching_child
            if shared == len(child.fragment):
                # The cached edge is entirely consumed; continue below it.
                parent = child
                tail = tail[shared:]
                continue

            # Existing [a,b,c] and incoming [a,b,d] become shared [a,b] with
            # two suffix children [c] and [d]. This is the key radix-tree step.
            shared_node = RadixNode(fragment=child.fragment[:shared])
            child.fragment = child.fragment[shared:]
            shared_node.children.append(child)
            child_position = parent.children.index(child)
            parent.children[child_position] = shared_node
            parent = shared_node
            tail = tail[shared:]

        # The incoming sequence ended exactly at an existing node boundary.
        parent.cache_key = cache_key

    def match(self, prompt: Sequence[Token]) -> PrefixMatch:
        """Find the longest leading token region indexed by the tree.

        A partial match inside a compressed edge is still reusable conceptually:
        the original cached computation contains KV states for every token in
        that edge. ``deepest_cache_key`` is only set when a full node boundary
        is reached, so callers do not mistake it for production-ready refcount
        management.
        """
        remaining = tuple(prompt)
        parent = self.root
        hit = 0
        deepest_key: Optional[str] = None

        while remaining:
            child = next(
                (node for node in parent.children if lcp_length(node.fragment, remaining)),
                None,
            )
            if child is None:
                break
            shared = lcp_length(child.fragment, remaining)
            hit += shared
            if shared < len(child.fragment):
                break
            remaining = remaining[shared:]
            parent = child
            if child.cache_key is not None:
                deepest_key = child.cache_key

        prompt_tuple = tuple(prompt)
        return PrefixMatch(
            hit_length=hit,
            reusable_prefix=prompt_tuple[:hit],
            miss_suffix=prompt_tuple[hit:],
            deepest_cache_key=deepest_key,
        )

    def edge_fragments(self) -> List[TokenSeq]:
        """Return a deterministic shallow view for the learning trace."""
        result: List[TokenSeq] = []

        def visit(node: RadixNode) -> None:
            for child in node.children:
                result.append(child.fragment)
                visit(child)

        visit(self.root)
        return result


# ---------------------------------------------------------------------------
# Part 3. Prefix hit turns into an explicit chunked-prefill work plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrefillChunk:
    request_id: str
    ordinal: int
    tokens: TokenSeq


def make_prefill_plan(
    request_id: str, unmatched_suffix: Sequence[Token], chunk_size: int
) -> List[PrefillChunk]:
    """Split only the cache miss region; the hit region must not be recomputed."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    suffix = tuple(unmatched_suffix)
    return [
        PrefillChunk(request_id, ordinal, suffix[start : start + chunk_size])
        for ordinal, start in enumerate(range(0, len(suffix), chunk_size), start=1)
    ]


@dataclass(frozen=True)
class BatchPlan:
    """One serving iteration under an explicit token budget."""

    decode_request_ids: Tuple[str, ...]
    prefill_chunks: Tuple[PrefillChunk, ...]
    used_tokens: int
    token_budget: int


def schedule_one_iteration(
    pending_decode_ids: Sequence[str],
    pending_prefills: Sequence[PrefillChunk],
    token_budget: int,
) -> BatchPlan:
    """Prioritize one decode token per live request, then fill budget with prefills.

    This is intentionally a transparent policy, not a claim about any serving
    engine's current default. The key invariant is ``used_tokens <= token_budget``.
    """
    if token_budget <= 0:
        raise ValueError("token_budget must be positive")

    decode_ids = tuple(pending_decode_ids[:token_budget])
    budget_left = token_budget - len(decode_ids)
    admitted: List[PrefillChunk] = []
    for chunk in pending_prefills:
        if len(chunk.tokens) <= budget_left:
            admitted.append(chunk)
            budget_left -= len(chunk.tokens)
        else:
            break
    return BatchPlan(
        decode_request_ids=decode_ids,
        prefill_chunks=tuple(admitted),
        used_tokens=token_budget - budget_left,
        token_budget=token_budget,
    )


# ---------------------------------------------------------------------------
# Part 4. Value-aware retention and stale-safe eviction heap
# ---------------------------------------------------------------------------


@dataclass
class CacheRecord:
    key: str
    blocks: int
    hits: int
    last_used: int
    score: float = 0.0


class ValueAwareKVCache:
    """Finite cache where low-value records leave first under capacity pressure.

    Scores use a logarithmic reuse term to avoid allowing one historically hot
    object to dominate forever, a reciprocal recency term, and a size cost.
    Heap entries are appended lazily; stale entries are verified at pop time.
    """

    def __init__(self, capacity_blocks: int) -> None:
        if capacity_blocks <= 0:
            raise ValueError("capacity_blocks must be positive")
        self.capacity_blocks = capacity_blocks
        self.used_blocks = 0
        self.clock = 0
        self.entries: Dict[str, CacheRecord] = {}
        self._eviction_heap: List[Tuple[float, int, str]] = []
        self.events: List[str] = []

    def _score(self, record: CacheRecord) -> float:
        age = max(self.clock - record.last_used, 0)
        reuse_value = math.log2(1 + record.hits)
        recency_value = 1.0 / (1.0 + age)
        normalized_size_cost = record.blocks / self.capacity_blocks
        return reuse_value + 0.75 * recency_value - 0.50 * normalized_size_cost

    def _refresh(self, record: CacheRecord) -> None:
        record.score = self._score(record)
        heapq.heappush(self._eviction_heap, (record.score, record.last_used, record.key))

    def _evict_until_fit(self, required_blocks: int) -> None:
        while self.used_blocks + required_blocks > self.capacity_blocks:
            if not self.entries:
                raise MemoryError("no cache record can be evicted to make room")
            while self._eviction_heap:
                score, observed_last_used, key = heapq.heappop(self._eviction_heap)
                record = self.entries.get(key)
                is_stale = record is None or (
                    score != record.score or observed_last_used != record.last_used
                )
                if not is_stale:
                    break
            else:
                # Defensive fallback: recreate a valid heap from current entries.
                self._eviction_heap = []
                for record in self.entries.values():
                    self._refresh(record)
                continue

            self.entries.pop(record.key)
            self.used_blocks -= record.blocks
            self.events.append(f"evict:{record.key}")

    def touch(self, key: str, blocks: int) -> None:
        """Record a prefix access; a hit changes value, a miss may cause eviction."""
        if blocks <= 0:
            raise ValueError("blocks must be positive")
        if blocks > self.capacity_blocks:
            raise ValueError("a single cache entry exceeds total capacity")

        self.clock += 1
        existing = self.entries.get(key)
        if existing is not None:
            existing.hits += 1
            existing.last_used = self.clock
            self._refresh(existing)
            self.events.append(f"reuse:{key}")
            return

        self._evict_until_fit(blocks)
        record = CacheRecord(key=key, blocks=blocks, hits=1, last_used=self.clock)
        self.entries[key] = record
        self.used_blocks += blocks
        self._refresh(record)
        self.events.append(f"add:{key}")

    def snapshot(self) -> List[Tuple[str, int, int, float]]:
        """Current records ordered from most worth retaining to least."""
        return [
            (record.key, record.blocks, record.hits, round(record.score, 4))
            for record in sorted(
                self.entries.values(),
                key=lambda record: (-record.score, -record.last_used, record.key),
            )
        ]


# ---------------------------------------------------------------------------
# Verification and human-readable trace
# ---------------------------------------------------------------------------


def verify_paged_pool() -> SequenceState:
    pool = PagedKVPool(num_blocks=8, block_size=4)
    alpha = pool.prefill("alpha", [10, 11, 12, 13, 14, 15])
    beta = pool.prefill("beta", [90, 91, 92, 93])
    pool.append_decode_token(beta, 94)  # beta consumes an intervening free page.
    pool.append_decode_token(alpha, 16)
    pool.append_decode_token(alpha, 17)
    pool.append_decode_token(alpha, 18)  # alpha now crosses a page boundary.

    assert alpha.block_table == [7, 6, 3], alpha.block_table
    assert pool.materialize(alpha) == [10, 11, 12, 13, 14, 15, 16, 17, 18]
    assert pool.materialize(beta) == [90, 91, 92, 93, 94]
    assert pool.used_blocks == 5
    return alpha


def verify_radix_index() -> PrefixMatch:
    index = RadixPrefixIndex()
    index.insert([1, 2, 3, 4], cache_key="chat-A-turn-1")
    index.insert([1, 2, 3, 9], cache_key="chat-B-turn-1")
    index.insert([8, 8], cache_key="few-shot-template")

    match = index.match([1, 2, 3, 9, 42, 43])
    assert match.hit_length == 4
    assert match.reusable_prefix == (1, 2, 3, 9)
    assert match.miss_suffix == (42, 43)
    assert match.deepest_cache_key == "chat-B-turn-1"
    assert set(index.edge_fragments()) == {(1, 2, 3), (4,), (9,), (8, 8)}
    return match


def verify_chunked_schedule(match: PrefixMatch) -> BatchPlan:
    chunks = make_prefill_plan("chat-B-turn-2", match.miss_suffix + (44, 45, 46), 2)
    assert [chunk.tokens for chunk in chunks] == [(42, 43), (44, 45), (46,)]

    plan = schedule_one_iteration(
        pending_decode_ids=["alpha", "beta"],
        pending_prefills=chunks,
        token_budget=5,
    )
    assert plan.decode_request_ids == ("alpha", "beta")
    assert [chunk.tokens for chunk in plan.prefill_chunks] == [(42, 43)]
    assert plan.used_tokens == 4 <= plan.token_budget
    return plan


def verify_value_aware_eviction() -> ValueAwareKVCache:
    cache = ValueAwareKVCache(capacity_blocks=6)
    for key, blocks in [
        ("common-system", 2),
        ("tenant-a-history", 3),
        ("common-system", 2),  # raises reuse value
        ("tenant-b-history", 3),  # forces one eviction
    ]:
        cache.touch(key, blocks)

    assert cache.used_blocks <= cache.capacity_blocks
    assert any(event.startswith("reuse:common-system") for event in cache.events)
    assert any(event.startswith("evict:") for event in cache.events)
    assert "common-system" in cache.entries
    return cache


def format_rows(rows: Iterable[Sequence[object]]) -> str:
    """Small dependency-free formatter for the trace printed by ``main``."""
    return "\n".join("  " + " | ".join(str(item) for item in row) for row in rows)


def main() -> None:
    alpha = verify_paged_pool()
    match = verify_radix_index()
    plan = verify_chunked_schedule(match)
    cache = verify_value_aware_eviction()

    print("All invariants passed. This is a structural CPU simulation, not a GPU benchmark.\n")
    print("[1] Paged KV mapping")
    print(f"  alpha logical length: {alpha.length}")
    print(f"  alpha block table:    {alpha.block_table}  (intentionally non-contiguous)")
    print("\n[2] Radix longest-prefix match")
    print(f"  hit / miss: {match.reusable_prefix} / {match.miss_suffix}")
    print(f"  matched cache key: {match.deepest_cache_key}")
    print("\n[3] Decode-first batch plan")
    print(f"  decode requests: {plan.decode_request_ids}")
    print(f"  prefill chunks:  {[chunk.tokens for chunk in plan.prefill_chunks]}")
    print(f"  budget used:     {plan.used_tokens}/{plan.token_budget} tokens")
    print("\n[4] Value-aware KV retention")
    print(f"  events: {cache.events}")
    print("  key | blocks | hits | retention_score")
    print(format_rows(cache.snapshot()))


if __name__ == "__main__":
    main()
