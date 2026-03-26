"""In-memory prefix snapshot and restore helpers for siliconrt.

This module is the first real siliconrt prototype on top of mlx_lm.

It does not implement paged attention or a new KV layout. It does something
smaller and more important first:

- own the prefix snapshot/restore seam explicitly
- keep reusable prefix caches in memory
- make restore/eviction policy measurable
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import copy
import hashlib
from importlib.metadata import version
from typing import Any, Iterable, Optional, Sequence

import mlx.core as mx

from mlx_lm.generate import generate_step
from mlx_lm.models import cache as mlx_cache


def _clone_state(value: Any) -> Any:
    """Deep-copy an mlx_lm cache state into fresh MLX arrays.

    `mx.array(existing_array)` creates a new array object with copied contents,
    which is what we need here. Reusing the same underlying arrays would cause
    aliasing once the cache is extended by a later request.
    """

    if isinstance(value, mx.array):
        return mx.array(value)
    if isinstance(value, list):
        return [_clone_state(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_state(v) for v in value)
    if isinstance(value, dict):
        return {k: _clone_state(v) for k, v in value.items()}
    return value


def _clone_prompt_cache(cache: Sequence[Any]) -> list[Any]:
    """Clone an mlx_lm prompt cache in memory without using disk."""

    cloned = []
    for entry in cache:
        entry_state = _clone_state(entry.state)
        entry_meta = copy.deepcopy(entry.meta_state)
        cloned.append(type(entry).from_state(entry_state, entry_meta))
    return cloned


def _prompt_cache_nbytes(cache: Sequence[Any]) -> int:
    return sum(c.nbytes for c in cache)


def _collect_arrays(value: Any, out: list[mx.array]) -> None:
    if isinstance(value, mx.array):
        out.append(value)
        return
    if isinstance(value, list):
        for v in value:
            _collect_arrays(v, out)
        return
    if isinstance(value, tuple):
        for v in value:
            _collect_arrays(v, out)
        return
    if isinstance(value, dict):
        for v in value.values():
            _collect_arrays(v, out)


def _materialize_prompt_cache(cache: Sequence[Any]) -> None:
    arrays: list[mx.array] = []
    for entry in cache:
        _collect_arrays(entry.state, arrays)
    if arrays:
        mx.eval(arrays)


def _build_prefix_key(
    *,
    model_key: str,
    prefix_tokens: Sequence[int],
    max_kv_size: Optional[int],
    kv_bits: Optional[int],
    kv_group_size: int,
) -> str:
    h = hashlib.sha256()
    h.update(model_key.encode("utf-8"))
    h.update(b"\0")
    h.update(str(max_kv_size).encode("utf-8"))
    h.update(b"\0")
    h.update(str(kv_bits).encode("utf-8"))
    h.update(b"\0")
    h.update(str(kv_group_size).encode("utf-8"))
    h.update(b"\0")
    for tok in prefix_tokens:
        h.update(int(tok).to_bytes(4, "little", signed=False))
    return h.hexdigest()


def _cache_class_names(cache: Sequence[Any]) -> tuple[str, ...]:
    return tuple(type(entry).__name__ for entry in cache)


def _cache_kind(max_kv_size: Optional[int]) -> str:
    return "dense" if max_kv_size is None else "bounded_contiguous"


def _batch_merge_compatible(cache: Sequence[Any]) -> bool:
    def entry_ok(entry: Any) -> bool:
        if type(entry).__name__ == "KVCache":
            return True
        if type(entry).__name__ == "RotatingKVCache":
            return getattr(entry, "keep", 0) == 0
        if type(entry).__name__ == "CacheList":
            return all(entry_ok(sub) for sub in entry.caches)
        return False

    return all(entry_ok(entry) for entry in cache)


@dataclass
class PrefixSnapshot:
    key: str
    prefix_token_hash: str
    model_key: str
    prefix_tokens: int
    cache_kind: str
    cache_class_names: tuple[str, ...]
    batch_merge_compatible: bool
    cache: list[Any]
    max_kv_size: Optional[int]
    kv_bits: Optional[int]
    kv_group_size: int
    mlx_version: str
    mlx_lm_version: str
    nbytes: int

    def restore(self, *, materialize: bool = False) -> list[Any]:
        restored = _clone_prompt_cache(self.cache)
        if materialize:
            _materialize_prompt_cache(restored)
        return restored

    def restore_many(self, n: int, *, materialize: bool = False) -> list[list[Any]]:
        return [self.restore(materialize=materialize) for _ in range(n)]


class PrefixStore:
    """A tiny in-memory LRU store for reusable prefix snapshots."""

    def __init__(self, byte_budget: Optional[int] = None):
        self.byte_budget = byte_budget
        self._entries: OrderedDict[str, PrefixSnapshot] = OrderedDict()
        self._nbytes = 0
        self._hits = 0
        self._misses = 0
        self._puts = 0
        self._evictions = 0
        self._evicted_bytes = 0

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def puts(self) -> int:
        return self._puts

    @property
    def evictions(self) -> int:
        return self._evictions

    @property
    def evicted_bytes(self) -> int:
        return self._evicted_bytes

    def keys(self) -> list[str]:
        return list(self._entries.keys())

    def contains(self, key: str) -> bool:
        return key in self._entries

    def stats(self) -> dict[str, int | None]:
        return {
            "entries": len(self._entries),
            "resident_bytes": self._nbytes,
            "byte_budget": self.byte_budget,
            "hits": self._hits,
            "misses": self._misses,
            "puts": self._puts,
            "evictions": self._evictions,
            "evicted_bytes": self._evicted_bytes,
        }

    def get(self, key: str) -> Optional[PrefixSnapshot]:
        snapshot = self._entries.get(key)
        if snapshot is None:
            self._misses += 1
            return None
        self._hits += 1
        self._entries.move_to_end(key)
        return snapshot

    def put(self, snapshot: PrefixSnapshot) -> None:
        self._puts += 1
        existing = self._entries.pop(snapshot.key, None)
        if existing is not None:
            self._nbytes -= existing.nbytes

        self._entries[snapshot.key] = snapshot
        self._entries.move_to_end(snapshot.key)
        self._nbytes += snapshot.nbytes
        self._evict_if_needed()

    def checkout(self, key: str, *, materialize: bool = False) -> list[Any]:
        snapshot = self.get(key)
        if snapshot is None:
            raise KeyError(f"Unknown prefix snapshot key: {key}")
        return snapshot.restore(materialize=materialize)

    def capture(
        self,
        *,
        model,
        model_key: str,
        prefix_tokens: Sequence[int],
        max_kv_size: Optional[int] = None,
        kv_bits: Optional[int] = None,
        kv_group_size: int = 64,
    ) -> PrefixSnapshot:
        snapshot = _build_prefix_snapshot(
            model=model,
            model_key=model_key,
            prefix_tokens=prefix_tokens,
            max_kv_size=max_kv_size,
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
        )
        self.put(snapshot)
        return snapshot

    def _evict_if_needed(self) -> None:
        if self.byte_budget is None:
            return
        while self._nbytes > self.byte_budget and self._entries:
            _key, snapshot = self._entries.popitem(last=False)
            self._nbytes -= snapshot.nbytes
            self._evictions += 1
            self._evicted_bytes += snapshot.nbytes


def _build_prefix_snapshot(
    *,
    model,
    model_key: str,
    prefix_tokens: Sequence[int],
    max_kv_size: Optional[int] = None,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
) -> PrefixSnapshot:
    """Build an in-memory reusable prefix snapshot for later restore."""

    if max_kv_size is not None and kv_bits is not None:
        raise ValueError(
            "siliconrt v1 does not support bounded/rotating KV together with kv_bits yet."
        )

    prompt_cache = mlx_cache.make_prompt_cache(model, max_kv_size=max_kv_size)
    for _ in generate_step(
        mx.array(prefix_tokens),
        model,
        max_tokens=0,
        prompt_cache=prompt_cache,
        max_kv_size=max_kv_size,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
    ):
        pass

    stored_cache = _clone_prompt_cache(prompt_cache)
    prefix_token_hash = hashlib.sha256()
    for tok in prefix_tokens:
        prefix_token_hash.update(int(tok).to_bytes(4, "little", signed=False))
    prefix_token_hash_hex = prefix_token_hash.hexdigest()

    return PrefixSnapshot(
        key=_build_prefix_key(
            model_key=model_key,
            prefix_tokens=prefix_tokens,
            max_kv_size=max_kv_size,
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
        ),
        prefix_token_hash=prefix_token_hash_hex,
        model_key=model_key,
        prefix_tokens=len(prefix_tokens),
        cache_kind=_cache_kind(max_kv_size),
        cache_class_names=_cache_class_names(stored_cache),
        batch_merge_compatible=_batch_merge_compatible(stored_cache),
        cache=stored_cache,
        max_kv_size=max_kv_size,
        kv_bits=kv_bits,
        kv_group_size=kv_group_size,
        mlx_version=version("mlx"),
        mlx_lm_version=version("mlx-lm"),
        nbytes=_prompt_cache_nbytes(stored_cache),
    )


# Backward-compatible alias while the repo is still tiny.
PrefixCacheStore = PrefixStore
