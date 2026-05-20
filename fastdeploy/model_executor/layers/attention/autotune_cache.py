"""
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

from __future__ import annotations

import json
import os
import statistics
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import paddle
from paddleformers.utils.log import logger

if TYPE_CHECKING:
    from fastdeploy.model_executor.layers.attention.base_attention_backend import AttentionBackend

_COUNT_CANDIDATES = [1, 2, 4, 16, 32, 64, 128, 192, 256, 384, 512]


def make_tokens_bins(max_tokens: int) -> list:
    """Token bins: denser at small values, sparser at large. Capped by max_tokens."""
    # Exponential growth for small values, then linear for large
    bins = []
    v = 128
    while v < max_tokens:
        bins.append(v)
        v = v * 2 if v < 4096 else v + 4096
    bins.append(max_tokens)
    return bins


def make_count_bins(max_seqs: int) -> list:
    """Count bins from predefined candidates, capped by max_seqs."""
    return [c for c in _COUNT_CANDIDATES if c <= max_seqs]


@dataclass
class BatchProfile:
    total_prefill_tokens: int
    num_prefill_seqs: int


def bucketize(value, bins):
    """Return the bin value that `value` falls into."""
    idx = bisect_right(bins, value)
    idx = min(idx, len(bins) - 1)
    return bins[idx]


class AutotuneCache:
    def __init__(self, candidates: list, tokens_bins: list, count_bins: list):
        self.candidates = candidates
        self.tokens_bins = tokens_bins
        self.count_bins = count_bins
        self.cache: dict[tuple, object] = {}

    def compute_key(self, profile: BatchProfile) -> tuple:
        return (
            bucketize(profile.total_prefill_tokens, self.tokens_bins),
            bucketize(profile.num_prefill_seqs, self.count_bins),
        )

    def get_best(self, key: tuple) -> Optional[object]:
        return self.cache.get(key)

    def tune(self, key, q, k, v, qkv, compressed_kv, k_pe, layer, forward_meta) -> object:
        """Benchmark all candidates for this key, cache the fastest."""
        results = []
        for backend in self.candidates:
            # warmup
            backend.forward_mixed(q, k, v, qkv, compressed_kv, k_pe, layer, forward_meta)
            paddle.cuda.synchronize()
            # timed run: 10 iterations, take median of middle 5
            times = []
            for _ in range(10):
                start_event = paddle.cuda.Event(enable_timing=True)
                end_event = paddle.cuda.Event(enable_timing=True)
                start_event.record()
                backend.forward_mixed(q, k, v, qkv, compressed_kv, k_pe, layer, forward_meta)
                end_event.record()
                paddle.cuda.synchronize()
                times.append(start_event.elapsed_time(end_event))
            times.sort()
            elapsed = statistics.median(times[2:7])
            results.append((elapsed, backend))

        best = min(results, key=lambda x: x[0])[1]
        self.cache[key] = best
        detail = ", ".join(f"{type(b).__name__}={t:.3f}ms" for t, b in results)
        logger.info(f"AutoAttn tune key={key}: {detail} -> best={type(best).__name__}")
        return best

    def save(self, path: str):
        """Persist cache to JSON file."""
        data = {}
        for key, backend in self.cache.items():
            data[str(key)] = type(backend).__name__
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str, name_to_backend: dict):
        """Load cache from JSON file."""
        if not os.path.exists(path):
            return
        with open(path, "r") as f:
            data = json.load(f)
        for key_str, backend_name in data.items():
            if backend_name in name_to_backend:
                key = tuple(eval(key_str))
                self.cache[key] = name_to_backend[backend_name]
