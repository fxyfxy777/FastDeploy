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

import os
import time
import types
from typing import TYPE_CHECKING

import paddle
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.layers.attention.append_attn_backend import AppendAttentionBackend
from fastdeploy.model_executor.layers.attention.autotune_cache import (
    AutotuneCache,
    BatchProfile,
    make_tokens_bins,
    make_count_bins,
)
from fastdeploy.model_executor.layers.attention.base_attention_backend import AttentionBackend
from fastdeploy.model_executor.layers.attention.flash_attn_backend import FlashAttentionBackend
from fastdeploy.model_executor.utils import get_sm_version

if TYPE_CHECKING:
    from fastdeploy.model_executor.forward_meta import ForwardMeta
    from fastdeploy.model_executor.layers.attention.attention import Attention


class AutoAttentionBackend(AttentionBackend):
    """Dynamically selects the best attention backend per batch profile."""

    __infer_dynamic_dims_fields__ = ["attention_metadata"]

    def __init__(
        self,
        fd_config: FDConfig,
        kv_num_heads: int,
        num_heads: int,
        head_dim: int,
        encoder_block_shape_q: int = -1,
        decoder_block_shape_q: int = -1,
    ):
        super().__init__()
        self._fd_config = fd_config
        self._kv_num_heads = kv_num_heads
        self._num_heads = num_heads
        self._head_dim = head_dim
        init_args = (fd_config, kv_num_heads, num_heads, head_dim, encoder_block_shape_q, decoder_block_shape_q)

        self.append_backend = AppendAttentionBackend(*init_args)
        self.flash_backend = FlashAttentionBackend(*init_args)
        self.candidates = [self.append_backend, self.flash_backend]

        sm_version = get_sm_version()
        if sm_version >= 90:
            try:
                from fastdeploy.model_executor.layers.attention.flash_mask_attn_backend import (
                    FlashMaskAttentionBackend,
                )
                self.flash_mask_backend = FlashMaskAttentionBackend(*init_args)
                self.candidates.append(self.flash_mask_backend)
            except Exception:
                self.flash_mask_backend = None
        else:
            self.flash_mask_backend = None

        # Build bins based on scheduler config
        max_batched_tokens = fd_config.scheduler_config.max_num_batched_tokens
        max_num_seqs = fd_config.scheduler_config.max_num_seqs
        self._tokens_bins = make_tokens_bins(max_batched_tokens)
        self._count_bins = make_count_bins(max_num_seqs)

        self.autotune_cache = AutotuneCache(self.candidates, self._tokens_bins, self._count_bins)
        self._current_backend = None
        self._cache_path = self._get_cache_path(num_heads, head_dim, max_batched_tokens, max_num_seqs)

        # Load persisted cache (unless recache requested)
        name_to_backend = {type(b).__name__: b for b in self.candidates}
        if os.getenv("FD_AUTO_ATTN_RECACHE", "0") != "1":
            self.autotune_cache.load(self._cache_path, name_to_backend)

        logger.info(
            f"AutoAttentionBackend initialized with candidates: "
            f"{[type(b).__name__ for b in self.candidates]}, "
            f"tokens_bins={self._tokens_bins}, count_bins={self._count_bins}, "
            f"loaded {len(self.autotune_cache.cache)} cached entries."
        )

    def _get_cache_path(self, num_heads, head_dim, max_batched_tokens, max_num_seqs) -> str:
        gpu_name = paddle.device.cuda.get_device_name(0).replace(" ", "_")
        return os.path.expanduser(
            f"~/.fastdeploy/autotune_attn_{gpu_name}_{num_heads}_{head_dim}_t{max_batched_tokens}_b{max_num_seqs}.json"
        )

    # ------------------------------------------------------------------
    # Offline benchmark (called during server warmup, before serving)
    # ------------------------------------------------------------------
    def run_benchmark(self, fd_config: FDConfig):
        """Run offline benchmark for all key combinations using synthetic data."""
        tokens_bins = self._tokens_bins
        count_bins = self._count_bins
        total_keys = len(tokens_bins) * len(count_bins)

        # Force recache if requested
        if os.getenv("FD_AUTO_ATTN_RECACHE", "0") == "1":
            self.autotune_cache.cache.clear()
            try:
                os.remove(self._cache_path)
            except FileNotFoundError:
                pass
            logger.info("AutoAttn: FD_AUTO_ATTN_RECACHE=1, clearing existing cache.")

        if len(self.autotune_cache.cache) >= total_keys:
            logger.info(f"AutoAttn: cache already covers all {total_keys} keys, skipping benchmark.")
            return

        logger.info(f"AutoAttn: running offline benchmark for {total_keys} key combinations "
                    f"(tokens_bins={tokens_bins}, count_bins={count_bins})...")
        start_time = time.perf_counter()
        benchmarked = 0

        for tokens in tokens_bins:
            for count in count_bins:
                key = (tokens, count)
                if self.autotune_cache.get_best(key) is not None:
                    continue

                prefill_per_seq = max(1, tokens // count)
                try:
                    fm, layer, q, k, v, qkv = self._build_synthetic_inputs(
                        fd_config, prefill_per_seq, tokens, count
                    )
                except Exception as e:
                    logger.warning(f"AutoAttn: skip key={key}, build failed: {e}")
                    self.autotune_cache.cache[key] = self.append_backend
                    continue

                if fm is None:
                    self.autotune_cache.cache[key] = self.append_backend
                    continue

                try:
                    best = self.autotune_cache.tune(
                        key, q, k, v, qkv, None, None, layer, fm
                    )
                except Exception as e:
                    logger.warning(f"AutoAttn: skip key={key}, tune failed: {e}")
                    self.autotune_cache.cache[key] = self.append_backend
                    best = self.append_backend

                benchmarked += 1
                # Free GPU memory
                del fm, layer, q, k, v, qkv
                paddle.device.cuda.empty_cache()
                if benchmarked % 5 == 0:
                    logger.info(f"AutoAttn: benchmarked {benchmarked}/{total_keys} keys...")

        self.autotune_cache.save(self._cache_path)
        elapsed = time.perf_counter() - start_time
        logger.info(f"AutoAttn: benchmark done in {elapsed:.1f}s, saved to {self._cache_path}")

    def _build_synthetic_inputs(self, fd_config, prefill_per_seq, total_prefill_tokens, num_prefill_seqs):
        """Construct synthetic forward_meta + tensors for a given batch profile."""
        from fastdeploy.model_executor.layers.attention.append_attn_backend import allocate_launch_related_buffer
        from fastdeploy.model_executor.layers.rotary_embedding import get_rope
        from fastdeploy.model_executor.ops.gpu import get_padding_offset

        mc = fd_config.model_config
        block_size = fd_config.cache_config.block_size
        head_dim = mc.head_dim
        kv_num_heads = self._kv_num_heads
        num_heads = self._num_heads

        batch_size = num_prefill_seqs

        # Cap batch size to avoid OOM during benchmark
        max_batch = int(os.getenv("FD_AUTO_ATTN_BENCH_MAX_BATCH", "64"))
        if batch_size > max_batch:
            num_prefill_seqs = max_batch
            batch_size = max_batch
            prefill_per_seq = total_prefill_tokens // num_prefill_seqs

        # Build seq_lens (pure prefill batch for benchmark)
        seq_lens_encoder = paddle.full([batch_size], prefill_per_seq, dtype="int32")
        seq_lens_decoder = paddle.zeros([batch_size], dtype="int32")
        seq_lens_this_time = seq_lens_encoder.clone()

        # Block tables & KV cache (only 1 layer needed for benchmark)
        alloc_blocks = (prefill_per_seq + block_size - 1) // block_size + 1
        max_blocks_per_seq = (mc.max_model_len + block_size - 1) // block_size
        total_blocks = alloc_blocks * batch_size
        cache_shape = (total_blocks, kv_num_heads, block_size, head_dim)

        caches = [
            paddle.zeros(cache_shape, dtype="bfloat16"),
            paddle.zeros(cache_shape, dtype="bfloat16"),
        ]

        block_tables = paddle.zeros([batch_size, max_blocks_per_seq], dtype="int32")
        for i in range(batch_size):
            for j in range(alloc_blocks):
                block_tables[i, j] = i * alloc_blocks + j

        # Rotary embeddings
        pos_ids = paddle.arange(mc.max_model_len).reshape([1, -1])
        rotary_embs = get_rope(
            rotary_dim=head_dim,
            position_ids=pos_ids,
            base=mc.rope_theta if mc.rope_theta else 10000.0,
            model_config=mc,
            partial_rotary_factor=getattr(mc, "partial_rotary_factor", 1.0),
        )

        # Padding offset
        token_num = int(seq_lens_this_time.sum().item())
        input_ids = paddle.zeros([batch_size, mc.max_model_len], dtype="int64")
        ids_remove_padding, batch_id_per_token, cu_seqlens_q, cu_seqlens_k = get_padding_offset(
            input_ids, seq_lens_this_time, seq_lens_encoder, seq_lens_decoder, None, token_num
        )

        # Allocate launch-related buffers needed by append/flash backends
        encoder_block_shape_q = self.append_backend.encoder_block_shape_q
        decoder_block_shape_q = self.append_backend.decoder_block_shape_q
        launch_buffers = allocate_launch_related_buffer(
            max_batch_size=batch_size,
            max_model_len=mc.max_model_len,
            encoder_block_shape_q=encoder_block_shape_q,
            decoder_block_shape_q=decoder_block_shape_q,
            decoder_step_token_num=1,
            num_heads=num_heads,
            kv_num_heads=kv_num_heads,
            block_size=block_size,
        )

        # Forward mode stub
        forward_mode = types.SimpleNamespace(
            is_mixed=lambda: True,
            is_decode=lambda: False,
            is_native=lambda: False,
            is_extend=lambda: False,
        )

        # Build forward_meta as SimpleNamespace
        fm = types.SimpleNamespace(
            seq_lens_encoder=seq_lens_encoder,
            seq_lens_decoder=seq_lens_decoder,
            seq_lens_this_time=seq_lens_this_time,
            ids_remove_padding=ids_remove_padding,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            batch_id_per_token=batch_id_per_token,
            block_tables=block_tables,
            caches=caches,
            rotary_embs=rotary_embs,
            attn_backend=self.flash_backend,
            attn_mask=None,
            attn_mask_offsets=None,
            is_dummy_or_profile_run=True,
            forward_mode=forward_mode,
            rope_already_applied=False,
            **launch_buffers,
        )

        # Synthetic Q/K/V tensors
        q = paddle.randn([token_num, num_heads, head_dim], dtype="bfloat16")
        k = paddle.randn([token_num, kv_num_heads, head_dim], dtype="bfloat16")
        v = paddle.randn([token_num, kv_num_heads, head_dim], dtype="bfloat16")
        qkv = paddle.concat([
            q.reshape([token_num, -1]),
            k.reshape([token_num, -1]),
            v.reshape([token_num, -1]),
        ], axis=-1)

        # Fake layer
        layer = types.SimpleNamespace(
            layer_id=0,
            num_heads=num_heads,
            kv_num_heads=kv_num_heads,
            head_dim=head_dim,
            cache_quant_type_str="none",
            qkv_bias=None,
            qkv_scale=None,
            linear_shift=None,
            linear_smooth=None,
            use_neox_rotary_style=True,
            rms_norm_eps=1e-6,
            sliding_window=0,
            use_qk_norm=False,
            out_scale=-1.0,
        )

        # Init metadata for all candidates
        for backend in self.candidates:
            try:
                backend.init_attention_metadata(fm)
            except Exception:
                pass

        return fm, layer, q, k, v, qkv

    @property
    def attention_metadata(self):
        return self.flash_backend.attention_metadata

    @attention_metadata.setter
    def attention_metadata(self, value):
        self.flash_backend.attention_metadata = value

    def get_kv_cache_shape(self, max_num_blocks, kv_cache_quant_type=None):
        return self.append_backend.get_kv_cache_shape(max_num_blocks, kv_cache_quant_type)

    def init_attention_metadata(self, forward_meta: ForwardMeta):
        # Each backend needs its own metadata type
        self.append_backend.init_attention_metadata(forward_meta)
        self.flash_backend.init_attention_metadata(forward_meta)
        if self.flash_mask_backend is not None:
            self.flash_mask_backend.init_attention_metadata(forward_meta)

    def forward_mixed(
        self,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        qkv: paddle.Tensor,
        compressed_kv: paddle.Tensor,
        k_pe: paddle.Tensor,
        layer: Attention,
        forward_meta: ForwardMeta,
    ) -> paddle.Tensor:
        if layer.layer_id == 0:
            # During CUDAGraph capture / profile, skip dynamic selection
            if getattr(forward_meta, 'is_dummy_or_profile_run', False):
                self._current_backend = self.append_backend
            else:
                self._current_backend = self._select(forward_meta)
        return self._current_backend.forward_mixed(q, k, v, qkv, compressed_kv, k_pe, layer, forward_meta)

    def forward_decode(
        self,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        qkv: paddle.Tensor,
        compressed_kv: paddle.Tensor,
        k_pe: paddle.Tensor,
        layer: Attention,
        forward_meta: ForwardMeta,
    ) -> paddle.Tensor:
        return self.append_backend.forward_decode(q, k, v, qkv, compressed_kv, k_pe, layer, forward_meta)

    def _select(self, forward_meta):
        profile = self._extract_profile(forward_meta)

        # Pure decode -> always append
        if profile.total_prefill_tokens == 0:
            return self.append_backend

        key = self.autotune_cache.compute_key(profile)

        # Cache hit
        cached = self.autotune_cache.get_best(key)
        if cached is not None:
            logger.info(
                f"AutoAttn: key={key} -> {type(cached).__name__} "
                f"(prefill_tokens={profile.total_prefill_tokens}, count={profile.num_prefill_seqs})"
            )
            return cached

        # Fallback for uncovered keys
        fallback = self._heuristic_select(profile)
        logger.info(f"AutoAttn: key={key} -> {type(fallback).__name__} (heuristic)")
        return fallback

    def _extract_profile(self, forward_meta) -> BatchProfile:
        seq_lens_encoder = forward_meta.seq_lens_encoder

        # Use paddle ops to avoid sync during CUDAGraph capture
        prefill_lens = seq_lens_encoder[seq_lens_encoder > 0]
        num_prefill = prefill_lens.shape[0]
        if num_prefill > 0:
            total_prefill_tokens = int(prefill_lens.sum())
        else:
            total_prefill_tokens = 0

        return BatchProfile(
            total_prefill_tokens=total_prefill_tokens,
            num_prefill_seqs=num_prefill,
        )

    def _heuristic_select(self, profile: BatchProfile):
        if profile.total_prefill_tokens <= 256:
            return self.append_backend
        if profile.total_prefill_tokens >= 4096 and profile.num_prefill_seqs <= 4:
            return self.flash_backend
        return self.append_backend
