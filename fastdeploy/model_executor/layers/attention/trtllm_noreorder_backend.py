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
from dataclasses import dataclass
from typing import TYPE_CHECKING

import paddle

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.attention.base_attention_backend import (
    AttentionBackend,
    AttentionMetadata,
)
from fastdeploy.model_executor.layers.attention.ops import (
    append_attention,
    get_block_shape_and_split_kv_block,
    gqa_rope_write_cache,
    pre_cache_len_concat,
)
from fastdeploy.model_executor.layers.attention.utils import init_rank_and_device_id

if TYPE_CHECKING:
    from fastdeploy.model_executor.forward_meta import ForwardMeta

from fastdeploy.platforms import current_platform

paddle.compat.enable_torch_proxy(scope={"flashinfer"})
from flashinfer.prefill import trtllm_batch_context_with_kv_cache


@dataclass
class TrtllmNoReorderAttentionMetadata(AttentionMetadata):
    """
    Attention metadata for TrtllmNoReorder backend.
    No prefill/decode reorder required.
    """

    num_running_requests: int = 0
    num_decode_tokens: int = 0
    num_prefill_tokens: int = 0
    total_tokens: int = 0
    is_pure_decode: bool = False

    cu_seqlens_k: paddle.Tensor = None

    pre_cache_batch_ids = None
    pre_cache_tile_ids_per_batch = None
    pre_cache_num_blocks_cpu = None
    kv_token_num_cpu = None

    _fuse_kernel_compute_dtype: str = "bf16"
    _dtype: paddle.dtype = paddle.bfloat16

    kv_signal_data_list: list = None

    # context kernel params
    block_tables: paddle.Tensor = None
    seq_lens_kv: paddle.Tensor = None
    max_q_len: int = 0
    max_kv_len: int = 0
    cum_seq_lens_q: paddle.Tensor = None
    cum_seq_lens_kv: paddle.Tensor = None
    batch_size: int = 0


class TrtllmNoReorderAttentionBackend(AttentionBackend):
    """
    TrtllmNoReorder attention backend using flashinfer's trtllm kernels.

    Key design:
    - enable_ids_reorder = False (no prefill/decode reorder needed)
    - Prefill / mixed batch: gqa_rope_write_cache + trtllm_batch_context_with_kv_cache
      (context kernel supports variable q_len per request)
    - Pure decode batch: append_attention (fused RoPE + KV write + attention)
    - Best suited for large batch / long sequence prefill scenarios (total_tokens > 32K)
    """

    __infer_dynamic_dims_fields__ = ["attention_metadata"]
    attention_metadata: TrtllmNoReorderAttentionMetadata
    flash_attn_func: callable = None
    use_output: bool = True
    enable_ids_reorder: bool = False

    def __init__(
        self,
        fd_config: FDConfig,
        kv_num_heads: int,
        num_heads: int,
        head_dim: int,
        encoder_block_shape_q: int = -1,
        decoder_block_shape_q: int = -1,
        sliding_window: int | None = None,
    ):
        super().__init__()
        self.attention_metadata: TrtllmNoReorderAttentionMetadata = None
        self.max_seq_len = fd_config.model_config.max_model_len
        self.causal = getattr(fd_config.model_config, "causal", True)

        self.kv_num_heads = kv_num_heads
        self.num_heads = num_heads
        self.group_size: int = self.num_heads // self.kv_num_heads
        self.head_dim = fd_config.model_config.head_dim
        self.attn_outputsize_tp = self.num_heads * self.head_dim
        self.block_size = fd_config.cache_config.block_size
        self.num_layers: int = fd_config.model_config.num_hidden_layers
        self.encoder_block_shape_q: int = encoder_block_shape_q
        self.decoder_block_shape_q: int = decoder_block_shape_q

        self.speculative_method = fd_config.speculative_config.method
        self.use_speculate = self.speculative_method is not None
        self.speculate_max_draft_token_num = fd_config.speculative_config.num_speculative_tokens
        self.keep_pd_step_flag: bool = fd_config.speculative_config.model_type == "mtp"
        self.num_layers_draft_model: int = int(fd_config.speculative_config.method in ["mtp"])

        self.pd_disaggregation_mode: str = fd_config.parallel_config.pd_disaggregation_mode

        self.start_layer_index: int = fd_config.model_config.start_layer_index

        self.rank, self.device_id = init_rank_and_device_id(fd_config)

        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.window_left = self.sliding_window[0] if self.sliding_window is not None else -1

        self.rope_3d: bool = getattr(fd_config, "enable_rope_3d_runtime", False)
        self.max_partition_size: int = int(os.getenv("FLAGS_max_partition_size", 1024))
        self.zero_seq_enc_lens_for_decode = paddle.zeros(
            shape=[fd_config.scheduler_config.max_num_seqs, 1], dtype=paddle.int32
        )
        self.block_kv_indptr_gpu = paddle.zeros(
            shape=[fd_config.scheduler_config.max_num_seqs + 1], dtype=paddle.int32
        )
        self.workspace_buffer = paddle.empty(394 * 1024 * 1024, dtype=paddle.int8)

    def get_attention_meta(self):
        """Get the current attention metadata."""
        return self.attention_metadata

    def get_kv_cache_shape(
        self,
        max_num_blocks: int,
        kv_cache_quant_type: str = None,
    ):
        """Calculate kv cache shape."""
        key_cache_shape = [max_num_blocks, self.kv_num_heads, self.block_size, self.head_dim]
        value_cache_shape = key_cache_shape
        return key_cache_shape, value_cache_shape

    def init_attention_metadata(self, forward_meta: ForwardMeta):
        """Initialize attention metadata from forward_meta."""
        metadata = TrtllmNoReorderAttentionMetadata()

        num_running_requests = forward_meta.seq_lens_this_time.shape[0]
        metadata.num_running_requests = num_running_requests
        metadata.batch_size = num_running_requests

        seq_lens_encoder = forward_meta.seq_lens_encoder[:num_running_requests]
        seq_lens_this_time = forward_meta.seq_lens_this_time[:num_running_requests]
        seq_lens_decoder = forward_meta.seq_lens_decoder[:num_running_requests]

        total_tokens = paddle.sum(seq_lens_this_time).item()
        metadata.total_tokens = total_tokens

        # decode: seq_lens_encoder == 0; prefill: seq_lens_encoder > 0
        decode_mask = (seq_lens_encoder.squeeze(-1) == 0)
        num_decode_tokens = paddle.sum(seq_lens_this_time.squeeze(-1) * decode_mask.astype("int32")).item()
        num_prefill_tokens = total_tokens - num_decode_tokens
        metadata.num_decode_tokens = num_decode_tokens
        metadata.num_prefill_tokens = num_prefill_tokens
        metadata.is_pure_decode = (num_prefill_tokens == 0)

        # Common ops for write cache (block shape split etc.)
        get_block_shape_and_split_kv_block(
            forward_meta.seq_lens_encoder,
            forward_meta.seq_lens_decoder,
            forward_meta.seq_lens_this_time,
            forward_meta.decoder_batch_ids,
            forward_meta.decoder_tile_ids_per_batch,
            forward_meta.decoder_num_blocks_cpu,
            forward_meta.decoder_num_blocks_device,
            forward_meta.decoder_chunk_size_device,
            forward_meta.max_len_tensor_cpu,
            forward_meta.encoder_batch_ids,
            forward_meta.encoder_tile_ids_per_batch,
            forward_meta.encoder_num_blocks_x_cpu,
            forward_meta.kv_batch_ids,
            forward_meta.kv_tile_ids_per_batch,
            forward_meta.kv_num_blocks_x_cpu,
            self.encoder_block_shape_q,
            self.decoder_block_shape_q,
            self.group_size,
            self.block_size,
        )

        (
            metadata.cu_seqlens_k,
            metadata.pre_cache_batch_ids,
            metadata.pre_cache_tile_ids_per_batch,
            metadata.pre_cache_num_blocks_cpu,
            metadata.kv_token_num_cpu,
        ) = pre_cache_len_concat(
            forward_meta.seq_lens_encoder,
            forward_meta.seq_lens_decoder,
            forward_meta.seq_lens_this_time,
            forward_meta.max_len_tensor_cpu[2],
            self.block_size,
        )

        metadata.kv_signal_data_list = [None] * self.num_layers

        if metadata._dtype == "bfloat16":
            metadata._fuse_kernel_compute_dtype = "bf16"
        elif metadata._dtype == "float16":
            metadata._fuse_kernel_compute_dtype = "fp16"
        elif metadata._dtype == "float32":
            metadata._fuse_kernel_compute_dtype = "fp32"

        # Total kv length per request = seq_lens_decoder + seq_lens_this_time
        total_seq_len = seq_lens_decoder + seq_lens_this_time  # [num_running, 1]

        if not metadata.is_pure_decode:
            # Context kernel params (for prefill or mixed batch)
            metadata.cum_seq_lens_q = forward_meta.cu_seqlens_q[:num_running_requests + 1]

            # cum_seq_lens_kv: prefix sum of block counts (block-level indptr)
            num_blocks = (total_seq_len + (self.block_size - 1)) // self.block_size
            self.block_kv_indptr_gpu[1:num_running_requests + 1] = paddle.cumsum(num_blocks)
            metadata.cum_seq_lens_kv = self.block_kv_indptr_gpu[:num_running_requests + 1]

            metadata.block_tables = forward_meta.block_tables[:num_running_requests]
            metadata.seq_lens_kv = total_seq_len.squeeze(-1)

            metadata.max_q_len = paddle.max(seq_lens_this_time).item()
            metadata.max_kv_len = paddle.max(total_seq_len).item()

        self.attention_metadata: AttentionMetadata = metadata

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
    ):
        """Forward pass for mixed prefill + decode batch."""
        metadata = self.attention_metadata
        workspace_buffer = self.workspace_buffer
        bmm1_scale = float(1.0 / (self.head_dim ** 0.5))

        if metadata.is_pure_decode:
            # Pure decode: use append_attention (fused RoPE + KV write + attention)
            # Same approach as flash_attn_backend / append_attn_backend for decode
            cache_k = forward_meta.caches[2 * layer.layer_id]
            cache_v = forward_meta.caches[2 * layer.layer_id + 1]

            res = append_attention(
                qkv,
                cache_k,
                cache_v,
                forward_meta.seq_lens_encoder,
                forward_meta.seq_lens_decoder,
                forward_meta.seq_lens_this_time,
                forward_meta.batch_id_per_token,
                forward_meta.cu_seqlens_q,
                forward_meta.block_tables,
                forward_meta.encoder_batch_ids,
                forward_meta.encoder_tile_ids_per_batch,
                forward_meta.encoder_num_blocks_x_cpu,
                forward_meta.kv_batch_ids,
                forward_meta.kv_tile_ids_per_batch,
                forward_meta.kv_num_blocks_x_cpu,
                forward_meta.decoder_batch_ids,
                forward_meta.decoder_tile_ids_per_batch,
                forward_meta.decoder_num_blocks_cpu,
                forward_meta.max_len_tensor_cpu,
                forward_meta.rotary_embs,
                forward_meta.attn_mask,
                layer.qkv_bias,
                layer.qkv_scale,
                getattr(layer, "cache_k_scale", None),
                getattr(layer, "cache_v_scale", None),
                getattr(layer, "cache_k_out_scale", None),
                getattr(layer, "cache_v_out_scale", None),
                getattr(layer, "cache_k_zp", None),
                getattr(layer, "cache_v_zp", None),
                layer.linear_shift,
                layer.linear_smooth,
                forward_meta.attn_mask_offsets,
                metadata.kv_signal_data_list[layer.layer_id],
                getattr(layer, "q_norm_weight", None),
                getattr(layer, "k_norm_weight", None),
                getattr(layer, "sinks", None),
                getattr(layer, "rms_norm_eps", 1e-6),
                metadata._fuse_kernel_compute_dtype,
                getattr(layer, "cache_quant_type_str", "none"),
                layer.use_neox_rotary_style,
                self.rope_3d,
                self.max_seq_len,
                getattr(layer, "quant_max_bound", 0.0),
                getattr(layer, "quant_min_bound", 0.0),
                getattr(layer, "out_scale", -1.0),
                self.encoder_block_shape_q,
                self.decoder_block_shape_q,
                self.max_partition_size,
                self.max_seq_len,
                self.speculate_max_draft_token_num + 1,
                self.causal,
                self.speculative_method is not None,
            )
            return res
        else:
            # Prefill / mixed batch: gqa_rope_write_cache + trtllm context kernel
            q, _, _, _ = gqa_rope_write_cache(
                qkv,
                forward_meta.caches[2 * layer.layer_id],
                forward_meta.caches[2 * layer.layer_id + 1],
                forward_meta.cu_seqlens_q,
                metadata.cu_seqlens_k,
                forward_meta.rotary_embs,
                forward_meta.seq_lens_this_time,
                forward_meta.seq_lens_encoder,
                forward_meta.seq_lens_decoder,
                forward_meta.batch_id_per_token,
                forward_meta.block_tables,
                forward_meta.kv_batch_ids,
                forward_meta.kv_tile_ids_per_batch,
                forward_meta.kv_num_blocks_x_cpu,
                metadata.pre_cache_batch_ids,
                metadata.pre_cache_tile_ids_per_batch,
                metadata.pre_cache_num_blocks_cpu,
                getattr(layer, "q_norm_weight", None),
                getattr(layer, "k_norm_weight", None),
                getattr(layer, "cache_k_scale", None),
                getattr(layer, "cache_v_scale", None),
                getattr(layer, "cache_k_out_scale", None),
                getattr(layer, "cache_v_out_scale", None),
                getattr(layer, "cache_k_zp", None),
                getattr(layer, "cache_v_zp", None),
                None,
                metadata.kv_token_num_cpu[0].item(),
                self.max_seq_len,
                getattr(layer, "rms_norm_eps", 1e-6),
                layer.use_neox_rotary_style,
                getattr(layer, "cache_quant_type_str", "none"),
                self.rope_3d,
            )

            # Allocate output buffer
            output = paddle.empty([metadata.total_tokens, self.num_heads, self.head_dim], dtype=q.dtype)

            # Context kernel: supports variable q_len per request
            all_q = q[:metadata.total_tokens].reshape([-1, self.num_heads, self.head_dim])

            trtllm_batch_context_with_kv_cache(
                query=all_q,
                kv_cache=(forward_meta.caches[2 * layer.layer_id], forward_meta.caches[2 * layer.layer_id + 1]),
                workspace_buffer=workspace_buffer,
                block_tables=metadata.block_tables,
                seq_lens=metadata.seq_lens_kv,
                max_q_len=metadata.max_q_len,
                max_kv_len=metadata.max_kv_len,
                bmm1_scale=bmm1_scale,
                bmm2_scale=1.0,
                batch_size=metadata.batch_size,
                cum_seq_lens_q=metadata.cum_seq_lens_q,
                cum_seq_lens_kv=metadata.cum_seq_lens_kv,
                out=output[:metadata.total_tokens],
            )

            return output.reshape([output.shape[0], -1])
