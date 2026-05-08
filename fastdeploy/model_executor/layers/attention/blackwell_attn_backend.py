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

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import paddle

try:
    from paddleformers.utils.log import logger
except ImportError:
    logger = logging.getLogger(__name__)

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.attention.base_attention_backend import (
    AttentionBackend,
    AttentionMetadata,
)
from fastdeploy.model_executor.layers.attention.utils import init_rank_and_device_id

if TYPE_CHECKING:
    from fastdeploy.model_executor.forward_meta import ForwardMeta

try:
    import blackwell_ops
except ImportError as e:
    logger.warning(f"blackwell_ops not available: {e}")
    blackwell_ops = None


@dataclass
class BlackwellAttentionMetadata(AttentionMetadata):
    """
    Per-forward metadata for BlackwellAttentionBackend.

    All CPU-side work (.numpy, .tolist, .item) is done in init_attention_metadata()
    so that forward_mixed() contains only pure GPU ops and is CUDA Graph compatible.

    FastDeploy convention:
      seq_lens_encoder[i] = new tokens this step (prompt len for prefill, 1 for decode)
      seq_lens_decoder[i] = cached tokens already in KV cache (0 for prefill, >0 for decode)

    Dispatch: dec==0 -> prefill path, dec>0 -> decode path.
    """
    # ── Prefill sub-batch (None when no prefill this step) ────────────────────
    pf_ei: Optional[paddle.Tensor] = None       # [pf_et]  int64  token indices into packed input
    pf_cu_q: Optional[paddle.Tensor] = None     # [npf+1]  int32  cumulative Q seqlens
    pf_sl_enc: Optional[paddle.Tensor] = None   # [npf]    int32  encoder seq lens
    pf_sl_dec: Optional[paddle.Tensor] = None   # [npf]    int32  zeros (no cache for prefill)
    pf_bidx: Optional[paddle.Tensor] = None     # [npf]    int64  batch indices
    pf_mask: Optional[paddle.Tensor] = None     # [pf_et]  int32  causal position per token
    pf_max_enc: int = 0                         # max prefill seq len (Python int)
    pf_et: int = 0                              # total prefill tokens

    # ── Decode sub-batch (None when no decode this step) ──────────────────────
    dc_di: Optional[paddle.Tensor] = None       # [dc_dt]    int64  token indices into packed input
    dc_cu_q: Optional[paddle.Tensor] = None     # [ndc+1]    int32  cumulative Q seqlens
    dc_sl_kv_enc: Optional[paddle.Tensor] = None  # [ndc]    int32  zeros (blackwell decode convention)
    dc_sl_kv_dec: Optional[paddle.Tensor] = None  # [ndc]    int32  cached history token counts
    dc_bidx: Optional[paddle.Tensor] = None     # [ndc]      int64  batch indices
    dc_act_cu_k: Optional[paddle.Tensor] = None # [ndc+1]    int32  actual cumulative K seqlens
    dc_sl_kv: Optional[paddle.Tensor] = None    # [ndc_pad+4] int32 FastDivmod packed lens
    dc_max_seq_k: int = 0                       # max total KV len across decode requests
    dc_max_new: int = 0                         # max new tokens this step (1 for standard decode)
    dc_dt: int = 0                              # total decode tokens


class BlackwellAttentionBackend(AttentionBackend):
    """
    Attention backend using blackwell_ops (Blackwell / B200 custom CUDA kernels).

    Activated by: FD_ATTENTION_BACKEND=BLACKWELL_ATTN
    """

    def __init__(
        self,
        fd_config: FDConfig,
        kv_num_heads: int,
        num_heads: int,
        head_dim: int,
        encoder_block_shape_q: int = -1,  # accepted for API compat, unused by blackwell_ops
        decoder_block_shape_q: int = -1,  # accepted for API compat, unused by blackwell_ops
    ):
        super().__init__()
        if blackwell_ops is None:
            raise RuntimeError(
                "BlackwellAttentionBackend requires blackwell_ops. "
                "Build with: cd blackwell && python setup.py install"
            )
        self.kv_num_heads = kv_num_heads
        self.num_heads = num_heads
        self.head_dim = fd_config.model_config.head_dim
        self.block_size = fd_config.cache_config.block_size
        if self.block_size != 128:
            raise ValueError(
                f"BlackwellAttentionBackend requires block_size=128 "
                f"(blackwell_ops kernels hard-code kBlockSize=128), "
                f"got block_size={self.block_size}. "
                f"Add --block-size 128 to your launch command."
            )
        self.max_seq_len = fd_config.model_config.max_model_len
        self.num_layers = fd_config.model_config.num_hidden_layers
        self.rank, self.device_id = init_rank_and_device_id(fd_config)

        # Pre-allocate identity rotary (cos=1, sin=0) for decode path
        identity_rotary = paddle.zeros([2, self.max_seq_len, self.head_dim // 2], dtype="float32")
        identity_rotary[0, :, :] = 1.0  # cos = 1
        self._identity_rotary = identity_rotary

    def get_kv_cache_shape(self, max_num_blocks: int, kv_cache_quant_type: str = None):
        shape = [max_num_blocks, self.kv_num_heads, self.block_size, self.head_dim]
        return shape, shape

    def get_attention_meta(self):
        return self.attention_metadata

    def _rotary_embs_for_encoder(self, rotary_embs: paddle.Tensor) -> paddle.Tensor:
        """
        Encoder kernel (neox native) expects [2, max_seq_len, rotary_dim/2].
        FastDeploy produces [2, 1, max_seq_len, 1, rotary_dim/2]. Just squeeze.
        """
        if rotary_embs.ndim == 5:
            rotary_embs = rotary_embs.squeeze([1, 3])  # [2, max_seq_len, rotary_dim/2]
        return rotary_embs

    def _rotary_embs_for_decoder(self, rotary_embs: paddle.Tensor) -> paddle.Tensor:
        """
        Decoder kernel (GPT-J style) expects [2, max_seq_len, head_dim/2].
        When partial_rotary_factor < 1.0 (e.g. GLM4: rotary_dim=64, head_dim=128),
        FastDeploy only stores rotary_dim/2=32 values. Pad to head_dim/2=64:
          cos pad with 1.0 (identity), sin pad with 0.0 (identity).
        """
        if rotary_embs.ndim == 5:
            rotary_embs = rotary_embs.squeeze([1, 3])  # [2, max_seq_len, rotary_dim/2]

        target_dim = self.head_dim // 2
        actual_dim = rotary_embs.shape[-1]
        if actual_dim < target_dim:
            pad = target_dim - actual_dim
            max_seq_len = rotary_embs.shape[1]
            cos_pad = paddle.ones([1, max_seq_len, pad], dtype=rotary_embs.dtype)
            sin_pad = paddle.zeros([1, max_seq_len, pad], dtype=rotary_embs.dtype)
            rotary_embs = paddle.concat([
                paddle.concat([rotary_embs[0:1], cos_pad], axis=-1),
                paddle.concat([rotary_embs[1:2], sin_pad], axis=-1),
            ], axis=0)

        return rotary_embs  # [2, max_seq_len, head_dim/2]

    # ------------------------------------------------------------------
    # init_attention_metadata: classify prefill/decode on GPU, no D2H bulk copy
    # ------------------------------------------------------------------
    def init_attention_metadata(self, forward_meta: "ForwardMeta"):
        meta = BlackwellAttentionMetadata()

        seq_enc   = forward_meta.seq_lens_encoder    # [bsz] int32
        seq_dec   = forward_meta.seq_lens_decoder    # [bsz] int32
        seq_now   = forward_meta.seq_lens_this_time  # [bsz] int32
        batch_ids = forward_meta.batch_id_per_token  # [total_tokens] int32

        # ── GPU classification masks (no sync) ────────────────────────────────
        is_pf = (seq_dec == 0) & (seq_now > 0)  # [bsz] bool
        is_dc = seq_dec > 0                      # [bsz] bool

        # Scalar counts — minimal syncs needed for dynamic branching and buffer alloc
        npf = int(is_pf.cast("int32").sum().item())
        ndc = int(is_dc.cast("int32").sum().item())

        # ── Prefill metadata ──────────────────────────────────────────────────
        if npf > 0:
            pf_bidx = paddle.nonzero(is_pf).flatten().cast("int64")  # [npf]

            # Token indices: tokens whose batch item is a prefill request
            tok_is_pf = paddle.gather(is_pf.cast("int32"), batch_ids.cast("int64"))
            pf_ei = paddle.nonzero(tok_is_pf.cast("bool")).flatten().cast("int64")  # [pf_et]

            pf_enc_lens = paddle.gather(seq_now, pf_bidx.cast("int32")).cast("int32")  # [npf]
            pf_cu = paddle.concat([
                paddle.zeros([1], dtype="int32"),
                paddle.cumsum(pf_enc_lens).cast("int32"),
            ])

            meta.pf_et      = int(pf_enc_lens.sum().item())

            # Causal mask: 1-indexed position of each token within its sequence
            # e.g. for two seqs of len 3 and 5: [1,2,3, 1,2,3,4,5]
            pf_positions = paddle.arange(meta.pf_et, dtype="int32")
            batch_of_token = paddle.searchsorted(pf_cu[1:], pf_positions, right=True)
            token_seq_start = paddle.gather(pf_cu, batch_of_token.cast("int64"))
            pf_mask_t = (pf_positions - token_seq_start + 1).cast("int32")  # [pf_et]
            meta.pf_max_enc = int(pf_enc_lens.max().item())
            meta.pf_ei      = pf_ei
            meta.pf_cu_q    = pf_cu
            meta.pf_sl_enc  = pf_enc_lens
            meta.pf_sl_dec  = paddle.zeros([npf], dtype="int32")
            meta.pf_bidx    = pf_bidx
            meta.pf_mask    = pf_mask_t

        # ── Decode metadata ───────────────────────────────────────────────────
        if ndc > 0:
            dc_bidx = paddle.nonzero(is_dc).flatten().cast("int64")  # [ndc]

            # Token indices: tokens whose batch item is a decode request
            tok_is_dc = paddle.gather(is_dc.cast("int32"), batch_ids.cast("int64"))
            dc_di = paddle.nonzero(tok_is_dc.cast("bool")).flatten().cast("int64")  # [dc_dt]

            dc_new_lens  = paddle.gather(seq_now, dc_bidx.cast("int32")).cast("int32")  # [ndc]
            dc_hist_lens = paddle.gather(seq_dec, dc_bidx.cast("int32")).cast("int32")  # [ndc]

            dc_cu = paddle.concat([
                paddle.zeros([1], dtype="int32"),
                paddle.cumsum(dc_new_lens).cast("int32"),
            ])
            kv_per_b = dc_hist_lens + dc_new_lens
            cu_kd = paddle.concat([
                paddle.zeros([1], dtype="int32"),
                paddle.cumsum(kv_per_b).cast("int32"),
            ])

            dc_sl_kv_enc = paddle.zeros([ndc], dtype="int32")
            act_cu_k     = paddle.zeros([ndc + 1], dtype="int32")
            sl_kv        = paddle.zeros([(ndc + 3) // 4 * 4 + 4], dtype="int32")

            max_token_cpu = blackwell_ops.flash_attn_get_qk_token(
                dc_sl_kv_enc, dc_hist_lens,
                dc_cu, cu_kd,
                act_cu_k, sl_kv,
                self.kv_num_heads,
            )[0]

            meta.dc_dt        = int(dc_new_lens.sum().item())
            meta.dc_max_new   = int(dc_new_lens.max().item())
            meta.dc_di        = dc_di
            meta.dc_cu_q      = dc_cu
            meta.dc_sl_kv_enc = dc_sl_kv_enc
            meta.dc_sl_kv_dec = dc_hist_lens
            meta.dc_bidx      = dc_bidx
            meta.dc_act_cu_k  = act_cu_k
            meta.dc_sl_kv     = sl_kv
            meta.dc_max_seq_k = int(max_token_cpu[1].item()) or int(kv_per_b.max().item())

        logger.debug(f"[BW] PREFILL n={npf}  DECODE n={ndc}")
        self.attention_metadata = meta

    # ------------------------------------------------------------------
    # forward_mixed: pure GPU ops — CUDA Graph compatible
    # ------------------------------------------------------------------
    def forward_mixed(
        self,
        q, k, v, qkv,
        compressed_kv, k_pe,
        layer: Attention,
        forward_meta: "ForwardMeta",
    ) -> paddle.Tensor:
        meta = self.attention_metadata
        nH, nKVH, hd = self.num_heads, self.kv_num_heads, self.head_dim

        cache_quant_type_str = getattr(layer, "cache_quant_type_str", "none")
        if cache_quant_type_str == "block_wise_fp8":
            cache_k = forward_meta.caches[4 * layer.layer_id]
            cache_v = forward_meta.caches[4 * layer.layer_id + 1]
        else:
            cache_k = forward_meta.caches[2 * layer.layer_id]
            cache_v = forward_meta.caches[2 * layer.layer_id + 1]

        norm_after_rope_in_kernel = not getattr(layer, "qk_norm_before_rope", False)
        q_norm_weight = getattr(layer, "q_norm_weight", None) if norm_after_rope_in_kernel else None
        k_norm_weight = getattr(layer, "k_norm_weight", None) if norm_after_rope_in_kernel else None

        D_type = paddle.float16 if qkv.dtype == paddle.float16 else paddle.bfloat16
        total_tokens = qkv.shape[0]
        # Unpack: [total_tokens, (nH+2*nKVH)*hd] -> [total_tokens, nH+2*nKVH, hd]
        qkv_3d = qkv.reshape([total_tokens, nH + 2 * nKVH, hd]).cast(D_type)

        block_tables = forward_meta.block_tables

        result = paddle.zeros([total_tokens, nH * hd], dtype=D_type)

        # ── Prefill path (encoder kernel: neox native, no data conversion) ────
        if meta.pf_ei is not None:
            pf_qkv = paddle.index_select(qkv_3d,      meta.pf_ei,   axis=0)
            bt_pf  = paddle.index_select(block_tables, meta.pf_bidx, axis=0)

            rotary_embs_enc = self._rotary_embs_for_encoder(forward_meta.rotary_embs)

            # RoPE + write KV cache; static_op returns (q_e, k_e, v_e) as tensors
            q_e = paddle.empty([meta.pf_et, nH,   hd], dtype=D_type)
            k_e = paddle.empty([meta.pf_et, nKVH, hd], dtype=D_type)
            v_e = paddle.empty([meta.pf_et, nKVH, hd], dtype=D_type)
            q_e, k_e, v_e = blackwell_ops.static_op_flash_attn_write_cache_kv_encoder(
                pf_qkv, meta.pf_cu_q, meta.pf_cu_q,
                rotary_embs_enc,
                meta.pf_sl_enc, meta.pf_sl_dec,
                cache_k, cache_v, bt_pf,
                q_e, k_e, v_e,
                None,                          # kv_dequant_scale
                q_norm_weight, k_norm_weight,
                nH, nKVH, hd,
                meta.pf_max_enc, meta.pf_max_enc, self.max_seq_len,
                cache_quant_type_str,
            )

            enc_out = paddle.empty([meta.pf_et, nH, hd], dtype=D_type)
            blackwell_ops.static_op_flash_encoder_attn_fwd(
                q_e, k_e, v_e,
                meta.pf_cu_q, meta.pf_cu_q,
                enc_out,
                None,  # standard causal mask (text-only model)
            )

            result = paddle.scatter(result, meta.pf_ei, enc_out.reshape([meta.pf_et, nH * hd]))

        # ── Decode path ──────────────────────────────────────────────────────
        # The encoder kernel stores K to cache in neox element order with neox
        # RoPE.  The decoder kernel (write_cache_kv_decoder) internally uses
        # GPT-J style RoPE which changes the element order when combined with
        # the Python neox→gptj conversion.  To keep cache K consistent between
        # prefill-written and decode-written blocks, we:
        #   1. Apply neox RoPE to Q/K in Python (matching the encoder).
        #   2. Pass identity rotary_embs (cos=1, sin=0) to the decoder kernel
        #      so it only does GQA packing of Q and K/V cache write without
        #      modifying values.
        if meta.dc_di is not None:
            dc_qkv = paddle.index_select(qkv_3d,      meta.dc_di,   axis=0)
            bt_dc  = paddle.index_select(block_tables, meta.dc_bidx, axis=0)

            # Apply neox RoPE in Python for Q and K heads
            rotary_embs_raw = forward_meta.rotary_embs  # [2, 1, max_seq_len, 1, rotary_dim/2] or [2, max_seq_len, rotary_dim/2]
            if rotary_embs_raw.ndim == 5:
                rotary_embs_2d = rotary_embs_raw.squeeze([1, 3])  # [2, max_seq_len, rotary_dim/2]
            else:
                rotary_embs_2d = rotary_embs_raw
            half_rot = rotary_embs_2d.shape[-1]  # rotary_dim / 2
            rotary_dim = half_rot * 2

            # Gather per-token cos/sin based on position (= cached history length)
            # dc_sl_kv_dec[i] = position of new token for decode batch item i
            positions = meta.dc_sl_kv_dec  # [ndc] int32, positions for each decode seq
            # Expand positions per token: for each token, its position = seq_pos + offset_within_seq
            # For standard decode (1 token/seq, dc_max_new==1): tok_positions = positions directly
            if meta.dc_max_new == 1:
                tok_positions = positions  # [ndc] = [dc_dt]
            else:
                # Multi-token decode: build per-token positions via GPU ops
                dc_new_lens = paddle.diff(meta.dc_cu_q)  # [ndc] int32
                # token_seq_id[t] = which seq this token belongs to
                token_seq_id = paddle.searchsorted(meta.dc_cu_q[1:], paddle.arange(meta.dc_dt, dtype="int32"), right=True)
                # offset within seq
                token_offset = paddle.arange(meta.dc_dt, dtype="int32") - paddle.gather(meta.dc_cu_q, token_seq_id.cast("int64")).cast("int32")
                tok_positions = paddle.gather(positions, token_seq_id.cast("int64")).cast("int32") + token_offset

            # cos/sin: [dc_dt, half_rot]
            cos_all = paddle.index_select(rotary_embs_2d[0], tok_positions.cast("int64"), axis=0)  # [dc_dt, half_rot]
            sin_all = paddle.index_select(rotary_embs_2d[1], tok_positions.cast("int64"), axis=0)  # [dc_dt, half_rot]

            # Apply neox RoPE to Q and K (compute in float32, cast back to D_type)
            qk_dc = dc_qkv[:, :nH + nKVH, :]  # [dc_dt, nH+nKVH, hd]
            first_half  = qk_dc[:, :, :half_rot].cast("float32")           # [dc_dt, nH+nKVH, half_rot]
            second_half = qk_dc[:, :, half_rot:rotary_dim].cast("float32") # [dc_dt, nH+nKVH, half_rot]
            cos_e = cos_all.unsqueeze(1)  # [dc_dt, 1, half_rot]
            sin_e = sin_all.unsqueeze(1)  # [dc_dt, 1, half_rot]
            new_first  = (first_half * cos_e - second_half * sin_e).cast(D_type)
            new_second = (second_half * cos_e + first_half * sin_e).cast(D_type)
            if rotary_dim < hd:
                qk_dc = paddle.concat([new_first, new_second, qk_dc[:, :, rotary_dim:]], axis=-1)
            else:
                qk_dc = paddle.concat([new_first, new_second], axis=-1)
            dc_qkv = paddle.concat([qk_dc, dc_qkv[:, nH + nKVH:, :]], axis=1)

            # GQA packing + K/V cache write (no RoPE applied by kernel)
            q_dec, _ = blackwell_ops.static_op_flash_attn_write_cache_kv_decoder(
                dc_qkv, meta.dc_cu_q,
                meta.dc_sl_kv_enc, meta.dc_sl_kv_dec,
                self._identity_rotary,
                cache_k, cache_v, bt_dc,
                None,                          # kv_dequant_scale
                None, None,                    # no norm in kernel (already done or not needed)
                nH, nKVH, hd,
                self.max_seq_len,
                cache_quant_type_str,
            )

            dec_out = paddle.zeros([meta.dc_dt, nH, hd], dtype=D_type)
            blackwell_ops.static_op_flash_decoder_attn_fwd(
                q_dec, meta.dc_cu_q, meta.dc_sl_kv,
                meta.dc_sl_kv_enc, meta.dc_sl_kv_dec,
                cache_k, cache_v, bt_dc,
                dec_out,
                None, None,                    # q_dequant_scale, kv_dequant_scale
                nH, nKVH, hd,
                meta.dc_max_seq_k, meta.dc_max_new - 1,
                cache_quant_type_str,
            )

            result = paddle.scatter(result, meta.dc_di, dec_out.reshape([meta.dc_dt, nH * hd]))

        return result

    def forward_extend(self, q, k, v, qkv, compressed_kv, k_pe,
                       layer: Attention, forward_meta: "ForwardMeta") -> paddle.Tensor:
        return self.forward_mixed(q, k, v, qkv, compressed_kv, k_pe, layer, forward_meta)

    def forward_decode(self, q, k, v, qkv, compressed_kv, k_pe,
                       layer: Attention, forward_meta: "ForwardMeta") -> paddle.Tensor:
        return self.forward_mixed(q, k, v, qkv, compressed_kv, k_pe, layer, forward_meta)
