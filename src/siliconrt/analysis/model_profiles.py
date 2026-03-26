"""Model-specific cache math profiles used for CPU-only design work."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json


@dataclass(frozen=True)
class Qwen35TextCacheEstimate:
    seq_len: int
    batch_size: int
    bytes_per_element: float
    effective_full_attention_seq_len: int
    full_attention_layers: int
    linear_attention_layers: int
    full_attention_bytes_per_token_total: float
    full_attention_total_bytes: float
    linear_attention_bytes_per_layer_constant: float
    linear_attention_total_bytes: float
    total_bytes: float
    total_gb: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def estimate_qwen35_9b_text_cache(
    *,
    seq_len: int,
    batch_size: int = 1,
    bytes_per_element: float = 2.0,
    full_attention_window_len: int | None = None,
) -> Qwen35TextCacheEstimate:
    """Estimate text-side cache memory for Qwen3.5-9B.

    This profile matches the local MLX config currently in use:

    - 32 text layers total
    - 8 full-attention layers with KV cache
    - 24 linear-attention layers with:
      - conv state: [B, kernel_size - 1, conv_dim]
      - recurrent state: [B, Hv, Dv, Dk]

    The linear-attention cache is effectively constant with respect to sequence
    length. The full-attention cache grows linearly with sequence length.
    """

    if seq_len <= 0:
        raise ValueError("seq_len must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if full_attention_window_len is not None and full_attention_window_len <= 0:
        raise ValueError("full_attention_window_len must be positive when provided.")

    full_attention_layers = 8
    linear_attention_layers = 24

    # Full-attention layers
    num_key_value_heads = 4
    head_dim = 256
    effective_full_attention_seq_len = (
        min(seq_len, full_attention_window_len)
        if full_attention_window_len is not None
        else seq_len
    )
    full_attention_bytes_per_token_total = (
        full_attention_layers
        * num_key_value_heads
        * head_dim
        * 2  # K and V
        * bytes_per_element
    )
    full_attention_total_bytes = (
        full_attention_bytes_per_token_total
        * effective_full_attention_seq_len
        * batch_size
    )

    # Linear-attention layers
    linear_num_key_heads = 16
    linear_key_head_dim = 128
    linear_num_value_heads = 32
    linear_value_head_dim = 128
    linear_conv_kernel_dim = 4

    key_dim = linear_num_key_heads * linear_key_head_dim
    value_dim = linear_num_value_heads * linear_value_head_dim
    conv_dim = key_dim * 2 + value_dim

    conv_state_bytes = (
        (linear_conv_kernel_dim - 1) * conv_dim * bytes_per_element
    )
    recurrent_state_bytes = (
        linear_num_value_heads
        * linear_value_head_dim
        * linear_key_head_dim
        * bytes_per_element
    )
    linear_attention_bytes_per_layer_constant = (
        conv_state_bytes + recurrent_state_bytes
    )
    linear_attention_total_bytes = (
        linear_attention_layers
        * linear_attention_bytes_per_layer_constant
        * batch_size
    )

    total_bytes = full_attention_total_bytes + linear_attention_total_bytes

    return Qwen35TextCacheEstimate(
        seq_len=seq_len,
        batch_size=batch_size,
        bytes_per_element=bytes_per_element,
        effective_full_attention_seq_len=effective_full_attention_seq_len,
        full_attention_layers=full_attention_layers,
        linear_attention_layers=linear_attention_layers,
        full_attention_bytes_per_token_total=full_attention_bytes_per_token_total,
        full_attention_total_bytes=full_attention_total_bytes,
        linear_attention_bytes_per_layer_constant=linear_attention_bytes_per_layer_constant,
        linear_attention_total_bytes=linear_attention_total_bytes,
        total_bytes=total_bytes,
        total_gb=total_bytes / 1e9,
    )
