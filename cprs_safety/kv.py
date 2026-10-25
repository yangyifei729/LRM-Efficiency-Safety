"""Minimal SDPA SnapKV/PyramidKV adapter for Transformers 4.49.0.

Adapted from Zefan Cai's PyramidKV (MIT); see THIRD_PARTY.md and licenses/.
Modified: shared Llama/Qwen2 implementation, instance-local patching, modern
DynamicCache API, parameter checks, and removal of unused attention backends.
The original implementation expands GQA heads before storing the prompt cache;
we preserve that convention. Only the prefill cache is compressed.
"""

import math
from types import MethodType

import torch
import torch.nn.functional as F


def compress_prompt(key, query, value, method, window, capacity, kernel,
                    pooling, layer_index, layer_count):
    length, dimension = query.shape[-2:]
    if length < capacity:
        return key, value
    keep = capacity - window
    if method == "pyramidkv" and length >= 2 * keep:
        minimum = keep // 20
        maximum = 2 * keep - minimum
        if maximum >= length - window:
            maximum = length - window
            minimum = 2 * keep - maximum
        step = (maximum - minimum) // max(layer_count - 1, 1)
        keep = maximum - layer_index * step
    keep = max(0, min(keep, length - window))
    scores = query[..., -window:, :] @ key.transpose(-2, -1) / math.sqrt(dimension)
    causal = torch.full((window, window), float("-inf"), device=query.device).triu(1)
    scores[..., -window:] += causal.to(scores.dtype)
    scores = scores.softmax(dim=-1, dtype=torch.float32).to(query.dtype)
    importance = scores[..., :-window].sum(dim=-2)
    pool = F.max_pool1d if pooling == "maxpool" else F.avg_pool1d
    importance = pool(importance, kernel, stride=1, padding=kernel // 2)
    indices = importance.topk(keep, dim=-1).indices.unsqueeze(-1)
    indices = indices.expand(*indices.shape[:-1], dimension)
    compressed_key = torch.cat((key[..., :-window, :].gather(2, indices), key[..., -window:, :]), dim=2)
    compressed_value = torch.cat((value[..., :-window, :].gather(2, indices), value[..., -window:, :]), dim=2)
    return compressed_key, compressed_value


def _attention(self, hidden_states, position_embeddings, attention_mask=None,
               past_key_value=None, cache_position=None, **kwargs):
    from transformers.cache_utils import DynamicCache
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

    if hidden_states.shape[0] != 1:
        raise ValueError("KV compression supports one unpadded prompt at a time")
    if kwargs.get("output_attentions"):
        raise ValueError("This SDPA adapter does not return attention matrices")
    if past_key_value is not None and not isinstance(past_key_value, DynamicCache):
        raise ValueError("KV compression requires DynamicCache")
    shape = (*hidden_states.shape[:-1], -1, self.head_dim)
    query = self.q_proj(hidden_states).view(shape).transpose(1, 2)
    key = self.k_proj(hidden_states).view(shape).transpose(1, 2)
    value = self.v_proj(hidden_states).view(shape).transpose(1, 2)
    cosine, sine = position_embeddings
    query, key = apply_rotary_pos_emb(query, key, cosine, sine)
    key, value = (repeat_kv(tensor, self.num_key_value_groups) for tensor in (key, value))
    if past_key_value is not None:
        if past_key_value.get_seq_length(self.layer_idx) == 0:
            compressed = compress_prompt(key, query, value, *self._cprs_kv)
            past_key_value.update(*compressed, self.layer_idx)
            # Prefill attends to the complete prompt; eviction affects decode.
        else:
            if query.shape[-2] != 1:
                raise ValueError("Chunked prefill is unsupported by this minimal adapter")
            key, value = past_key_value.update(key, value, self.layer_idx)
    # For single-token decoding every retained key is in the causal past.
    mask = attention_mask[..., :key.shape[-2]] if attention_mask is not None else None
    output = F.scaled_dot_product_attention(
        query.contiguous(), key.contiguous(), value.contiguous(), attn_mask=mask,
        dropout_p=0.0, is_causal=mask is None and query.shape[-2] > 1,
    )
    output = output.transpose(1, 2).reshape(*hidden_states.shape[:-1], -1).contiguous()
    return self.o_proj(output), None


def enable_kv(model, method, window=32, capacity=64, kernel=7, pooling="maxpool"):
    import transformers

    if transformers.__version__ != "4.49.0":
        raise ValueError("The KV adapter requires transformers==4.49.0")
    if method not in ("snapkv", "pyramidkv"):
        raise ValueError("Unknown KV method")
    if model.config.model_type not in ("llama", "qwen2"):
        raise ValueError("KV compression supports Llama and Qwen2 model architectures")
    if window < 1 or capacity <= window or kernel < 1 or kernel % 2 == 0:
        raise ValueError("Require 0 < window < capacity and a positive odd kernel")
    if pooling not in ("maxpool", "avgpool"):
        raise ValueError("Unknown pooling method")
    if getattr(model.config, "use_sliding_window", False):
        raise ValueError("Disable sliding-window attention for this adapter")
    layers = model.model.layers
    for index, layer in enumerate(layers):
        layer.self_attn._cprs_kv = (method, window, capacity, kernel, pooling, index, len(layers))
        layer.self_attn.forward = MethodType(_attention, layer.self_attn)
