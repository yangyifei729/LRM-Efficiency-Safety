"""Measure physical tensor memory and synchronized prefill/decode latency."""

import time

from .io import seed_all, write_json
from .model import input_device, load_from_args


def memory_by_device(model):
    """Count unique physical storages, including quantization buffers and zeros."""
    result, seen = {}, set()
    for tensor in list(model.parameters()) + list(model.buffers()):
        if tensor.device.type == "meta":
            continue
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr())
        if key not in seen:
            seen.add(key)
            result[str(tensor.device)] = result.get(str(tensor.device), 0) + storage.nbytes()
    return result


def synchronize():
    import torch

    for index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(index)


def profile(args):
    import torch

    seed_all(args.seed)
    model, tokenizer = load_from_args(args)
    device = input_device(model)
    # Synthetic unpadded tokens; no benchmark outputs or fixed paper measurements.
    token = tokenizer.encode("Hello", add_special_tokens=False)[0]
    ids = torch.full((1, args.length), token, dtype=torch.long, device=device)
    tensor_bytes = memory_by_device(model)
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)
    synchronize()
    with torch.inference_mode():
        begin = time.perf_counter()
        result = model.model(input_ids=ids, use_cache=True)
        token = model.lm_head(result.last_hidden_state[:, -1:, :].to(model.lm_head.weight.device)).argmax(-1).to(device)
        synchronize()
        ttft = time.perf_counter() - begin
        cache = result.past_key_values
        del result
        synchronize()
        begin = time.perf_counter()
        for index in range(args.decode_tokens):
            position = torch.tensor([args.length + index], device=device)
            result = model.model(input_ids=token, past_key_values=cache, cache_position=position, use_cache=True)
            token = model.lm_head(result.last_hidden_state[:, -1:, :].to(model.lm_head.weight.device)).argmax(-1).to(device)
            cache = result.past_key_values
            del result
        synchronize()
        duration = time.perf_counter() - begin
    allocated = {f"cuda:{index}": torch.cuda.max_memory_allocated(index)
                 for index in range(torch.cuda.device_count())}
    reserved = {f"cuda:{index}": torch.cuda.max_memory_reserved(index)
                for index in range(torch.cuda.device_count())}
    cache_bytes = sum(tensor.numel() * tensor.element_size() for layer in cache for tensor in layer)
    write_json(args.output, {"model": args.model, "kv": args.kv, "prompt_tokens": args.length,
        "decode_tokens": args.decode_tokens, "cuda_available": torch.cuda.is_available(),
        "model_storage_bytes_by_device": tensor_bytes, "final_kv_tensor_bytes": cache_bytes,
        "cuda_peak_allocated_bytes_by_device": allocated, "cuda_peak_reserved_bytes_by_device": reserved,
        "ttft_seconds": ttft, "decode_tokens_per_second": args.decode_tokens / duration,
        "notes": "Single cold prefill followed by fixed-length greedy decode; model loading excluded. Summed per-device peaks are not a simultaneous global peak."})
    print(f"TTFT {ttft:.4f}s; decode {args.decode_tokens / duration:.2f} tokens/s")
