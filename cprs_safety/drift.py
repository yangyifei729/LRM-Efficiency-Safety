"""Compare category-averaged last-token hidden states on identical token paths."""

import gc

from .io import pearson, read_jsonl, seed_all, write_json
from .model import encode, input_device, load_model, load_from_args


def trace(model, ids, continuation=None, steps=0):
    import torch

    vectors = {}
    hooks = []
    for index, layer in enumerate(model.model.layers):
        def capture(module, inputs, output, index=index):
            vectors[index] = output[0][:, -1, :].detach().float().cpu()[0]
        hooks.append(layer.register_forward_hook(capture))
    tokens = [] if continuation is None else continuation
    try:
        with torch.inference_mode():
            result = model.model(input_ids=ids, use_cache=True)
            for index in range(steps):
                if continuation is None:
                    next_token = model.lm_head(result.last_hidden_state[:, -1:, :].to(model.lm_head.weight.device)).argmax(-1)
                    tokens.append(int(next_token.item()))
                token = torch.tensor([[tokens[index]]], device=ids.device)
                position = torch.tensor([ids.shape[1] + index], device=ids.device)
                result = model.model(input_ids=token, past_key_values=result.past_key_values,
                                     cache_position=position, use_cache=True)
    finally:
        for hook in hooks:
            hook.remove()
    return vectors, tokens


def drift(args):
    import torch
    import torch.nn.functional as F

    if args.kv != "none" and args.continuation_tokens == 0:
        raise ValueError("Use --continuation-tokens > 0 to measure KV eviction during decode")
    samples = [(category, row) for category, path in (("reasoning", args.reasoning_data),
               ("jailbreak", args.jailbreak_data)) for row in read_jsonl(path)]
    seed_all(args.seed)
    model, tokenizer = load_model(args.reference, args.dtype, args.device_map)
    reference, token_paths = [], []
    for category, row in samples:
        ids = encode(tokenizer, row["prompt"]).input_ids.to(input_device(model))
        vectors, continuation = trace(model, ids, steps=args.continuation_tokens)
        reference.append(vectors)
        token_paths.append((ids.cpu(), continuation))
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model, tokenizer = load_from_args(args)
    comparison = []
    for (_, row), (ids, continuation) in zip(samples, token_paths):
        check = encode(tokenizer, row["prompt"]).input_ids
        if not torch.equal(check, ids):
            raise ValueError("Reference and comparison tokenizers differ on the supplied prompts")
        vectors, _ = trace(model, ids.to(input_device(model)), continuation, args.continuation_tokens)
        comparison.append(vectors)
    categories = {}
    for category in ("reasoning", "jailbreak"):
        indices = [index for index, (name, _) in enumerate(samples) if name == category]
        values = []
        layers = len(reference[0])
        if any(len(vectors) != layers for vectors in comparison):
            raise ValueError("Models must have equal layer counts and hidden dimensions")
        for layer in range(layers):
            original = torch.stack([reference[index][layer] for index in indices]).mean(0)
            compressed = torch.stack([comparison[index][layer] for index in indices]).mean(0)
            if original.shape != compressed.shape:
                raise ValueError("Models have different hidden dimensions")
            values.append(float((1 - F.cosine_similarity(original, compressed, dim=0)).clamp(0, 2)))
        categories[category] = values
    write_json(args.output, {"reference": args.reference, "model": args.model, "kv": args.kv,
        "continuation_tokens": args.continuation_tokens, "drift_by_layer": categories,
        "pearson_r": pearson(categories["reasoning"], categories["jailbreak"]),
        "prompt_counts": {category: sum(name == category for name, _ in samples) for category in categories}})
    print("Saved layerwise cosine drift and Pearson correlation")
