"""Layerwise Wanda/SparseGPT pruning and optional GPTQ/AWQ quantization.

SparseGPT error compensation is adapted from IST-DASLab/sparsegpt, Apache-2.0.
Modified: Linear-only implementation, CPU support, exact mask sizes, common
Llama/Qwen2 runner, and removal of unrelated evaluation/quantization code.
"""

import random
from pathlib import Path

from .io import read_jsonl, seed_all


def calibration_texts(args):
    if args.calibration:
        rows = read_jsonl(args.calibration)
        texts = [row.get("text", row.get("prompt", row.get("problem", ""))) for row in rows]
        texts = [text for text in texts if isinstance(text, str) and text.strip()]
        if len(texts) < args.samples:
            raise ValueError("Calibration file has fewer texts than --samples")
        return texts[:args.samples]
    from datasets import load_dataset

    dataset = load_dataset("allenai/c4", data_files="en/c4-train.00000-of-01024.json.gz",
                           split="train", streaming=True)
    texts = []
    for row in dataset:
        texts.append(row["text"])
        if len(texts) == args.samples:
            break
    return texts


def calibration_tokens(tokenizer, texts, length, seed):
    import torch

    rng = random.Random(seed)
    # Concatenate short documents so every sample has the requested token length.
    tokens = []
    for text in texts:
        tokens.extend(tokenizer.encode(text, add_special_tokens=False))
        if tokenizer.eos_token_id is not None:
            tokens.append(tokenizer.eos_token_id)
    if len(tokens) < length:
        raise ValueError("Not enough calibration tokens; supply more/longer texts or reduce --length")
    batches = []
    for _ in texts:
        start = rng.randrange(len(tokens) - length + 1)
        batches.append(torch.tensor([tokens[start:start + length]], dtype=torch.long))
    return batches


class Statistics:
    def __init__(self, layer, method):
        import torch

        self.layer, self.method, self.samples = layer, method, 0
        columns = layer.weight.shape[1]
        self.moment = torch.zeros((columns, columns) if method == "sparsegpt" else columns,
                                  device=layer.weight.device, dtype=torch.float32)

    def add(self, inputs):
        x = inputs.reshape(-1, inputs.shape[-1]).float()
        self.samples += inputs.shape[0]
        self.moment += x.T @ x if self.method == "sparsegpt" else x.square().sum(dim=0)

    def prune(self, sparsity, blocksize=128, damping=0.01):
        import torch

        if sparsity == 0:
            return
        weight = self.layer.weight.detach().float().clone()
        if self.method == "wanda":
            importance = weight.abs() * (self.moment / max(1, self.samples)).sqrt()
            indices = importance.argsort(dim=1, stable=True)[:, :int(weight.shape[1] * sparsity)]
            weight.scatter_(1, indices, 0)
        else:
            hessian = self.moment * (2 / max(1, self.samples))
            dead = hessian.diagonal() == 0
            hessian[dead, dead] = 1
            weight[:, dead] = 0
            diagonal = torch.arange(hessian.shape[0], device=weight.device)
            hessian[diagonal, diagonal] += damping * hessian.diagonal().mean().clamp_min(1e-8)
            inverse = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(hessian)), upper=True)
            for start in range(0, weight.shape[1], blocksize):
                end = min(start + blocksize, weight.shape[1])
                block = weight[:, start:end].clone()
                errors, quantized = torch.zeros_like(block), torch.zeros_like(block)
                hblock = inverse[start:end, start:end]
                importance = block.square() / hblock.diagonal().square().unsqueeze(0)
                mask = torch.zeros_like(block, dtype=torch.bool).flatten()
                count = int(mask.numel() * sparsity)
                mask[importance.flatten().argsort(stable=True)[:count]] = True
                mask = mask.reshape_as(block)
                for column in range(end - start):
                    old = block[:, column].clone()
                    new = old.masked_fill(mask[:, column], 0)
                    quantized[:, column] = new
                    error = (old - new) / hblock[column, column]
                    block[:, column:] -= error.unsqueeze(1) @ hblock[column, column:].unsqueeze(0)
                    errors[:, column] = error
                weight[:, start:end] = quantized
                weight[:, end:] -= errors @ inverse[start:end, end:]
        self.layer.weight.data.copy_(weight.to(self.layer.weight.dtype))


def move_tree(value, device):
    import torch

    if torch.is_tensor(value):
        return value.detach().to(device)
    if isinstance(value, dict):
        return {key: move_tree(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move_tree(item, device) for item in value)
    return value


def prune_model(model, batches, method, sparsity, device, blocksize=128, damping=0.01):
    import torch

    if model.config.model_type not in ("llama", "qwen2"):
        raise ValueError("Pruning supports Llama and Qwen2 architectures")
    if getattr(model.config, "quantization_config", None):
        raise ValueError("Prune dense weights before packing them with GPTQ/AWQ")
    if not 0 <= sparsity < 1:
        raise ValueError("Sparsity must be in [0, 1)")
    layers, captured = model.model.layers, []
    original_first = layers[0]
    use_cache = model.config.use_cache
    model.config.use_cache = False

    class Captured(Exception):
        pass

    class Catcher(torch.nn.Module):
        def forward(self, hidden_states, **kwargs):
            captured.append((hidden_states.detach().cpu(), move_tree(kwargs, "cpu")))
            raise Captured

    model.model.embed_tokens.to(device)
    model.model.rotary_emb.to(device)
    layers[0] = Catcher()
    try:
        with torch.inference_mode():
            for batch in batches:
                try:
                    model(batch.to(device), use_cache=False)
                except Captured:
                    pass
    finally:
        layers[0] = original_first
        model.model.embed_tokens.cpu()
        model.model.rotary_emb.cpu()
    try:
        with torch.inference_mode():
            for index, layer in enumerate(layers):
                layer.to(device)
                linear = {name: module for name, module in layer.named_modules()
                          if isinstance(module, torch.nn.Linear)}
                groups = [["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
                          ["self_attn.o_proj"], ["mlp.up_proj", "mlp.gate_proj"], ["mlp.down_proj"]]
                for names in groups:
                    stats = {name: Statistics(linear[name], method) for name in names}
                    hooks = [linear[name].register_forward_hook(
                        lambda module, inputs, output, stat=stat: stat.add(inputs[0]))
                        for name, stat in stats.items()]
                    try:
                        for hidden, kwargs in captured:
                            layer(hidden.to(device), **move_tree(kwargs, device))
                    finally:
                        for hook in hooks:
                            hook.remove()
                    for stat in stats.values():
                        stat.prune(sparsity, blocksize, damping)
                    del stats
                next_inputs = []
                for hidden, kwargs in captured:
                    output = layer(hidden.to(device), **move_tree(kwargs, device))[0]
                    next_inputs.append((output.cpu(), kwargs))
                captured = next_inputs
                layer.cpu()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"Pruned layer {index + 1}/{len(layers)}", flush=True)
    finally:
        model.config.use_cache = use_cache


def compress(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new, empty checkpoint output directory")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; choose --device cpu for a small-model smoke test")
    seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    texts = calibration_texts(args)
    if args.method in ("wanda", "sparsegpt"):
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype),
                                                    device_map="cpu", attn_implementation="sdpa").eval()
        batches = calibration_tokens(tokenizer, texts, args.length, args.seed)
        prune_model(model, batches, args.method, args.sparsity, args.device, args.blocksize, args.damping)
        model.save_pretrained(output, safe_serialization=True)
    elif args.method == "gptq":
        from gptqmodel import GPTQModel, QuantizeConfig

        batches = calibration_tokens(tokenizer, texts, args.length, args.seed)
        config = QuantizeConfig(bits=4, group_size=128, device=args.device)
        model = GPTQModel.load(args.model, config, torch_dtype=getattr(torch, args.dtype))
        calibration = [{"input_ids": batch, "attention_mask": torch.ones_like(batch)} for batch in batches]
        model.quantize(calibration, batch_size=1)
        model.save(str(output))
    else:
        from awq import AutoAWQForCausalLM

        model = AutoAWQForCausalLM.from_pretrained(args.model, torch_dtype=getattr(torch, args.dtype))
        batches = calibration_tokens(tokenizer, texts, args.length, args.seed)
        calibration = [batch[0].tolist() for batch in batches]
        model.quantize(tokenizer, quant_config={"zero_point": True, "q_group_size": 128,
                       "w_bit": 4, "version": "GEMM"}, calib_data=calibration,
                       max_calib_samples=args.samples, max_calib_seq_len=args.length)
        model.save_quantized(str(output))
    tokenizer.save_pretrained(output)
    print(f"Saved {args.method} checkpoint to {output}")
