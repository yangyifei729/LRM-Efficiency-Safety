"""Explicit model loading, shared by all local workflows."""


def load_model(name, dtype="float16", device_map="auto", loader="hf", kv="none",
               window=32, capacity=64, kernel=7, pooling="maxpool"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    precision = getattr(torch, dtype)
    if loader == "gptq":
        from gptqmodel import GPTQModel

        model = GPTQModel.load(name, device_map=device_map, torch_dtype=precision).model
    elif loader == "awq":
        from awq import AutoAWQForCausalLM

        model = AutoAWQForCausalLM.from_quantized(
            name, fuse_layers=False, device_map=device_map, torch_dtype=precision,
        ).model
    else:
        model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=precision, device_map=device_map, attn_implementation="sdpa",
        )
    tokenizer = AutoTokenizer.from_pretrained(name, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    # Optional loaders may default to a different attention backend.
    model.config._attn_implementation = "sdpa"
    for layer in model.model.layers:
        layer.self_attn.config._attn_implementation = "sdpa"
    if kv != "none":
        from .kv import enable_kv

        enable_kv(model, kv, window, capacity, kernel, pooling)
    return model, tokenizer


def load_from_args(args, name=None, kv=None):
    return load_model(name or args.model, args.dtype, args.device_map, args.loader,
                      args.kv if kv is None else kv, args.window, args.capacity,
                      args.kernel, args.pooling)


def encode(tokenizer, prompt, task="safety"):
    messages = []
    if task == "math":
        messages.append({"role": "system", "content": r"Please reason step by step, and put your final answer within \boxed{}."})
    messages.append({"role": "user", "content": prompt})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, return_tensors="pt", add_special_tokens=False, return_token_type_ids=False)


def input_device(model):
    return model.get_input_embeddings().weight.device
