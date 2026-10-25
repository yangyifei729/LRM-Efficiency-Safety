"""Generate one response per prompt and temperature, without scoring side effects."""

from .io import append_record, final_response, new_output, read_jsonl, seed_all
from .model import encode, input_device, load_from_args


def generate(args):
    import torch

    rows = read_jsonl(args.data)
    if args.limit is not None:
        rows = rows[:args.limit]
    if len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("Input IDs must be unique")
    model, tokenizer = load_from_args(args)
    with new_output(args.output) as stream, torch.inference_mode():
        for index, row in enumerate(rows):
            inputs = encode(tokenizer, row["prompt"], args.task).to(input_device(model))
            for temperature in args.temperatures:
                seed_all(args.seed + index)
                options = {"do_sample": temperature > 0, "max_new_tokens": args.max_new_tokens,
                           "use_cache": True, "pad_token_id": tokenizer.pad_token_id}
                if temperature > 0:
                    options.update(temperature=temperature, top_p=1.0, top_k=0)
                tokens = model.generate(**inputs, **options)[0, inputs.input_ids.shape[1]:]
                raw = tokenizer.decode(tokens, skip_special_tokens=True)
                append_record(stream, {**row, "model": args.model, "kv": args.kv,
                    "temperature": temperature, "seed": args.seed + index,
                    "max_new_tokens": args.max_new_tokens, "raw_response": raw,
                    "response": final_response(raw), "generated_tokens": len(tokens)})
            print(f"Generated {index + 1}/{len(rows)}", flush=True)
