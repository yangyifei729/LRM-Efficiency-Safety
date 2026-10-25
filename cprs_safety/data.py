"""Normalize externally obtained benchmark inputs; redistribute no datasets."""

from .io import append_record, new_output, read_jsonl


def normalize(rows, dataset):
    normalized, seen = [], set()
    for index, row in enumerate(rows):
        if dataset == "hcot":
            prompt = row.get("Full_Input (H-CoT + Request)", row.get("prompt"))
            goal = row.get("Goal", row.get("goal"))
        elif dataset == "mousetrap":
            prompt, goal = row.get("prompt"), row.get("Goal", row.get("goal"))
        else:
            prompt, goal = row.get("problem", row.get("prompt")), None
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Row {index}: missing benchmark prompt")
        identity = str(row.get("id", row.get("unique_id", index)))
        if identity in seen:
            raise ValueError(f"Duplicate sample ID: {identity}")
        seen.add(identity)
        record = {"id": identity, "benchmark": dataset, "prompt": prompt}
        if dataset in ("hcot", "mousetrap"):
            if not isinstance(goal, str) or not goal.strip():
                raise ValueError(f"Row {index}: missing original harmful goal")
            record["goal"] = goal
        else:
            if row.get("answer") is None:
                raise ValueError(f"Row {index}: missing reference answer")
            record["answer"] = str(row["answer"])
        normalized.append(record)
    return normalized


def prepare(args):
    if args.source:
        rows = read_jsonl(args.source)
    else:
        from datasets import load_dataset

        defaults = {"hcot": ("DukeCEICenter/Malicious_Educator_hcot_o1", "train"),
                    "math500": ("HuggingFaceH4/MATH-500", "test")}
        if args.dataset not in defaults:
            raise ValueError("Use --source with the original benchmark JSONL file")
        name, split = defaults[args.dataset]
        rows = list(load_dataset(name, split=split))
    rows = normalize(rows, args.dataset)
    if args.limit is not None:
        rows = rows[:args.limit]
    with new_output(args.output) as stream:
        for row in rows:
            append_record(stream, row)
    print(f"Prepared {len(rows)} {args.dataset} prompts")
