"""Small, dependency-free utilities shared by the command-line tools."""

import json
import math
import random
from collections import Counter
from pathlib import Path


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{number}: expected a JSON object")
                rows.append(row)
    if not rows:
        raise ValueError(f"No records in {path}")
    return rows


def new_output(path):
    """Refuse accidental overwrites, including an input used as output."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("x", encoding="utf-8")


def write_json(path, value):
    with new_output(path) as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def append_record(stream, record):
    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()


def final_response(text):
    """Keep the final answer; never silently discard untagged generations."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip()
    # An unfinished thinking section is recorded as no final answer.
    if "<think>" in text:
        return ""
    return text.strip()


def vote(values):
    """Majority vote; ties fall back to the first (greedy) sample."""
    if not values:
        raise ValueError("Cannot vote over an empty list")
    count = Counter(values)
    best, frequency = count.most_common(1)[0]
    return best if frequency > len(values) / 2 else values[0]


def seed_all(seed):
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pearson(x, y):
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Pearson correlation requires paired observations")
    mx, my = sum(x) / len(x), sum(y) / len(y)
    xx = sum((v - mx) ** 2 for v in x)
    yy = sum((v - my) ** 2 for v in y)
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(xx * yy) if xx and yy else None
