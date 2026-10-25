"""Run `python -m cprs_safety --help` for the available workflows."""

import argparse
import math
from pathlib import Path


def positive(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def model_options(parser):
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local checkpoint directory")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--device-map", default="auto", help="auto, cpu, cuda:0, ...")
    parser.add_argument("--loader", choices=("hf", "gptq", "awq"), default="hf")
    parser.add_argument("--kv", choices=("none", "snapkv", "pyramidkv"), default="none")
    parser.add_argument("--window", type=positive, default=32)
    parser.add_argument("--capacity", type=positive, default=64)
    parser.add_argument("--kernel", type=positive, default=7)
    parser.add_argument("--pooling", choices=("maxpool", "avgpool"), default="maxpool")
    parser.add_argument("--seed", type=int, default=0)


def build_parser():
    parser = argparse.ArgumentParser(description="Minimal efficiency/safety experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Normalize external benchmark inputs")
    prepare.add_argument("--dataset", choices=("hcot", "mousetrap", "aime", "math500"), required=True)
    prepare.add_argument("--source", help="Original benchmark JSONL; required for AIME/Mousetrap")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--limit", type=positive)

    generate = commands.add_parser("generate", help="Generate benchmark responses locally")
    model_options(generate)
    generate.add_argument("--task", choices=("safety", "math"), required=True)
    generate.add_argument("--data", required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument("--temperatures", nargs="+", type=float, default=[0.0, 0.3, 0.6])
    generate.add_argument("--max-new-tokens", type=positive, default=32768)
    generate.add_argument("--limit", type=positive)

    judge = commands.add_parser("judge", help="Score harmfulness and classify response behavior")
    judge.add_argument("--input", required=True)
    judge.add_argument("--output", required=True)
    judge.add_argument("--judge-model", default="gpt-4o")
    judge.add_argument("--timeout", type=positive, default=120)
    judge.add_argument("--limit", type=positive)

    summarize = commands.add_parser("summarize", help="Aggregate the three temperature samples per prompt")
    summarize.add_argument("--task", choices=("safety", "math"), required=True)
    summarize.add_argument("--input", required=True)
    summarize.add_argument("--output", required=True)
    summarize.add_argument("--temperatures", nargs="+", type=float, default=[0.0, 0.3, 0.6])
    summarize.add_argument("--grader", choices=("math-verify", "exact"), default="math-verify")

    compress = commands.add_parser("compress", help="Create a pruned or quantized checkpoint locally")
    compress.add_argument("--model", required=True)
    compress.add_argument("--method", choices=("wanda", "sparsegpt", "gptq", "awq"), required=True)
    compress.add_argument("--output", required=True)
    compress.add_argument("--calibration", help="JSONL with text fields; otherwise stream C4")
    compress.add_argument("--samples", type=positive, default=128)
    compress.add_argument("--length", type=positive, default=2048)
    compress.add_argument("--sparsity", type=float, default=0.25)
    compress.add_argument("--blocksize", type=positive, default=128)
    compress.add_argument("--damping", type=float, default=0.01)
    compress.add_argument("--device", default="cuda:0")
    compress.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    compress.add_argument("--seed", type=int, default=0)

    profile = commands.add_parser("profile", help="Measure physical memory and prefill/decode speed")
    model_options(profile)
    profile.add_argument("--length", type=positive, default=2048)
    profile.add_argument("--decode-tokens", type=positive, default=128)
    profile.add_argument("--output", required=True)

    drift = commands.add_parser("drift", help="Measure hidden-state drift on matched token paths")
    model_options(drift)
    drift.add_argument("--reference", required=True)
    drift.add_argument("--reasoning-data", required=True)
    drift.add_argument("--jailbreak-data", required=True)
    drift.add_argument("--continuation-tokens", type=int, default=0)
    drift.add_argument("--output", required=True)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "temperatures") and (any(value < 0 or not math.isfinite(value) for value in args.temperatures)
            or len(set(args.temperatures)) != len(args.temperatures) or min(args.temperatures) != 0):
        parser.error("temperatures must be unique, nonnegative, and include greedy temperature 0")
    if getattr(args, "continuation_tokens", 0) < 0:
        parser.error("continuation-tokens must be nonnegative")
    if args.command != "compress" and Path(args.output).exists():
        parser.error("output already exists; choose a new output file")
    if args.command == "compress" and (not 0 <= args.sparsity < 1 or args.damping <= 0):
        parser.error("require 0 <= sparsity < 1 and damping > 0")
    if args.command == "prepare":
        from .data import prepare
        prepare(args)
    elif args.command == "generate":
        from .generate import generate
        generate(args)
    elif args.command in ("judge", "summarize"):
        from .evaluate import judge, summarize
        (judge if args.command == "judge" else summarize)(args)
    elif args.command == "compress":
        from .compress import compress
        compress(args)
    elif args.command == "profile":
        from .profile import profile
        profile(args)
    else:
        from .drift import drift
        drift(args)


if __name__ == "__main__":
    main()
