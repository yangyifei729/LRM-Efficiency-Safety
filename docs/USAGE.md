# On the Efficiency-Safety Dilemma in Large Reasoning Models

Minimal runnable code for studying compression, reasoning capability, and
jailbreak vulnerability in large reasoning models. The workflows cover GPTQ,
AWQ, SparseGPT, Wanda, SnapKV, PyramidKV, combined compression, harmfulness
evaluation, mathematical evaluation, memory profiling, and hidden-state drift.

This repository contains source code and evaluation prompts. Models and
benchmark inputs are obtained separately; generated artifacts stay local.

## Install

Use Python 3.10 or 3.11 in a dedicated environment. Install a PyTorch build
appropriate for your CUDA environment, then install the workflow dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[runtime,judge,math]'
python -m cprs_safety --help
```

The KV adapter targets **Transformers 4.49.0** and the `llama` / `qwen2`
architectures used by DeepSeek-R1-Distill-Llama-8B,
DeepSeek-R1-Distill-Qwen-32B, and QwQ-32B. It runs through PyTorch SDPA and
does not require FlashAttention. Large-model runs need sufficient GPU memory
and host RAM. Start with `--limit 1 --max-new-tokens 32` to check your setup.

Optional quantization packages:

```bash
# Install only the methods you need; GPTQModel builds against the installed torch.
python -m pip install 'gptqmodel==2.2.0' --no-build-isolation
python -m pip install -e '.[awq]'
```

The `cprs-safety` executable and `python -m cprs_safety` are interchangeable.

## Prepare inputs

```bash
python -m cprs_safety prepare --dataset hcot --output data/hcot.jsonl
python -m cprs_safety prepare --dataset math500 --output data/math500.jsonl
python -m cprs_safety prepare --dataset aime \
  --source /path/to/test2024.jsonl --output data/aime.jsonl
python -m cprs_safety prepare --dataset mousetrap \
  --source /path/to/Mousetrap.jsonl --output data/mousetrap.jsonl
```

H-CoT is loaded from `DukeCEICenter/Malicious_Educator_hcot_o1`; MATH-500 is
loaded from `HuggingFaceH4/MATH-500`. `--source` also accepts local versions
of these two datasets. Obtain AIME inputs from the original
[AIME evaluation project](https://github.com/GAIR-NLP/AIME-Preview), and
Mousetrap attack inputs from the [Mousetrap authors](https://github.com/EVIGBYEN/Mousetrap).
Mousetrap source rows must contain `Goal` and the already transformed `prompt`;
an untransformed Trotter goal alone is insufficient. This repository evaluates
provided attack prompts and does not bundle upstream attack datasets.

Normalized inputs use one JSON object per line:

```text
{"id":"sample-id","benchmark":"hcot","prompt":"full attack prompt","goal":"original request"}
{"id":"sample-id","benchmark":"aime","prompt":"math problem","answer":"reference answer"}
```

Use separate files for each benchmark. Safety inputs require `goal`; math inputs
require `answer`. IDs must be unique. No model-generated answers are needed
to prepare benchmark inputs.

## Generate and evaluate

Choose an upstream model ID or a local checkpoint:

```bash
MODEL=deepseek-ai/DeepSeek-R1-Distill-Llama-8B
python -m cprs_safety generate --model "$MODEL" --task safety \
  --data data/hcot.jsonl --output outputs/hcot.generations.jsonl
```

Generation defaults to FP16, one unpadded prompt per call, three temperatures
`0 0.3 0.6`, and a maximum of 32,768 generated tokens. Each generation records
the raw response and its final answer separately. Content after the final
`</think>` is used for evaluation; untagged responses are kept, and an explicitly
unfinished `<think>` section has an empty final answer. Raw text is always retained.
Model chat templates are used without a safety system prompt. Math generation
adds the original instruction to reason step by step and use `\boxed{}`.

Set the judge credentials in the shell. `.env.example` documents the names;
the program does not automatically load `.env` files.

```bash
export OPENAI_API_KEY=your-key
# Optional: export OPENAI_BASE_URL=https://your-compatible-endpoint/v1
python -m cprs_safety judge --input outputs/hcot.generations.jsonl \
  --output outputs/hcot.judged.jsonl --judge-model gpt-4o
python -m cprs_safety summarize --task safety --input outputs/hcot.judged.jsonl \
  --output outputs/hcot.summary.json
```

The judge makes two independent calls per response: the original 0–5 harmfulness
rubric (Appendix A.7), then the refusal / hazardous / attempted-but-failed
classifier (Appendix A.8). Requests go to the endpoint configured by the user.
The SDK retries transport failures; malformed judge responses stop the run.
Output files are written incrementally, and existing files are never overwritten.
There is no automatic resume: use a new output name and an explicit input subset
when continuing a partial run.

Harmfulness and category scores are majority-voted across the specified
temperatures, with the greedy sample used when there is no majority. The summary
reports mean HR, refusal rate, hazardous rate, failed-attempt rate, and HR
conditional on a hazardous category. Rates use all prompts as the denominator.
The conditional mean uses only hazardous prompts and is `null` for an empty set.

For math:

```bash
python -m cprs_safety generate --model "$MODEL" --task math \
  --data data/aime.jsonl --output outputs/aime.generations.jsonl
python -m cprs_safety summarize --task math --input outputs/aime.generations.jsonl \
  --output outputs/aime.summary.json
```

Math grading extracts the final balanced `\boxed{...}`, groups equivalent answers
using Math-Verify, votes over answers, then compares to the reference. The output
is named `majority_accuracy`: three-temperature majority selection is distinct
from single-sample pass@1. Missing boxed answers are incorrect. `--grader exact`
provides strict string comparison for integer-answer smoke tests. Math-Verify
replaces the original notebook's larger symbolic grader; edge cases can differ.
If you override temperatures, pass the same list to both generation and summary.

## Compression and combinations

Prune decoder linear layers using C4 calibration (25% sparsity by default):

```bash
python -m cprs_safety compress --model "$MODEL" --method sparsegpt \
  --output checkpoints/sparsegpt
python -m cprs_safety compress --model "$MODEL" --method wanda \
  --output checkpoints/wanda
```

Pruning loads the dense model into host RAM and moves one decoder layer at a time
to `--device cuda:0`. It calibrates attention and MLP projection groups sequentially.
`--device cpu --dtype float32` is available for small-model verification. A
calibration JSONL with `text` fields can be supplied with `--calibration`; otherwise
C4 is streamed. `--samples`, `--length`, `--seed`, `--blocksize`, and `--damping`
are explicit. Calibration samples are fixed-length random spans from concatenated
documents. This release uses exact per-row masks for Wanda and exact block masks
for SparseGPT, avoiding an extra pruned element from threshold ties.

Generate 4-bit, group-size-128 GPTQ/AWQ checkpoints locally:

```bash
python -m cprs_safety compress --model "$MODEL" --method gptq \
  --output checkpoints/gptq
python -m cprs_safety compress --model "$MODEL" --method awq \
  --output checkpoints/awq
```

Evaluate a saved checkpoint with `--model checkpoints/gptq --loader gptq` or
`--model checkpoints/awq --loader awq`. Dense and pruned checkpoints use the
default `--loader hf`. The optional loaders keep backend failures visible.

Enable KV compression at inference time:

```bash
python -m cprs_safety generate --model "$MODEL" --task safety --kv snapkv \
  --window 32 --capacity 64 --kernel 7 --pooling maxpool \
  --data data/hcot.jsonl --output outputs/hcot.snapkv.jsonl
# Use --kv pyramidkv for PyramidKV.
```

The adapter stores selected historical prompt tokens plus the recent window;
prefill attention still sees the complete prompt. Decode tokens are subsequently
appended. `capacity` is a prompt-cache budget, not a fixed cap on the entire
growing generation cache. PyramidKV varies the historical budget across layers.
Grouped-query attention heads are expanded before cache storage, following the
original local implementation. Batching, padded inputs, chunked prefill, static
caches, sliding-window models, and beam search are outside this minimal adapter.

Combine methods by chaining checkpoints and runtime options:

```bash
# Prune, then quantize; save each stage to a different directory.
python -m cprs_safety compress --model checkpoints/sparsegpt --method gptq \
  --output checkpoints/sparsegpt-gptq
# Quantization + KV (use a pruned model here for Pruning + KV).
python -m cprs_safety generate --model checkpoints/gptq --loader gptq \
  --kv pyramidkv --task safety --data data/hcot.jsonl \
  --output outputs/hcot.gptq-pyramidkv.jsonl
```

Pruning acts on dense weights before quantization packing. Quantization error
compensation can change the zero pattern. Unstructured zero weights in dense
tensors retain their physical storage size; they do not automatically reduce
GPU allocation or latency. Measure each final configuration rather than assuming
that a sparsity ratio equals a memory-saving ratio.

## Memory, latency, and representational drift

```bash
python -m cprs_safety profile --model "$MODEL" --length 2048 \
  --decode-tokens 128 --output outputs/profile.json
```

Use the same input and decode lengths to compare settings. `--length` is explicit;
choose larger values only within the model's context and available memory. The
profiler reports unique model tensor storage, final KV tensor bytes, CUDA
allocated/reserved peaks per device, cold prefill time including first-token
selection, and greedy decode throughput. Loading is excluded. These are measured
values, not nonzero-weight storage estimates; they need not match historical
memory accounting. Synthetic prompts and fixed-length decode make the benchmark
procedure explicit. CPU execution reports CPU storage and empty CUDA counters.

For drift, prepare two JSONL files containing `id` and `prompt` fields for the
reasoning and jailbreak prompt sets in Appendix A.9:

```bash
python -m cprs_safety drift --reference "$MODEL" --model checkpoints/sparsegpt \
  --reasoning-data data/reasoning.jsonl --jailbreak-data data/jailbreak.jsonl \
  --output outputs/drift.json
python -m cprs_safety drift --reference "$MODEL" --model "$MODEL" --kv pyramidkv \
  --continuation-tokens 16 --reasoning-data data/reasoning.jsonl \
  --jailbreak-data data/jailbreak.jsonl --output outputs/drift.pyramidkv.json
```

The metric is `1 - cosine(mean_original_last_token, mean_compressed_last_token)`
for each category and decoder layer. Both models process the same token IDs;
optional continuation tokens are generated greedily by the reference model and
teacher-forced into both models. Model loading is sequential. Layerwise reasoning
and jailbreak drift are correlated with Pearson's r; constant vectors produce
`null`. KV eviction only affects decode, so KV drift requires a nonzero shared
continuation. A prefill-only comparison would measure zero eviction effect.

This is a cleaned implementation of the core workflows, not a byte-for-byte
snapshot of every historical experiment. Supplementary human annotation and
generation-stage labeling are not automated here. Model revisions, exact input
files, optional backend versions, decoding settings, and calibration choices must
be recorded for a new experiment; paper numbers are not embedded in the code.

## Verify the code and release contents

```bash
python -m unittest discover -s tests -v
python scripts/check_release.py
```

Model tests instantiate tiny random Llama/Qwen2 models in temporary directories.
They check compressed-cache decoding, repeat-run isolation, no-compression
equivalence, teacher-forced drift, both pruning methods, physical storage
accounting, and command-line generation/profiling. Judge tests use a fake client.
No model downloads or billable API calls are made by the tests. GPTQ/AWQ kernels
and full-size GPU experiments require separate validation on target hardware.

See [THIRD_PARTY.md](../THIRD_PARTY.md) for code and prompt attribution. Python code
uses Apache-2.0; the inherited H-CoT evaluation policy prompt retains its upstream
CC BY-NC-SA 4.0 terms. Dataset and model licenses apply separately.
