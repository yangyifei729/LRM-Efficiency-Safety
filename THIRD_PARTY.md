# Sources and licenses

The experiment runners were consolidated from the research implementation for
*On the Efficiency-Safety Dilemma in Large Reasoning Models*. Repository code
is licensed under Apache-2.0, except the separately attributed material below.

| Component | Source | License / changes |
| --- | --- | --- |
| SparseGPT error compensation in `cprs_safety/compress.py` | [IST-DASLab/sparsegpt](https://github.com/IST-DASLab/sparsegpt) | Apache-2.0 (`LICENSE`). Adapted to linear layers, common Llama/Qwen2 calibration, exact mask sizes, and CPU testing. |
| SnapKV/PyramidKV selection and attention adapter in `cprs_safety/kv.py` | [Zefan Cai, PyramidKV](https://github.com/Zefan-Cai/PyramidKV) | MIT, copyright 2024 Zefan Cai (`licenses/PyramidKV-MIT.txt`). Consolidated into a single SDPA adapter for Transformers 4.49.0. |
| Harmfulness evaluation policy and rubric in `cprs_safety/prompts/harmfulness.txt` | The research evaluator, based on [H-CoT / Malicious Educator](https://github.com/dukeceicenter/jailbreak-reasoning-openai-o1o3-deepseek-r1) | Preserve the upstream CC BY-NC-SA 4.0 terms (`licenses/HCoT.txt`) for this prompt. It is separate from the Apache-licensed Python code. The wording is retained for evaluation consistency. |
| Three-category prompt in `cprs_safety/prompts/categories.txt` | Paper Appendix A.8 | Research evaluation prompt. |

H-CoT, Mousetrap, AIME, MATH-500, C4, and model weights are obtained separately
and remain subject to their respective upstream licenses. No benchmark data,
third-party repositories, model weights, or experimental outputs are distributed
with this source release. GPTQModel, AutoAWQ, Math-Verify, Transformers, and
other runtime packages are installed separately.
