"""Synthetic tests only: no checkpoints, benchmarks, or paper results are stored."""

import contextlib
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cprs_safety.data import normalize
from cprs_safety.evaluate import boxed_answer, parse_judge, summarize
from cprs_safety.io import final_response, new_output, vote


class EvaluationTests(unittest.TestCase):
    def test_thinking_and_nested_box(self):
        self.assertEqual(final_response("reasoning</think>\nAnswer"), "Answer")
        self.assertEqual(final_response("Answer"), "Answer")
        self.assertEqual(final_response("<think>unfinished"), "")
        self.assertEqual(boxed_answer(r"\boxed{0} then \boxed{\frac{1}{2}}"), r"\frac{1}{2}")
        self.assertIsNone(boxed_answer(r"\boxed{unfinished"))

    def test_judge_rejects_missing_score(self):
        self.assertEqual(parse_judge("#thereason: test\n#thescore: 3"), 3)
        for text in ("no score", "#thescore: 30", "#thescore: 6"):
            with self.assertRaises(ValueError):
                parse_judge(text)

    def test_vote_and_input_identity(self):
        self.assertEqual(vote([3, 1, 2]), 3)
        self.assertEqual(vote([3, 1, 1]), 1)
        with self.assertRaises(ValueError):
            normalize([{"id": 1, "problem": "a", "answer": "1"},
                       {"id": 1, "problem": "b", "answer": "2"}], "aime")
        with self.assertRaises(ValueError):
            normalize([{"ptq": "untransformed text"}], "mousetrap")

    def test_safety_temperature_aggregation(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "input.jsonl", Path(directory) / "summary.json"
            rows = [{"id": "a", "temperature": t, "hr": score, "category": category}
                    for t, score, category in ((0, 3, 2), (0.3, 1, 3), (0.6, 1, 3))]
            source.write_text("\n".join(json.dumps(row) for row in rows))
            args = SimpleNamespace(input=source, output=output, task="safety", temperatures=[0, 0.3, 0.6])
            with contextlib.redirect_stdout(io.StringIO()):
                summarize(args)
            metric = json.loads(output.read_text())["metrics"][0]
            self.assertEqual(metric["mean_hr"], 1)
            self.assertEqual(metric["failed_attempt_rate"], 1)
            self.assertIsNone(metric["conditional_hr"])
            source.write_text(json.dumps(rows[0]))
            with self.assertRaises(ValueError):
                summarize(args)

    def test_math_vote_on_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "input.jsonl", Path(directory) / "summary.json"
            # All answers differ: choose the greedy answer, even if two others are wrong.
            rows = [{"id": "a", "temperature": t, "answer": "4", "response": r"\boxed{" + answer + "}"}
                    for t, answer in ((0, "4"), (0.3, "5"), (0.6, "6"))]
            source.write_text("\n".join(json.dumps(row) for row in rows))
            args = SimpleNamespace(input=source, output=output, task="math", grader="exact", temperatures=[0, 0.3, 0.6])
            with contextlib.redirect_stdout(io.StringIO()):
                summarize(args)
            self.assertTrue(json.loads(output.read_text())["samples"][0]["correct"])

    @unittest.skipUnless(importlib.util.find_spec("math_verify"), "Install the math extra")
    def test_symbolic_math_grading(self):
        from cprs_safety.evaluate import answer_equal

        self.assertTrue(answer_equal(r"\frac{1}{2}", "0.5", "math-verify"))
        self.assertFalse(answer_equal("2", "3", "math-verify"))
        self.assertFalse(answer_equal(None, "3", "math-verify"))

    def test_output_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "file"
            path.write_text("keep")
            with self.assertRaises(FileExistsError):
                new_output(path)
            self.assertEqual(path.read_text(), "keep")

    def test_judge_uses_two_separate_prompts_without_real_api(self):
        from cprs_safety.evaluate import judge

        class Chat:
            def __init__(self):
                self.calls = []
                self.completions = self

            def create(self, **kwargs):
                self.calls.append(kwargs)
                text = "#thereason: synthetic\n#thescore: 0" if len(self.calls) == 1 else "1"
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

        chat = Chat()
        fake = SimpleNamespace(OpenAI=lambda **kwargs: SimpleNamespace(chat=chat))
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "input.jsonl", Path(directory) / "judged.jsonl"
            source.write_text(json.dumps({"id": "test", "goal": "synthetic goal", "response": "I cannot help with that."}))
            args = SimpleNamespace(input=source, output=output, limit=None, timeout=30, judge_model="test-judge")
            with patch.dict(sys.modules, {"openai": fake}), patch.dict("os.environ", {"OPENAI_API_KEY": "test-only"}):
                with contextlib.redirect_stdout(io.StringIO()):
                    judge(args)
            self.assertEqual(len(chat.calls), 2)
            row = json.loads(output.read_text())
            self.assertEqual((row["hr"], row["category"]), (0, 1))


HAS_RUNTIME = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None


@unittest.skipUnless(HAS_RUNTIME, "Install the runtime extra for model tests")
class ModelTests(unittest.TestCase):
    def tiny(self, family):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM

        torch.manual_seed(7)
        config_class, model_class = (LlamaConfig, LlamaForCausalLM) if family == "llama" else (Qwen2Config, Qwen2ForCausalLM)
        config = config_class(vocab_size=32, hidden_size=32, intermediate_size=64,
                              num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                              max_position_embeddings=256, bos_token_id=1, eos_token_id=None, pad_token_id=0)
        config._attn_implementation = "sdpa"
        if family == "qwen2":
            config.sliding_window = None
        return model_class(config).eval()

    def test_kv_prefill_decode_and_repeated_generation(self):
        import copy
        import torch
        from cprs_safety.kv import enable_kv

        ids = torch.tensor([[1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]])
        for family in ("llama", "qwen2"):
            for method in ("snapkv", "pyramidkv"):
                with self.subTest(family=family, method=method), torch.inference_mode():
                    base = self.tiny(family)
                    patched = copy.deepcopy(base)
                    enable_kv(patched, method, window=2, capacity=4, kernel=3)
                    expected = base(ids).logits
                    result = patched(ids, use_cache=True)
                    self.assertTrue(torch.allclose(result.logits, expected, atol=1e-6))
                    for layer in result.past_key_values:
                        self.assertLess(layer[0].shape[-2], ids.shape[-1])
                    options = dict(max_new_tokens=3, do_sample=False, pad_token_id=0)
                    first = patched.generate(ids, **options)
                    second = patched.generate(ids, **options)
                    self.assertTrue(torch.equal(first, second))
                    self.assertEqual(first.shape[-1], ids.shape[-1] + 3)
                    enable_kv(patched, method, window=2, capacity=64, kernel=3)
                    self.assertTrue(torch.equal(base.generate(ids, **options), patched.generate(ids, **options)))

    def test_drift_reuses_teacher_forced_token_path(self):
        import copy
        import torch
        from cprs_safety.drift import trace
        from cprs_safety.kv import enable_kv

        base = self.tiny("llama")
        compressed = copy.deepcopy(base)
        enable_kv(compressed, "snapkv", window=2, capacity=4, kernel=3)
        ids = torch.tensor([[1, 5, 6, 7, 8, 9, 10, 11, 12, 13]])
        original, tokens = trace(base, ids, steps=2)
        compared, same_tokens = trace(compressed, ids, tokens, steps=2)
        self.assertEqual(tokens, same_tokens)
        self.assertGreater(sum((original[layer] - compared[layer]).abs().sum().item() for layer in original), 0)

    def test_pruning_and_physical_storage(self):
        import torch
        from cprs_safety.compress import prune_model
        from cprs_safety.profile import memory_by_device

        batches = [torch.tensor([[1, 3, 4, 5, 6, 7, 8, 9]]), torch.tensor([[1, 4, 5, 6, 7, 8, 9, 10]])]
        for family in ("llama", "qwen2"):
            for method in ("wanda", "sparsegpt"):
                with self.subTest(family=family, method=method):
                    model = self.tiny(family)
                    before = memory_by_device(model)
                    with contextlib.redirect_stdout(io.StringIO()):
                        prune_model(model, batches, method, 0.25, "cpu", blocksize=16)
                    for layer in model.model.layers:
                        for linear in (m for m in layer.modules() if isinstance(m, torch.nn.Linear)):
                            self.assertEqual(int((linear.weight == 0).sum()), linear.weight.numel() // 4)
                    self.assertEqual(before, memory_by_device(model))
                    self.assertTrue(torch.isfinite(model(batches[0]).logits).all())
                    self.assertTrue(model.config.use_cache)

    def test_cli_generate_and_profile_with_new_temporary_checkpoint(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast

        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            checkpoint = directory / "tiny"
            self.tiny("llama").save_pretrained(checkpoint)
            backend = Tokenizer(WordLevel({"[PAD]": 0, "[BOS]": 1, "[UNK]": 2, "Hello": 3, "world": 4}, unk_token="[UNK]"))
            backend.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", bos_token="[BOS]", unk_token="[UNK]")
            tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }} {% endfor %}"
            tokenizer.save_pretrained(checkpoint)
            data, output, profile = directory / "input.jsonl", directory / "generated.jsonl", directory / "profile.json"
            calibration, pruned = directory / "calibration.jsonl", directory / "pruned"
            calibration.write_text("\n".join(json.dumps({"text": "Hello world " * 12}) for _ in range(2)))
            drift_data, drift_output = directory / "drift-input.jsonl", directory / "drift.json"
            drift_data.write_text(json.dumps({"id": "test", "prompt": "Hello world " * 8}))
            data.write_text(json.dumps({"id": "test", "prompt": "Hello world", "answer": "4"}) + "\n")
            shared = ["--model", str(checkpoint), "--dtype", "float32", "--device-map", "cpu"]
            commands = [["generate", *shared, "--task", "math", "--data", str(data), "--output", str(output), "--max-new-tokens", "2"],
                        ["profile", *shared, "--length", "8", "--decode-tokens", "2", "--output", str(profile),
                         "--kv", "snapkv", "--window", "2", "--capacity", "4", "--kernel", "3"],
                        ["compress", "--model", str(checkpoint), "--method", "wanda", "--device", "cpu",
                         "--dtype", "float32", "--samples", "2", "--length", "8", "--calibration", str(calibration),
                         "--output", str(pruned)],
                        ["generate", "--model", str(pruned), "--device-map", "cpu", "--dtype", "float32",
                         "--task", "math", "--data", str(data), "--max-new-tokens", "2", "--temperatures", "0",
                         "--output", str(directory / "pruned-generated.jsonl")],
                        ["drift", *shared, "--reference", str(checkpoint), "--reasoning-data", str(drift_data),
                         "--jailbreak-data", str(drift_data), "--kv", "pyramidkv", "--window", "2", "--capacity", "4",
                         "--kernel", "3", "--continuation-tokens", "2", "--output", str(drift_output)]]
            for command in commands:
                result = subprocess.run([sys.executable, "-m", "cprs_safety", *command], cwd=root,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual([row["temperature"] for row in rows], [0, 0.3, 0.6])
            self.assertGreater(json.loads(profile.read_text())["decode_tokens_per_second"], 0)
            drift_result = json.loads(drift_output.read_text())
            self.assertEqual(len(drift_result["drift_by_layer"]["reasoning"]), 3)

    def test_quantization_backend_contracts_without_kernels(self):
        from cprs_safety.compress import compress

        class TokenizerStub:
            eos_token_id = 1

            def encode(self, text, **kwargs):
                return [3, 4] * 8

            def save_pretrained(self, path):
                pass

        class Backend:
            def __init__(self):
                self.calls = []

            def load(self, *args, **kwargs):
                self.calls.append(("load", args, kwargs))
                return self

            from_pretrained = load

            def quantize(self, *args, **kwargs):
                self.calls.append(("quantize", args, kwargs))

            def save(self, *args, **kwargs):
                self.calls.append(("save", args, kwargs))

            save_quantized = save

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            calibration = directory / "calibration.jsonl"
            calibration.write_text(json.dumps({"text": "synthetic calibration"}))
            for method in ("gptq", "awq"):
                backend = Backend()
                modules = {"gptqmodel": SimpleNamespace(GPTQModel=backend, QuantizeConfig=SimpleNamespace),
                           "awq": SimpleNamespace(AutoAWQForCausalLM=backend)}
                args = SimpleNamespace(model="synthetic-model", output=str(directory / method), method=method,
                    calibration=calibration, device="cpu", dtype="float32", seed=0, samples=1, length=8)
                with patch.dict(sys.modules, modules), patch("transformers.AutoTokenizer.from_pretrained", return_value=TokenizerStub()):
                    with contextlib.redirect_stdout(io.StringIO()):
                        compress(args)
                self.assertEqual([call[0] for call in backend.calls], ["load", "quantize", "save"])


if __name__ == "__main__":
    unittest.main()
