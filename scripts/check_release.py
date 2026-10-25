"""Check a source release and its paper framework figure before publishing."""

import re
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    excluded = {".git", "__pycache__", ".pytest_cache", ".ruff_cache"}
    documentation_assets = {"assets/framework.png"}
    forbidden = {".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".onnx",
                 ".pkl", ".pickle", ".ipynb", ".pdf", ".png", ".jpg", ".jpeg",
                 ".json", ".jsonl", ".csv", ".parquet", ".log", ".zip", ".gz"}
    secrets = re.compile(rb"(?:sk-[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)")
    machine_paths = re.compile(rb"/(?:data/share|data2|home)/[A-Za-z0-9_/-]+")
    issues, count, size = [], 0, 0
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part in excluded or part.startswith(".venv") for part in relative.parts):
            continue
        if path.is_symlink():
            issues.append(f"Symlink: {relative}")
            continue
        if not path.is_file():
            continue
        count += 1
        size += path.stat().st_size
        if (path.suffix.lower() in forbidden and relative.as_posix() not in documentation_assets) or path.name == ".env":
            issues.append(f"Artifact or local configuration: {relative}")
        if path.stat().st_size > 1024 * 1024:
            issues.append(f"Unexpected large file: {relative}")
            continue
        content = path.read_bytes()
        if relative.as_posix() in documentation_assets:
            if not content.startswith(b"\x89PNG\r\n\x1a\n"):
                issues.append(f"Invalid documentation image: {relative}")
            continue
        if secrets.search(content):
            issues.append(f"Credential-like content: {relative}")
        if machine_paths.search(content):
            issues.append(f"Machine-specific path: {relative}")
    if issues:
        raise SystemExit("\n".join(issues))
    print(f"Release check passed: {count} files, {size:,} bytes; no weights, data files, existing outputs, credentials, or machine paths detected.")


if __name__ == "__main__":
    main()
