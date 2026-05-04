#!/usr/bin/env python3
"""Check that the public repository does not contain private artifacts."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

import yaml


FORBIDDEN_DIR_NAMES = {
    ".claude",
    "__pycache__",
    "dpo-al",
    "evaluation_results",
    "evaluation_results_paper",
    "logs",
    "outputs",
    "outputs_last",
    "outputs_paper",
    "outputs_sanity",
    "outputs_sanity_local",
    "project_report",
    "venv",
    "wandb",
}

FORBIDDEN_EXTENSIONS = {
    ".bin",
    ".ckpt",
    ".jsonl",
    ".pt",
    ".pth",
    ".safetensors",
    ".tar",
}

PRIVATE_PATTERNS = [
    re.compile(r"/" + r"mnt/"),
    re.compile(r"/" + r"home/"),
    re.compile(r"def-" + r"dsuth"),
    re.compile(r"~/" + r"aa\b"),
    re.compile(r"LMOD" + r"_PKG"),
    re.compile(r"WANDB_MODE:\s*[\"']?online[\"']?"),
    re.compile(r"^\s*" + "HISTORY" + "_PRIVATE" + r":\s*[\"']?1[\"']?\s*$", re.MULTILINE),
    re.compile(r"^\s*" + "HISTORY" + "_HUB_NS" + r":", re.MULTILINE),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)(api[_-]?key|secret|password)\s*[:=]\s*['\"][^'\"]+['\"]"),
]

SKIP_DIR_NAMES = {".git"}
TEXT_SUFFIXES = {".cfg", ".cff", ".ini", ".json", ".md", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml"}


def iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        yield path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    root = args.root.resolve()
    errors: list[str] = []

    required = [
        "README.md",
        "LICENSE",
        "CITATION.cff",
        "pyproject.toml",
        "requirements.txt",
        "setup.py",
        "trl",
        "active_learning",
        "scripts/active_queue_runner.py",
        "commands/run_active_envaware.sh",
        "commands/run_active_queue.sh",
        "commands/run_reproduce.sh",
        "configs/paper_sweep_A_harmful_multiseed.yaml",
        "configs/paper_sweep_B_controls.yaml",
    ]
    for rel in required:
        if not (root / rel).exists():
            errors.append(f"missing required public file or directory: {rel}")

    for path in iter_files(root):
        rel = path.relative_to(root)
        if path.is_dir() and path.name in FORBIDDEN_DIR_NAMES:
            errors.append(f"forbidden directory present: {rel}")
            continue
        if path.is_file() and path.suffix in FORBIDDEN_EXTENSIONS:
            errors.append(f"forbidden artifact extension present: {rel}")
            continue
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        if rel == Path("scripts/verify_public_repo.py"):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in PRIVATE_PATTERNS:
            if pattern.search(text):
                errors.append(f"private/public-unsafe pattern {pattern.pattern!r} in {rel}")

    for config in [
        root / "configs/paper_sweep_A_harmful_multiseed.yaml",
        root / "configs/paper_sweep_B_controls.yaml",
    ]:
        try:
            data = yaml.safe_load(config.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"cannot parse YAML config {config.relative_to(root)}: {exc}")
            continue
        base_env = data.get("defaults", {}).get("base_env", {}) if isinstance(data, dict) else {}
        if base_env.get("WANDB_MODE") != "disabled":
            errors.append(f"{config.relative_to(root)} must default WANDB_MODE to disabled")
        if "HISTORY" + "_HUB_NS" in base_env or str(base_env.get("HISTORY" + "_PRIVATE", "0")) == "1":
            errors.append(f"{config.relative_to(root)} must not enable history hub upload by default")

    if errors:
        print("Public repo verification failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("Public repo verification passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
