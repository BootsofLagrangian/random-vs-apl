#!/usr/bin/env python3
"""
Orchestrate Active DAP training sweeps from a YAML configuration.

This runner expands the Cartesian product described in
`configs/active_experiment_queue.yaml`, launches up to N experiments in
parallel (pinning each to a specific GPU via `CUDA_VISIBLE_DEVICES`),
and logs success / failure metadata to a CSV file beside the per-run
stdout logs.

Example usage:
    python scripts/active_queue_runner.py \
        --config configs/active_experiment_queue.yaml

Use `--dry-run` to preview the queue without launching jobs, or
`--max-concurrency`/`--gpus` to override the defaults in the config.

Add `--sanity-check` to execute every combination as a 5-step smoke
test (logging each step, evaluating twice) with outputs redirected to a
separate debug directory. GPU IDs are pinned automatically for each
job; override the pool with `--gpus` if needed.
"""

from __future__ import annotations

import argparse
import csv
import json
import hashlib
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


@dataclass
class Experiment:
    run_id: str
    dataset: str
    model: str
    alignment_name: str
    lora_name: str
    query_name: str
    env: Dict[str, str]


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def sanitize(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-")


def short_name(value: str) -> str:
    parts = value.split("/")
    return sanitize(parts[-1] if parts else value)


def slugify_identifier(value: str) -> str:
    if not value:
        return "unknown"
    slug = sanitize(value.replace("/", "-"))
    if slug.startswith("activeDap-"):
        slug = slug[len("activeDap-") :] or slug
    return slug or "unknown"


def flatten_env(env: Dict[str, str]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(env.items()))


def detect_system_gpu_count() -> int:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            capture_output=True,
            text=True,
        )
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        return len(lines) if lines else 1
    except Exception:
        return 1


DEFAULT_ENV = {
    "DATASET": "activeDap/sft-hh-data",
    "MODEL": "MohamadBazzi/gemma-bpo-sft",
    "ALIGNMENT": "online_dpo",
    "LOSS_TYPE": "sigmoid",
    "EVAL_JUDGE": "OpenAssistant/reward-model-deberta-v3-large-v2",
    "QUERY_STRATEGY": "random",
    "NUM_EVAL_PROMPTS": "500",
    "SEQ_LEN": "256",
    "PER_DEVICE_BATCH_SIZE": "16",
    "UPDATES_PER_SAMPLE": "1",
    "LEARNING_RATE": "5.0e-5",
    "SEED": "1",
    "FILTER_OUT": "none",
    "GRADIENT_ACCUMULATION_STEPS": "1",
    "QUERY_FREQ_FACTOR": "1",
}


def dataset_abbreviation(dataset: str) -> str:
    if not dataset:
        return "unknown"
    if dataset.startswith("activeDap/"):
        slug = dataset[len("activeDap/") :]
        return slugify_identifier(slug) or "unknown"
    mapping = {
        "Anthropic/hh-rlhf": "hh",
        "won-bae/bpo_preference_hh_data": "hh",
        "activeDap/sft-hh-data": "help-bpo",
        "activeDap/sft-harm-data": "harm-bpo",
        "activeDap/sft-tldr-data": "tldr",
        "activeDap/ultrafeedback_chosen": "ultrafeedback",
        "yuasosnin/imdb-dpo": "imdb",
        "UCL-DARK/openai-tldr-summarisation-preferences": "tldr",
        "trl-lib/tldr": "tldr",
        "trl-lib/ultrafeedback_binarized": "ultrafeedback",
    }
    return mapping.get(dataset) or slugify_identifier(dataset)


def model_abbreviation(model: str) -> str:
    mapping = {
        "Qwen/Qwen2.5-3B-Instruct": "qwen3b",
        "edbeeching/gpt2-large-imdb": "gpt2",
        "trl-lib/pythia-1b-deduped-tldr-sft": "pythia1b",
        "trl-lib/pythia-2.8b-deduped-tldr-sft": "pythia2.8b",
        "sahandrez/sft-Qwen2.5-1.5B-ultrafeedback": "qwen1.5b",
        "MohamadBazzi/gemma-bpo-sft": "gemma2b",
        "google/gemma-2b": "gemma2b-vanilla",
    }
    if model.startswith("activeDap/"):
        return slugify_identifier(model)
    return mapping.get(model) or slugify_identifier(model)


def eval_judge_abbreviation(judge: str) -> str:
    mapping = {
        "gpt-4o-mini": "openai",
        "pair_rm": "rm",
        "hf": "hf",
        "trl-lib/pythia-1b-deduped-tldr-rm": "pythia1b",
        "trl-lib/pythia-2.8b-deduped-tldr-rm": "pythia2.8b",
        "OpenAssistant/reward-model-deberta-v3-large-v2": "deberta",
        "Skywork/Skywork-Reward-V2-Qwen3-8B": "skywork-v2-qwen3",
        "Skywork/Skywork-Reward-V2-Llama-3.1-8B": "skywork-v2-llama31",
        "PKU-Alignment/beaver-7b-v1.0-reward": "beaver7b",
        "RLHFlow/pair-preference-model-LLaMA3-8B": "llama3-pairrm",
        "nvidia/Qwen-3-Nemotron-32B-Reward": "nemotron32",
    }
    return mapping.get(judge) or slugify_identifier(judge)


def determine_lora_tag(env: Dict[str, str]) -> str:
    extra_args = env.get("EXTRA_TRAINING_ARGS", "")
    enable_lora = env.get("ENABLE_LORA")
    if extra_args and "--use_peft" in extra_args:
        return "lora"
    if enable_lora is None:
        return "lora"
    if enable_lora.lower() in {"false", "0", "no", "off"}:
        return "nolora"
    return "lora"


def parse_int(value: str, default: int) -> int:
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == "":
            return default
        return int(float(value))
    except ValueError:
        return default


def infer_output_dir(env: Dict[str, str], repo_root: Path) -> Path:
    output_dir = env.get("OUTPUT_DIR")
    if output_dir:
        path = Path(output_dir)
        if not path.is_absolute():
            path = (repo_root / path).resolve()
        return path

    merged = dict(DEFAULT_ENV)
    merged.update(env)

    output_root = merged.get("OUTPUT_ROOT_DIR", "outputs")
    output_root_path = Path(output_root)
    if not output_root_path.is_absolute():
        output_root_path = (repo_root / output_root_path).resolve()

    dataset = merged["DATASET"]
    model = merged["MODEL"]
    alignment = merged["ALIGNMENT"]
    loss_type = merged["LOSS_TYPE"]
    eval_judge = merged.get("EVAL_JUDGE", DEFAULT_ENV["EVAL_JUDGE"])
    query_strategy = merged["QUERY_STRATEGY"]
    updates_per_sample = merged.get("UPDATES_PER_SAMPLE", DEFAULT_ENV["UPDATES_PER_SAMPLE"])
    per_device_batch_size = parse_int(merged.get("PER_DEVICE_BATCH_SIZE"), 16)
    gradient_accumulation = parse_int(merged.get("GRADIENT_ACCUMULATION_STEPS"), 1)
    query_freq_factor = parse_int(merged.get("QUERY_FREQ_FACTOR"), 1)
    num_query_env = merged.get("NUM_QUERY")

    cuda_visible = env.get("CUDA_VISIBLE_DEVICES") or os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        num_gpus = len([token for token in cuda_visible.split(",") if token.strip() != ""])
    else:
        num_gpus = parse_int(merged.get("NUM_GPUS"), 1)
    if num_gpus <= 0:
        num_gpus = 1

    # Optionally keep global batch constant across GPUs for naming/defaults.
    fix_global_raw = str(merged.get("FIX_GLOBAL_BATCH", "true")).lower()
    fix_global = fix_global_raw not in {"false", "0", "no", "off"}
    base_num_gpus = parse_int(merged.get("BASE_NUM_GPUS"), 1)
    effective_pdbs = per_device_batch_size
    effective_ga = gradient_accumulation
    if fix_global and num_gpus > 0:
        try:
            effective_pdbs = max(1, (per_device_batch_size * base_num_gpus) // num_gpus)
            denom = max(1, effective_pdbs * num_gpus)
            base_global = per_device_batch_size * gradient_accumulation * base_num_gpus
            effective_ga = max(1, (base_global + denom - 1) // denom)  # ceil
        except Exception:
            effective_pdbs = per_device_batch_size
            effective_ga = gradient_accumulation

    if num_query_env is None:
        num_query = effective_pdbs * num_gpus * query_freq_factor * effective_ga
    else:
        num_query = parse_int(num_query_env, per_device_batch_size)

    # Prefer steps if provided; otherwise track epochs for naming (HF default is 3 epochs)
    max_steps_raw = merged.get("MAX_STEPS")
    num_epochs_raw = merged.get("NUM_TRAIN_EPOCHS")
    training_len_part: str
    if max_steps_raw is not None and str(max_steps_raw).strip() != "":
        try:
            max_steps = int(float(max_steps_raw))
        except Exception:
            max_steps = 0
        training_len_part = f"steps{max_steps}"
    elif num_epochs_raw is not None and str(num_epochs_raw).strip() != "":
        try:
            num_epochs = int(float(num_epochs_raw))
        except Exception:
            num_epochs = 3
        training_len_part = f"epochs{num_epochs}"
    else:
        training_len_part = "epochs3"

    dataset_abbrev = dataset_abbreviation(dataset)
    model_abbrev = model_abbreviation(model)
    judge_abbrev = eval_judge_abbreviation(eval_judge)
    lora_tag = determine_lora_tag(merged)

    dataset_addr = slugify_identifier(dataset)
    model_addr = slugify_identifier(model)

    dir_name = (
        f"{model_addr}_{judge_abbrev}_{alignment}_{query_strategy}"
        f"_query{num_query}_update_per_sample{updates_per_sample}_{training_len_part}"
        f"_batch{effective_pdbs}_gpu{num_gpus}_loss_{loss_type}_{lora_tag}"
    )

    return (output_root_path / dataset_addr / dir_name).resolve()


def prepare_evaluation_invocation(
    evaluation_settings: Dict[str, Any],
    overrides_env: Dict[str, str],
    output_dir: Path,
    gpu_id: str,
) -> Optional[Dict[str, Any]]:
    if not evaluation_settings.get("enabled"):
        return None
    script_path: Optional[Path] = evaluation_settings.get("script")
    if script_path is None:
        return None

    mode = evaluation_settings.get("mode") or "single"
    generate_csv = evaluation_settings.get("generate_csv", "false")
    eval_env_overrides = evaluation_settings.get("env", {})

    lora_enabled = determine_lora_tag(overrides_env) == "lora"
    pretrained_arg = overrides_env.get("MODEL") if lora_enabled else "auto"
    if not pretrained_arg:
        pretrained_arg = "auto"

    command = ["bash", str(script_path), pretrained_arg, str(output_dir)]
    if mode is not None:
        command.append(str(mode))
    if generate_csv is not None:
        command.append(str(generate_csv).lower())

    def truncate_slug(value: str, limit: int = 80) -> str:
        if len(value) <= limit:
            return value
        return value[:limit].rstrip("-_")

    dataset_slug = truncate_slug(slugify_identifier(overrides_env.get("DATASET", "dataset")))
    model_slug = truncate_slug(slugify_identifier(overrides_env.get("MODEL", "model")))
    alignment_slug = slugify_identifier(overrides_env.get("ALIGNMENT", "alignment"))
    lora_tag = determine_lora_tag(overrides_env)
    loss_slug = slugify_identifier(overrides_env.get("LOSS_TYPE", "loss"))
    query_slug = slugify_identifier(overrides_env.get("QUERY_STRATEGY", "query"))

    base_slug_parts = [
        f"align-{alignment_slug}",
        f"lora-{lora_tag}",
        f"loss-{loss_slug}",
        f"query-{query_slug}",
    ]
    config_slug = "__".join(base_slug_parts)
    config_slug = truncate_slug(config_slug, 120)
    hash_input = "|".join(
        [
            overrides_env.get("RUN_NAME", ""),
            overrides_env.get("WANDB_NAME", ""),
            overrides_env.get("ALIGNMENT", ""),
            overrides_env.get("ENABLE_LORA", ""),
            overrides_env.get("LOSS_TYPE", ""),
            overrides_env.get("QUERY_STRATEGY", ""),
            overrides_env.get("EVAL_JUDGE", ""),
            overrides_env.get("DATASET", ""),
            overrides_env.get("MODEL", ""),
        ]
    )
    config_hash = hashlib.sha1(hash_input.encode("utf-8")).hexdigest()[:8]

    eval_env = os.environ.copy()
    for key, value in eval_env_overrides.items():
        eval_env[str(key)] = str(value)
    if gpu_id:
        eval_env["CUDA_VISIBLE_DEVICES"] = gpu_id
    eval_env.setdefault("EVAL_DATASET_SLUG", dataset_slug)
    eval_env.setdefault("EVAL_MODEL_SLUG", model_slug)
    eval_env.setdefault("EVAL_CONFIG_SLUG", config_slug)
    eval_env.setdefault("EVAL_CONFIG_HASH", config_hash)

    return {
        "command": command,
        "env": eval_env,
        "cwd": script_path.parent,
    }


def compute_baseline_evaluation_output_dir(
    evaluation_settings: Dict[str, Any],
    overrides_env: Dict[str, str],
) -> Optional[Path]:
    """Locate the expected lm-eval output directory for a step-0 baseline run."""
    script_path_raw = evaluation_settings.get("script")
    script_path: Optional[Path] = (
        Path(script_path_raw)
        if isinstance(script_path_raw, (str, Path))
        else None
    )
    if script_path is None:
        return None

    results_subdir = evaluation_settings.get("env", {}).get("EVAL_RESULTS_SUBDIR")
    if not results_subdir:
        results_subdir = "evaluation_results"

    base_dir_primary = (script_path.parent / str(results_subdir)).resolve()
    base_dir_fallback = (script_path.parent.parent / str(results_subdir)).resolve()
    base_dir_candidates = [base_dir_primary, base_dir_fallback]

    def truncate_slug(value: str, limit: int = 80) -> str:
        if len(value) <= limit:
            return value
        return value[:limit].rstrip("-_")

    dataset_slug = truncate_slug(slugify_identifier(overrides_env.get("DATASET", "dataset")))
    model_slug = truncate_slug(slugify_identifier(overrides_env.get("MODEL", "model")))
    config_slug = str(evaluation_settings.get("baseline_config_slug") or "baseline")
    config_hash = str(evaluation_settings.get("baseline_config_hash") or "00000000")

    for base_dir in base_dir_candidates:
        candidate = base_dir / dataset_slug / model_slug / f"{config_slug}-{config_hash}"
        if candidate.exists():
            return candidate
    # Return the primary candidate path even if it doesn't exist yet.
    return base_dir_primary / dataset_slug / model_slug / f"{config_slug}-{config_hash}"


def baseline_evaluation_outputs_present(
    evaluation_settings: Dict[str, Any],
    overrides_env: Dict[str, str],
) -> bool:
    eval_dir = compute_baseline_evaluation_output_dir(evaluation_settings, overrides_env)
    if eval_dir is None or not eval_dir.exists():
        return False
    try:
        for item in eval_dir.rglob("*"):
            if item.is_file():
                return True
    except Exception:
        return False
    return False


def prepare_baseline_evaluation_invocation(
    evaluation_settings: Dict[str, Any],
    overrides_env: Dict[str, str],
    dummy_dir: Path,
    gpu_id: str,
) -> Optional[Dict[str, Any]]:
    """Build a run_eval.sh invocation for the base SFT model (no adapter)."""
    if not evaluation_settings.get("enabled"):
        return None
    script_path: Optional[Path] = evaluation_settings.get("script")
    if script_path is None:
        return None

    pretrained_model = overrides_env.get("MODEL", "")
    if not pretrained_model:
        return None

    mode = str(evaluation_settings.get("baseline_mode") or "single")
    generate_csv = str(evaluation_settings.get("baseline_generate_csv") or evaluation_settings.get("generate_csv") or "false")
    eval_env_overrides = evaluation_settings.get("env", {})

    command = ["bash", str(script_path), pretrained_model, str(dummy_dir), mode, generate_csv.lower()]

    def truncate_slug(value: str, limit: int = 80) -> str:
        if len(value) <= limit:
            return value
        return value[:limit].rstrip("-_")

    dataset_slug = truncate_slug(slugify_identifier(overrides_env.get("DATASET", "dataset")))
    model_slug = truncate_slug(slugify_identifier(pretrained_model))
    config_slug = str(evaluation_settings.get("baseline_config_slug") or "baseline")
    config_hash = str(evaluation_settings.get("baseline_config_hash") or "00000000")

    eval_env = os.environ.copy()
    for key, value in eval_env_overrides.items():
        eval_env[str(key)] = str(value)
    if gpu_id:
        eval_env["CUDA_VISIBLE_DEVICES"] = gpu_id

    eval_env["EVAL_DATASET_SLUG"] = dataset_slug
    eval_env["EVAL_MODEL_SLUG"] = model_slug
    eval_env["EVAL_CONFIG_SLUG"] = config_slug
    eval_env["EVAL_CONFIG_HASH"] = config_hash

    return {
        "command": command,
        "env": eval_env,
        "cwd": script_path.parent,
    }


def launch_evaluation_process(
    evaluation_settings: Dict[str, Any],
    experiment: Experiment,
    overrides_env: Dict[str, str],
    output_dir: Path,
    log_path: Path,
    gpu_label: str,
    gpu_id: str,
    row_base: Dict[str, Any],
    console_label: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    prepared = prepare_evaluation_invocation(evaluation_settings, overrides_env, output_dir, gpu_id)
    if prepared is None:
        return None

    if console_label:
        print(console_label)

    log_file = log_path.open("a", encoding="utf-8")
    log_file.write("\n# --- evaluation stage ---\n")
    log_file.write(f"# Evaluation command: {' '.join(prepared['command'])}\n")
    log_file.flush()

    try:
        process = subprocess.Popen(
            prepared["command"],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=prepared["env"],
            cwd=prepared["cwd"],
        )
    except Exception as exc:
        log_file.write(f"# Failed to launch evaluation: {exc}\n")
        log_file.flush()
        log_file.close()
        return None

    return {
        "stage": "eval",
        "experiment": experiment,
        "process": process,
        "log_file": log_file,
        "log_path": log_path,
        "gpu": gpu_label,
        "gpu_id": str(gpu_id),
        "start_time": datetime.utcnow(),
        "output_dir": output_dir,
        "overrides_env": overrides_env,
        "row_base": dict(row_base),
    }


def compute_evaluation_output_dir(
    evaluation_settings: Dict[str, Any],
    overrides_env: Dict[str, str],
    output_dir: Path,
) -> Optional[Path]:
    script_path_raw = evaluation_settings.get("script")
    script_path: Optional[Path] = (
        Path(script_path_raw)
        if isinstance(script_path_raw, (str, Path))
        else None
    )
    if script_path is None:
        return None

    # The lm-evaluation-harness repo lives one directory above the active_dap
    # repo. Its run_eval.sh writes results under:
    #   <lm-evaluation-harness>/<RESULTS_SUBDIR>
    # where RESULTS_SUBDIR defaults to 'evaluation_results' but can be
    # overridden via the EVAL_RESULTS_SUBDIR environment variable passed in
    # `evaluation_settings["env"]`.
    results_subdir = (
        evaluation_settings.get("env", {}).get("EVAL_RESULTS_SUBDIR")
        if isinstance(evaluation_settings, dict)
        else None
    )
    if not results_subdir:
        results_subdir = "evaluation_results"
    # Primary: results are written relative to the harness repo root (cwd when
    # run_eval.sh is invoked).
    base_dir_primary = (script_path.parent / str(results_subdir)).resolve()
    # Fallback: support legacy layouts where results were written relative to a
    # different working directory (one level above the harness repo).
    base_dir_fallback = (script_path.parent.parent / str(results_subdir)).resolve()
    base_dir_candidates = [base_dir_primary, base_dir_fallback]

    # 1) First, try to locate outputs produced via the EVAL_* slug layout that
    #    `prepare_evaluation_invocation` sets when calling run_eval.sh.
    def truncate_slug(value: str, limit: int = 80) -> str:
        if len(value) <= limit:
            return value
        return value[:limit].rstrip("-_")

    dataset_slug = truncate_slug(slugify_identifier(overrides_env.get("DATASET", "dataset")))
    model_slug = truncate_slug(slugify_identifier(overrides_env.get("MODEL", "model")))
    alignment_slug = slugify_identifier(overrides_env.get("ALIGNMENT", "alignment"))
    lora_tag = determine_lora_tag(overrides_env)
    loss_slug = slugify_identifier(overrides_env.get("LOSS_TYPE", "loss"))
    query_slug = slugify_identifier(overrides_env.get("QUERY_STRATEGY", "query"))

    base_slug_parts = [
        f"align-{alignment_slug}",
        f"lora-{lora_tag}",
        f"loss-{loss_slug}",
        f"query-{query_slug}",
    ]
    config_slug = "__".join(base_slug_parts)
    config_slug = truncate_slug(config_slug, 120)
    hash_input = "|".join(
        [
            overrides_env.get("RUN_NAME", ""),
            overrides_env.get("WANDB_NAME", ""),
            overrides_env.get("ALIGNMENT", ""),
            overrides_env.get("ENABLE_LORA", ""),
            overrides_env.get("LOSS_TYPE", ""),
            overrides_env.get("QUERY_STRATEGY", ""),
            overrides_env.get("EVAL_JUDGE", ""),
            overrides_env.get("DATASET", ""),
            overrides_env.get("MODEL", ""),
        ]
    )
    config_hash = hashlib.sha1(hash_input.encode("utf-8")).hexdigest()[:8]

    for base_dir in base_dir_candidates:
        eval_dir_new = base_dir / dataset_slug / model_slug / f"{config_slug}-{config_hash}"
        if eval_dir_new.exists():
            return eval_dir_new
    # 2) Fallback: legacy layout used when RESULT_* env vars were not provided.
    plain_pretrain = None

    adapter_path = output_dir / "adapter_config.json"
    if adapter_path.exists():
        try:
            import json

            data = json.loads(adapter_path.read_text())
            base_model = data.get("base_model_name_or_path")
            if base_model:
                plain_pretrain = slugify_identifier(base_model)
        except Exception:
            plain_pretrain = None

    if plain_pretrain is None:
        pretrained_env = overrides_env.get("MODEL") or overrides_env.get("PRETRAINED_MODEL_NAME_OR_PATH")
        if pretrained_env and pretrained_env != "auto":
            plain_pretrain = slugify_identifier(pretrained_env)

    if plain_pretrain is not None:
        for base_dir in base_dir_candidates:
            candidate = base_dir / plain_pretrain / output_dir.name
            if candidate.exists():
                return candidate

    target_name = output_dir.name
    for base_dir in base_dir_candidates:
        if not base_dir.exists():
            continue
        for child in base_dir.iterdir():
            if not child.is_dir():
                continue
            candidate = child / target_name
            if candidate.exists():
                return candidate
    return None


def evaluation_outputs_present(
    evaluation_settings: Dict[str, Any],
    overrides_env: Dict[str, str],
    output_dir: Path,
) -> bool:
    eval_dir = compute_evaluation_output_dir(evaluation_settings, overrides_env, output_dir)
    if eval_dir is None or not eval_dir.exists():
        return False
    try:
        for item in eval_dir.rglob("*"):
            if item.is_file():
                return True
    except Exception:
        return False
    return False


def cleanup_checkpoints(output_dir: Path, log_path: Path) -> Dict[str, Any]:
    summary = {
        "removed": 0,
        "bytes": 0,
        "status": "skipped",
    }
    if not output_dir.exists():
        return summary

    checkpoint_dirs = sorted(p for p in output_dir.glob("checkpoint-*") if p.is_dir())
    if not checkpoint_dirs:
        return summary

    summary["status"] = "started"
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("\n# --- checkpoint cleanup ---\n")
        for checkpoint_dir in checkpoint_dirs:
            try:
                size_bytes = sum(
                    child.stat().st_size
                    for child in checkpoint_dir.rglob("*")
                    if child.is_file()
                )
            except (FileNotFoundError, PermissionError):
                size_bytes = 0
            try:
                shutil.rmtree(checkpoint_dir)
                summary["removed"] += 1
                summary["bytes"] += size_bytes
                log_file.write(
                    f"# Removed {checkpoint_dir.name}, reclaimed ~{size_bytes / (1024**3):.2f} GB\n"
                )
            except Exception as exc:
                log_file.write(f"# Failed to remove {checkpoint_dir}: {exc}\n")
        log_file.write(
            f"# Cleanup summary: removed {summary['removed']} directories, reclaimed ~{summary['bytes'] / (1024**3):.2f} GB\n"
        )

    summary["status"] = "completed"
    return summary


def build_experiments(config: Dict[str, Any]) -> List[Experiment]:
    datasets = config.get("datasets", {})
    alignments = config.get("alignments", [])
    lora_modes = config.get("lora_modes", [])
    query_strategies = config.get("query_strategies", [])
    seeds_raw = config.get("seeds")

    if not datasets:
        raise ValueError("No datasets defined in the configuration.")
    if not alignments:
        raise ValueError("No alignment modes defined in the configuration.")
    if not lora_modes:
        raise ValueError("No LoRA modes defined in the configuration.")
    if not query_strategies:
        raise ValueError("No query strategies defined in the configuration.")

    seeds: List[Optional[int]] = [None]
    if seeds_raw is not None:
        if not isinstance(seeds_raw, (list, tuple)):
            raise ValueError("Config key 'seeds' must be a list of integers.")
        seeds = []
        for item in seeds_raw:
            try:
                seeds.append(int(item))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid seed value in config: {item!r}") from exc
        if not seeds:
            raise ValueError("Config key 'seeds' must contain at least one seed.")

    experiments: List[Experiment] = []

    for seed in seeds:
        for query in query_strategies:
            for lora in lora_modes:
                for dataset, spec in datasets.items():
                    models = spec.get("models", [])
                    if not models:
                        raise ValueError(f"Dataset '{dataset}' does not list any models.")
                    dataset_env = spec.get("env", {})

                    for model in models:
                        for alignment in alignments:
                            env: Dict[str, str] = {}
                            env.update(dataset_env)
                            env["DATASET"] = dataset
                            env["MODEL"] = model
                            if seed is not None:
                                env["SEED"] = str(seed)
                            env.update(alignment.get("env", {}))
                            env.update(lora.get("env", {}))
                            env.update(query.get("env", {}))

                            run_id_parts = [
                                short_name(dataset),
                                short_name(model),
                                sanitize(alignment.get("name", "align")),
                                sanitize(lora.get("name", "lora")),
                                sanitize(query.get("name", "query")),
                            ]
                            if seed is not None:
                                run_id_parts.append(f"seed{seed}")

                            run_id = "__".join(run_id_parts)

                            experiments.append(
                                Experiment(
                                    run_id=run_id,
                                    dataset=dataset,
                                    model=model,
                                    alignment_name=alignment.get("name", ""),
                                    lora_name=lora.get("name", ""),
                                    query_name=query.get("name", ""),
                                    env=env,
                                )
                            )
    return experiments


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def append_csv(row: Dict[str, Any], csv_path: Path, fieldnames: Iterable[str]) -> None:
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Queue Active DAP experiments.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/active_experiment_queue.yaml"),
        help="Path to the queue configuration YAML.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned experiments and exit without launching anything.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Override the maximum number of concurrent runs (defaults to config value).",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated list of GPU indices to cycle through (default inferred from concurrency).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip experiments whose log files already exist.",
    )
    parser.add_argument(
        "--sanity-check",
        action="store_true",
        help="Run each experiment as a short 5-step sanity check with heavy logging.",
    )
    return parser.parse_args()


def build_override_env(
    base_env: Dict[str, Any],
    experiment_env: Dict[str, Any],
    run_id: str,
    gpu: Optional[str] = None,
    extra_env: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    overrides.update({k: str(v) for k, v in base_env.items()})
    overrides.update({k: str(v) for k, v in experiment_env.items()})
    if extra_env:
        overrides.update({k: str(v) for k, v in extra_env.items()})
    overrides.setdefault("RUN_NAME", run_id)
    overrides.setdefault("WANDB_NAME", run_id)
    if gpu is not None:
        overrides["CUDA_VISIBLE_DEVICES"] = gpu
    return overrides


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)

    defaults = config.get("defaults", {})
    script_path = Path(defaults.get("script_path", "commands/run_active_envaware.sh"))
    if not script_path.exists():
        raise FileNotFoundError(f"Cannot find launcher script at {script_path}")

    log_dir = Path(defaults.get("log_dir", "logs/active_queue")).resolve()
    csv_path = log_dir / "experiment_status.csv"
    repo_root = script_path.parent.parent.resolve()
    include_seed_in_output = bool(defaults.get("include_seed_in_output", False))

    evaluation_completed: set[str] = set()
    if csv_path.exists():
        try:
            with csv_path.open("r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    if row.get("run_id") and row.get("evaluation_status") == "success":
                        evaluation_completed.add(row["run_id"])
        except Exception:
            evaluation_completed = set()

    evaluation_raw = defaults.get("evaluation", {})
    evaluation_enabled = bool(evaluation_raw.get("enabled"))
    evaluation_settings = {
        "enabled": evaluation_enabled,
        "script": None,
        "mode": str(evaluation_raw.get("mode", "single")),
        "generate_csv": str(evaluation_raw.get("generate_csv", "false")),
        "baseline_enabled": bool(evaluation_raw.get("baseline", False)),
        "baseline_mode": str(evaluation_raw.get("baseline_mode", "single")),
        "baseline_generate_csv": str(evaluation_raw.get("baseline_generate_csv", evaluation_raw.get("generate_csv", "false"))),
        "baseline_config_slug": str(evaluation_raw.get("baseline_config_slug", "baseline")),
        "baseline_config_hash": str(evaluation_raw.get("baseline_config_hash", "00000000")),
        "env": {str(k): str(v) for k, v in evaluation_raw.get("env", {}).items()},
        "cleanup_checkpoints": bool(evaluation_raw.get("cleanup_checkpoints", False)),
    }
    if evaluation_enabled:
        eval_script_path = Path(os.path.expandvars(str(evaluation_raw.get("script_path", ""))))
        if not eval_script_path.is_absolute():
            eval_script_path = (repo_root / eval_script_path).resolve()
        if not eval_script_path.exists():
            raise FileNotFoundError(f"Evaluation script not found at {eval_script_path}")
        evaluation_settings["script"] = eval_script_path

    system_gpu_count = detect_system_gpu_count()

    sanity_common_env: Dict[str, str] = {}
    sanity_output_base: Optional[Path] = None
    if args.sanity_check:
        # Keep sanity outputs inside the repo by default to avoid relying on
        # external symlinks (some environments map outputs_sanity elsewhere).
        sanity_output_base = (repo_root / "outputs_sanity_local").resolve()
        sanity_common_env = {
            "MAX_STEPS": "5",
            "SAVE_STEPS": "1000000",
            "LOGGING_STEPS": "2",
            "EVAL_STEPS": "2",
            # Keep win-rate eval lightweight; this is just a smoke test.
            "NUM_EVAL_PROMPTS": "20",
            "WANDB_MODE": "offline",
            "UPLOAD_TO_HF": "false",
            # Never upload replay histories during sanity checks.
            "HISTORY_LOG_MODE": "none",
            "SANITY_CHECK": "1",
        }
        ensure_dir(sanity_output_base)

        # Sanity checks are intended to validate training loops quickly; skip
        # expensive lm-evaluation-harness stages (and baseline eval) entirely.
        evaluation_settings["enabled"] = False
        evaluation_settings["baseline_enabled"] = False

    if args.gpus:
        gpu_pool = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    else:
        defaults_gpu_entry = defaults.get("gpus")
        if isinstance(defaults_gpu_entry, str):
            gpu_pool = [gpu.strip() for gpu in defaults_gpu_entry.split(",") if gpu.strip()]
        elif isinstance(defaults_gpu_entry, (list, tuple)):
            gpu_pool = [str(gpu) for gpu in defaults_gpu_entry]
        else:
            gpu_pool = [str(i) for i in range(system_gpu_count)]

    if not gpu_pool:
        gpu_pool = ["0"]

    config_max = defaults.get("max_concurrency")
    try:
        config_max_int = int(config_max) if config_max is not None else 0
    except (TypeError, ValueError):
        config_max_int = 0
    if config_max_int <= 0:
        config_max_int = len(gpu_pool)

    if args.max_concurrency is not None:
        max_concurrency = args.max_concurrency
    else:
        max_concurrency = config_max_int

    if max_concurrency < 1:
        raise ValueError("max_concurrency must be >= 1")

    if max_concurrency > len(gpu_pool):
        print(
            f"[WARN] max_concurrency={max_concurrency} reduced to match available GPUs ({len(gpu_pool)})."
        )
        max_concurrency = len(gpu_pool)

    base_env = defaults.get("base_env", {})

    experiments = build_experiments(config)

    print(f"Loaded {len(experiments)} experiments from {args.config}")

    if args.dry_run:
        for idx, exp in enumerate(experiments):
            extra_env = None
            if args.sanity_check:
                assert sanity_output_base is not None
                extra_env = dict(sanity_common_env)
                extra_env["OUTPUT_DIR"] = str((sanity_output_base / exp.run_id).resolve())
            if extra_env is None:
                extra_env = {}
            gpu_id = gpu_pool[idx % len(gpu_pool)]
            num_gpus_for_run = len(gpu_id.split(",")) if gpu_id else 1
            extra_env.setdefault("NUM_GPUS", str(num_gpus_for_run))
            overrides_env = build_override_env(base_env, exp.env, exp.run_id, gpu_id, extra_env=extra_env)
            if include_seed_in_output and "OUTPUT_DIR" not in overrides_env:
                seed_value = overrides_env.get("SEED")
                if seed_value is not None and str(seed_value).strip() != "":
                    base_output_dir = infer_output_dir(overrides_env, repo_root)
                    overrides_env["OUTPUT_DIR"] = str(
                        base_output_dir.with_name(f"{base_output_dir.name}_seed{seed_value}")
                    )

            # Infer training output directory and existing artifacts so that
            # dry-run can report which code path would be taken in a real run.
            output_dir = infer_output_dir(overrides_env, repo_root)
            win_rates_path = output_dir / "win_rates.json"
            has_win_rates = win_rates_path.exists()

            log_path = log_dir / f"{exp.run_id}.log"
            has_log = log_path.exists()

            # Determine whether this run would enter the eval-only path,
            # the train path (fresh or resume), or be skipped immediately
            # due to an existing win_rates.json.
            mode = "train_fresh"
            reason = ""

            eval_enabled = evaluation_settings.get("enabled")
            eval_only_eligible = False
            if eval_enabled and exp.run_id not in evaluation_completed:
                if has_win_rates and any(output_dir.glob("checkpoint-*")):
                    if not evaluation_outputs_present(evaluation_settings, overrides_env, output_dir):
                        eval_only_eligible = True

            if eval_only_eligible:
                mode = "eval_only"
                reason = "existing win_rates.json, checkpoints present, no eval outputs yet"
            elif has_win_rates:
                if args.resume and has_log:
                    mode = "skip_train_existing_win_rates_with_resume"
                    reason = "log and win_rates.json already exist; --resume would skip training"
                else:
                    mode = "skip_train_existing_win_rates"
                    reason = "win_rates.json already exists; training would be skipped"
            else:
                if args.resume and has_log:
                    mode = "train_resume"
                    reason = "log exists but no win_rates.json; training would resume"
                else:
                    mode = "train_fresh"
                    reason = "no win_rates.json; training would start from scratch"

            print(
                f"[DRY] {exp.run_id}: dataset={exp.dataset} model={exp.model} "
                f"align={exp.alignment_name} lora={exp.lora_name} query={exp.query_name}"
            )
            print(f"      env overrides: {flatten_env(overrides_env)}")
            print(f"      output_dir: {output_dir}")
            print(f"      mode_if_run: {mode} ({reason})")
        return 0

    ensure_dir(log_dir)

    # Optional: run step-0 lm-eval for the base (SFT) model once per dataset×model.
    if (
        evaluation_settings.get("enabled")
        and evaluation_settings.get("baseline_enabled")
        and not args.sanity_check
    ):
        baseline_dummy_dir = (repo_root / "outputs_paper" / "_baseline_dummy").resolve()
        ensure_dir(baseline_dummy_dir)
        baseline_log_dir = (log_dir / "baseline_eval").resolve()
        ensure_dir(baseline_log_dir)

        seen_keys: set[tuple[str, str]] = set()
        baseline_queue: deque[dict[str, Any]] = deque()
        for exp in experiments:
            key = (exp.dataset, exp.model)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            baseline_env = {"DATASET": exp.dataset, "MODEL": exp.model}
            if baseline_evaluation_outputs_present(evaluation_settings, baseline_env):
                continue
            baseline_queue.append(
                {
                    "dataset": exp.dataset,
                    "model": exp.model,
                    "env": baseline_env,
                }
            )

        if baseline_queue:
            print(f"Running baseline lm-eval for {len(baseline_queue)} dataset×model pairs...")

            available_gpus = deque(gpu_pool)
            active_baseline: list[dict[str, Any]] = []

            while baseline_queue or active_baseline:
                while baseline_queue and available_gpus and len(active_baseline) < max_concurrency:
                    task = baseline_queue.popleft()
                    gpu_id = available_gpus.popleft()
                    prepared = prepare_baseline_evaluation_invocation(
                        evaluation_settings,
                        task["env"],
                        baseline_dummy_dir,
                        gpu_id,
                    )
                    if prepared is None:
                        print(f"[BASELINE][SKIP] Could not prepare eval for {task['model']} ({task['dataset']})")
                        available_gpus.append(gpu_id)
                        continue

                    dataset_slug = slugify_identifier(task["dataset"])
                    model_slug = slugify_identifier(task["model"])
                    log_path = baseline_log_dir / f"baseline__{dataset_slug}__{model_slug}.log"
                    log_file = log_path.open("w", encoding="utf-8")
                    log_file.write(f"# Baseline lm-eval (step 0)\n")
                    log_file.write(f"# Command: {' '.join(prepared['command'])}\n")
                    log_file.write(f"# CUDA_VISIBLE_DEVICES={gpu_id}\n\n")
                    log_file.flush()

                    try:
                        process = subprocess.Popen(
                            prepared["command"],
                            stdout=log_file,
                            stderr=subprocess.STDOUT,
                            env=prepared["env"],
                            cwd=prepared["cwd"],
                        )
                    except Exception as exc:
                        log_file.write(f"# Failed to launch baseline eval: {exc}\n")
                        log_file.flush()
                        log_file.close()
                        available_gpus.append(gpu_id)
                        continue

                    active_baseline.append(
                        {
                            "task": task,
                            "process": process,
                            "log_file": log_file,
                            "log_path": log_path,
                            "gpu": gpu_id,
                        }
                    )
                    print(
                        f"[BASELINE] {task['model']} ({task['dataset']}) on GPU {gpu_id} -> {log_path}"
                    )

                # Poll for completions.
                finished: list[dict[str, Any]] = []
                for entry in active_baseline:
                    proc = entry["process"]
                    code = proc.poll()
                    if code is None:
                        continue
                    finished.append(entry)
                    try:
                        entry["log_file"].write(f"\n# Exit code: {code}\n")
                        entry["log_file"].flush()
                    except Exception:
                        pass
                    try:
                        entry["log_file"].close()
                    except Exception:
                        pass
                    available_gpus.append(entry["gpu"])
                    task = entry["task"]
                    status = "OK" if code == 0 else f"FAIL({code})"
                    print(f"[BASELINE][{status}] {task['model']} ({task['dataset']})")

                if finished:
                    for entry in finished:
                        active_baseline.remove(entry)
                else:
                    time.sleep(1.0)

    pending = deque(experiments)
    active: List[Dict[str, Any]] = []
    interrupted = False

    def handle_sigint(signum, frame):
        nonlocal interrupted
        print("Received interrupt; waiting for active jobs to finish...")
        interrupted = True

    signal.signal(signal.SIGINT, handle_sigint)

    fieldnames = [
        "run_id",
        "dataset",
        "model",
        "alignment",
        "lora_mode",
        "query_strategy",
        "loss_type",
        "enable_lora",
        "start_time",
        "end_time",
        "duration_sec",
        "gpu",
        "exit_code",
        "status",
        "log_path",
        "sanity_check",
        "evaluation_status",
        "evaluation_exit_code",
        "evaluation_duration_sec",
        "checkpoints_removed",
        "cleanup_bytes_reclaimed_gb",
    ]

    def process_evaluation_only(gpu_label: str) -> bool:
        if not evaluation_settings.get("enabled"):
            return False
        for exp in list(pending):
            extra_env = None
            if args.sanity_check:
                assert sanity_output_base is not None
                extra_env = dict(sanity_common_env)
                extra_env["OUTPUT_DIR"] = str((sanity_output_base / exp.run_id).resolve())
            if extra_env is None:
                extra_env = {}
            num_gpus_for_dir = len(gpu_label.split(",")) if gpu_label else 1
            extra_env.setdefault("NUM_GPUS", str(num_gpus_for_dir))
            overrides_env = build_override_env(base_env, exp.env, exp.run_id, gpu_label, extra_env=extra_env)
            if include_seed_in_output and "OUTPUT_DIR" not in overrides_env:
                seed_value = overrides_env.get("SEED")
                if seed_value is not None and str(seed_value).strip() != "":
                    base_output_dir = infer_output_dir(overrides_env, repo_root)
                    overrides_env["OUTPUT_DIR"] = str(
                        base_output_dir.with_name(f"{base_output_dir.name}_seed{seed_value}")
                    )
            output_dir = infer_output_dir(overrides_env, repo_root)
            win_rates_path = output_dir / "win_rates.json"
            if not win_rates_path.exists():
                continue
            if evaluation_outputs_present(evaluation_settings, overrides_env, output_dir):
                # Evaluation artifacts already present for this run in the
                # configured results directory; skip eval-only.
                continue
            if not any(output_dir.glob("checkpoint-*")):
                continue

            pending.remove(exp)
            log_path = log_dir / f"{exp.run_id}.log"
            ensure_dir(log_path.parent)

            row_base = {
                "run_id": exp.run_id,
                "dataset": exp.dataset,
                "model": exp.model,
                "alignment": exp.alignment_name,
                "lora_mode": exp.lora_name,
                "query_strategy": exp.query_name,
                "loss_type": exp.env.get("LOSS_TYPE", ""),
                "enable_lora": exp.env.get("ENABLE_LORA", ""),
                "gpu": gpu_label,
                "exit_code": "",
                "status": "eval_only_pending",
                "log_path": str(log_path),
                "sanity_check": "true" if args.sanity_check else "false",
            }

            entry = launch_evaluation_process(
                evaluation_settings,
                exp,
                overrides_env,
                output_dir,
                log_path,
                gpu_label,
                gpu_label,
                row_base,
                console_label=(
                    f"[EVAL-ONLY] {exp.run_id} on GPU {gpu_label} "
                    "(lm-evaluation-harness before resuming training)"
                ),
            )

            if entry is None:
                now = datetime.utcnow()
                row = row_base.copy()
                row["start_time"] = now.isoformat()
                row["end_time"] = now.isoformat()
                row["duration_sec"] = "0.0"
                row["status"] = "eval_only_failed"
                row["evaluation_status"] = "error"
                row["evaluation_exit_code"] = ""
                row["evaluation_duration_sec"] = "0.0"
                row["checkpoints_removed"] = ""
                row["cleanup_bytes_reclaimed_gb"] = ""
                append_csv(row, csv_path, fieldnames)
                print(
                    f"[EVAL_ONLY_FAILED] {exp.run_id} (evaluation launch failed) -> {log_path}"
                )
                continue

            entry["eval_only"] = True
            active.append(entry)
            return True
        return False

    while (pending and not interrupted) or active:
        # Launch new experiments if slots are free.
        while (
            not interrupted
            and pending
            and len(active) < max_concurrency
        ):
            used_gpus = {entry["gpu_id"] for entry in active}
            gpu_id = next((gpu for gpu in gpu_pool if gpu not in used_gpus), None)
            if gpu_id is None:
                break
            gpu_label = gpu_id
            override_gpu = gpu_id
            num_gpus_for_dir = len(gpu_id.split(",")) if gpu_id else 1

            if process_evaluation_only(gpu_label):
                if len(gpu_pool) > 1:
                    gpu_pool = gpu_pool[1:] + gpu_pool[:1]
                continue

            exp = pending.popleft()
            log_path = log_dir / f"{exp.run_id}.log"

            extra_env = None
            if args.sanity_check:
                assert sanity_output_base is not None
                extra_env = dict(sanity_common_env)
                extra_env["OUTPUT_DIR"] = str((sanity_output_base / exp.run_id).resolve())
            if extra_env is None:
                extra_env = {}
            extra_env.setdefault("NUM_GPUS", str(num_gpus_for_dir))

            overrides_env = build_override_env(base_env, exp.env, exp.run_id, override_gpu, extra_env=extra_env)
            if include_seed_in_output and "OUTPUT_DIR" not in overrides_env:
                seed_value = overrides_env.get("SEED")
                if seed_value is not None and str(seed_value).strip() != "":
                    base_output_dir = infer_output_dir(overrides_env, repo_root)
                    overrides_env["OUTPUT_DIR"] = str(
                        base_output_dir.with_name(f"{base_output_dir.name}_seed{seed_value}")
                    )
            output_dir = infer_output_dir(overrides_env, repo_root)
            win_rates_path = output_dir / "win_rates.json"

            has_log = log_path.exists()
            has_win_rates = win_rates_path.exists()

            if has_win_rates:
                if args.resume and has_log:
                    print(f"[SKIP] {exp.run_id} (log and win_rates.json already exist; --resume set)")
                else:
                    print(f"[SKIP] {exp.run_id} (found existing win_rates.json at {win_rates_path})")
                continue

            if args.resume and has_log:
                # If checkpoints exist, resume from the latest one by injecting
                # `--resume_from_checkpoint` into EXTRA_TRAINING_ARGS.
                latest_checkpoint: Optional[Path] = None
                try:
                    checkpoint_dirs = [p for p in output_dir.glob("checkpoint-*") if p.is_dir()]
                    if checkpoint_dirs:
                        checkpoint_dirs.sort(
                            key=lambda p: int(re.sub(r"^checkpoint-", "", p.name)) if re.match(r"^checkpoint-\d+$", p.name) else -1
                        )
                        candidate = checkpoint_dirs[-1]
                        if candidate.exists():
                            latest_checkpoint = candidate.resolve()
                except Exception:
                    latest_checkpoint = None

                if latest_checkpoint is not None:
                    extra_args = overrides_env.get("EXTRA_TRAINING_ARGS", "").strip()
                    if "--resume_from_checkpoint" not in extra_args:
                        extra_args = (extra_args + f" --resume_from_checkpoint {latest_checkpoint}").strip()
                        overrides_env["EXTRA_TRAINING_ARGS"] = extra_args
                    print(
                        f"[RESUME] {exp.run_id} resuming from {latest_checkpoint}"
                    )
                else:
                    print(
                        f"[RESUME] {exp.run_id} has prior log but no win_rates.json; re-running."
                    )

            ensure_dir(log_path.parent)
            log_file = log_path.open("w", encoding="utf-8")

            print(f"[LAUNCH] {exp.run_id} on GPU {gpu_label}")
            print(f"        env overrides: {flatten_env(overrides_env)}")
            log_file.write(f"# Run ID: {exp.run_id}\n")
            log_file.write(f"# Launch time (UTC): {datetime.utcnow().isoformat()}\n")
            log_file.write(f"# GPU: {gpu_label}\n")
            log_file.write(f"# Environment overrides: {flatten_env(overrides_env)}\n")
            log_file.write("# --- subprocess output ---\n")
            log_file.flush()

            env = os.environ.copy()
            env.update(overrides_env)

            process = subprocess.Popen(
                ["bash", str(script_path)],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
            )

            active.append(
                {
                    "stage": "train",
                    "experiment": exp,
                    "process": process,
                    "log_file": log_file,
                    "log_path": log_path,
                    "gpu": gpu_label,
                    "gpu_id": gpu_id,
                    "start_time": datetime.utcnow(),
                    "overrides_env": overrides_env,
                    "output_dir": output_dir,
                }
            )

        # Poll active processes.
        for entry in list(active):
            proc: subprocess.Popen = entry["process"]
            retcode = proc.poll()
            if retcode is None:
                continue

            proc.wait()
            entry["log_file"].close()
            active.remove(entry)

            stage = entry.get("stage", "train")
            exp: Experiment = entry["experiment"]

            if stage == "eval":
                eval_end = datetime.utcnow()
                start_time = entry["start_time"]
                row = dict(entry.get("row_base", {}))
                row.setdefault("run_id", exp.run_id)
                row.setdefault("dataset", exp.dataset)
                row.setdefault("model", exp.model)
                row.setdefault("alignment", exp.alignment_name)
                row.setdefault("lora_mode", exp.lora_name)
                row.setdefault("query_strategy", exp.query_name)
                row.setdefault("loss_type", exp.env.get("LOSS_TYPE", ""))
                row.setdefault("enable_lora", exp.env.get("ENABLE_LORA", ""))
                row.setdefault("gpu", entry.get("gpu", ""))
                if not row.get("start_time"):
                    row["start_time"] = start_time.isoformat()
                if not row.get("end_time"):
                    row["end_time"] = eval_end.isoformat()
                if not row.get("duration_sec"):
                    row["duration_sec"] = f"{(eval_end - start_time).total_seconds():.1f}"
                row.setdefault("log_path", str(entry["log_path"]))
                row.setdefault("sanity_check", "true" if args.sanity_check else "false")
                eval_status = "success" if retcode == 0 else "failed"
                row["evaluation_status"] = eval_status
                row["evaluation_exit_code"] = "" if retcode is None else str(retcode)
                row["evaluation_duration_sec"] = f"{(eval_end - start_time).total_seconds():.1f}"

                checkpoints_removed = ""
                cleanup_bytes = ""
                if (
                    eval_status == "success"
                    and evaluation_settings.get("cleanup_checkpoints")
                ):
                    cleanup_result = cleanup_checkpoints(entry.get("output_dir", Path(".")), entry["log_path"])
                    checkpoints_removed = str(cleanup_result.get("removed", ""))
                    reclaimed_bytes = cleanup_result.get("bytes")
                    if isinstance(reclaimed_bytes, (int, float)) and reclaimed_bytes > 0:
                        cleanup_bytes = f"{reclaimed_bytes / (1024**3):.2f}"
                row["checkpoints_removed"] = checkpoints_removed
                row["cleanup_bytes_reclaimed_gb"] = cleanup_bytes

                if eval_status == "success":
                    evaluation_completed.add(exp.run_id)

                if entry.get("eval_only"):
                    row["status"] = "eval_only_success" if eval_status == "success" else "eval_only_failed"
                    row.setdefault("exit_code", "")
                else:
                    if eval_status != "success":
                        row["status"] = "eval_failed"
                    else:
                        row["status"] = row.get("status", "success")

                append_csv(row, csv_path, fieldnames)
                train_exit = row.get('exit_code', '') or '-'
                eval_exit = row.get('evaluation_exit_code', '') or '-'
                print(
                    f"[{row['status'].upper()}] {exp.run_id} "
                    f"(train exit {train_exit}, eval exit {eval_exit}, "
                    f"removed {checkpoints_removed or '0'} ckpts) -> {entry['log_path']}"
                )
                continue

            # Training stage completion
            end_time = datetime.utcnow()
            start_time = entry["start_time"]
            status = "success" if retcode == 0 else "failed"

            loss_type = exp.env.get("LOSS_TYPE", "")
            enable_lora = exp.env.get("ENABLE_LORA", "")

            row_base = {
                "run_id": exp.run_id,
                "dataset": exp.dataset,
                "model": exp.model,
                "alignment": exp.alignment_name,
                "lora_mode": exp.lora_name,
                "query_strategy": exp.query_name,
                "loss_type": loss_type,
                "enable_lora": enable_lora,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_sec": f"{(end_time - start_time).total_seconds():.1f}",
                "gpu": entry["gpu"],
                "exit_code": str(retcode),
                "status": status,
                "log_path": str(entry["log_path"]),
                "sanity_check": "true" if args.sanity_check else "false",
            }

            if retcode != 0:
                row = row_base.copy()
                row["evaluation_status"] = "skipped"
                row["evaluation_exit_code"] = ""
                row["evaluation_duration_sec"] = ""
                row["checkpoints_removed"] = ""
                row["cleanup_bytes_reclaimed_gb"] = ""
                append_csv(row, csv_path, fieldnames)
                print(
                    f"[{status.upper()}] {exp.run_id} (train exit {retcode}) -> {entry['log_path']}"
                )
                continue

            eval_entry = None
            if evaluation_settings.get("enabled"):
                eval_entry = launch_evaluation_process(
                    evaluation_settings,
                    exp,
                    entry.get("overrides_env", {}),
                    entry.get("output_dir", Path(".")),
                    entry["log_path"],
                    entry.get("gpu", ""),
                    entry.get("gpu_id", ""),
                    row_base,
                    console_label=(
                        f"[EVAL] {exp.run_id} on GPU {entry.get('gpu_id') or ''} (lm-evaluation-harness)"
                    ),
                )
                if eval_entry is not None:
                    eval_entry["eval_only"] = False
                    active.append(eval_entry)
                    continue

            row = row_base.copy()
            if evaluation_settings.get("enabled"):
                row["evaluation_status"] = "error"
                row["status"] = "eval_failed"
            else:
                row["evaluation_status"] = "skipped"
            row["evaluation_exit_code"] = ""
            row["evaluation_duration_sec"] = ""
            row["checkpoints_removed"] = ""
            row["cleanup_bytes_reclaimed_gb"] = ""
            append_csv(row, csv_path, fieldnames)
            print(
                f"[{row['status'].upper()}] {exp.run_id} (train exit {retcode}, eval {row['evaluation_status']}) -> {entry['log_path']}"
            )

        if active and not interrupted:
            time.sleep(5)

    if interrupted:
        print("Queue interrupted by user. Remaining experiments were not started.")
        return 1

    print("Queue completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
