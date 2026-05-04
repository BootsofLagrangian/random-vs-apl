# random-vs-apl

Reproduction code for **Random Is Hard to Beat: Active Selection in online DPO with Modern LLMs** (arXiv:2604.02766).

This repository vendors a modified TRL fork used for the paper. The `trl/` directory is included intentionally because the paper experiments depend on paper-specific trainer and active-learning changes that are not available in upstream TRL.

## What Is Included

- `trl/`: the modified TRL fork used by the experiments
- `active_learning/`: APL, random, and related query strategy code
- `scripts/active_queue_runner.py`: sweep orchestration for paper experiments
- `commands/run_active_envaware.sh`: environment-aware launcher used by the queue runner
- `commands/run_active_queue.sh`: convenience queue runner wrapper
- `commands/run_reproduce.sh`: public reproduction entry point
- `configs/paper_sweep_A_harmful_multiseed.yaml`: harmfulness stability and reward-model dependence sweep
- `configs/paper_sweep_B_controls.yaml`: helpfulness and UltraFeedback control sweep
- `scripts/verify_public_repo.py`: repository hygiene verifier

Artifacts, checkpoints, logs, result CSVs, raw replay files, and generated model weights are not included. Runs regenerate outputs locally under ignored directories such as `outputs_paper/`, `logs/`, and `evaluation_results_paper/`.

## Requirements

Full reproduction requires:

- Python 3.9+
- CUDA GPU environment
- Hugging Face access to the listed base and reward models
- Optional: an `lm-evaluation-harness` checkout for post-training evaluation

Install the Python dependencies in your preferred environment:

```bash
pip install -r requirements.txt
pip install -e .
```

The default public configs set `WANDB_MODE=disabled`, do not upload histories or models, and write generated files to repo-local ignored directories.

## Quick Verification

```bash
python3 scripts/verify_public_repo.py
python3 scripts/active_queue_runner.py --config configs/paper_sweep_A_harmful_multiseed.yaml --dry-run
python3 scripts/active_queue_runner.py --config configs/paper_sweep_B_controls.yaml --dry-run
python3 -m compileall trl active_learning scripts
```

## Reproduction Commands

Preview Sweep A:

```bash
python3 scripts/active_queue_runner.py --config configs/paper_sweep_A_harmful_multiseed.yaml --dry-run
bash commands/run_reproduce.sh --sweep A --dry-run
```

Preview Sweep B:

```bash
python3 scripts/active_queue_runner.py --config configs/paper_sweep_B_controls.yaml --dry-run
bash commands/run_reproduce.sh --sweep B --dry-run
```

Launch a full sweep after confirming GPU and model access:

```bash
bash commands/run_reproduce.sh --sweep A
bash commands/run_reproduce.sh --sweep B
```

Use `--max-concurrency` and `--gpus` to control scheduling:

```bash
bash commands/run_reproduce.sh --sweep A --max-concurrency 2 --gpus 0,1
```

## Optional Evaluation Harness

The queue configs keep external lm-evaluation disabled by default so dry-runs and training setup do not require another repository. To enable it, edit the `defaults.evaluation.enabled` field in the target config and provide harness paths via environment variables:

```bash
export LM_EVAL_ROOT=/path/to/lm-evaluation-harness
export LM_EVAL_BIN="$LM_EVAL_ROOT/venv/bin/lm_eval"
export PYTHON_BIN="$LM_EVAL_ROOT/venv/bin/python"
export EVAL_RESULTS_SUBDIR=evaluation_results_paper
```

Evaluation outputs should remain under ignored local directories and should not be committed.

## Licensing

The vendored TRL code keeps the upstream Apache-2.0 license terms. See `LICENSE`.

## Citation

```bibtex
@misc{oh2026randomhardbeat,
  title = {Random Is Hard to Beat: Active Selection in online DPO with Modern LLMs},
  author = {Oh, Giyeong and Lee, Junghyun and Park, Jaehyun and Yu, Youngjae and Bae, Wonho and Noh, Junhyug},
  year = {2026},
  eprint = {2604.02766},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url = {https://arxiv.org/abs/2604.02766}
}
```

## AI Assistance Disclosure

Public repository cleanup and documentation were organized with AI assistance from GPT-5.5 high Codex.
