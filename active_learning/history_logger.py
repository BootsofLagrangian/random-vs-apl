import os
import json
import time
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from datasets import load_dataset, Dataset
except Exception:
    load_dataset = None
    Dataset = None


def _now_iso() -> str:
    try:
        import pandas as pd
        return pd.Timestamp.utcnow().isoformat()
    except Exception:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _slugify(value: str) -> str:
    import re
    value = (value or "").strip()
    value = value.replace("/", "-")
    value = re.sub(r"[^0-9A-Za-z._+-]+", "-", value)
    value = re.sub(r"-+", "-", value)
    return value.strip("-") or "unknown"


class HistoryLogger:
    """
    Lightweight JSONL logger for online DPO/XPO training histories with
    optional final push to the Hugging Face Hub as a dataset.

    Configuration is primarily via environment variables to avoid touching
    the existing CLI/dataclasses:

      - HISTORY_LOG_MODE:    'none' | 'local' | 'hub' | 'both' (default: 'none')
      - HISTORY_LOCAL_DIR:   local directory root (default: outputs_crossover/replay)
      - history hub namespace env var: org/user namespace for dataset push
      - history private env var: '1' or '0' (default: '0')
      - HISTORY_FLUSH_STEPS: integer batch flush interval (default: 1)
    """

    def __init__(self, run_meta: Dict[str, Any]):
        self.run_meta = dict(run_meta or {})
        env = os.environ
        self.mode = env.get("HISTORY_LOG_MODE", "none").lower()
        # Root under which we place replay/<dataset>/<model>/<algo>
        self.local_root = Path(env.get("HISTORY_LOCAL_DIR", "outputs_crossover")).expanduser()
        self.hub_ns = env.get("HISTORY" + "_HUB_NS", "")
        self.hub_private = (env.get("HISTORY" + "_PRIVATE", "0") == "1")
        self.flush_steps = int(env.get("HISTORY_FLUSH_STEPS", "1") or 1)

        # Derive a stable address for the run under local_root: replay/<dataset>/<model>/<algo>
        dataset_slug = _slugify(self.run_meta.get("dataset", "dataset"))
        model_slug = _slugify(self.run_meta.get("model", "model"))
        algo_slug = _slugify(self.run_meta.get("query", "algo"))
        align_slug = _slugify(self.run_meta.get("alignment", "align"))

        # Put alignment into folder to avoid collisions across methods
        self.local_run_dir = self.local_root / "replay" / dataset_slug / model_slug / f"{align_slug}__{algo_slug}"
        self.local_run_dir.mkdir(parents=True, exist_ok=True)

        ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        self.history_path = self.local_run_dir / f"history_{ts}.jsonl"

        # Write a small meta file for reproducibility
        meta = dict(self.run_meta)
        meta.update({
            "created_at": _now_iso(),
            "history_file": str(self.history_path),
        })
        try:
            (self.local_run_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        except Exception:
            pass

        self._buffer: List[Dict[str, Any]] = []
        self._written = 0

    def _make_record(self, step: int, prompt: str, completion_first: str, completion_second: str,
                      chosen_index: int, metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        chosen = completion_first if int(chosen_index) == 0 else completion_second
        rejected = completion_second if int(chosen_index) == 0 else completion_first
        record = {
            "timestamp": _now_iso(),
            "step": int(step),
            "prompt": prompt,
            "completion_first": completion_first,
            "completion_second": completion_second,
            "chosen_index": int(chosen_index),
            "chosen": chosen,
            "rejected": rejected,
            "run": self.run_meta,
        }
        if metrics:
            record["metrics"] = metrics
        return record

    def log_batch(
        self,
        step: int,
        prompts: List[str],
        completion_first_list: List[str],
        completion_second_list: List[str],
        chosen_mask_list: List[bool],
        metrics_per_sample: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        n = len(prompts)
        if not (len(completion_first_list) == len(completion_second_list) == len(chosen_mask_list) == n):
            return
        for i in range(n):
            metrics = metrics_per_sample[i] if metrics_per_sample and i < len(metrics_per_sample) else None
            rec = self._make_record(
                step=step,
                prompt=str(prompts[i]),
                completion_first=str(completion_first_list[i]),
                completion_second=str(completion_second_list[i]),
                chosen_index=0 if bool(chosen_mask_list[i]) else 1,
                metrics=metrics,
            )
            self._buffer.append(rec)

        if len(self._buffer) >= self.flush_steps:
            self._flush_to_disk()

    def _flush_to_disk(self) -> None:
        if not self._buffer:
            return
        with self.history_path.open("a", encoding="utf-8") as f:
            for rec in self._buffer:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                self._written += 1
        self._buffer.clear()

    def finalize(self) -> Dict[str, Any]:
        """Flush remaining logs and, if requested, push to the Hub as a dataset."""
        out: Dict[str, Any] = {"written": self._written}
        self._flush_to_disk()

        # Optional Hub push
        if self.mode in {"hub", "both"} and load_dataset is not None:
            if os.environ.get("HF_HUB_OFFLINE", "0") == "1":
                out["hub"] = "skipped_offline"
                return out
            try:
                # Build a reasonably stable repo id
                parts = [
                    "replay",
                    _slugify(self.run_meta.get("dataset", "dataset")),
                    _slugify(self.run_meta.get("model", "model")),
                    _slugify(self.run_meta.get("alignment", "align")),
                    _slugify(self.run_meta.get("query", "algo")),
                ]
                base = "__".join(parts)
                hash_input = json.dumps(self.run_meta, sort_keys=True).encode("utf-8")
                suffix = hashlib.sha1(hash_input).hexdigest()[:8]
                repo_id = f"{self.hub_ns}/{base}-{suffix}"

                # Load as dataset and push (Hub stores Arrow/Parquet files).
                ds = load_dataset("json", data_files={"train": str(self.history_path)})["train"]
                if isinstance(ds, Dataset):
                    ds.push_to_hub(repo_id=repo_id, private=self.hub_private)
                else:  # pragma: no cover
                    # older datasets versions may return dict-like
                    Dataset.from_list(list(ds)).push_to_hub(repo_id=repo_id, private=self.hub_private)

                # Additionally upload the raw JSONL history and a small README
                # with the first few samples for easier inspection from the UI.
                try:
                    from huggingface_hub import HfApi, create_repo  # type: ignore

                    api = HfApi()
                    create_repo(
                        repo_id,
                        repo_type="dataset",
                        private=self.hub_private,
                        exist_ok=True,
                    )

                    # Upload raw history JSONL under its original file name.
                    api.upload_file(
                        path_or_fileobj=str(self.history_path),
                        path_in_repo=self.history_path.name,
                        repo_id=repo_id,
                        repo_type="dataset",
                    )

                    # Build README with a short meta summary and first 3 samples.
                    try:
                        n_preview = min(3, len(ds))
                        preview_rows: List[Dict[str, Any]] = []
                        if n_preview > 0:
                            # ds.select returns a Dataset; cast to list of dicts.
                            preview_rows = [ds[int(i)] for i in range(n_preview)]
                    except Exception:
                        preview_rows = []

                    header_lines = [
                        "# Active DAP replay history",
                        "",
                        "This dataset was generated automatically from an online DPO/XPO run.",
                        "",
                    ]
                    meta_lines = [
                        "## Run metadata",
                        "",
                        "```json",
                        json.dumps(self.run_meta, indent=2, ensure_ascii=False),
                        "```",
                        "",
                    ]
                    samples_lines: List[str] = ["## Preview (first 3 samples)", ""]
                    if preview_rows:
                        samples_lines.extend(
                            [
                                "```json",
                                json.dumps(preview_rows, indent=2, ensure_ascii=False),
                                "```",
                                "",
                            ]
                        )
                    else:
                        samples_lines.append("_No samples available in this history._\n")

                    readme_text = "\n".join(header_lines + meta_lines + samples_lines)
                    readme_path = self.local_run_dir / "README.md"
                    try:
                        readme_path.write_text(readme_text, encoding="utf-8")
                    except Exception:
                        # If local write fails, still try to upload from memory.
                        pass

                    api.upload_file(
                        path_or_fileobj=str(readme_path),
                        path_in_repo="README.md",
                        repo_id=repo_id,
                        repo_type="dataset",
                    )
                except Exception:
                    # README / extra file upload failures should not break training.
                    pass

                out["hub"] = {"status": "pushed", "repo_id": repo_id}
            except Exception as exc:
                out["hub"] = {"status": "error", "error": str(exc)}
        return out
