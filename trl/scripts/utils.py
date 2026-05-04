# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import importlib
import inspect
import logging
import os
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Optional, Union, List

import yaml
import torch
import gc
from transformers import HfArgumentParser, PreTrainedModel, PreTrainedTokenizer
from transformers.hf_argparser import DataClass, DataClassType
from transformers.utils import is_rich_available
from torch.cuda.amp import autocast


logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    """
    Arguments common to all scripts.

    Args:
        dataset_name (`str`):
            Dataset name.
        dataset_config (`str` or `None`, *optional*, defaults to `None`):
            Dataset configuration name. Corresponds to the `name` argument of the [`~datasets.load_dataset`] function.
        dataset_train_split (`str`, *optional*, defaults to `"train"`):
            Dataset split to use for training.
        dataset_test_split (`str`, *optional*, defaults to `"test"`):
            Dataset split to use for evaluation.
        gradient_checkpointing_use_reentrant (`bool`, *optional*, defaults to `False`):
            Whether to apply `use_reentrant` for gradient checkpointing.
        ignore_bias_buffers (`bool`, *optional*, defaults to `False`):
            Debug argument for distributed training. Fix for DDP issues with LM bias/mask buffers - invalid scalar
            type, inplace operation. See https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992.
    """

    dataset_name: Optional[str] = field(default=None, metadata={"help": "Dataset name."})
    dataset_config: Optional[str] = field(
        default=None,
        metadata={
            "help": "Dataset configuration name. Corresponds to the `name` argument of the `datasets.load_dataset` "
            "function."
        },
    )
    dataset_train_split: str = field(default="train", metadata={"help": "Dataset split to use for training."})
    dataset_test_split: str = field(default="test", metadata={"help": "Dataset split to use for evaluation."})
    gradient_checkpointing_use_reentrant: bool = field(
        default=False,
        metadata={"help": "Whether to apply `use_reentrant` for gradient checkpointing."},
    )
    ignore_bias_buffers: bool = field(
        default=False,
        metadata={
            "help": "Debug argument for distributed training. Fix for DDP issues with LM bias/mask buffers - invalid "
            "scalar type, inplace operation. See "
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992."
        },
    )
    judge_count: int = field(
        default=0, 
        metadata={
            "help": "GPU count for judge model"
        }
    )
    judge_index: Optional[List[int]] = field(
        default=list, 
        metadata={
            "help": "GPU index to assign to the judge model"
        }
    )
    training_index: Optional[List[int]] = field(
        default=list, 
        metadata={
            "help": "GPU index to assign to training model"
        }
    )

@dataclass
class ActiveLearningArguments:
    """
    Arguments for active learning.
    """

    updates_per_sample: int = field(
        default=3, metadata={"help": "number of expected updates per sample"},
    )
    query_strategy: str = field(
        default="random", metadata={"help": "query strategy for active learning"},
    )   
    num_query: int = field(
        default=64, metadata={"help": "number of query per round"},
    )
    
    extractor_type: str = field(
        default="difference_vectors",
        metadata={"help": "feature extractor types: difference_vectors, separate_concat, sentence_transformer, llm"}
    )
    
    embedding_input_type: str = field(
        default="difference_vectors",
        metadata={"help": "embedding input types: prompt, concat, template, difference_vectors, separate_concat"}
    )
    
    radius: float = field(
        default=1.0, metadata={"help": "radius parameter for a kernel"},
    )
    
    normalize: bool = field(
        default=True, metadata={"help": "normalize embeddings"},
    )
    
    filter_out: str = field(
        default=None, metadata={"help": "filter out selected samples. choose between ['none', 'wrong', 'right']"},
    )
    
    # ADPO-specific parameters
    subset_size: int = field(
        default=1000,
        metadata={"help": "Number of candidate samples to evaluate in each ADPO query round"}
    )

    apl_n: int = field(
        default=2,
        metadata={
            "help": (
                "Number of completions per prompt for APL-style querying. "
                "Original paper uses 20, but this can be very expensive; "
                "we default to 2 for practical sweeps."
            )
        },
    )

    query_batch_size: int = field(
        default=0,
        metadata={
            "help": (
                "Per-device batch size for scoring the unlabeled pool in active querying "
                "(APL / reward-margin / diversity, etc.). "
                "If 0, fall back to training batch size or strategy-specific defaults."
            )
        },
    )
    
    epsilon: float = field(
        default=1e-3,
        metadata={"help": "Ridge regularization parameter for covariance matrix in ADPO"}
    )
    
    linear_pool: str = field(
        default="cls",
        metadata={"help": "Pooling strategy for linearized policy: 'mean' or 'cls'"}
    )
    
    fitting_steps: int = field(
        default=1000,
        metadata={"help": "Number of steps to fit the linear head in ADPO before selection"}
    )

    optimal_batch: int = field(
        default=0,
        metadata={"help": "If > 0, auto-tune by probing GPU memory."}
    )
    
    optimal_batch_size: int = field(
        default=32,
        metadata={"help": "If > 0, auto-tune by probing GPU memory."}
    )


def init_zero_verbose():
    """
    Perform zero verbose init - use this method on top of the CLI modules to make
    logging and warning output cleaner. Uses Rich if available, falls back otherwise.
    """
    import logging
    import warnings

    FORMAT = "%(message)s"

    if is_rich_available():
        from rich.logging import RichHandler

        handler = RichHandler()
    else:
        handler = logging.StreamHandler()

    logging.basicConfig(format=FORMAT, datefmt="[%X]", handlers=[handler], level=logging.ERROR)

    # Custom warning handler to redirect warnings to the logging system
    def warning_handler(message, category, filename, lineno, file=None, line=None):
        logging.warning(f"{filename}:{lineno}: {category.__name__}: {message}")

    # Add the custom warning handler - we need to do that before importing anything to make sure the loggers work well
    warnings.showwarning = warning_handler


class TrlParser(HfArgumentParser):
    """
    A subclass of [`transformers.HfArgumentParser`] designed for parsing command-line arguments with dataclass-backed
    configurations, while also supporting configuration file loading and environment variable management.

    Args:
        dataclass_types (`Union[DataClassType, Iterable[DataClassType]]` or `None`, *optional*, defaults to `None`):
            Dataclass types to use for argument parsing.
        **kwargs:
            Additional keyword arguments passed to the [`transformers.HfArgumentParser`] constructor.

    Examples:

    ```yaml
    # config.yaml
    env:
        VAR1: value1
    arg1: 23
    ```

    ```python
    # main.py
    import os
    from dataclasses import dataclass
    from trl import TrlParser

    @dataclass
    class MyArguments:
        arg1: int
        arg2: str = "alpha"

    parser = TrlParser(dataclass_types=[MyArguments])
    training_args = parser.parse_args_and_config()

    print(training_args, os.environ.get("VAR1"))
    ```

    ```bash
    $ python main.py --config config.yaml
    (MyArguments(arg1=23, arg2='alpha'),) value1

    $ python main.py --arg1 5 --arg2 beta
    (MyArguments(arg1=5, arg2='beta'),) None
    ```
    """

    def __init__(
        self,
        dataclass_types: Optional[Union[DataClassType, Iterable[DataClassType]]] = None,
        **kwargs,
    ):
        # Make sure dataclass_types is an iterable
        if dataclass_types is None:
            dataclass_types = []
        elif not isinstance(dataclass_types, Iterable):
            dataclass_types = [dataclass_types]

        # Check that none of the dataclasses have the "config" field
        for dataclass_type in dataclass_types:
            if "config" in dataclass_type.__dataclass_fields__:
                raise ValueError(
                    f"Dataclass {dataclass_type.__name__} has a field named 'config'. This field is reserved for the "
                    f"config file path and should not be used in the dataclass."
                )

        super().__init__(dataclass_types=dataclass_types, **kwargs)

    def parse_args_and_config(
        self, args: Optional[Iterable[str]] = None, return_remaining_strings: bool = False
    ) -> tuple[DataClass, ...]:
        """
        Parse command-line args and config file into instances of the specified dataclass types.

        This method wraps [`transformers.HfArgumentParser.parse_args_into_dataclasses`] and also parses the config file
        specified with the `--config` flag. The config file (in YAML format) provides argument values that replace the
        default values in the dataclasses. Command line arguments can override values set by the config file. The
        method also sets any environment variables specified in the `env` field of the config file.
        """
        args = list(args) if args is not None else sys.argv[1:]
        if "--config" in args:
            # Get the config file path from
            config_index = args.index("--config")
            args.pop(config_index)  # remove the --config flag
            config_path = args.pop(config_index)  # get the path to the config file
            with open(config_path) as yaml_file:
                config = yaml.safe_load(yaml_file)

            # Set the environment variables specified in the config file
            if "env" in config:
                env_vars = config.pop("env", {})
                if not isinstance(env_vars, dict):
                    raise ValueError("`env` field should be a dict in the YAML file.")
                for key, value in env_vars.items():
                    os.environ[key] = str(value)

            # Set the defaults from the config values
            config_remaining_strings = self.set_defaults_with_config(**config)
        else:
            config_remaining_strings = []

        # Parse the arguments from the command line
        output = self.parse_args_into_dataclasses(args=args, return_remaining_strings=return_remaining_strings)

        # Merge remaining strings from the config file with the remaining strings from the command line
        if return_remaining_strings:
            args_remaining_strings = output[-1]
            return output[:-1] + (config_remaining_strings + args_remaining_strings,)
        else:
            return output

    def set_defaults_with_config(self, **kwargs) -> list[str]:
        """
        Overrides the parser's default values with those provided via keyword arguments, including for subparsers.

        Any argument with an updated default will also be marked as not required
        if it was previously required.

        Returns a list of strings that were not consumed by the parser.
        """

        def apply_defaults(parser, kw):
            used_keys = set()
            for action in parser._actions:
                # Handle subparsers recursively
                if isinstance(action, argparse._SubParsersAction):
                    for subparser in action.choices.values():
                        used_keys.update(apply_defaults(subparser, kw))
                elif action.dest in kw:
                    action.default = kw[action.dest]
                    action.required = False
                    used_keys.add(action.dest)
            return used_keys

        used_keys = apply_defaults(self, kwargs)
        # Remaining args not consumed by the parser
        remaining = [
            item for key, value in kwargs.items() if key not in used_keys for item in (f"--{key}", str(value))
        ]
        return remaining


def get_git_commit_hash(package_name):
    try:
        # Import the package to locate its path
        package = importlib.import_module(package_name)
        # Get the path to the package using inspect
        package_path = os.path.dirname(inspect.getfile(package))

        # Navigate up to the Git repository root if the package is inside a subdirectory
        git_repo_path = os.path.abspath(os.path.join(package_path, ".."))
        git_dir = os.path.join(git_repo_path, ".git")

        if os.path.isdir(git_dir):
            # Run the git command to get the current commit hash
            commit_hash = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=git_repo_path).strip().decode("utf-8")
            )
            return commit_hash
        else:
            return None
    except Exception as e:
        return f"Error: {str(e)}"





def find_optimal_batch_size(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt_texts: list[str],
    length_percentile: float = 0.9,
    max_batch_size: int = 512,
    device: str = "cuda",
    use_amp: bool = False,
    max_length: int = 128,
) -> int:
    """
    Estimate the largest batch size that fits into GPU memory using binary search.
    
    - Samples a representative token length at the given percentile from prompt_texts.
    - Clears intermediate tensors and GPU cache after each test to avoid fragmentation.
    - Optionally uses automatic mixed-precision (AMP) for the test.
    
    Args:
        model: Hugging Face model to evaluate.
        tokenizer: Hugging Face tokenizer for input preparation.
        prompt_texts: A list of example prompts to sample length distribution.
        length_percentile: Between 0 and 1, the percentile in token length distribution to use.
        max_batch_size: Upper bound on batch size to search.
        device: Device string, e.g. "cuda" or "cuda:0".
        use_amp: Whether to wrap forward pass in autocast for mixed precision.
        max_length: Maximum token length when truncating.
    
    Returns:
        The maximum batch size that does not raise an OOM error.
    """
    # Move model to device and set to evaluation mode
    model.to(device).eval()
    
    # Compute token lengths for all sample prompts, then pick the given percentile
    lengths = []
    for txt in prompt_texts:
        enc = tokenizer(txt, add_special_tokens=True, truncation=True, max_length=max_length)
        lengths.append(len(enc["input_ids"]))
    target_length = int(torch.quantile(torch.tensor(lengths, dtype=torch.float32), length_percentile).item())
    target_length = max(1, target_length)

    # Tokenize one prompt to get the input tensor shape
    sample = tokenizer(
        prompt_texts[0],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=target_length
    )
    def make_batch(bs: int):
        return {k: v.expand(bs, -1).to(device) for k, v in sample.items()}

    # Binary search over batch sizes
    low, high = 1, max_batch_size
    best = 1

    while low <= high:
        mid = (low + high) // 2
        try:
            batch = make_batch(mid)
            with torch.no_grad():
                if use_amp:
                    with autocast(device_type=device):
                        out = model(**batch)
                else:
                    out = model(**batch)

            # Clean up intermediate tensors and free GPU memory
            del out, batch
            torch.cuda.empty_cache()
            gc.collect()

            best = mid
            low = mid + 1
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # On OOM, clear cache and search lower half
                del batch
                torch.cuda.empty_cache()
                gc.collect()
                high = mid - 1
            else:
                # Reraise unexpected errors
                raise

    return best
