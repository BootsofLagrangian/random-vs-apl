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

import os
import textwrap
import warnings
from functools import wraps
from typing import Any, Callable, Optional, Union

import datasets
import jinja2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import transformers
from datasets import Dataset
from packaging import version
from torch.utils.data import DataLoader, IterableDataset
from transformers import (
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    StoppingCriteriaList,
    Trainer,
    TrainerCallback,
    is_apex_available,
    is_wandb_available,
    DebertaV2ForSequenceClassification
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, EvalPrediction, seed_worker
from transformers.training_args import OptimizerNames
from transformers.utils import is_peft_available, is_sagemaker_mp_enabled, logging

from ..data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from ..environment.base_environment import StringStoppingCriteria
from ..import_utils import is_vllm_available
from ..models import create_reference_model
from ..models.utils import unwrap_model_for_generation
from .judges import BasePairwiseJudge
from .online_dpo_config import OnlineDPOConfig
from .utils import (
    SIMPLE_CHAT_TEMPLATE,
    DPODataCollatorWithPadding,
    disable_dropout_in_model,
    empty_cache,
    generate_model_card,
    get_comet_experiment_url,
    get_reward,
    compute_reward_scores,
    prepare_deepspeed,
    truncate_right,
    build_stop_token_sequences,
    truncate_on_stop_sequences,
    #
    prepare_weights_for_vllm
)
###  GRPO' 
from ..extras.vllm_client import VLLMClient
from ..extras.profiling import profiling_context, profiling_decorator
from contextlib import nullcontext
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


if is_peft_available():
    from peft import PeftModel, get_peft_model

if is_apex_available():
    from apex import amp


if is_sagemaker_mp_enabled():
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

else:
    IS_SAGEMAKER_MP_POST_1_10 = False


if is_vllm_available():
    from vllm import LLM, SamplingParams

if is_wandb_available():
    import wandb

logger = logging.get_logger(__name__)


class OnlineDPOTrainer(Trainer):
    r"""
    Initialize OnlineDPOTrainer.

    Args:
        model (`transformers.PreTrainedModel` or `torch.nn.Module`):
            The model to train, preferably an `AutoModelForCausalLM`.
        ref_model (`transformers.PreTrainedModel` or `torch.nn.Module` or `None`):
            The reference model to use for training. If None is specified, the reference model will be created from
            the model.
        reward_model (`transformers.PreTrainedModel` or `torch.nn.Module` or `None`):
            The reward model to score completions with, preferably an `AutoModelForSequenceClassification`.
        judge (`BasePairwiseJudge`):
            The judge to use for pairwise comparison of model completions.
        args (`OnlineDPOConfig`):
            The online DPO config arguments to use for training.
        data_collator (`transformers.DataCollator`):
            The data collator to use for training. If None is specified, the default data collator (`DPODataCollatorWithPadding`) will be used
            which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        processing_class (`PreTrainedTokenizerBase` or `BaseImageProcessor` or `FeatureExtractionMixin` or `ProcessorMixin`, *optional*):
            Processing class used to process the data. If provided, will be used to automatically process the inputs
            for the model, and it will be saved along the model to make it easier to rerun an interrupted training or
            reuse the fine-tuned model.
        peft_config (`dict`):
            The peft config to use for training.
        compute_metrics (`Callable[[EvalPrediction], dict]`, *optional*):
            The function to use to compute the metrics. Must take a `EvalPrediction` and return
            a dictionary string to metric values.
        callbacks (`list[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
    """

    _tag_names = ["trl", "online-dpo"]

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module],
        ref_model: Union[PreTrainedModel, nn.Module, None] = None,
        reward_model: Union[PreTrainedModel, nn.Module, None] = None,
        judge: Optional[BasePairwiseJudge] = None,
        args: Optional[OnlineDPOConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset, "datasets.Dataset"]] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset], "datasets.Dataset"]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        reward_processing_class: Optional[PreTrainedTokenizerBase] = None,
        peft_config: Optional[dict] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        if ref_model is model:
            raise ValueError(
                "`model` and `ref_model` cannot be the same object. If you want `ref_model` to be the "
                "same as `model`, either omit the `ref_model` argument or pass `None`."
            )

        self.ref_model = ref_model

        if reward_model is not None and judge is not None:
            warnings.warn(
                "Both `reward_model` and `judge` are provided. Please choose provide only one of them. "
                "Ignoring `judge` and using `reward_model`.",
                UserWarning,
            )
            judge = None
        elif reward_model is None and judge is None:
            raise ValueError("Either `reward_model` or `judge` must be provided.")

        self.reward_model = reward_model
        self.reward_processing_class = reward_processing_class
        self.judge = judge
        self.is_encoder_decoder = model.config.is_encoder_decoder

        if args.missing_eos_penalty is not None and judge is not None:
            raise ValueError("`missing_eos_penalty` is not supported when `judge` is provided.")

        if args is None:
            raise ValueError("`args` must be provided.")

        # Check that the processing_class is provided
        if processing_class is None:
            raise ValueError("`processing_class` must be provided.")

        # Convert to PEFT model if peft_config is provided
        if peft_config is not None:
            # Check if PEFT is available
            if not is_peft_available():
                raise ImportError(
                    "PEFT is not available and passed `peft_config`. Please install PEFT with "
                    "`pip install peft` to use it."
                )

            # If the model is already a PeftModel, we need to merge and unload it.
            # Further information here: https://huggingface.co/docs/trl/dpo_trainer#reference-model-considerations-with-peft
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()

            # Get peft model with the given config
            model = get_peft_model(model, peft_config)

        # Disable dropout in the model and reference model
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        # Handle the ref_model
        # Usually, the user wants the ref model to be the initial version of the model. When using PEFT, it's easy to
        # get the ref model, as it's just the model with a disabled adapter. When not using PEFT, we need to create
        # the ref model from the model by copying it and disable the gradients and set it in evaluation mode.
        if ref_model is None:  # No ref model provided, the most common case
            if peft_config is None:
                self.ref_model = create_reference_model(model)  # copy, disable gradients, set eval mode
            else:
                self.ref_model = None  # we don't need a ref model here, we can just disable the adapter.
        else:  # rare case, the user provided a ref model
            self.ref_model = ref_model
            self.ref_model.eval()

        # Disable the gradient and set the reward model in eval mode
        if self.reward_model is not None:
            self.reward_model.eval()
        # Define the collator is not provided
        if data_collator is None:
            data_collator = DPODataCollatorWithPadding(pad_token_id=processing_class.pad_token_id)

        self.max_length = args.max_length

        self.stats = {
            "objective/kl": [],
            "objective/entropy": [],
            "objective/non_score_reward": [],
            "rewards/chosen": [],
            "rewards/rejected": [],
            "rewards/accuracies": [],
            "rewards/margins": [],
            "logps/chosen": [],
            "logps/rejected": [],
            "val/contain_eos_token": [],
            "beta": [],
        }
        if self.reward_model is not None:
            self.stats["objective/rlhf_reward"] = []
            self.stats["objective/scores_margin"] = []
            self.stats["objective/scores"] = []


        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in Online DPO, the sampled data does not include
        # the "input_ids" key. As a result, the trainer issues the warning: "Could not estimate the number of tokens
        # of the input, floating-point operations will not be computed." To suppress this warning, we set the
        # "estimate_tokens" key in the model's "warnings_issued" dictionary to True. This acts as a flag to indicate
        # that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Add tags for models that have been loaded with the correct transformers version
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        self._beta = args.beta
        if args.stop_strings is None:
            self.stop_strings = ["user:", " user:", "user", " user"]
        else:
            self.stop_strings = [s.strip() for s in str(args.stop_strings).split(",") if s.strip()]
        self.stop_token_sequences = build_stop_token_sequences(self.processing_class, self.stop_strings)

        self.use_vllm = args.use_vllm
        self.vllm_mode = "colocate"  # | args.vllm_mode |  # default to colocate mode
        self.vllm_gpu_memory_utilization = args.gpu_memory_utilization  # args.vllm_tensor_parallel_size
        self.vllm_tensor_parallel_size = self.accelerator.num_processes # args.vllm_tensor_parallel_size

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        if self.use_vllm:
            # for debugging purposes, we can change it later
            self.vllm_mode = "colocate" # for debugging purposes, we can change it to "server" later
            self.vllm_tensor_parallel_size = self.accelerator.num_processes # for debugging purposes, we can change it later
            self.vllm_gpu_memory_utilization = args.gpu_memory_utilization
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )

            if self.vllm_mode == "server" and self.accelerator.is_main_process:
                if args.vllm_server_base_url is not None:
                    base_url = args.vllm_server_base_url
                else:
                    base_url = f"http://{args.vllm_server_host}:{args.vllm_server_port}"
                self.vllm_client = VLLMClient(base_url=base_url, connection_timeout=args.vllm_server_timeout)
                self.vllm_client.init_communicator()

            elif self.vllm_mode == "colocate":
                # Make sure vllm_tensor_parallel_size group size evenly divides the world size - each group should have
                # the same number of ranks
                if not self.accelerator.num_processes % self.vllm_tensor_parallel_size == 0:
                    raise ValueError(
                        f"vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}) must divide world size "
                        f"({self.accelerator.num_processes}) evenly."
                    )

                if self.vllm_tensor_parallel_size > 1:
                    # Create subgroups of ranks for TP, each group with `vllm_tensor_parallel_size` ranks.
                    # For example, if world_size=8 and vllm_tensor_parallel_size=2 → groups: [0,1], [2,3], [4,5], [6,7]
                    self.tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                        [
                            list(range(i * self.vllm_tensor_parallel_size, (i + 1) * self.vllm_tensor_parallel_size))
                            for i in range(self.accelerator.num_processes // self.vllm_tensor_parallel_size)
                        ]
                    )

                self.llm = LLM(
                    model=model.name_or_path,
                    dtype=model.dtype,
                    tensor_parallel_size=self.vllm_tensor_parallel_size,
                    gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                    max_num_seqs=self.args.per_device_eval_batch_size
                    # * self.vllm_tensor_parallel_size
                    * self.args.gradient_accumulation_steps,
                    # max_model_len=self.max_prompt_length + self.max_completion_length, # debugging purposes, we can change it later
                    distributed_executor_backend="external_launcher",
                    # Feed identical seed for tp groups to ensure sampling results are the same across workers
                    seed=self.accelerator.process_index // self.vllm_tensor_parallel_size,
                    disable_custom_all_reduce=True,
                )
            self.generation_config = SamplingParams(
                n=2,  # 2 generations per prompt
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=50,
                top_p=1.0,
                detokenize=False,  # to avoid vllm to decode (we don't need it)
                stop=self.stop_strings if self.stop_strings else None,
            )

            # vLLM specific sampling arguments
            # self.guided_decoding_regex = args.vllm_guided_decoding_regex

            self._last_loaded_step = -1  # tag to avoid useless loading during grad accumulation

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            # self.accelerator.print("[DEBUG] succesfully initialized vLLM, synchronizing all processes...")
            self.accelerator.wait_for_everyone()

        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=50,
                top_p=1.0,
                do_sample=True,
                use_cache=False if args.gradient_checkpointing else True,
            )
            
            self.greedy_generation_config = GenerationConfig(
                max_new_tokens=args.max_new_tokens,
                top_k=50,
                top_p=1.0,
                do_sample=False,
                use_cache=False if args.gradient_checkpointing else True,
            )

        # Placed after the super().__init__ because we need self.is_deepspeed_enabled and self.accelerator
        if self.is_deepspeed_enabled:
            if self.reward_model is not None:
                self.reward_model = prepare_deepspeed(
                    self.reward_model, args.per_device_train_batch_size, args.fp16, args.bf16
                )
            if self.ref_model is not None:
                self.ref_model = prepare_deepspeed(
                    self.ref_model, args.per_device_train_batch_size, args.fp16, args.bf16
                )
        else:
            if self.ref_model is not None:
                self.ref_model = self.ref_model.to(self.accelerator.device)
            if self.reward_model is not None:
                self.reward_model = self.reward_model.to(self.accelerator.device)

        if self.args.loss_type == "simpo":
            if args.simpo_gamma is None:
                raise ValueError("`simpo_gamma` must be provided when `loss_type` is `simpo`.")
            if args.simpo_gamma < 0:
                raise ValueError("`simpo_gamma` must be non-negative.")
            self.simpo_gamma = args.simpo_gamma
            
    @property
    def beta(self):
        if isinstance(self._beta, list):
            epoch = self.state.epoch
            return self._beta[epoch] if epoch < len(self._beta) else self._beta[-1]
        else:
            return self._beta

    @staticmethod
    def tokenize_row(feature, is_encoder_decoder: bool, tokenizer: PreTrainedTokenizerBase) -> dict[str, Any]:
        """Tokenize a single row from a DPO specific dataset."""
        if not is_encoder_decoder:
            batch = tokenizer(feature["prompt"], add_special_tokens=False)
            # Add BOS token to head of prompt. Avoid adding if it's already there
            if tokenizer.bos_token_id is not None:
                prompt_len_input_ids = len(batch["input_ids"])
                if prompt_len_input_ids == 0 or tokenizer.bos_token_id != batch["input_ids"][0]:
                    batch["input_ids"] = [tokenizer.bos_token_id] + batch["input_ids"]
                    batch["attention_mask"] = [1] + batch["attention_mask"]
        else:
            batch = tokenizer(feature["prompt"], add_special_tokens=True)
        batch = {f"prompt_{key}": value for key, value in batch.items()}
        return batch

    # Same as Trainer.get_train_dataloader but skip the "remove_unused_columns".
    @wraps(Trainer.get_train_dataloader)
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    # Same as Trainer.get_eval_dataloader but skip the "remove_unused_columns".
    @wraps(Trainer.get_eval_dataloader)
    def get_eval_dataloader(self, eval_dataset: Optional[Union[str, Dataset]] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self.accelerator.prepare(self._eval_dataloaders[dataloader_key])

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = eval_dataloader
            else:
                self._eval_dataloaders = {dataloader_key: eval_dataloader}

        return self.accelerator.prepare(eval_dataloader)


    def _sync_fsdp_params_to_vllm(self, module: nn.Module, prefix: str = "", visited=None):
        """Memory-efficient post-order traversal of FSDP modules to extract full parameters and sync with vLLM."""
        if visited is None:
            visited = set()

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            self._sync_fsdp_params_to_vllm(
                child_module, prefix=child_prefix, visited=visited
            )  # recurse into the child

        if isinstance(module, FSDP):
            with FSDP.summon_full_params(module, recurse=False, writeback=False):
                for param_name, param in module.named_parameters():
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    for extra in ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module."):
                        full_name = full_name.replace(extra, "")

                    if full_name in visited:
                        continue  # skip FSDP subtrees already traversed
                    visited.add(full_name)

                    if self.vllm_mode == "server" and self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(full_name, param.data)
                    elif self.vllm_mode == "colocate":
                        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                        llm_model.load_weights([(full_name, param.data)])

    # @profiling_decorator
    def _move_model_to_vllm(self, model: nn.Module = None):
        # For DeepSpeed ZeRO-3 and FSDP, we need to gather all parameters before operations
        import gc
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext
        if model is None:
            model = self.model
        with unwrap_model_for_generation(model, accelerator=self.accelerator) as unwrapped_model:

            # self.accelerator.print(f"[DEBUG] moving model:", unwrapped_model.__class__.__name__, "to vLLM", self.vllm_mode, "mode")
            if is_peft_model(unwrapped_model):
                # With PEFT and FSDP/DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging, as
                # merging adapters in a sharded manner is not supported.
                # TODO: does this work with FSDP?
                if getattr(unwrapped_model, "is_quantized", False):
                    model = prepare_weights_for_vllm(unwrapped_model)
                    # Use the dequantized model's parameters instead of the quantized model's parameters
                    with gather_if_zero3(list(model.parameters())):
                        for name, param in model.named_parameters():
                            if self.vllm_mode == "server": # and self.accelerator.is_main_process:
                                self.vllm_client.update_named_param(name, param.data)
                            elif self.vllm_mode == "colocate":
                                llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                                llm_model.load_weights([(name, param.data)])
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
                else:
                    with gather_if_zero3(list(unwrapped_model.parameters())):
                        unwrapped_model.merge_adapter()

                        # Update vLLM weights while parameters are gathered
                        if self.is_fsdp_enabled:  # note if using FSDP, gather_if_zero3 is nullcontext
                            # Update vLLM weights while parameters are gathered
                            # For PEFT with FSDP we need to use the memory efficient post-order traversal
                            self._sync_fsdp_params_to_vllm(unwrapped_model)
                        else:
                            # DeepSpeed ZeRO-3 with PEFT
                            for name, param in unwrapped_model.named_parameters():
                                # When using PEFT, we need to recover the original parameter name and discard some parameters
                                name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                                if unwrapped_model.prefix in name:
                                    continue
                                # When module to save, remove its prefix and discard the original module
                                if "original_module" in name:
                                    continue
                                name = name.replace("modules_to_save.default.", "")

                                if self.vllm_mode == "server" and self.accelerator.is_main_process:
                                    self.vllm_client.update_named_param(name, param.data)
                                elif self.vllm_mode == "colocate":
                                    llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                                    llm_model.load_weights([(name, param.data)])
                        # Unmerge adapters while parameters are still gathered
                        unwrapped_model.unmerge_adapter()
                        # Parameters will automatically be repartitioned when exiting the context
            else:
                # For non-PEFT models, simply gather (if needed) and update each parameter individually.
                if self.is_fsdp_enabled:
                    self._sync_fsdp_params_to_vllm(unwrapped_model)  # use memory-efficient post-order traversal for FSDP
                else:
                    for name, param in unwrapped_model.named_parameters():
                        with gather_if_zero3([param]):
                            if self.vllm_mode == "server" and self.accelerator.is_main_process:
                                self.vllm_client.update_named_param(name, param.data)
                            elif self.vllm_mode == "colocate":
                                llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                                llm_model.load_weights([(name, param.data)])

        # Reset cache on vLLM
        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == "colocate":
            self.llm.reset_prefix_cache()


    def _generate_vllm(self, model, prompts):
        eos_token_id = self.processing_class.eos_token_id
        pad_token_id = self.processing_class.pad_token_id

        # Load the latest weights
        # llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        # model = prepare_weights_for_vllm(model, accelerator=self.accelerator)
        # llm_model.load_weights(model.state_dict().items())
        if self._last_loaded_step != self.state.global_step:
            
            self._move_model_to_vllm(model)
            # Reset the last loaded step to the current global step
            self._last_loaded_step = self.state.global_step

        if self.vllm_tensor_parallel_size > 1:
            # Gather prompts from all ranks in the TP group and flatten.
            # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
            orig_size = len(prompts)
            gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
            torch.distributed.all_gather_object(gathered_prompts, prompts, group=self.tp_group)
            all_prompts_text = [p for sublist in gathered_prompts for p in sublist]
        else:
            all_prompts_text = prompts

        with profiling_context(self, "vLLM.generate"):
            if is_conversational({"prompt": all_prompts_text[0]}):
                all_outputs = self.llm.chat(all_prompts_text, self.generation_config, use_tqdm=False)
            else:
                all_outputs = self.llm.generate(all_prompts_text, self.generation_config, use_tqdm=False)
        
        # Slice the outputs to match the local rank in the TP group
        # This is necessary because each rank in the TP group will only see its own outputs.
        # If vllm_tensor_parallel_size == 1, all_outputs will be the same as the original prompts.
        if self.vllm_tensor_parallel_size > 1:
            local_rank_in_group = torch.distributed.get_rank(group=self.tp_group)
            tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
            all_outputs = all_outputs[tp_slice] # slice the outputs to match the local rank in the TP group
        
        # Handling N=2 outputs per prompt
        completion_ids = [list(output.outputs[i].token_ids) for i in range(2) for output in all_outputs ]
        prompt_ids = [list(output.prompt_token_ids) for _ in range(2) for output in all_outputs ]
        
        # Create mask and pad the prompt and completion
        max_prompt_length = max(len(ids) for ids in prompt_ids)
        prompt_mask = [[0] * (max_prompt_length - len(ids)) + [1] * len(ids) for ids in prompt_ids]
        prompt_ids = [[pad_token_id] * (max_prompt_length - len(ids)) + ids for ids in prompt_ids]
        max_tokens = self.generation_config.max_tokens
        completion_mask = [[1] * len(ids) + [0] * (max_tokens - len(ids)) for ids in completion_ids]
        completion_ids = [
            ids + [eos_token_id] if ids[-1] != eos_token_id and len(ids) < max_tokens else ids
            for ids in completion_ids
        ]
        completion_ids = [ids + [pad_token_id] * (max_tokens - len(ids)) for ids in completion_ids]

        # Convert to tensors
        prompt_ids = torch.tensor(prompt_ids, device=self.accelerator.device)
        prompt_mask = torch.tensor(prompt_mask, device=self.accelerator.device)
        completion_ids = torch.tensor(completion_ids, device=self.accelerator.device)
        completion_mask = torch.tensor(completion_mask, device=self.accelerator.device)

        completion_ids, completion_mask = truncate_on_stop_sequences(
            completion_ids, self.stop_token_sequences, pad_token_id
        )
    
        return prompt_ids, prompt_mask, completion_ids, completion_mask

    def _build_stop_criteria(self):
        if not self.stop_strings:
            return None
        return StoppingCriteriaList([StringStoppingCriteria(self.stop_strings, self.processing_class)])

    def _generate(self, model, prompts, generate_greedy=False):
        eos_token_id = self.processing_class.eos_token_id
        pad_token_id = self.processing_class.pad_token_id

        # Apply chat template and tokenize the input. We do this on-the-fly to enable the use of reward models and
        # policies with different tokenizers / chat templates.
        inputs = [{"prompt": prompt} for prompt in prompts]
        inputs = [maybe_apply_chat_template(x, self.processing_class) for x in inputs]
        inputs = [self.tokenize_row(x, self.is_encoder_decoder, self.processing_class) for x in inputs]
        inputs = self.data_collator(inputs)

        if generate_greedy:
            # Sample 1 greedy completion and non-greedy per prompt of size `max_new_tokens` from the model
            inputs = self._prepare_inputs(inputs)
            prompt_ids = inputs["prompt_input_ids"]
            prompt_mask = inputs["prompt_attention_mask"]
            with unwrap_model_for_generation(
                model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
            ) as unwrapped_model, self.accelerator.autocast():
                stop_criteria = self._build_stop_criteria()
                output = unwrapped_model.generate(
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    generation_config=self.generation_config,
                    stopping_criteria=stop_criteria,
                )
                
                greedy_stop_criteria = self._build_stop_criteria()
                greedy_output = unwrapped_model.generate(
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    generation_config=self.greedy_generation_config,
                    stopping_criteria=greedy_stop_criteria,
                )

            completion_ids = torch.cat([greedy_output[:, prompt_ids.size(1) :], output[:, prompt_ids.size(1) :]], dim=0)
            completion_ids, completion_mask = truncate_right(completion_ids, eos_token_id, pad_token_id)
            completion_ids, completion_mask = truncate_on_stop_sequences(
                completion_ids, self.stop_token_sequences, pad_token_id
            )
            prompt_ids = prompt_ids.repeat(2, 1)
            prompt_mask = prompt_mask.repeat(2, 1)
        else:
            # Sample 2 completions per prompt of size `max_new_tokens` from the model
            inputs = self._prepare_inputs(inputs)
            prompt_ids = inputs["prompt_input_ids"].repeat(2, 1)
            prompt_mask = inputs["prompt_attention_mask"].repeat(2, 1)
            with unwrap_model_for_generation(
                model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
            ) as unwrapped_model:
                stop_criteria = self._build_stop_criteria()
                output = unwrapped_model.generate(
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    generation_config=self.generation_config,
                    stopping_criteria=stop_criteria,
                )

            completion_ids = output[:, prompt_ids.size(1) :]
            completion_ids, completion_mask = truncate_right(completion_ids, eos_token_id, pad_token_id)
            completion_ids, completion_mask = truncate_on_stop_sequences(
                completion_ids, self.stop_token_sequences, pad_token_id
            )

        return prompt_ids, prompt_mask, completion_ids, completion_mask

    def _forward(self, model, prompt_ids, prompt_mask, completion_ids, completion_mask):
        # Get the number of tokens to truncate from prompt
        num_tokens_to_truncate = max(prompt_ids.size(1) + completion_ids.size(1) - self.max_length, 0)

        # Truncate left to avoid oom
        prompt_ids = prompt_ids[:, num_tokens_to_truncate:]
        prompt_mask = prompt_mask[:, num_tokens_to_truncate:]

        # Concat the prompt and completion
        prompt_completion_ids = torch.cat((prompt_ids, completion_ids), dim=1)
        prompt_completion_mask = torch.cat((prompt_mask, completion_mask), dim=1)

        # Get the logprobs of the completions from the model
        output = model(prompt_completion_ids, attention_mask=prompt_completion_mask)

        # There is 1 offset, because the model predict the next token
        logits = output.logits[:, prompt_ids.size(1) - 1 : -1]

        # Take the completion tokens logprob
        logprobs = torch.take_along_dim(logits.log_softmax(dim=-1), completion_ids.unsqueeze(-1), dim=2).squeeze(-1)
        return logprobs

    def training_step(
        self, model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], num_items_in_batch: Optional[int] = None
    ) -> torch.Tensor:
        model.train()

        prompts = inputs["prompt"]
        # Preserve original prompts for logging (avoid later template mutations)
        orig_prompts = list(prompts)
        batch_size = len(prompts)
        if self.active_iter:
            prompts, prompt_ids, prompt_mask, completion_ids, completion_mask  = self._load_generated_inputs(inputs)
            batch_size = len(prompts)
        else:
            if self.args.use_vllm:
                prompt_ids, prompt_mask, completion_ids, completion_mask = self._generate_vllm(model, prompts)
            else:
                prompt_ids, prompt_mask, completion_ids, completion_mask = self._generate(model, prompts)

        contain_eos_token = torch.any(completion_ids == self.processing_class.eos_token_id, dim=-1)

        logprobs = self._forward(model, prompt_ids, prompt_mask, completion_ids, completion_mask)
        with torch.no_grad():
            if self.ref_model is not None:
                ref_logprobs = self._forward(self.ref_model, prompt_ids, prompt_mask, completion_ids, completion_mask)
            else:  # peft case: we just need to disable the adapter
                with self.model.disable_adapter():
                    ref_logprobs = self._forward(self.model, prompt_ids, prompt_mask, completion_ids, completion_mask)

        # Decode the completions, and format them if the input is conversational
        device = logprobs.device
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational({"prompt": prompts[0]}):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        # Get the reward from the reward model or judge
        if self.judge is not None:
            # Once formatted, conversational data may contain special tokens (such as <|im_start|>) that are not
            # directly understandable by the judge and could alter its judgment. To avoid this and make the judge
            # independent of the model's chat template, we use the raw conversation data, and apply our own chat
            # template to it.
            if is_conversational({"prompt": prompts[0]}):
                environment = jinja2.Environment()
                template = environment.from_string(SIMPLE_CHAT_TEMPLATE)
                prompts = [template.render(messages=prompt) for prompt in prompts]
                completions = [template.render(messages=completion) for completion in completions]

            ranks_of_first_completion = self.judge.judge(
                prompts, list(zip(completions[:batch_size], completions[batch_size:]))
            )

            # convert ranks to a True/False mask:
            # when rank == 0, it means the first completion is the best
            # when rank == 1, it means the second completion is the best
            mask = torch.tensor([rank == 0 for rank in ranks_of_first_completion], device=device)
        else:
            # The reward model may not have the same chat template or tokenizer as the model, so we need to use the
            # raw data (string), apply the chat template (if needed), and tokenize it with the reward processing class.
            prompts = 2 * prompts  # repeat the prompt: [prompt0, prompt1] -> [prompt0, prompt1, prompt0, prompt1]
            if is_conversational({"prompt": prompts[0]}):
                examples = [{"prompt": p, "completion": c} for p, c in zip(prompts, completions)]
                examples = [apply_chat_template(example, self.reward_processing_class) for example in examples]
                prompts = [example["prompt"] for example in examples]
                completions = [example["completion"] for example in examples]

            reward_max_length = int(getattr(self.args, "reward_max_length", self.max_length))

            # Tokenize the prompts/completions for the reward model.
            # NOTE: UltraFeedback prompts can be very long; without truncation this can easily OOM during RM scoring.
            prev_trunc_side = getattr(self.reward_processing_class, "truncation_side", "right")
            try:
                # Keep the *end* of long prompts (closest to the completion).
                self.reward_processing_class.truncation_side = "left"
                prompts_ids = self.reward_processing_class(
                    prompts,
                    padding=True,
                    truncation=True,
                    max_length=reward_max_length,
                    return_tensors="pt",
                    padding_side="left",
                )["input_ids"].to(device)

                # Keep the *start* of completions (common practice for RM scoring).
                self.reward_processing_class.truncation_side = "right"
                completions_ids = self.reward_processing_class(
                    completions,
                    padding=True,
                    truncation=True,
                    max_length=reward_max_length,
                    return_tensors="pt",
                    padding_side="right",
                )["input_ids"].to(device)
            finally:
                self.reward_processing_class.truncation_side = prev_trunc_side

            # Enforce (prompt + completion) <= reward_max_length by truncating from the left of the prompt.
            num_tokens_to_truncate = max(prompts_ids.size(1) + completions_ids.size(1) - reward_max_length, 0)
            if num_tokens_to_truncate > 0:
                prompts_ids = prompts_ids[:, num_tokens_to_truncate:]

            context_length = prompts_ids.shape[1]
            prompt_completion_ids = torch.cat((prompts_ids, completions_ids), dim=1)
            reward_batch_size = max(1, getattr(self.args, "reward_batch_size", 2))

            with torch.inference_mode():
                scores = compute_reward_scores(
                    self.reward_model,
                    self.reward_processing_class,
                    prompts=prompts,
                    completions=completions,
                    device=device,
                    prompt_completion_ids=prompt_completion_ids,
                    context_length=context_length,
                    contain_eos_mask=contain_eos_token,
                    missing_eos_penalty=self.args.missing_eos_penalty,
                    reward_batch_size=reward_batch_size,
                )

            # Split the scores in 2 (the prompts of the first half are the same as the second half)
            first_half, second_half = scores.split(batch_size)

            # Get the indices of the chosen and rejected examples
            mask = first_half >= second_half
        
        batch_range = torch.arange(batch_size, device=device)
        chosen_indices = batch_range + (~mask * batch_size)
        rejected_indices = batch_range + (mask * batch_size)

        # Build tensor so that the first half is the chosen examples and the second half the rejected examples
        cr_indices = torch.cat((chosen_indices, rejected_indices), dim=0)  # cr = chosen and rejected
        cr_logprobs = logprobs[cr_indices]
        cr_ref_logprobs = ref_logprobs[cr_indices]

        # mask out the padding tokens
        padding_mask = ~completion_mask.bool()
        cr_padding_mask = padding_mask[cr_indices]

        cr_logprobs_sum = (cr_logprobs * ~cr_padding_mask).sum(1)
        cr_ref_logprobs_sum = (cr_ref_logprobs * ~cr_padding_mask).sum(1)

        # Split the chosen and rejected examples
        chosen_logprobs_sum, rejected_logprobs_sum = torch.split(cr_logprobs_sum, batch_size)
        chosen_ref_logprobs_sum, rejected_ref_logprobs_sum = torch.split(cr_ref_logprobs_sum, batch_size)
        pi_logratios = chosen_logprobs_sum - rejected_logprobs_sum
        ref_logratios = chosen_ref_logprobs_sum - rejected_ref_logprobs_sum

        logits = pi_logratios - ref_logratios

        if self.args.loss_type == "sigmoid":
            losses = -F.logsigmoid(self.beta * logits)
        elif self.args.loss_type == "ipo":
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.args.loss_type == "simpo":
            gamma_logratios = self.simpo_gamma / self.beta
            logits = logits - gamma_logratios
            # This reduces to Equation 3 from the CPO paper when label_smoothing -> 0.
            losses = -F.logsigmoid(self.beta * logits)
        else:
            raise NotImplementedError(f"invalid loss type {self.loss_type}")

        loss = losses.mean()

        # === History logging (main process only) ===
        try:
            if hasattr(self, "history_logger") and self.history_logger is not None:
                from accelerate.utils import gather_object

                # Decode raw completions (string, not conversational dicts)
                base_completions = self.processing_class.batch_decode(
                    completion_ids, skip_special_tokens=True
                )
                first_text = base_completions[:batch_size]
                second_text = base_completions[batch_size:]

                # Sum logprobs per sample for first/second and ref
                padding_mask = ~completion_mask.bool()
                all_lp_sum = (logprobs * ~padding_mask).sum(1)
                all_ref_sum = (ref_logprobs * ~padding_mask).sum(1)
                first_lp = all_lp_sum[:batch_size]
                second_lp = all_lp_sum[batch_size:]
                first_ref_lp = all_ref_sum[:batch_size]
                second_ref_lp = all_ref_sum[batch_size:]

                # Rewards if available
                reward_first = None
                reward_second = None
                if 'scores' in locals():
                    reward_first = scores[:batch_size]
                    reward_second = scores[batch_size:]

                # Gather across processes
                prompts_all = gather_object(orig_prompts)
                first_all = gather_object(list(map(str, first_text)))
                second_all = gather_object(list(map(str, second_text)))
                mask_all = gather_object([bool(x) for x in mask.detach().cpu().tolist()])

                # Build metrics per sample and gather as python objects
                metrics_local: list[dict] = []
                for i in range(batch_size):
                    m: Dict[str, float] = {
                        "model_logprobs_first": float(first_lp[i].detach().cpu().item()),
                        "model_logprobs_second": float(second_lp[i].detach().cpu().item()),
                        "ref_logprobs_first": float(first_ref_lp[i].detach().cpu().item()),
                        "ref_logprobs_second": float(second_ref_lp[i].detach().cpu().item()),
                    }
                    if reward_first is not None:
                        m["reward_first"] = float(reward_first[i].detach().cpu().item())
                        m["reward_second"] = float(reward_second[i].detach().cpu().item())
                        try:
                            _margin = reward_first[i] - reward_second[i]
                            m["reward_margin"] = float(_margin.detach().cpu().item())
                        except Exception:
                            pass
                    metrics_local.append(m)

                metrics_all = gather_object(metrics_local)

                if self.accelerator.is_main_process:
                    self.history_logger.log_batch(
                        step=int(self.state.global_step),
                        prompts=prompts_all,
                        completion_first_list=first_all,
                        completion_second_list=second_all,
                        chosen_mask_list=mask_all,
                        metrics_per_sample=metrics_all,
                    )
        except Exception:
            # Never break training on logging failure
            pass

        # Log everything
        if self.reward_model is not None:
            scores_margin = scores[chosen_indices] - scores[rejected_indices]
            self.stats["objective/scores_margin"].append(
                self.accelerator.gather_for_metrics(scores_margin.mean()).mean().item()
            )
            self.stats["objective/scores"].append(self.accelerator.gather_for_metrics(scores.mean()).mean().item())
        self.stats["val/contain_eos_token"].append(contain_eos_token.float().mean().item())
        self.stats["logps/chosen"].append(self.accelerator.gather_for_metrics(chosen_logprobs_sum).mean().item())
        self.stats["logps/rejected"].append(self.accelerator.gather_for_metrics(rejected_logprobs_sum).mean().item())

        kl = logprobs - ref_logprobs
        mean_kl = kl.sum(1).mean()
        self.stats["objective/kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        non_score_reward = (-self.beta * kl).sum(1)
        mean_non_score_reward = non_score_reward.mean()
        self.stats["objective/non_score_reward"].append(
            self.accelerator.gather_for_metrics(mean_non_score_reward).mean().item()
        )
        if self.reward_model is not None:
            rlhf_reward = scores + non_score_reward
            self.stats["objective/rlhf_reward"].append(self.accelerator.gather_for_metrics(rlhf_reward).mean().item())
        mean_entropy = -logprobs.sum(1).mean()
        self.stats["objective/entropy"].append(self.accelerator.gather_for_metrics(mean_entropy).mean().item())
        chosen_rewards = self.beta * (chosen_logprobs_sum - chosen_ref_logprobs_sum)
        gathered_chosen_rewards = self.accelerator.gather_for_metrics(chosen_rewards)
        self.stats["rewards/chosen"].append(gathered_chosen_rewards.mean().item())
        rejected_rewards = self.beta * (rejected_logprobs_sum - rejected_ref_logprobs_sum)
        gathered_rejected_rewards = self.accelerator.gather_for_metrics(rejected_rewards)
        self.stats["rewards/rejected"].append(gathered_rejected_rewards.mean().item())
        margin = gathered_chosen_rewards - gathered_rejected_rewards
        self.stats["rewards/margins"].append(margin.mean().item())
        accuracy = margin > 0
        self.stats["rewards/accuracies"].append(accuracy.float().mean().item())
        self.stats["beta"].append(self.beta)


        # TODO: psuedo labeling - 1. generate more completions, 2. compute rewards, 3. determine chosen or rejected, 4. compute additional losses
        # if False:
        #     print('Add additional pseudo labeled samples...')
        #     # 1. generate more completions
        #     if self.args.use_vllm:
        #         extra_prompt_ids, extra_prompt_mask, extra_completion_ids, extra_completion_mask = self._generate_vllm(model, prompts)
        #     else:
        #         extra_prompt_ids, extra_prompt_mask, extra_completion_ids, extra_completion_mask = self._generate(model, prompts, generate_greedy=False)
        
        #     #2. compute rewards
        #     logprobs = self._forward(model, prompt_ids, prompt_mask, completion_ids, completion_mask)
        #     with torch.no_grad():
        #         if self.ref_model is not None:
        #             ref_logprobs = self._forward(self.ref_model, prompt_ids, prompt_mask, completion_ids, completion_mask)
        #         else:  # peft case: we just need to disable the adapter
        #             with self.model.disable_adapter():
        #                 ref_logprobs = self._forward(self.model, prompt_ids, prompt_mask, completion_ids, completion_mask)
                        
        #     cr_logprobs = logprobs
        #     cr_ref_logprobs = ref_logprobs
            
        #     # mask out the padding tokens
        #     padding_mask = ~completion_mask.bool()
        #     cr_padding_mask = padding_mask
    
        #     cr_logprobs_sum = (cr_logprobs * ~cr_padding_mask).sum(1)
        #     cr_ref_logprobs_sum = (cr_ref_logprobs * ~cr_padding_mask).sum(1)
            
        #     first_logprobs_sum, second_logprobs_sum = torch.split(cr_logprobs_sum, batch_size)
        #     first_ref_logprobs_sum, second_ref_logprobs_sum = torch.split(cr_ref_logprobs_sum, batch_size)
            
        #     first_rewards = self.beta * (first_logprobs_sum - first_ref_logprobs_sum)
        #     second_rewards = self.beta * (second_logprobs_sum - second_ref_logprobs_sum)
            
        #     # 3. determine chosen or rejected
        #     is_chosen_more_probable = (chosen_rewards > rejected_rewards)
        #     first_chosen_bool = torch.logical_and(
        #         first_rewards > chosen_rewards, torch.logical_not(is_chosen_more_probable))
        #     first_rejected_bool = torch.logical_and(
        #         first_rewards > rejected_rewards, is_chosen_more_probable)
        #     second_chosen_bool = torch.logical_and(
        #         second_rewards > chosen_rewards, torch.logical_not(is_chosen_more_probable))
        #     second_rejected_bool = torch.logical_and(
        #         second_rewards > rejected_rewards, is_chosen_more_probable)
            
        #     # 4. compute additional losses
        #     # first <-> chosen
        #     first_chosen_pi_logratios = chosen_logprobs_sum - first_logprobs_sum
        #     first_chosen_ref_logratios = chosen_ref_logprobs_sum - first_ref_logprobs_sum
        #     first_chosen_logits = first_chosen_pi_logratios - first_chosen_ref_logratios
            
        #     if self.args.loss_type == "sigmoid":
        #         first_chosen_losses = -F.logsigmoid(self.beta * first_chosen_logits)
        #     elif self.args.loss_type == "ipo":
        #         first_chosen_losses = (first_chosen_logits - 1 / (2 * self.beta)) ** 2
        #     else:
        #         raise NotImplementedError(f"invalid loss type {self.loss_type}")
            
        #     first_chosen_losses = first_chosen_losses[first_chosen_bool]
            
        #     # first <-> rejected
        #     first_rejected_pi_logratios =  first_logprobs_sum - rejected_logprobs_sum
        #     first_rejected_ref_logratios = first_ref_logprobs_sum - rejected_ref_logprobs_sum
        #     first_rejected_logits = first_rejected_pi_logratios - first_rejected_ref_logratios
            
        #     if self.args.loss_type == "sigmoid":
        #         first_rejected_losses = -F.logsigmoid(self.beta * first_rejected_logits)
        #     elif self.args.loss_type == "ipo":
        #         first_rejected_losses = (first_rejected_logits - 1 / (2 * self.beta)) ** 2
        #     else:
        #         raise NotImplementedError(f"invalid loss type {self.loss_type}")
            
        #     first_rejected_losses = first_rejected_losses[first_rejected_bool]
            
        #     # second <-> chosen
        #     second_chosen_pi_logratios = chosen_logprobs_sum - second_logprobs_sum
        #     second_chosen_ref_logratios = chosen_ref_logprobs_sum - second_ref_logprobs_sum
        #     second_chosen_logits = second_chosen_pi_logratios - second_chosen_ref_logratios
            
        #     if self.args.loss_type == "sigmoid":
        #         second_chosen_losses = -F.logsigmoid(self.beta * second_chosen_logits)
        #     elif self.args.loss_type == "ipo":
        #         second_chosen_losses = (second_chosen_logits - 1 / (2 * self.beta)) ** 2
        #     else:
        #         raise NotImplementedError(f"invalid loss type {self.loss_type}")
            
        #     second_chosen_losses = second_chosen_losses[second_chosen_bool]
            
        #     # second <-> rejected
        #     second_rejected_pi_logratios =  second_logprobs_sum - rejected_logprobs_sum
        #     second_rejected_ref_logratios = second_ref_logprobs_sum - rejected_ref_logprobs_sum
        #     second_rejected_logits = second_rejected_pi_logratios - second_rejected_ref_logratios
            
        #     if self.args.loss_type == "sigmoid":
        #         second_rejected_losses = -F.logsigmoid(self.beta * second_rejected_logits)
        #     elif self.args.loss_type == "ipo":
        #         second_rejected_losses = (second_rejected_logits - 1 / (2 * self.beta)) ** 2
        #     else:
        #         raise NotImplementedError(f"invalid loss type {self.loss_type}")
            
        #     second_rejected_losses = second_rejected_losses[second_rejected_bool]
            
        #     prev_loss_num = losses.shape[0]
        #     losses = torch.cat([losses, first_chosen_losses, first_rejected_losses, second_chosen_losses, second_rejected_losses])
        #     loss = losses.mean()  
        #     # print(f'number of losses increased from {prev_loss_num} to {losses.shape[0]}')      
        
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            empty_cache()

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss, **kwargs)

        return loss.detach() / self.args.gradient_accumulation_steps

    # Same as Trainer._maybe_log_save_evaluate but log our metrics
    # start_time defaults to None to allow compatibility with transformers<=4.46
    def _maybe_log_save_evaluate(
        self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time=None, learning_rate=None
    ):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            logs: dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            if learning_rate is not None:
                logs["learning_rate"] = learning_rate
            else:
                logs["learning_rate"] = self._get_learning_rate()

            # Add our metrics
            for key, val in self.stats.items():
                if len(val) > 0:  # Only compute average if there are values
                    logs[key] = sum(val) / len(val)
            self.stats = {key: [] for key in self.stats}  # reset stats

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
                self.log(logs, start_time)
            else:  # transformers<=4.46
                self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)

            if self.args.save_strategy == "best":
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    # Copy-pasted from transformers.Trainer to maintain compatibility with earlier versions.
    # This can be removed once the minimum transformers version is updated to 4.47.
    # Refer to https://github.com/huggingface/trl/pull/2288 for more details.
    def _determine_best_metric(self, metrics, trial):
        """
        Determine if the model should be saved based on the evaluation metrics.
        If args.metric_for_best_model is not set, the loss is used.
        Returns:
            bool: True if a new best metric was found, else False
        """
        is_new_best_metric = False

        if self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model

            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"

            try:
                metric_value = metrics[metric_to_check]
            except KeyError as exc:
                raise KeyError(
                    f"The `metric_for_best_model` training argument is set to '{metric_to_check}', which is not found in the evaluation metrics. "
                    f"The available evaluation metrics are: {list(metrics.keys())}. Consider changing the `metric_for_best_model` via the TrainingArguments."
                ) from exc

            operator = np.greater if self.args.greater_is_better else np.less

            if self.state.best_metric is None:
                self.state.best_metric = float("-inf") if self.args.greater_is_better else float("inf")

            if operator(metric_value, self.state.best_metric):
                run_dir = self._get_output_dir(trial=trial)
                checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
                output_dir = os.path.join(run_dir, checkpoint_folder)
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

                is_new_best_metric = True

        return is_new_best_metric

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent("""\
        @article{guo2024direct,
            title        = {{Direct Language Model Alignment from Online AI Feedback}},
            author       = {Shangmin Guo and Biao Zhang and Tianlin Liu and Tianqi Liu and Misha Khalman and Felipe Llinares and Alexandre Ram{\'{e}} and Thomas Mesnard and Yao Zhao and Bilal Piot and Johan Ferret and Mathieu Blondel},
            year         = 2024,
            eprint       = {arXiv:2402.04792}
        }""")

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="Online DPO",
            trainer_citation=citation,
            paper_title="Direct Language Model Alignment from Online AI Feedback",
            paper_id="2402.04792",
        )
        model_card.save(os.path.join(self.args.output_dir, "README.md"))
