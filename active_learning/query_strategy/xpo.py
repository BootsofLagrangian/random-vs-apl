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
from typing import Any, Callable, Optional, Union

import jinja2
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, IterableDataset
from transformers import (
    BaseImageProcessor,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    TrainerCallback,
    is_apex_available,
    is_wandb_available,
)
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import OptimizerNames
from transformers.utils import is_peft_available

from trl.data_utils import is_conversational, maybe_apply_chat_template, apply_chat_template
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.utils import (
    SIMPLE_CHAT_TEMPLATE,
    empty_cache,
    get_reward,
    selective_log_softmax,
    truncate_right,
)

import numpy as np
from .strategy import Strategy
from ..utils import merge_prompt_or_completion, safe_gather, broadcast_subset, gather_2d_once
from accelerate.utils import gather_object
from tqdm import tqdm

if is_apex_available():
    from apex import amp

if is_peft_available():
    from peft import PeftModel


class XPO(Strategy):
    def __init__(self, trainer, system_prompt, config):
        super(XPO, self).__init__(trainer, system_prompt, config)
        
        self._alpha = getattr(config, 'alpha', 1e-5)
        self.trainer = trainer
        self.config = config
        
        self.trainer.training_step = self.training_step

        self.trainer.stats = {
            "loss/dpo": [],
            "loss/xpo": [],
            "objective/kl": [],
            "objective/entropy": [],
            "rewards/chosen": [],
            "rewards/rejected": [],
            "rewards/accuracies": [],
            "rewards/margins": [],
            "logps/chosen": [],
            "logps/rejected": [],
            "alpha": [],
            "beta": [],
        }
        if self.trainer.reward_model is not None:
            self.trainer.stats["objective/model_scores"] = []
            self.trainer.stats["objective/ref_scores"] = []
            self.trainer.stats["objective/scores_margin"] = []

    
    @property
    def alpha(self):
        if isinstance(self._alpha, list):
            epoch = self.trainer.state.epoch
            return self._alpha[epoch] if epoch < len(self._alpha) else self._alpha[-1]
        else:
            return self._alpha

    def query(self, n):
        unlabeled_indices = self.trainer.extract_unlabeled_indices()
        accelerator = self.trainer.accelerator
        per_device_bs = self.query_batch_size or getattr(self.trainer.args, 'per_device_train_batch_size', 4)
        subset_unlabeled_indices = broadcast_subset(
            unlabeled_indices,
            desired_size=int(n),
            accelerator=accelerator,
            per_device_bs=per_device_bs,
            k=1,
            seed=getattr(self.trainer.args, "seed", 42),
            enforce_multiple=False,
        ).cpu().numpy()
        unlabeled_dataloader = self.trainer.get_unlabeled_dataloader(subset_unlabeled_indices, per_device_bs)
        accelerator = self.trainer.accelerator
        model = self.prepare_model(unlabeled_dataloader)

        total_prompts_local = []  
        total_completion_ids_local = []
        total_completion_mask_local = []

        for inputs in unlabeled_dataloader:
            with torch.no_grad():
                # Generate completions from both model and reference
                model_output, ref_output = self._generate_completions(model, inputs)

                # Extract completion parts for dataset storage
                model_completion_ids = model_output["completion_ids"]
                model_completion_mask = model_output["completion_mask"]
                ref_completion_ids = ref_output["completion_ids"]
                ref_completion_mask = ref_output["completion_mask"]
                
                # Combine model and ref completions (2 completions per prompt)
                batch_size = len(model_output["prompt"])
                combined_completion_ids = torch.stack([model_completion_ids, ref_completion_ids], dim=1).reshape(batch_size * 2, -1)
                combined_completion_mask = torch.stack([model_completion_mask, ref_completion_mask], dim=1).reshape(batch_size * 2, -1)

                # Accumulate locally; gather once at the end
                total_prompts_local += model_output["prompt"]
                total_completion_ids_local.append(combined_completion_ids.detach().cpu())
                total_completion_mask_local.append(combined_completion_mask.detach().cpu())

        # Concatenate all completions via single gather
        total_batch_size = len(subset_unlabeled_indices)
        total_prompts = gather_object(total_prompts_local)[: total_batch_size]
        total_completion_ids_g, _ = gather_2d_once(total_completion_ids_local, accelerator, pad_token_id=self.trainer.processing_class.pad_token_id)
        total_completion_mask_g, _ = gather_2d_once(total_completion_mask_local, accelerator, pad_token_id=0)
        total_completion_ids = merge_prompt_or_completion([total_completion_ids_g], accelerator.num_processes, total_batch_size=total_batch_size)
        total_completion_mask = merge_prompt_or_completion([total_completion_mask_g], accelerator.num_processes, total_batch_size=total_batch_size)
        
        args_to_update = {
            'topk_indices': torch.arange(total_batch_size),
            'completion_ids': total_completion_ids,
            'completion_mask': total_completion_mask,
            'prompt': total_prompts,
            'selected_indices': subset_unlabeled_indices,
        }
        
        args_to_update = self._prepare_for_data_update(args_to_update, total_batch_size)
        
        return args_to_update

    def _generate_completions(self, model, inputs, decode_completion=False):

        if "prompt" in inputs:
            prompts = inputs["prompt"]
        elif "prompt_input_ids" in inputs:
            prompt_ids = inputs["prompt_input_ids"]
            if isinstance(prompt_ids, torch.Tensor):
                prompt_ids = prompt_ids.cpu().tolist()
            prompts = self.trainer.processing_class.batch_decode(
                prompt_ids, skip_special_tokens=True
            )
        else:
            raise KeyError(
                f"Expected 'prompt' or 'prompt_input_ids' in inputs, got keys: {list(inputs.keys())}"
            )
        
        batch_size = len(prompts)
        inputs = [{k: v[i] for k, v in inputs.items()} for i in range(batch_size)]
        inputs = [maybe_apply_chat_template(x, self.trainer.processing_class) for x in inputs]
        inputs = [self.trainer.tokenize_row(x, self.trainer.model.config.is_encoder_decoder, self.trainer.processing_class) for x in inputs]
        inputs = self.trainer.data_collator(inputs)
        # need the prompt_ only
        inputs = self.trainer._prepare_inputs(inputs)

        with unwrap_model_for_generation(model, self.trainer.accelerator) as unwrapped_policy_model_for_gen:
            model_output = unwrapped_policy_model_for_gen.generate(
                input_ids=inputs["prompt_input_ids"],
                attention_mask=inputs["prompt_attention_mask"],
                generation_config=self.trainer.generation_config,
            )
        model_completion_ids = model_output[:, inputs["prompt_input_ids"].size(1):]
        model_completion_ids, model_completion_mask = truncate_right(
            model_completion_ids, self.trainer.processing_class.eos_token_id, self.trainer.processing_class.pad_token_id
        )
        
        actual_model_for_ref_generation: torch.nn.Module
        if self.trainer.ref_model is None:
            unwrapped_main_model_for_ref_logic = self.trainer.accelerator.unwrap_model(model)

            if is_peft_available() and isinstance(unwrapped_main_model_for_ref_logic, PeftModel):
                actual_model_for_ref_generation = unwrapped_main_model_for_ref_logic.get_base_model()
            else:
                raise ValueError("Reference model needs PeftModel")
                actual_model_for_ref_generation = unwrapped_main_model_for_ref_logic
        else:
            actual_model_for_ref_generation = self.trainer.accelerator.unwrap_model(self.trainer.ref_model)

        with unwrap_model_for_generation(actual_model_for_ref_generation, self.trainer.accelerator) as unwrapped_ref_model_for_gen:
            ref_output = unwrapped_ref_model_for_gen.generate(
                input_ids=inputs["prompt_input_ids"],
                attention_mask=inputs["prompt_attention_mask"],
                generation_config=self.trainer.generation_config,
            )
        ref_completion_ids = ref_output[:, inputs["prompt_input_ids"].size(1):]
        ref_completion_ids, ref_completion_mask = truncate_right(
            ref_completion_ids, self.trainer.processing_class.eos_token_id, self.trainer.processing_class.pad_token_id
        )
        
        model_output = {
            "prompt": prompts,
            "prompt_ids": inputs["prompt_input_ids"],
            "prompt_mask": inputs["prompt_attention_mask"],
            "completion_ids": model_completion_ids,
            "completion_mask": model_completion_mask,
        }
        
        ref_output = {
            "prompt": prompts,
            "prompt_ids": inputs["prompt_input_ids"],
            "prompt_mask": inputs["prompt_attention_mask"],
            "completion_ids": ref_completion_ids,
            "completion_mask": ref_completion_mask,
        }
        
        if decode_completion:
            completions = self.trainer.processing_class.batch_decode(model_completion_ids, skip_special_tokens=True)
            if is_conversational({"prompt": prompts[0]}):
                completions = [[{"role": "assistant", "content": completion}] for completion in completions]
            model_output["completions"] = completions
            ref_output["completions"] = completions

        return model_output, ref_output

    def _compute_losses(
        self,
        model_logprobs_model_data,
        model_logprobs_ref_data,
        ref_logprobs_ref_data,
        ref_logprobs_model_data,
        chosen_mask,
    ):
        model_logprobs_model_data_sum = model_logprobs_model_data.sum(1)
        model_logprobs_ref_data_sum = model_logprobs_ref_data.sum(1)
        ref_logprobs_ref_data_sum = ref_logprobs_ref_data.sum(1)
        ref_logprobs_model_data_sum = ref_logprobs_model_data.sum(1)

        chosen_model_logprobs = torch.where(chosen_mask, model_logprobs_model_data_sum, model_logprobs_ref_data_sum)
        chosen_ref_logprobs = torch.where(chosen_mask, ref_logprobs_model_data_sum, ref_logprobs_ref_data_sum)
        chosen_log_ratios = chosen_model_logprobs - chosen_ref_logprobs

        rejected_model_logprobs = torch.where(~chosen_mask, model_logprobs_model_data_sum, model_logprobs_ref_data_sum)
        rejected_ref_logprobs = torch.where(~chosen_mask, ref_logprobs_model_data_sum, ref_logprobs_ref_data_sum)
        rejected_log_ratios = rejected_model_logprobs - rejected_ref_logprobs

        logits = chosen_log_ratios - rejected_log_ratios

        if self.trainer.args.loss_type == "sigmoid":
            dpo_losses = -F.logsigmoid(self.trainer.beta * logits)
        elif self.trainer.args.loss_type == "ipo":
            dpo_losses = (logits - 1 / (2 * self.trainer.beta)) ** 2
        else:
            raise NotImplementedError(f"invalid loss type {self.trainer.args.loss_type}")

        xpo_losses = self.alpha * model_logprobs_ref_data_sum

        loss = (dpo_losses + xpo_losses).mean()

        return loss, dpo_losses, xpo_losses

    def _log_statistics(
        self,
        model_logprobs_model_data,
        model_logprobs_ref_data,
        ref_logprobs_ref_data,
        ref_logprobs_model_data,
        chosen_mask,
        dpo_losses,
        xpo_losses,
        model_scores=None,
        ref_scores=None,
    ):
        def gather_mean(tensor):
            return self.trainer.accelerator.gather_for_metrics(tensor).mean().item()

        self.trainer.stats["loss/dpo"].append(gather_mean(dpo_losses))
        self.trainer.stats["loss/xpo"].append(gather_mean(xpo_losses))

        if self.trainer.reward_model is not None:
            self.trainer.stats["objective/model_scores"].append(gather_mean(model_scores))
            self.trainer.stats["objective/ref_scores"].append(gather_mean(ref_scores))
            self.trainer.stats["objective/scores_margin"].append(gather_mean(model_scores - ref_scores))

        model_logprobs_model_data_sum = model_logprobs_model_data.sum(1)
        model_logprobs_ref_data_sum = model_logprobs_ref_data.sum(1)
        ref_logprobs_ref_data_sum = ref_logprobs_ref_data.sum(1)
        ref_logprobs_model_data_sum = ref_logprobs_model_data.sum(1)

        chosen_model_logprobs = torch.where(chosen_mask, model_logprobs_model_data_sum, model_logprobs_ref_data_sum)
        chosen_ref_logprobs = torch.where(chosen_mask, ref_logprobs_model_data_sum, ref_logprobs_ref_data_sum)
        chosen_log_ratios = chosen_model_logprobs - chosen_ref_logprobs

        rejected_model_logprobs = torch.where(~chosen_mask, model_logprobs_model_data_sum, model_logprobs_ref_data_sum)
        rejected_ref_logprobs = torch.where(~chosen_mask, ref_logprobs_model_data_sum, ref_logprobs_ref_data_sum)
        rejected_log_ratios = rejected_model_logprobs - rejected_ref_logprobs

        self.trainer.stats["logps/chosen"].append(gather_mean(chosen_model_logprobs.mean() + chosen_ref_logprobs.mean()))
        self.trainer.stats["logps/rejected"].append(gather_mean(rejected_model_logprobs.mean() + rejected_ref_logprobs.mean()))

        chosen_rewards = chosen_log_ratios * self.trainer.beta
        rejected_rewards = rejected_log_ratios * self.trainer.beta
        self.trainer.stats["rewards/chosen"].append(gather_mean(chosen_rewards.mean()))
        self.trainer.stats["rewards/rejected"].append(gather_mean(rejected_rewards.mean()))

        kl_model_data = model_logprobs_model_data - ref_logprobs_model_data
        kl_ref_data = model_logprobs_ref_data - ref_logprobs_ref_data
        mean_kl = (kl_model_data.sum(1) + kl_ref_data.sum(1)).mean() / 2
        self.trainer.stats["objective/kl"].append(gather_mean(mean_kl))

        entropy_model_data = -model_logprobs_model_data.sum(1)
        entropy_ref_data = -model_logprobs_ref_data.sum(1)
        mean_entropy = (entropy_model_data.mean() + entropy_ref_data.mean()) / 2
        self.trainer.stats["objective/entropy"].append(gather_mean(mean_entropy))

        margin = chosen_rewards - rejected_rewards
        self.trainer.stats["rewards/margins"].append(gather_mean(margin.mean()))

        accuracy = (margin > 0).float()
        self.trainer.stats["rewards/accuracies"].append(gather_mean(accuracy.mean()))

        self.trainer.stats["alpha"].append(self.alpha)
        self.trainer.stats["beta"].append(self.trainer.beta)

    def training_step(self, model, inputs, num_items_in_batch=None):

        model.train()

        prompts = inputs["prompt"]
        batch_size = len(prompts)
        
        if self.trainer.active_iter:
            prompts, prompt_ids, prompt_mask, completion_ids, completion_mask  = self.trainer._load_generated_inputs(inputs)
        else:
            raise ValueError("XPO requires pre-generated completions from query function. Ensure completion_ids and completion_mask are available.")

        contain_eos_token = torch.any(completion_ids == self.trainer.processing_class.eos_token_id, dim=-1)
        logprobs = self.trainer._forward(model, prompt_ids, prompt_mask, completion_ids, completion_mask)

        with torch.no_grad():
            if self.trainer.ref_model is not None:
                ref_logprobs = self.trainer._forward(self.trainer.ref_model, prompt_ids, prompt_mask, completion_ids, completion_mask)
            else:
                with self.trainer.model.disable_adapter():
                    ref_logprobs = self.trainer._forward(self.trainer.model, prompt_ids, prompt_mask, completion_ids, completion_mask)        

         # Decode the completions, and format them if the input is conversational
        device = logprobs.device
        completions = self.trainer.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational({"prompt": prompts[0]}):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        # Get the reward from the reward model or judge
        if self.trainer.judge is not None:
            # Once formatted, conversational data may contain special tokens (such as <|im_start|>) that are not
            # directly understandable by the judge and could alter its judgment. To avoid this and make the judge
            # independent of the model's chat template, we use the raw conversation data, and apply our own chat
            # template to it.
            if is_conversational({"prompt": prompts[0]}):
                environment = jinja2.Environment()
                template = environment.from_string(SIMPLE_CHAT_TEMPLATE)
                prompts = [template.render(messages=prompt) for prompt in prompts]
                completions = [template.render(messages=completion) for completion in completions]

            ranks_of_first_completion = self.trainer.judge.judge(
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
                examples = [apply_chat_template(example, self.trainer.reward_processing_class) for example in examples]
                prompts = [example["prompt"] for example in examples]
                completions = [example["completion"] for example in examples]

            # Tokenize the prompts
            prompts_ids = self.trainer.reward_processing_class(
                prompts, padding=True, return_tensors="pt", padding_side="left"
            )["input_ids"].to(device)
            context_length = prompts_ids.shape[1]

            # Tokenize the completions
            completions_ids = self.trainer.reward_processing_class(
                completions, padding=True, return_tensors="pt", padding_side="right"
            )["input_ids"].to(device)

            # Concatenate the prompts and completions and get the reward
            prompt_completion_ids = torch.cat((prompts_ids, completions_ids), dim=1)
            with torch.inference_mode():
                _, scores, _ = get_reward(
                    self.trainer.reward_model, prompt_completion_ids, self.trainer.reward_processing_class.pad_token_id, context_length
                )

                # Filter completion. Ensure that the sample contains stop_token_id
                # Completions not passing that filter will receive a lower score.
                if self.trainer.args.missing_eos_penalty is not None:
                    scores[~contain_eos_token] -= self.trainer.args.missing_eos_penalty

            # Split the scores in 2 (the prompts of the first half are the same as the second half)
            first_half, second_half = scores.split(batch_size)

            # Get the indices of the chosen and rejected examples
            mask = first_half >= second_half

        # Split logprobs into model and reference data components
        model_logprobs_model_data = logprobs[:batch_size]
        model_logprobs_ref_data = logprobs[batch_size:]
        ref_logprobs_model_data = ref_logprobs[:batch_size]
        ref_logprobs_ref_data = ref_logprobs[batch_size:]


        loss, dpo_losses, xpo_losses = self._compute_losses(
            model_logprobs_model_data,
            model_logprobs_ref_data,
            ref_logprobs_ref_data,
            ref_logprobs_model_data,
            mask,
        )

        self._log_statistics(
            model_logprobs_model_data.detach(),
            model_logprobs_ref_data.detach(),
            ref_logprobs_ref_data,
            ref_logprobs_model_data,
            mask,
            dpo_losses.detach(),
            xpo_losses.detach(),
            first_half,
            second_half,
        )

        if (
            self.trainer.args.torch_empty_cache_steps is not None
            and self.trainer.state.global_step % self.trainer.args.torch_empty_cache_steps == 0
        ):
            empty_cache()

        kwargs = {}
        if self.trainer.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self.trainer._get_learning_rate()

        if self.trainer.args.n_gpu > 1:
            loss = loss.mean()

        if self.trainer.use_apex:
            with amp.scale_loss(loss, self.trainer.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.trainer.accelerator.backward(loss, **kwargs)

        return loss.detach() / self.trainer.args.gradient_accumulation_steps
