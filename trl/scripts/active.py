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

"""
# Full training
python trl/scripts/dpo.py \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --learning_rate 5.0e-7 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50 \
    --output_dir Qwen2-0.5B-DPO \
    --no_remove_unused_columns

# LoRA:
python trl/scripts/dpo.py \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --learning_rate 5.0e-6 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --logging_steps 25 \
    --eval_strategy steps \
    --eval_steps 50 \
    --output_dir Qwen2-0.5B-DPO \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16
"""

import os
import argparse
import json

import torch
import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from utils import find_optimal_batch_size
from accelerate import Accelerator
from accelerate.utils import broadcast_object_list

from trl import (
    LogCompletionsCallback,
    WinRateCallback,
    ModelConfig,
    ScriptArguments,
    ActiveLearningArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

from trl.build_utils import (
    download_snapshots, get_trainer_config, get_active_trainer,
    get_judge, get_eval_judge, get_reward_model, wrap_reward_as_judge,
    get_win_rate_callback, get_unbiased_win_rate_callback, get_query_strategy_callback)
from active_learning import get_strategy
from active_learning.history_logger import HistoryLogger
from active_learning.dataset_utils import prepare_dataset_for_method, load_datasets


def main(script_args, active_args, training_args, model_args):
    download_snapshots(model_args, script_args, training_args)
    
    ################
    # Dataset
    ################
    local_files_only = True if os.environ['HF_HUB_OFFLINE'] == '1' else False
    if script_args.dataset_name in [
        "Anthropic/hh-rlhf",
        "yuasosnin/imdb-dpo",
        "UCL-DARK/openai-tldr-summarisation-preferences",
        "trl-lib/ultrafeedback_binarized",
    ]:
        try:
            # Use model_args.alignment and do not reference tokenizer yet
            dataset = prepare_dataset_for_method(
                dataset_name=script_args.dataset_name,
                dataset_config=script_args.dataset_config,
                alignment_method=model_args.alignment,
                split="train",
            )

            # Also prepare eval dataset if it exists
            try:
                eval_dataset = prepare_dataset_for_method(
                    dataset_name=script_args.dataset_name,
                    dataset_config=script_args.dataset_config,
                    alignment_method=model_args.alignment,
                    split="test",
                )
            except Exception as eval_e:
                print(f"Could not load eval dataset: {eval_e}")
                eval_dataset = None

        except Exception as e:
            print(f"Failed to use custom preprocessing: {e}")
            # Fallback to original loading'
            dataset = load_datasets(script_args.dataset_name, script_args.dataset_config, split="train")
            eval_dataset = load_datasets(script_args.dataset_name, script_args.dataset_config, split="test")
    else:
        # Original loading for other datasets
        dataset = load_dataset(script_args.dataset_name, script_args.dataset_config, split="train")
        eval_dataset = load_dataset(script_args.dataset_name, script_args.dataset_config, split="test")
    
    ###################
    # Model & Tokenizer
    ###################
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        local_files_only=local_files_only,
        cache_dir=os.environ["HF_HUB_CACHE"]
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )

    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
        )
    else:
        ref_model = None
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, padding_side="left", trust_remote_code=model_args.trust_remote_code,
        cache_dir=os.environ["HF_HUB_CACHE"], local_files_only=local_files_only
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]
    
    # Allow overriding generation length (e.g., for UF) via env.
    try:
        max_new_tokens_env = int(os.environ.get("MAX_NEW_TOKENS", "64"))
    except ValueError:
        max_new_tokens_env = 64
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens_env,
        temperature=1e-5,
        do_sample=True)

    
    ################
    # Optimal Batch Size
    ################
    batch_size = training_args.per_device_train_batch_size
    active_args.optimal_batch_size = batch_size
    if int(active_args.optimal_batch) > 0:
        prompt_samples = [ex["prompt"] for ex in dataset.shuffle().select(range(500))]
        accelerator = Accelerator()
        device = accelerator.device

        batch_size = find_optimal_batch_size(
                                                model, tokenizer, prompt_texts=prompt_samples, 
                                                length_percentile=0.9, max_batch_size=32, device=device,
                                                use_amp=training_args.fp16
                                            )
        active_args.optimal_batch_size = batch_size
        training_args.per_device_eval_batch_size = batch_size
        print(f'optimal_batch_size is set to {batch_size}')
        
        # prev_batch_size = training_args.per_device_train_batch_size
        # training_args.per_device_train_batch_size = batch_size
        

        # calculate num query
        # num_query = int(active_args.num_query / prev_batch_size * batch_size) # batch_size * accelerator.num_processes * training_args.gradient_accumulation_steps
        # active_args.num_query = num_query
        
        # print(f'num_query is set to {num_query}')
    
    
    ################
    # Training
    ################
    
    # construct arguments for a base trainer
    trainer_kwargs = {
        'model': model,
        'ref_model': ref_model,
        'args': training_args,
        'train_dataset': dataset,
        'eval_dataset': eval_dataset,
        'processing_class': tokenizer,
        'peft_config': peft_config,
    }
    
    # Add judge to trainer_kwargs if it's online_dpo or xpo
    # if judge is not None:
    if model_args.alignment in ["online_dpo", "xpo"]:
        if training_args.reward_model_path:
            judge = None
            reward_model, reward_tokenizer = get_reward_model(script_args, training_args)
        elif training_args.judge:
            judge = get_judge(script_args, training_args)
            reward_model, reward_tokenizer = None, None
        else:
            raise NotImplementedError('Either judge or reward model should be provided; not both.')
        
        trainer_kwargs['judge'] = judge
        trainer_kwargs['reward_model'] = reward_model
        trainer_kwargs['reward_processing_class'] = reward_tokenizer
        
    
    # initialized a base trainer e.g. DPOTrainer and wrap it with ActiveTrainer that can change the dataset
    trainer = get_active_trainer(model_args, **trainer_kwargs)
        
    # load a strategy
    strategy = get_strategy(active_args, script_args, trainer)

    # Attach history logger (env-driven). Never break if unavailable.
    try:
        lora_tag = "lora" if peft_config is not None else "nolora"
        run_meta = {
            "dataset": script_args.dataset_name,
            "model": model_args.model_name_or_path,
            "alignment": model_args.alignment,
            "loss": getattr(training_args, "loss_type", "unknown"),
            "lora": lora_tag,
            "query": getattr(active_args, "query_strategy", "unknown"),
            "output_dir": training_args.output_dir,
        }

        # Record reward model / judge configuration so that replay and
        # downstream analysis can distinguish between different reward
        # setups (e.g. Deberta vs Skywork-v2) even when sharing the same
        # dataset/model/query combination.
        try:
            run_meta["reward_model_path"] = getattr(training_args, "reward_model_path", None)
            run_meta["eval_judge"] = getattr(training_args, "eval_judge", None)
        except Exception:
            pass

        try:
            save_strategy_value = getattr(training_args, "save_strategy", None)
            if save_strategy_value is not None:
                save_strategy_value = str(save_strategy_value)

            training_snapshot = {
                "learning_rate": float(training_args.learning_rate) if training_args.learning_rate is not None else None,
                "per_device_train_batch_size": int(training_args.per_device_train_batch_size),
                "per_device_eval_batch_size": int(training_args.per_device_eval_batch_size),
                "gradient_accumulation_steps": int(training_args.gradient_accumulation_steps),
                "warmup_steps": int(training_args.warmup_steps) if training_args.warmup_steps is not None else None,
                "max_steps": int(training_args.max_steps) if training_args.max_steps is not None else None,
                "num_train_epochs": float(getattr(training_args, "num_train_epochs", 0.0)) if getattr(training_args, "num_train_epochs", None) is not None else None,
                "save_steps": training_args.save_steps,
                "logging_steps": training_args.logging_steps,
                "beta": float(getattr(training_args, "beta", 0.0)) if getattr(training_args, "beta", None) is not None else None,
                "loss_type": getattr(training_args, "loss_type", None),
                "save_strategy": save_strategy_value,
            }
            # Drop None values for cleanliness
            training_snapshot = {k: v for k, v in training_snapshot.items() if v is not None}
            if training_snapshot:
                run_meta["training"] = training_snapshot
        except Exception as exc_snapshot:
            print(f"History logger training snapshot warning: {exc_snapshot}")

        try:
            active_snapshot = {
                "num_query": int(getattr(active_args, 'num_query', 0)),
                "updates_per_sample": int(getattr(active_args, 'updates_per_sample', 1)),
                "query_strategy": getattr(active_args, 'query_strategy', None),
                "radius": float(getattr(active_args, 'radius', 0.0)) if getattr(active_args, 'radius', None) is not None else None,
                "normalize": getattr(active_args, 'normalize', None),
                "optimal_batch": float(getattr(active_args, 'optimal_batch', 0.0)) if getattr(active_args, 'optimal_batch', None) is not None else None,
            }
            active_snapshot = {k: v for k, v in active_snapshot.items() if v is not None}
            if active_snapshot:
                run_meta["active"] = active_snapshot
        except Exception as exc_active:
            print(f"History logger active snapshot warning: {exc_active}")

        trainer.history_logger = HistoryLogger(run_meta)
    except Exception as exc:
        print(f"History logger disabled: {exc}")
        trainer.history_logger = None
    
    ################
    # Eval Setup
    ################
    
    # create a judge for evaluation
    if training_args.eval_judge == training_args.judge:
        eval_judge = judge
    elif training_args.eval_judge == training_args.reward_model_path:
        eval_judge = wrap_reward_as_judge(trainer)
    else:
        eval_judge = get_eval_judge(script_args, training_args)
    
    win_rate_callback_config = {
        'eval_judge': eval_judge,
        'generation_config': generation_config
    }
    
    # load and add a win rate callback
    win_rate_callback = get_win_rate_callback(
        win_rate_callback_config, strategy.trainer, num_prompts=training_args.num_eval_prompts)
    # win_rate_callback = get_unbiased_win_rate_callback(win_rate_callback_config, strategy.trainer, num_prompts=1000)
    strategy.trainer.add_callback(win_rate_callback)
    
    ################
    # Train
    ################
    
    # load and add a query callback
    query_strategy_callback = get_query_strategy_callback(training_args, active_args, strategy)
    strategy.trainer.add_callback(query_strategy_callback)
    
    # init query
    args_to_update = strategy.query(active_args.num_query)
    if active_args.query_strategy != 'dummy':
        strategy.update_data(args_to_update)
    else: 
        strategy.trainer.labeled_indices = np.arange(len(dataset))
        strategy.trainer.active_iter = None
    
    # if strategy.trainer.accelerator.is_main_process:
    #     import IPython; IPython.embed()
    # strategy.trainer.accelerator.wait_for_everyone()

    strategy.trainer.train()

    # Ensure the win-rate callback records a point at the final global_step.
    # When `--eval_steps` is a fraction (e.g. 0.1), the internal rounding can
    # skip the last step (e.g. max_steps=625 -> logs at 567). A final evaluate()
    # call guarantees a last point at step=max_steps without changing training.
    try:
        accelerator = strategy.trainer.accelerator
        needs_final_eval = True
        if accelerator.is_main_process:
            last_eval_step = None
            for log in reversed(strategy.trainer.state.log_history):
                if "eval_win_rate" in log:
                    last_eval_step = log.get("step")
                    break
            needs_final_eval = last_eval_step != strategy.trainer.state.global_step
        flag = [bool(needs_final_eval)]
        flag = broadcast_object_list(flag, from_process=0)
        needs_final_eval = bool(flag[0])
        if needs_final_eval:
            _ = strategy.trainer.evaluate()
    except Exception as exc:
        print(f"Final win-rate evaluation skipped: {exc}")
        
    # Save win_rates to file
    if strategy.trainer.accelerator.is_main_process:
        eval_logs = [log for log in strategy.trainer.state.log_history if 'eval_win_rate' in log]
        win_rates = [log['eval_win_rate'] for log in eval_logs]
        steps = [log['step'] for log in eval_logs]
        
        win_rates_file = os.path.join(training_args.output_dir, 'win_rates.json')
        df = pd.DataFrame({
            "step": steps,
            "win_rate": win_rates
        })
        # Dedupe in case the final evaluate() repeats the last step.
        df = df.drop_duplicates(subset=["step"], keep="last").sort_values("step").reset_index(drop=True)
        df.to_json(win_rates_file, orient="records", lines=False)
        
        print(f'Win rates saved to {win_rates_file}')
        print(f'Win rates across all rounds:\n')
        print(df)
        print('\n')
        
        # Save and push to hub
        trainer.save_model(training_args.output_dir)
        if training_args.push_to_hub:
            trainer.push_to_hub(dataset_name=script_args.dataset_name)

    # Finalize history logging (flush + optional push)
    try:
        if hasattr(trainer, 'history_logger') and trainer.history_logger is not None and strategy.trainer.accelerator.is_main_process:
            result = trainer.history_logger.finalize()
            print(f"History logging summary: {result}")
    except Exception as exc:
        print(f"History finalize failed: {exc}")
    
    
def get_pre_args():
    parser = argparse.ArgumentParser(description="Trainer Configuration")

    parser.add_argument(
        "--alignment",
        type=str,
        choices=["dpo", "online_dpo", "xpo"],
        required=True,
        help="Specify the alignment strategy."
    )
    
    args, _ = parser.parse_known_args() 
    return args


def make_parser(pre_args, subparsers: argparse._SubParsersAction = None):
    trainer_config = get_trainer_config(pre_args.alignment)
    dataclass_types = (ScriptArguments, ActiveLearningArguments, trainer_config, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("dpo", help="Run the DPO training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
        
    return parser


if __name__ == "__main__":
    pre_args = get_pre_args()
    parser = make_parser(pre_args)
    script_args, active_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, active_args, training_args, model_args)
