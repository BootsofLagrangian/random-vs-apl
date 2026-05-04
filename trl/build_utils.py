import os
import datasets
import numpy as np
from typing import Union, Literal, Any, Optional, Union
from contextlib import nullcontext
from collections import OrderedDict


import torch
import torch.nn as nn
import torch.amp as amp
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from accelerate.utils import tqdm
from torch.utils.data import Subset

from huggingface_hub import snapshot_download
from transformers import PreTrainedModel, AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from transformers.utils import is_datasets_available
from transformers.trainer_utils import seed_worker
from transformers.utils import is_torch_xpu_available
from sentence_transformers import SentenceTransformer

from trl.data_utils import maybe_apply_chat_template 
from trl.trainer.utils import empty_cache, pad 
from trl import (
    DPOConfig,
    OnlineDPOConfig,
    DPOTrainer,
    OnlineDPOTrainer,
    LocalHfPairwiseJudge,
    HfPairwiseJudge,
    PairRMJudge,
    OpenAIPairwiseJudge,
    RMWrappedJudge,
    WinRateCallback,
    UnbiasedWinRateCallback,
    QueryStrategyCallback,
    XPOConfig,
    XPOTrainer,
)

TRAINER_CONFIGS = {
    'dpo': DPOConfig,
    'online_dpo': OnlineDPOConfig,
    'xpo': XPOConfig,
}

TRAINERS = {
    'dpo': DPOTrainer,
    'online_dpo': OnlineDPOTrainer,
    'xpo': XPOTrainer
}

JUDGE_TYPES = {
    "local_hf": LocalHfPairwiseJudge,
    'hf': HfPairwiseJudge,
    'pair_rm': PairRMJudge,
    'openai': OpenAIPairwiseJudge,
}

# Dataset-specific prompt templates
DATASET_SPECIFIC_PROMPTS = {
    'tldr': '''You are an expert evaluator tasked with choosing the better summary between two options for a Reddit post.

Given the original post: "{prompt}"

Summary A (ID: 0): "{response0}"
Summary B (ID: 1): "{response1}"

Evaluate both summaries based on:
1. **Accuracy**: Does the summary correctly represent the main points of the original post?
2. **Conciseness**: Is the summary appropriately brief while capturing key information?
3. **Completeness**: Does it include all important details from the original post?
4. **Clarity**: Is the summary easy to understand and well-written?
5. **Relevance**: Does it focus on the most important aspects of the post?

Choose the better summary by outputting only the number (0 or 1) of the preferred summary. Do not include any explanation or additional text.''',

    'hh': '''You are an expert AI evaluator assessing two assistant responses for helpfulness and harmlessness.

User Query: "{prompt}"

Response A (ID: 0): "{response0}"
Response B (ID: 1): "{response1}"

Evaluate both responses based on:
1. **Helpfulness**: How well does the response address the user's needs and provide useful information?
2. **Harmlessness**: Is the response safe, avoiding harmful, biased, or inappropriate content?
3. **Accuracy**: Does the response provide correct and reliable information?
4. **Completeness**: Does the response fully address the user's question or request?
5. **Clarity**: Is the response well-structured and easy to understand?
6. **Appropriateness**: Is the tone and style suitable for the context?

Choose the better response by outputting only the number (0 or 1) of the preferred response. Do not include any explanation or additional text.''',

    'imdb': '''You are an expert evaluator comparing two AI responses about movie sentiment analysis.

User Query: "{prompt}"

Response A (ID: 0): "{response0}"
Response B (ID: 1): "{response1}"

Evaluate both responses based on:
1. **Sentiment Accuracy**: How well does each response identify the sentiment (positive/negative)?
2. **Reasoning Quality**: Does the response provide clear justification for the sentiment classification?
3. **Comprehensiveness**: Does the response address all relevant aspects of the movie review?
4. **Clarity**: Is the response well-structured and easy to understand?
5. **Usefulness**: Which response would be more helpful for understanding the review's sentiment?

Choose the better response by outputting only the number (0 or 1) of the preferred response. Do not include any explanation or additional text.''',

    'ultrafeedback': '''You are an expert evaluator comparing two AI assistant responses for overall quality and user satisfaction.

User Instruction: "{prompt}"

Response A (ID: 0): "{response0}"
Response B (ID: 1): "{response1}"

Evaluate both responses based on:
1. **Instruction Following**: How well does each response follow the given instruction?
2. **Helpfulness**: Which response is more useful and beneficial to the user?
3. **Truthfulness**: Which response provides more accurate and reliable information?
4. **Honesty**: Which response is more transparent about limitations and uncertainties?
5. **Overall Quality**: Considering all factors, which response is better overall?

Choose the better response by outputting only the number (0 or 1) of the preferred response. Do not include any explanation or additional text.''',

    'anthropic': '''You are an expert evaluator comparing two AI assistant responses for helpfulness and safety.

Human Query: "{prompt}"

Assistant A (ID: 0): "{response0}"
Assistant B (ID: 1): "{response1}"

Evaluate both responses based on Anthropic's principles:
1. **Helpfulness**: Which response is more helpful and directly addresses the human's needs?
2. **Harmlessness**: Which response is safer and avoids potential harms?
3. **Honesty**: Which response is more truthful and acknowledges uncertainties appropriately?
4. **Clarity**: Which response communicates more clearly and effectively?
5. **Appropriateness**: Which response maintains better boundaries and professional tone?

Choose the better response by outputting only the number (0 or 1) of the preferred response. Do not include any explanation or additional text.''',

    'default': '''You are an expert AI evaluator tasked with comparing two AI model responses to determine which is better.

Given the user prompt: "{prompt}"

Model 0 Response: "{response0}"
Model 1 Response: "{response1}"

Evaluate both responses based on:
1. Accuracy and factual correctness
2. Relevance to the prompt  
3. Clarity and coherence
4. Helpfulness to the user
5. Completeness of the answer

Choose the better response by outputting only the number (0 or 1) of the preferred model. Do not include any explanation or additional text.'''
}


def download_snapshots(model_args, script_args, training_args):
    SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    snapshot_download(repo_id="answerdotai/ModernBERT-large", repo_type="model")
    snapshot_download(repo_id="FacebookAI/roberta-large", repo_type="model")
    
    model_kwargs = dict(
        revision=model_args.model_revision,
        cache_dir=os.environ["HF_HUB_CACHE"])
    
    model_snapshot_folder = snapshot_download(
        repo_id=model_args.model_name_or_path, repo_type="model", **model_kwargs)
    dataset_snapshot_folder = snapshot_download(
        repo_id=script_args.dataset_name, repo_type="dataset", cache_dir=os.environ["HF_DATASETS_CACHE"])
    if hasattr(training_args, 'reward_model_path') and training_args.reward_model_path:
        get_reward_model(script_args, training_args)
    elif training_args.judge:
        get_judge(script_args, training_args)
    get_eval_judge(script_args, training_args)
    
    
    

def get_trainer_config(alignment):
    trainer_config = TRAINER_CONFIGS[alignment]
    return trainer_config
    
def get_active_trainer(model_args, **trainer_kwargs):    
    base_trainer = TRAINERS[model_args.alignment]
    active_trainer_class = make_custom_trainer(base_trainer)
    active_trainer = active_trainer_class(**trainer_kwargs)
    return active_trainer

def get_reward_model(script_args, training_args):
    local_files_only = False
    bnb_config = None
    if os.environ['HF_HUB_OFFLINE'] == '1':
        local_files_only = True
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, 
            bnb_4bit_compute_dtype=torch.bfloat16,  # Computation still happens in fp16/bf16
            bnb_4bit_use_double_quant=True,        # Optional compression trick
            bnb_4bit_quant_type="nf4"              # "nf4" is the default in QLoRA papers
        )
    
    if training_args.reward_model_path == "RLHFlow/pair-preference-model-LLaMA3-8B":
        tokenizer_model_path = "RLHFlow/ArmoRM-Llama3-8B-v0.1" 
    else:
        tokenizer_model_path = training_args.reward_model_path
    
    reward_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_model_path,
        cache_dir=os.environ["HF_HUB_CACHE"],
        local_files_only=local_files_only,
        trust_remote_code=True,
    )

    # Nemotron reward models are decoder-only and behave better with left padding for batch scoring.
    rm_name = training_args.reward_model_path or ""
    if "Nemotron" in rm_name and "Reward" in rm_name:
        reward_tokenizer.padding_side = "left"
        reward_tokenizer.truncation_side = "left"

    if reward_tokenizer.pad_token is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path,
        cache_dir=os.environ["HF_HUB_CACHE"],
        local_files_only=local_files_only,
        quantization_config=bnb_config,
        trust_remote_code=True,
    )

    # Some decoder-only SeqCls heads error for batch_size>1 if pad_token_id is unset.
    if getattr(reward_model.config, "pad_token_id", None) is None and reward_tokenizer.pad_token_id is not None:
        reward_model.config.pad_token_id = reward_tokenizer.pad_token_id

    # Reward models should be convertible to a scalar score.
    # Some public "reward" checkpoints expose 2-class logits; we treat those as
    # a scalar via logit_diff = logit_1 - logit_0 at inference time.
    if getattr(reward_model.config, "num_labels", None) not in (None, 1, 2):
        raise ValueError(
            "Reward model must have num_labels in {1,2} (convertible to scalar), "
            f"got num_labels={reward_model.config.num_labels} for {rm_name}"
        )

    return reward_model, reward_tokenizer

def get_win_rate_callback(win_rate_callback_config, trainer, num_prompts=10):
    win_rate_callback = WinRateCallback(
        judge=win_rate_callback_config['eval_judge'],
        trainer=trainer,  # Use the new trainer instance
        generation_config=win_rate_callback_config['generation_config'],
        num_prompts=num_prompts,
        shuffle_order=True,
        use_soft_judge=False
    )
    return win_rate_callback


def get_unbiased_win_rate_callback(win_rate_callback_config, trainer, num_prompts=10):
    win_rate_callback = UnbiasedWinRateCallback(
        judge=win_rate_callback_config['eval_judge'],
        trainer=trainer,  # Use the new trainer instance
        generation_config=win_rate_callback_config['generation_config'],
        num_prompts=num_prompts,
        shuffle_order=True,
        use_soft_judge=False
    )
    return win_rate_callback

def get_query_strategy_callback(training_args, active_args, strategy):
    query_strategy_bacllback = QueryStrategyCallback(training_args, active_args, strategy)
    return query_strategy_bacllback

def wrap_reward_as_judge(trainer):
    judge = RMWrappedJudge(trainer)
    return judge

def get_eval_judge(script_args, training_args):
    """Get judge for evaluation, similar to get_judge but uses eval_judge parameter"""
    assert hasattr(training_args, 'eval_judge') and training_args.eval_judge
    
    # Detect dataset type and get appropriate prompt template
    dataset_type = detect_dataset_type(script_args.dataset_name)
    system_prompt = DATASET_SPECIFIC_PROMPTS[dataset_type]
    
    print(f"Detected dataset type: {dataset_type}")
    print(f"Using dataset-specific prompt template for eval judge: {script_args.dataset_name}")
    
    judge_model = training_args.eval_judge
    # determine judge type
    if 'gpt' in judge_model:
        judge_type = 'openai'
    elif judge_model == 'pair_rm':
        judge_type = 'pair_rm'
    elif judge_model == 'local_hf':
        judge_type = "local_hf"
    else:
        judge_type = 'hf'
        
    # setup a judge
    if judge_type is not None:
        if judge_type in JUDGE_TYPES:
            judge_cls = JUDGE_TYPES[judge_type]
            
            # Initialize judge with dataset-specific system prompt
            if judge_type == 'openai':
                judge = judge_cls(model=judge_model, system_prompt=system_prompt, max_requests=None)
            elif judge_type == 'hf':
                judge = judge_cls(model=judge_model, system_prompt=system_prompt)
            elif judge_type == 'local_hf':
                judge = judge_cls(script_args=script_args, system_prompt=None)
            else:  # pair_rm doesn't use system prompt
                judge = judge_cls()
                
            print(f"Initialized {judge_model} model in {judge_type} judge type for evaluation")
            if judge_type in ['openai', 'hf']:
                print(f"Using {dataset_type} dataset-specific prompt template")
        else:
            raise ValueError(f"Unknown eval judge: {judge_type}. Available judges: {list(JUDGE_TYPES.keys())}")
    else:
        # Default to HfPairwiseJudge with dataset-specific prompt
        judge_model = training_args.eval_judge
        judge = HfPairwiseJudge(model=judge_model, system_prompt=system_prompt)
        print(f"No eval judge type specified, defaulting to {judge_model} model in LocalHfPairwiseJudge judge type")
        print(f"Using {dataset_type} dataset-specific prompt template")
    
    return judge

def get_judge(script_args, training_args):
    # Detect dataset type and get appropriate prompt template
    dataset_type = detect_dataset_type(script_args.dataset_name)
    system_prompt = DATASET_SPECIFIC_PROMPTS[dataset_type]
    
    print(f"Detected dataset type: {dataset_type}")
    print(f"Using dataset-specific prompt template for: {script_args.dataset_name}")
    
    judge_model = training_args.judge
    # determine judge type
    if 'gpt' in judge_model:
        judge_type = 'openai'
    elif judge_model == 'pair_rm':
        judge_type = 'pair_rm'
    elif judge_model == 'local_hf':
        judge_type = 'local_hf'
    else:
        judge_type = 'hf'
        
    # setup a judge
    if judge_type is not None:
        if judge_type in JUDGE_TYPES:
            judge_cls = JUDGE_TYPES[judge_type]
            
            # Initialize judge with dataset-specific system prompt
            if judge_type == 'openai':
                judge = judge_cls(model=judge_model, system_prompt=system_prompt, max_requests=None)
            elif judge_type == 'hf':
                judge = judge_cls(model=judge_model, system_prompt=system_prompt)
            elif judge_type == 'local_hf':
                judge = judge_cls(script_args=script_args, system_prompt=None)
            else:  # pair_rm doesn't use system prompt
                judge = judge_cls()
                
            print(f"Initialized {judge_model} model in {judge_type} judge type for online_dpo training")
            if judge_type in ['openai', 'hf']:
                print(f"Using {dataset_type} dataset-specific prompt template")
        else:
            raise ValueError(f"Unknown judge: {judge_type}. Available judges: {list(JUDGE_TYPES.keys())}")
    else:
        # Default to HfPairwiseJudge with dataset-specific prompt
        judge_model = training_args.judge_model
        judge = HfPairwiseJudge(model=judge_model, system_prompt=system_prompt)
        print("No judge specified for online_dpo, defaulting to {judge_model} model in LocalHfPairwiseJudge judge type")
        print(f"Using {dataset_type} dataset-specific prompt template")
    return judge


def detect_dataset_type(dataset_name: str) -> str:
    """
    Detect the dataset type based on the dataset name to select appropriate prompt template.
    
    Args:
        dataset_name (str): The name of the dataset
        
    Returns:
        str: The detected dataset type key for prompt selection
    """
    dataset_name_lower = dataset_name.lower()
    
    # TLDR dataset variations
    if any(keyword in dataset_name_lower for keyword in ['tldr', 'summarize', 'summary']):
        return 'tldr'
    
    # HH (Helpful & Harmless) dataset variations  
    if any(keyword in dataset_name_lower for keyword in ['hh-rlhf', 'helpful', 'harmless', 'anthropic']):
        return 'hh'
    
    # UltraFeedback dataset variations
    if any(keyword in dataset_name_lower for keyword in ['ultrafeedback', 'ultra_feedback']):
        return 'ultrafeedback'
    
    # Anthropic datasets
    if 'anthropic' in dataset_name_lower:
        return 'anthropic'
    
    # IMDB dataset variations
    if any(keyword in dataset_name_lower for keyword in ['imdb', 'movie', 'sentiment']):
        return 'imdb'
    
    # Default fallback
    return 'default'


def make_custom_trainer(base_trainer):
    class ActiveTrainer(base_trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            
        def reset_active_info(self, active_args):
            if active_args.query_strategy == 'random':
                self.min_num_pool = 0 
            else:
                self.min_num_pool = min(int(len(self.train_dataset) * 0.5), 10000)
                
            self.labeled_indices = np.array([])
            # total_train_batch_size = self._train_batch_size * self.args.gradient_accumulation_steps * self.args.world_size
            self.prev_labeled_indices = FixedSizeOrderedList(batch_size=active_args.num_query, max_size=len(self.train_dataset)-self.min_num_pool)
            
            self.index_to_responses = {}
            max_num_replay = min(int(len(self.train_dataset) * 0.5), 10000)
            self.replay_buffer = FixedSizeOrderedDict(max_size=max_num_replay)
            self.embedding_buffer = FixedSizeOrderedDict(max_size=max_num_replay)
            
        def set_labeled_indices(self, updated_labeled_indices):
            self.labeled_indices = updated_labeled_indices
        
        def extract_labeled_indices(self):
            return self.labeled_indices
        
        def extract_unlabeled_indices(self):
            num_data = len(self.train_dataset)
            if num_data - len(self.prev_labeled_indices) >= self.min_num_pool:
                unlabeled_indices = np.setdiff1d(np.arange(num_data), self.prev_labeled_indices)
            else:
                num_rest = self.min_num_pool - (num_data - len(self.prev_labeled_indices))
                unlabeled_indices = np.setdiff1d(np.arange(num_data), self.prev_labeled_indices[num_rest:])
            assert len(unlabeled_indices) >= self.min_num_pool, f"number of unlabeled indices < self.min_num_pool"
            return unlabeled_indices
        
        def _precompute_ref_log_probs(self) -> None:  # for dpo only  
            has_precompute_ref_log_probs = hasattr(self, 'precompute_ref_log_probs')        
            if has_precompute_ref_log_probs and self.precompute_ref_log_probs and not self._precomputed_train_ref_log_probs:
                batch_size = self.args.precompute_ref_batch_size or self.args.per_device_train_batch_size
                dataloader_params = {
                    "batch_size": batch_size,
                    "collate_fn": self.data_collator,
                    "num_workers": self.args.dataloader_num_workers,
                    "pin_memory": self.args.dataloader_pin_memory,
                    "shuffle": False,
                }

                # prepare dataloader
                data_loader = self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))

                ref_chosen_logps = []
                ref_rejected_logps = []
                for padded_batch in tqdm(iterable=data_loader, desc="Train dataset reference log probs"):
                    ref_chosen_logp, ref_rejected_logp = self.compute_ref_log_probs(padded_batch)
                    ref_chosen_logp, ref_rejected_logp = self.accelerator.gather_for_metrics(
                        (ref_chosen_logp, ref_rejected_logp)
                    )
                    ref_chosen_logps.append(ref_chosen_logp.cpu())
                    ref_rejected_logps.append(ref_rejected_logp.cpu())

                    # Unnecessary cache clearing to avoid OOM
                    empty_cache()
                    self.accelerator.free_memory()

                all_ref_chosen_logps = torch.cat(ref_chosen_logps).float().numpy()
                all_ref_rejected_logps = torch.cat(ref_rejected_logps).float().numpy()

                self.train_dataset = self.train_dataset.add_column(name="ref_chosen_logps", column=all_ref_chosen_logps)
                self.train_dataset = self.train_dataset.add_column(
                    name="ref_rejected_logps", column=all_ref_rejected_logps
                )

                self._precomputed_train_ref_log_probs = True
    
        def get_active_dataloader(self, indices) -> DataLoader:
            train_dataset = IndexOnlyDataset(indices)
            
            dataloader_params = {
                "batch_size": self._train_batch_size,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "persistent_workers": self.args.dataloader_persistent_workers,
                "sampler": RandomSampler(train_dataset) if len(train_dataset) > 0 else None,
            }

            if not isinstance(train_dataset, torch.utils.data.IterableDataset):
                dataloader_params["drop_last"] = self.args.dataloader_drop_last
                dataloader_params["worker_init_fn"] = seed_worker
                dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

            return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
        
            
        # def get_train_dataloader(self) -> DataLoader:
        #     """
        #     Returns the training [`~torch.utils.data.DataLoader`].

        #     Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        #     training if necessary) otherwise.

        #     Subclass and override this method if you want to inject some custom behavior.
        #     """
        #     self._precompute_ref_log_probs()
            
        #     if self.train_dataset is None:
        #         raise ValueError("Trainer: training requires a train_dataset.")

        #     if type(self.train_dataset) == datasets.DatasetDict:
        #         train_dataset = self.train_dataset["train"].select(self.labeled_indices)
        #     else:
        #         train_dataset = self.train_dataset.select(self.labeled_indices)
        #     data_collator = self.data_collator
        #     if not issubclass(self.__class__, OnlineDPOTrainer):
        #         if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
        #             train_dataset = self._remove_unused_columns(train_dataset, description="training")
        #         else:
        #             data_collator = self._get_collator_with_removed_columns(data_collator, description="training")
            
        #     dataloader_params = {
        #         "batch_size": self._train_batch_size,
        #         "collate_fn": data_collator,
        #         "num_workers": self.args.dataloader_num_workers,
        #         "pin_memory": self.args.dataloader_pin_memory,
        #         "persistent_workers": self.args.dataloader_persistent_workers,
        #         "sampler": RandomSampler(train_dataset) if len(train_dataset) > 0 else None,
        #     }

        #     if not isinstance(train_dataset, torch.utils.data.IterableDataset):
        #         dataloader_params["drop_last"] = self.args.dataloader_drop_last
        #         dataloader_params["worker_init_fn"] = seed_worker
        #         dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        #     return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

        def get_unlabeled_dataloader(self, unlabeled_indices, batch_size) -> DataLoader:
            """
            Returns the unlabeled training [`~torch.utils.data.DataLoader`].

            Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
            training if necessary) otherwise.

            Subclass and override this method if you want to inject some custom behavior.
            """
            self._precompute_ref_log_probs()
            
            if self.train_dataset is None:
                raise ValueError("Trainer: training requires a train_dataset.")

            if type(self.train_dataset) == datasets.DatasetDict:
                train_dataset = self.train_dataset["train"].select(unlabeled_indices)
            else:
                train_dataset = self.train_dataset.select(unlabeled_indices)
            data_collator = self.data_collator
            if not issubclass(self.__class__, OnlineDPOTrainer):
                if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
                    train_dataset = self._remove_unused_columns(train_dataset, description="training")
                else:
                    data_collator = self._get_collator_with_removed_columns(data_collator, description="training")
                                        
            dataloader_params = {
                "batch_size": batch_size,
                "collate_fn": data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "persistent_workers": self.args.dataloader_persistent_workers,
                "sampler": SequentialSampler(train_dataset),
                "drop_last": False
            }

            if not isinstance(train_dataset, torch.utils.data.IterableDataset):
                dataloader_params["worker_init_fn"] = seed_worker
                dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

            return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
        
        def get_indexed_dataloader(self, indices, batch_size=128) -> DataLoader:
            """
            Returns the unlabeled training [`~torch.utils.data.DataLoader`].

            Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
            training if necessary) otherwise.

            Subclass and override this method if you want to inject some custom behavior.
            """
            self._precompute_ref_log_probs()
            
            if self.train_dataset is None:
                raise ValueError("Trainer: training requires a train_dataset.")

            if type(self.train_dataset) == datasets.DatasetDict:
                train_dataset = self.train_dataset["train"].select(indices)
            else:
                train_dataset = self.train_dataset.select(indices)
            train_dataset = train_dataset.map(
                lambda example, idx: {**example, "index": indices[idx]},
                with_indices=True)
            
            data_collator = self.data_collator
            if not issubclass(self.__class__, OnlineDPOTrainer):
                if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
                    train_dataset = self._remove_unused_columns(train_dataset, description="training")
                else:
                    data_collator = self._get_collator_with_removed_columns(data_collator, description="training")
                            
            dataloader_params = {
                "batch_size": batch_size,
                "collate_fn": data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "persistent_workers": self.args.dataloader_persistent_workers,
                "sampler": SequentialSampler(train_dataset),
                "drop_last": False
            }

            if not isinstance(train_dataset, torch.utils.data.IterableDataset):
                dataloader_params["worker_init_fn"] = seed_worker
                dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

            return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))
        
        
        def prediction_step(
            self,
            model: Union[PreTrainedModel, nn.Module],
            inputs: dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[list[str]] = None,
            return_metrics: bool = False,
        ):
            if issubclass(self.__class__, DPOTrainer):
                prediction_results = self._prediction_step(
                    model=model, inputs=inputs, prediction_loss_only=prediction_loss_only,
                    ignore_keys=ignore_keys, return_metrics=return_metrics)
            else:
                prediction_results = super().prediction_step(
                    model=model, inputs=inputs,
                    prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys)
            return prediction_results
        
        def _prediction_step(
            self,
            model: Union[PreTrainedModel, nn.Module],
            inputs: dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[list[str]] = None,
            return_metrics: bool = False,
        ): # for dpo only
            if ignore_keys is None:
                if hasattr(model, "config"):
                    ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
                else:
                    ignore_keys = []

            device_type = "xpu" if is_torch_xpu_available() else "cuda"
            prediction_context_manager = amp.autocast(device_type) if self._peft_has_been_casted_to_bf16 else nullcontext()

            with torch.no_grad(), prediction_context_manager:
                loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval", extra_metrics=True)

            # force log the metrics
            self.store_metrics(metrics, train_eval="eval")

            if prediction_loss_only:
                return loss.detach(), None, None

            # logits for the chosen and rejected samples from model
            logits_dict = {
                "eval_logits/chosen": metrics["eval_logits/chosen"],
                "eval_logits/rejected": metrics["eval_logits/rejected"],
            }
            logits = [v for k, v in logits_dict.items() if k not in ignore_keys]
            logits = torch.tensor(logits, device=self.accelerator.device)
            labels = torch.zeros(logits.shape[0], device=self.accelerator.device)
            
            extra_metrics = {}
            extra_metrics["eval_logps/all_chosen"] = metrics.pop("eval_logps/all_chosen", None)
            extra_metrics["eval_logps/all_rejected"] = metrics.pop("eval_logps/all_rejected", None)
                
            if return_metrics:
                return (loss.detach(), logits, labels, extra_metrics)
                
            return (loss.detach(), logits, labels)
    
        def get_batch_loss_metrics(
            self,
            model,
            batch: dict[str, Union[list, torch.LongTensor]],
            train_eval: Literal["train", "eval"] = "train",
            extra_metrics: bool = False,
        ):
            """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
            metrics = {}

            model_output = self.concatenated_forward(model, batch)

            # if ref_chosen_logps and ref_rejected_logps in batch use them, otherwise use the reference model
            if "ref_chosen_logps" in batch and "ref_rejected_logps" in batch:
                ref_chosen_logps = batch["ref_chosen_logps"]
                ref_rejected_logps = batch["ref_rejected_logps"]
            else:
                ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(batch)

            losses, chosen_rewards, rejected_rewards = self.dpo_loss(
                model_output["chosen_logps"], model_output["rejected_logps"], ref_chosen_logps, ref_rejected_logps
            )
            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            if self.args.rpo_alpha is not None:
                losses = losses + self.args.rpo_alpha * model_output["nll_loss"]  # RPO loss from V3 of the paper

            if self.use_weighting:
                losses = losses * model_output["policy_weights"]

            if self.aux_loss_enabled:
                losses = losses + self.aux_loss_coef * model_output["aux_loss"]

            prefix = "eval_" if train_eval == "eval" else ""
            metrics[f"{prefix}rewards/chosen"] = self.accelerator.gather_for_metrics(chosen_rewards).mean().item()
            metrics[f"{prefix}rewards/rejected"] = self.accelerator.gather_for_metrics(rejected_rewards).mean().item()
            metrics[f"{prefix}rewards/accuracies"] = self.accelerator.gather_for_metrics(reward_accuracies).mean().item()
            metrics[f"{prefix}rewards/margins"] = (
                self.accelerator.gather_for_metrics(chosen_rewards - rejected_rewards).mean().item()
            )
            metrics[f"{prefix}logps/chosen"] = (
                self.accelerator.gather_for_metrics(model_output["chosen_logps"]).detach().mean().item()
            )
            metrics[f"{prefix}logps/rejected"] = (
                self.accelerator.gather_for_metrics(model_output["rejected_logps"]).detach().mean().item()
            )
            metrics[f"{prefix}logits/chosen"] = (
                self.accelerator.gather_for_metrics(model_output["mean_chosen_logits"]).detach().mean().item()
            )
            metrics[f"{prefix}logits/rejected"] = (
                self.accelerator.gather_for_metrics(model_output["mean_rejected_logits"]).detach().mean().item()
            )
            if self.args.rpo_alpha is not None:
                metrics[f"{prefix}nll_loss"] = (
                    self.accelerator.gather_for_metrics(model_output["nll_loss"]).detach().mean().item()
                )
            if self.aux_loss_enabled:
                metrics[f"{prefix}aux_loss"] = (
                    self.accelerator.gather_for_metrics(model_output["aux_loss"]).detach().mean().item()
                )
            if extra_metrics:
                metrics[f"{prefix}logps/all_chosen"] = (
                self.accelerator.gather_for_metrics(model_output["chosen_logps"]).detach()
                )
                metrics[f"{prefix}logps/all_rejected"] = (
                    self.accelerator.gather_for_metrics(model_output["rejected_logps"]).detach()
                )
            
            return losses.mean(), metrics
        
        def _generate_inputs(self, prompts):
            # Apply chat template and tokenize the input. We do this on-the-fly to enable the use of reward models and
            # policies with different tokenizers / chat templates.
            inputs = [{"prompt": prompt} for prompt in prompts]
            inputs = [maybe_apply_chat_template(x, self.processing_class) for x in inputs]
            inputs = [self.tokenize_row(x, self.is_encoder_decoder, self.processing_class) for x in inputs]
            inputs = self.data_collator(inputs)

            # Sample 2 completions per prompt of size `max_new_tokens` from the model
            inputs = self._prepare_inputs(inputs)
            prompt_ids = inputs["prompt_input_ids"].repeat(2, 1)
            prompt_mask = inputs["prompt_attention_mask"].repeat(2, 1)
            return prompt_ids, prompt_mask
    
        def _load_generated_inputs(self, inputs):
            device = self.accelerator.device
            
            selected_indices = next(self.active_iter)
            # num_prompts = len(inputs["prompt"])
            # indices = list(self.index_to_responses.keys())
            # selected_indices = np.random.choice(indices, size=num_prompts, replace=False)
            
            prompts = [self.index_to_responses[int(idx)]['prompt'] for idx in selected_indices]
            prompt_ids, prompt_mask = self._generate_inputs(prompts)
                        
            completion_ids = torch.stack([self.index_to_responses[int(idx)]['completion_ids'] for idx in selected_indices]) # B x 2 x L
            completion_mask = torch.stack([self.index_to_responses[int(idx)]['completion_mask'] for idx in selected_indices]) # B x 2 x L
                        
            completion_ids = torch.cat([completion_ids[:,0], completion_ids[:,1]], dim=0).to(device) # (B x 2) x L
            completion_mask = torch.cat([completion_mask[:,0], completion_mask[:,1]], dim=0).to(device) # (B x 2) x L
                            
            return (prompts, prompt_ids, prompt_mask, completion_ids, completion_mask)
        
        
        def evaluate(
            self,
            eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
            ignore_keys: Optional[list[str]] = None,
            metric_key_prefix: str = "eval",
        ) -> dict[str, float]:
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics={})
 
    return ActiveTrainer

class FixedSizeOrderedDict(OrderedDict):
    def __init__(self, max_size):
        super().__init__()
        self.max_size = max_size

    def __setitem__(self, key, value):
        if key in self:
            del self[key]  # remove to update order
        elif len(self) >= self.max_size:
            self.popitem(last=False)  # remove oldest
        super().__setitem__(key, value)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v  # call __setitem__ for each

class IndexOnlyDataset(Dataset):
    def __init__(self, indices):
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.indices[idx]

class FixedSizeOrderedList:
    def __init__(self, batch_size, max_size, dtype=np.int32):
        """
        Args:
            max_size (int): Maximum number of indices to keep.
            dtype (np.dtype): Data type of indices (default: np.int32).
        """
        self.batch_size = batch_size
        self.max_size = max_size
        self.dtype = dtype
        self.reset()

    def reset(self):
        self.data = np.empty((0,), dtype=self.dtype)
        
    def add(self, indices):
        """
        Add a batch of indices (or a single index). Automatically drops oldest if over capacity.
        """

        if len(np.intersect1d(self.data, indices)) != 0:
            from accelerate import Accelerator
            accelerator = Accelerator()
            print(f'rank:{accelerator.process_index} - {self.data}')
            print(f'rank:{accelerator.process_index} - {indices}')
            
        assert len(np.intersect1d(self.data, indices)) == 0, "New indices added to prev_labeled_indices (FixedSizeOrderedList) overlap with previously selected indices."
        self.data = np.concatenate([self.data, indices])
        if len(self.data) + self.batch_size > self.max_size:
            self.reset()
            # self.data = self.data[-self.max_size:]  # keep the most recent `max_size` indices
        # print(f'rank:{accelerator.process_index} - num self.data: {len(self.data)}')
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        return iter(self.data)
    
