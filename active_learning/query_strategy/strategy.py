import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from trl.build_utils import get_active_trainer
from trl.data_utils import is_conversational, apply_chat_template
from trl import DPOTrainer, OnlineDPOTrainer, XPOTrainer

from typing import Any, Union, Optional
from ..utils import split_list, chunk_list, safe_gather, merge_prompt_or_completion
from accelerate.utils import gather_object
from transformers import DebertaV2ForSequenceClassification, AutoTokenizer, AutoModel, BitsAndBytesConfig
from trl.trainer.utils import get_reward
from sentence_transformers import SentenceTransformer


EXTRACTORS = {
    "roberta": "FacebookAI/roberta-large",
    "modernbert": "answerdotai/ModernBERT-large",
    "sentence_transformer": "sentence-transformers/all-MiniLM-L6-v2",
    "llm": None
}


class Strategy:
    def __init__(self, trainer, system_prompt, active_args):
        self.trainer = trainer
        self.trainer.reset_active_info(active_args)
        self.active_args = active_args
        self.is_offline = issubclass(self.trainer.__class__, DPOTrainer)
        self.query_batch_size = getattr(active_args, "query_batch_size", 0)
        
        self.system_prompt = system_prompt
        self.extractor_type = active_args.extractor_type
        self.embedding_input_type = active_args.embedding_input_type  # prompt, concat, template
        self.pooling_strategy = "mean" # mean, last, cls
        
        self.subset_factor = 64 # apl - imdb: 4_000 x 8 = 32_000, tldr: 2048 x 4 = 8_192
        self.device = self.trainer.accelerator.device
        
        # Initialize embedding models cache
        self._extractor = {}
        
        # Dataset-specific templates
        self.dataset_templates = self._get_dataset_templates()

    def _get_dataset_templates(self):
        """Get dataset-specific templates for different datasets."""
        return {
            'tldr': '''Post: {prompt}

Summary A: {response0}
Summary B: {response1}

Which summary better captures the key information from the post?''',
            
            'hh-rlhf': '''Human: {prompt}

Response A: {response0}
Response B: {response1}

Which response is more helpful and harmless?''',
            
            'imdb': '''Review: {prompt}

Response A: {response0}
Response B: {response1}

Which response better reflects the sentiment of the review?''',
            
            'ultrafeedback': '''Instruction: {prompt}

Response A: {response0}
Response B: {response1}

Which response better follows the instruction?''',
            
            'default': '''Prompt: {prompt}

Response A: {response0}
Response B: {response1}

Which response is better?''',

            'single': '''Prompt: {prompt}

Response: {response0}''',

            'prefill_a': '''Prompt: {prompt}

Response A: {response0}
Response B: {response1}

Which response is better? Response A''',

            'prefill_b': '''Prompt: {prompt}

Response A: {response0}
Response B: {response1}

Which response is better? Response B''',

        }

    def _detect_dataset_type(self, data):
        """Detect dataset type from the data structure or content."""
        if isinstance(data, dict) and 'prompt' in data and len(data['prompt']) > 0:
            prompt = data['prompt'][0]
            if isinstance(prompt, str):
                # Check for dataset-specific patterns
                if 'SUBREDDIT:' in prompt or 'TL;DR:' in prompt:
                    return 'tldr'
                elif 'Human:' in prompt:
                    return 'hh-rlhf'
                elif 'Review:' in prompt.lower():
                    return 'imdb'
                elif 'Instruction:' in prompt:
                    return 'ultrafeedback'
        return 'default'
        
    def init_labeled_indice(self):
        self.trainer.labeled_indices = np.array([])
        
    def init_train_dataset(self):
        if "completion_ids" in self.trainer.train_dataset.column_names:
            self.trainer.train_dataset.remove_columns("completion_ids")
        if "completion_mask" in self.trainer.train_dataset.column_names:
            self.trainer.train_dataset.remove_columns("completion_mask")
        
    def init_trainer(self, model_args, preserve_labeled_indices=True, preserve_train_dataset=True, **trainer_kwargs):
        labeled_indices = self.trainer.labeled_indices
        train_dataset = self.trainer.train_dataset
        self.trainer = get_active_trainer(model_args, **trainer_kwargs)
        if preserve_labeled_indices:
            self.trainer.labeled_indices = labeled_indices
        if preserve_train_dataset:
            self.trainer.train_dataset = train_dataset
        
    def prepare_model(self, dataloader):
        args = self.trainer.args
        model = self.trainer._wrap_model(self.trainer.model, training=False, dataloader=dataloader)
        
        if len(self.trainer.accelerator._models) == 0 and model is self.trainer.model:
            start_time = time.time()
            model = (
                self.trainer.accelerator.prepare(model)
                if self.trainer.is_deepspeed_enabled or (self.trainer.is_fsdp_enabled and self.trainer.accelerator.mixed_precision != "fp8")
                else self.trainer.accelerator.prepare_model(model, evaluation_mode=True)
            )
            self.trainer.model_preparation_time = round(time.time() - start_time, 4)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.trainer.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        model.eval()
        return model
        
    def generate_responses(self, model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]],
                           n: int = 2, decode_completion: bool = False) -> dict[str, Union[torch.Tensor, Any]]:
        assert n % 2 == 0, f'n={n} is not an even number.'

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
        
        if issubclass(self.trainer.__class__.__bases__[0], XPOTrainer):
            response_dict = self.trainer.generate_responses(model, prompts, n)
            if decode_completion:
                completions = self.trainer.processing_class.batch_decode(response_dict["completion_ids"], skip_special_tokens=True)
                if is_conversational({"prompt": inputs['prompt'][0]}):
                    completions = [[{"role": "assistant", "content": completion}] for completion in completions]
                response_dict["completions"] = completions
            return response_dict
        
        
        num_half_resp = max(1, n // 2)
        repeated_prompts = [p for p in prompts for _ in range(num_half_resp)]
        
        use_vllm = getattr(self.trainer.args, "use_vllm", False)
        if use_vllm:
            prompt_ids, prompt_mask, completion_ids, completion_mask = \
                self.trainer._generate_vllm(model, repeated_prompts)
        else:
            prompt_ids, prompt_mask, completion_ids, completion_mask = \
                self.trainer._generate(model, repeated_prompts)
        
        response_dict = {
            "prompt": prompts,
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
        }
        if decode_completion:
            completions = self.trainer.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
            if is_conversational({"prompt": prompts[0]}):
                completions = [[{"role": "assistant", "content": completion}] for completion in completions]
            response_dict["completions"] = completions
        
        return response_dict
        
    def compute_log_probs(self, model: nn.Module, response_dict: dict[str, Union[torch.Tensor, Any]], n: int) -> torch.Tensor:
        prompt_ids = response_dict["prompt_ids"]
        prompt_mask = response_dict["prompt_mask"]
        completion_ids = response_dict["completion_ids"]
        completion_mask = response_dict["completion_mask"]
        
        logprobs_per_tokens = self.trainer._forward(
            model, prompt_ids, prompt_mask, completion_ids, completion_mask) # (num_prompts x n) x num_tokens
        
        # reorder: (prompt 1, ..., prompt 1) n times, (prompt 2, ..., prompt 2) n times, ..., (prompt b, ..., prompt b) n times, 
        num_promts = prompt_ids.shape[0] // n
        num_half_resp = max(1, n // 2)
        reorder_indices = [[np.arange( i * num_half_resp, (i+1) * num_half_resp),
                            np.arange(num_promts * num_half_resp + i * num_half_resp, num_promts * num_half_resp + (i+1) * num_half_resp)] for i in range(num_promts)]
        reorder_indices = np.concatenate(np.concatenate(reorder_indices, axis=0), axis=0) 
        reorderd_logprobs_per_tokens = logprobs_per_tokens[reorder_indices]
        
        logprobs_per_response = reorderd_logprobs_per_tokens.sum(-1) # (num_prompts x n) x num_tokens -> (num_prompts x n,) 
        logprobs = logprobs_per_response.reshape(n, -1).mean(0) # (num_prompts x n,) -> n x num_prompts -> num_prompts
        return logprobs   
        
    def batched_compute_reward_margin(self, model: nn.Module, response_dict: dict[str, Union[torch.Tensor, Any],], n: int = 2, compute_log_probs=False, mini_bach_size=32) -> torch.Tensor:
        
        def batched_forward_logprobs(model, trainer, prompt_ids, prompt_mask, completion_ids, completion_mask, batch_size):
            total = prompt_ids.shape[0]
            logprobs_list = []

            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)

                prompt_ids_batch = prompt_ids[start:end]
                prompt_mask_batch = prompt_mask[start:end]
                completion_ids_batch = completion_ids[start:end]
                completion_mask_batch = completion_mask[start:end]

                with torch.inference_mode():
                    logprobs = trainer._forward(
                        model,
                        prompt_ids_batch,
                        prompt_mask_batch,
                        completion_ids_batch,
                        completion_mask_batch
                    )
                    logprobs_list.append(logprobs)

                torch.cuda.empty_cache()

            return torch.cat(logprobs_list, dim=0)

        
        prompt_ids = response_dict["prompt_ids"]
        prompt_mask = response_dict["prompt_mask"]
        completion_ids = response_dict["completion_ids"]
        completion_mask = response_dict["completion_mask"]
        
        batch_size = prompt_ids.shape[0] // n # assume that there are only two responses per prompt
        logprobs = batched_forward_logprobs(model, self.trainer, prompt_ids, prompt_mask, completion_ids, completion_mask, mini_bach_size)

        # logprobs = self.trainer._forward(model, prompt_ids, prompt_mask, completion_ids, completion_mask)
        with torch.inference_mode():
            if self.trainer.ref_model is not None:
                ref_logprobs = batched_forward_logprobs(self.trainer.ref_model, self.trainer, prompt_ids, prompt_mask, completion_ids, completion_mask, mini_bach_size)
                # ref_logprobs = self.trainer._forward(self.trainer.ref_model, prompt_ids, prompt_mask, completion_ids, completion_mask)
            else:  # peft case: we just need to disable the adapter
                with self.trainer.model.disable_adapter():
                    ref_logprobs = batched_forward_logprobs(self.trainer.model, self.trainer, prompt_ids, prompt_mask, completion_ids, completion_mask, mini_bach_size)
                    # ref_logprobs = self.trainer._forward(self.trainer.model, prompt_ids, prompt_mask, completion_ids, completion_mask)
            
        # note that the first half is a set of one response per prompt i.e. y_1 and the second half is a set of the other response per prompt i.e. y_2
        # cr_indices = torch.arange(batch_size * n) # simply [0, 1, 2,..., 15]
        cr_logprobs = logprobs
        cr_ref_logprobs = ref_logprobs
        
        # mask out the padding tokens
        padding_mask = ~completion_mask.bool()
        cr_padding_mask = padding_mask
 
        cr_logprobs_sum = (cr_logprobs * ~cr_padding_mask).sum(1)
        cr_ref_logprobs_sum = (cr_ref_logprobs * ~cr_padding_mask).sum(1)
        
        chosen_logprobs_sum, rejected_logprobs_sum = torch.split(cr_logprobs_sum, batch_size)
        chosen_ref_logprobs_sum, rejected_ref_logprobs_sum = torch.split(cr_ref_logprobs_sum, batch_size)
        
        chosen_rewards = self.trainer.beta * (chosen_logprobs_sum - chosen_ref_logprobs_sum)
        rejected_rewards = self.trainer.beta * (rejected_logprobs_sum - rejected_ref_logprobs_sum)
        
        margins = torch.abs(chosen_rewards - rejected_rewards)
        
        if compute_log_probs:
            # reorder: (prompt 1, ..., prompt 1) n times, (prompt 2, ..., prompt 2) n times, ..., (prompt b, ..., prompt b) n times, 
            num_promts = prompt_ids.shape[0] // n
            num_half_resp = max(1, n // 2)
            reorder_indices = [[np.arange( i * num_half_resp, (i+1) * num_half_resp),
                                np.arange(num_promts * num_half_resp + i * num_half_resp, num_promts * num_half_resp + (i+1) * num_half_resp)] for i in range(num_promts)]
            reorder_indices = np.concatenate(np.concatenate(reorder_indices, axis=0), axis=0) 
            reorderd_logprobs_per_tokens = logprobs[reorder_indices]
            
            logprobs_per_response = reorderd_logprobs_per_tokens.sum(-1) # (num_prompts x n) x num_tokens -> (num_prompts x n,) 
            logprobs = logprobs_per_response.reshape(n, -1).mean(0) # (num_prompts x n,) -> n x num_prompts -> num_prompts
            return margins, logprobs             
        return margins  

    def compute_reward_margin(self, model: nn.Module, response_dict: dict[str, Union[torch.Tensor, Any],], n: int = 2, compute_abs=True, compute_log_probs=False) -> torch.Tensor:
        prompt_ids = response_dict["prompt_ids"]
        prompt_mask = response_dict["prompt_mask"]
        completion_ids = response_dict["completion_ids"]
        completion_mask = response_dict["completion_mask"]
        
        batch_size = prompt_ids.shape[0] // n # assume that there are only two responses per prompt
        
        logprobs = self.trainer._forward(model, prompt_ids, prompt_mask, completion_ids, completion_mask)
        with torch.no_grad():
            if self.trainer.ref_model is not None:
                ref_logprobs = self.trainer._forward(self.trainer.ref_model, prompt_ids, prompt_mask, completion_ids, completion_mask)
            else:  # peft case: we just need to disable the adapter
                with self.trainer.model.disable_adapter():
                    ref_logprobs = self.trainer._forward(self.trainer.model, prompt_ids, prompt_mask, completion_ids, completion_mask)
            
        # note that the first half is a set of one response per prompt i.e. y_1 and the second half is a set of the other response per prompt i.e. y_2
        # cr_indices = torch.arange(batch_size * n) # simply [0, 1, 2,..., 15]
        cr_logprobs = logprobs
        cr_ref_logprobs = ref_logprobs
        
        # mask out the padding tokens
        padding_mask = ~completion_mask.bool()
        cr_padding_mask = padding_mask
 
        cr_logprobs_sum = (cr_logprobs * ~cr_padding_mask).sum(1)
        cr_ref_logprobs_sum = (cr_ref_logprobs * ~cr_padding_mask).sum(1)
        
        chosen_logprobs_sum, rejected_logprobs_sum = torch.split(cr_logprobs_sum, batch_size)
        chosen_ref_logprobs_sum, rejected_ref_logprobs_sum = torch.split(cr_ref_logprobs_sum, batch_size)
        
        chosen_rewards = self.trainer.beta * (chosen_logprobs_sum - chosen_ref_logprobs_sum)
        rejected_rewards = self.trainer.beta * (rejected_logprobs_sum - rejected_ref_logprobs_sum)
        
        margins = chosen_rewards - rejected_rewards
        if compute_abs:
            margins = torch.abs(margins)
        
        if compute_log_probs:
            # reorder: (prompt 1, ..., prompt 1) n times, (prompt 2, ..., prompt 2) n times, ..., (prompt b, ..., prompt b) n times, 
            num_promts = prompt_ids.shape[0] // n
            num_half_resp = max(1, n // 2)
            reorder_indices = [[np.arange( i * num_half_resp, (i+1) * num_half_resp),
                                np.arange(num_promts * num_half_resp + i * num_half_resp, num_promts * num_half_resp + (i+1) * num_half_resp)] for i in range(num_promts)]
            reorder_indices = np.concatenate(np.concatenate(reorder_indices, axis=0), axis=0) 
            reorderd_logprobs_per_tokens = logprobs[reorder_indices]
            
            logprobs_per_response = reorderd_logprobs_per_tokens.sum(-1) # (num_prompts x n) x num_tokens -> (num_prompts x n,) 
            logprobs = logprobs_per_response.reshape(n, -1).mean(0) # (num_prompts x n,) -> n x num_prompts -> num_prompts
            return margins, logprobs             
        return margins      
        
    def update_data(self, output_dict, merge_indices=False):
        if not output_dict: return
        selected_indices = output_dict['selected_indices']
        prompts = output_dict['prompt']
        completion_ids = output_dict['completion_ids'].cpu()
        completion_mask = output_dict['completion_mask'].cpu()
        
        if hasattr(selected_indices, "cpu"):
            selected_indices = selected_indices.cpu().numpy().tolist()

        # create index_to_responses map: response pairs are in the form of [{'role': 'assistant', 'content': '...'}, {'role': 'assistant', 'content': '...'}]
        index_to_responses = {
            int(data_idx): {
                'prompt': prompts[i],
                'completion_ids': completion_ids[i],
                'completion_mask': completion_mask[i],}
            for i, data_idx in enumerate(selected_indices)
        }
        self.trainer.index_to_responses = index_to_responses
        self.trainer.replay_buffer.update(index_to_responses)

        if merge_indices:
            prev_labeled_indices = self.trainer.labeled_indices 
            updated_labeled_indices = np.union1d(prev_labeled_indices, selected_indices).astype(int)
        else:
            updated_labeled_indices = selected_indices

        self.trainer.labeled_indices = updated_labeled_indices
        self.trainer.prev_labeled_indices.add(selected_indices)
        
        self.trainer.active_iter = multi_epoch_iterator(
            self.trainer.get_active_dataloader(self.trainer.labeled_indices), self.active_args.updates_per_sample)
        
    def judge(self, response_dict):
        prompts = response_dict["prompt"]
        completions = response_dict["completions"]
        completion_ids = response_dict["completion_ids"]
        
        batch_size = len(prompts)
        contain_eos_token = torch.any(completion_ids == self.trainer.processing_class.eos_token_id, dim=-1)
        
        # The reward model may not have the same chat template or tokenizer as the model, so we need to use the
        # raw data (string), apply the chat template (if needed), and tokenize it with the reward processing class.
        prompts = 2 * prompts  # repeat the prompt: [prompt0, prompt1] -> [prompt0, prompt1, prompt0, prompt1]
        if is_conversational({"prompt": prompts[0]}):
            examples = [{"prompt": p, "completion": c} for p, c in zip(prompts, completions)]
            examples = [apply_chat_template(example, self.reward_processing_class) for example in examples]
            prompts = [example["prompt"] for example in examples]
            completions = [example["completion"] for example in examples]

        # Tokenize the prompts
        prompts_ids = self.trainer.reward_processing_class(
            prompts, padding=True, return_tensors="pt", padding_side="left"
        )["input_ids"].to(self.device)
        context_length = prompts_ids.shape[1]

        # Tokenize the completions
        completions_ids = self.trainer.reward_processing_class(
            completions, padding=True, return_tensors="pt", padding_side="right"
        )["input_ids"].to(self.device)

        # Concatenate the prompts and completions and get the reward
        prompt_completion_ids = torch.cat((prompts_ids, completions_ids), dim=1)
        with torch.inference_mode():
            if isinstance(self.trainer.reward_model, DebertaV2ForSequenceClassification):
                inputs = self.trainer.reward_processing_class(prompts, completions, padding=True, return_tensors='pt')
                scores = self.trainer.reward_model(inputs['input_ids'].to(self.device), inputs['attention_mask'].to(self.device)).logits.reshape(-1)
            else:
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
        return mask.int()
        
    def get_labeled_embeddings(self, dataloader):
        total_labeled_embeddings = []
        for inputs in dataloader:
            indices = inputs["index"]
            
            prompt = []
            unordered_completion_ids = []
            missing_indices = []
            for i, idx in enumerate(indices):
                if idx not in self.trainer.embedding_buffer:
                    prompt.append(inputs["prompt"][i])
                    unordered_completion_ids.append(self.trainer.replay_buffer[idx]["completion_ids"])
                    missing_indices.append(idx)
                    
            if unordered_completion_ids:
                unordered_completion_ids = torch.stack(unordered_completion_ids, dim=0)
                completion_ids = torch.cat([unordered_completion_ids[:,0], unordered_completion_ids[:,1]])
    
                completions = self.trainer.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
                if is_conversational({"prompt": inputs['prompt'][0]}):
                    completions = [[{"role": "assistant", "content": completion}] for completion in completions]
                
                response_dict = {
                    'prompt': prompt,
                    'completions': completions
                }
                with torch.inference_mode():
                    missing_embeddings = self.get_embeddings(
                        response_dict, batch_size=len(prompt), extractor_type=self.extractor_type)
            
            missing_count = 0
            gathered_embedings = []
            for i, idx in enumerate(indices):
                if idx in missing_indices:
                    gathered_embedings.append(missing_embeddings[missing_count])
                    self.trainer.embedding_buffer[idx] = missing_embeddings[missing_count].cpu()
                    missing_count += 1
                else:
                    gathered_embedings.append(self.trainer.embedding_buffer[idx])
                    
            labeled_embeddings = torch.stack([embed.cpu() for embed in gather_object(gathered_embedings)], dim=0)
            total_labeled_embeddings.append(labeled_embeddings)
        
        total_labeled_embeddings = torch.cat(total_labeled_embeddings, dim=0)
        return total_labeled_embeddings
    
    
    def get_unlabeled_embeddings(self, dataloader, model, compute_log_probs=False):
        accelerator = self.trainer.accelerator
        # obtain embeddings for unlabeled data points
        total_prompts_local = []
        total_completion_ids_local = []
        total_completion_mask_local = []
        embeddings_local = []
        entropies_local = []
        for i, inputs in enumerate(dataloader):
            batch_size = len(inputs['prompt'])
            with torch.inference_mode():
                response_dict = self.generate_responses(
                    model, inputs, n=2, decode_completion=True) # estimation for entropy per prompt (refer to APL); shape: B
                embeddings = self.get_embeddings(
                    response_dict, batch_size=batch_size, extractor_type=self.extractor_type)
                if compute_log_probs:
                    mean_logprobs = self.compute_log_probs(model, response_dict, n=2)
            # accumulate locally
            total_prompts_local += inputs['prompt']
            total_completion_ids_local.append(response_dict['completion_ids'].detach().cpu())
            total_completion_mask_local.append(response_dict['completion_mask'].detach().cpu())
            embeddings_local.append(embeddings.detach().cpu())
            if compute_log_probs:
                entropies_local.append((-1.0 * mean_logprobs).detach().cpu())

        # Gather once across processes
        total_prompts = gather_object(total_prompts_local)
        # completions -> merge helper expects a list; we will return list
        from ..utils import gather_2d_once, gather_1d_once
        total_completion_ids_g, _ = gather_2d_once(total_completion_ids_local, accelerator, pad_token_id=self.trainer.processing_class.pad_token_id)
        total_completion_mask_g, _ = gather_2d_once(total_completion_mask_local, accelerator, pad_token_id=0)
        # embeddings are 2D of fixed width; reuse the same helper
        embeddings_g, _ = gather_2d_once(embeddings_local, accelerator, pad_token_id=0)
        if compute_log_probs:
            entropies_g, _ = gather_1d_once(entropies_local, accelerator, pad_index=float('-inf'))
            return (total_prompts, [total_completion_ids_g], [total_completion_mask_g], embeddings_g, entropies_g)
        return (total_prompts, [total_completion_ids_g], [total_completion_mask_g], embeddings_g)
    
    
    def _prepare_for_data_update(self, args_to_update, total_batch_size):
        final_topk_indices = args_to_update['topk_indices']
        total_completion_ids = args_to_update['completion_ids']
        total_completion_mask = args_to_update['completion_mask']
        
        total_completion_ids1, total_completion_ids2 = torch.split(total_completion_ids, total_batch_size) # (B x 2) x D -> B x D, B x D
        total_completion_id_reordered = torch.stack([total_completion_ids1, total_completion_ids2]).transpose(1, 0) # B x 2 x D
        
        total_completion_mask1, total_completion_mask2 = torch.split(total_completion_mask, total_batch_size) # (B x 2) x D -> B x D, B x D
        total_completion_mask_reordered = torch.stack([total_completion_mask1, total_completion_mask2]).transpose(1, 0) # B x 2 x D
    
        final_selected_completion_ids = total_completion_id_reordered[final_topk_indices] # k x 2 x D
        final_selected_completion_mask = total_completion_mask_reordered[final_topk_indices] # k x 2 x D

        args_to_update['completion_ids'] = final_selected_completion_ids
        args_to_update['completion_mask'] = final_selected_completion_mask
            
        return args_to_update
    
    def query(self, n):
        pass
    
    def get_labeled_count(self):
        return len(self.trainer.labeled_indices)

    def load_extractor(self, extractor_type):
        if extractor_type not in EXTRACTORS:
            raise NotImplementedError(f'extractor type: {extractor_type} does not exist')
        
        extractor_name = EXTRACTORS[extractor_type]
        if extractor_name in self._extractor:
            model, tokenizer = self._extractor[extractor_name]
            return model.to(self.device).eval(), tokenizer
        
        if extractor_type in ["roberta", "modernbert"]:
            tokenizer = AutoTokenizer.from_pretrained(extractor_name)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, 
                bnb_4bit_compute_dtype=torch.bfloat16,  # Computation still happens in fp16/bf16
                bnb_4bit_use_double_quant=True,        # Optional compression trick
                bnb_4bit_quant_type="nf4"              # "nf4" is the default in QLoRA papers
            )   
            model = AutoModel.from_pretrained(
                extractor_name, quantization_config=bnb_config).eval()
        elif extractor_type in ["sentence_transformer"]:
            tokenizer = None
            model = SentenceTransformer(extractor_name).eval()
        elif extractor_type == "llm":
            tokenizer = self.trainer.processing_class
            model = self.trainer.model.eval()
        else:
            raise NotImplementedError(f'extractor type: {extractor_type} does not exist')    

        self._extractor[extractor_name] = (model.to(self.device), tokenizer)
        return model, tokenizer
        

    def get_embeddings(self, data, extractor_type: Optional[str] = None, batch_size: int = 64, pooling_strategy: str = "mean"):
        # model_name: str = "answerdotai/ModernBERT-large", 
                    #    batch_size: int = 64, concat_mode: bool = False, pooling_strategy: str = "mask"):
        """
        Function that loads different embedding extractors with backward compatibility.
        
        Args:
            data: Input data (list of prompts/texts or generated responses)
            extractor_type: "sentence_transformer", "llm", "difference_vectors", "separate_concat", or specific model name
                           If None, uses default based on embedding_input_type
            model_name: Model name for BERT-based extractions
            batch_size: Batch size for processing
            concat_mode: For difference vectors, whether to concatenate [z0, z1, z0-z1]
            pooling_strategy: "mask" or "mean" for BERT-based methods
        
        Returns:
            torch.Tensor: Embeddings of shape (batch_size, embedding_dim)
        """        
        # Step1. load extractor
        model, tokenizer = self.load_extractor(extractor_type)
        
        # Step2. process text
        preprocessed_texts = self._preprocess_input(data)
        
        # Step3. extract embeddings and pool
        embeddings = self._extract_embedings(preprocessed_texts, model, tokenizer, pooling_strategy)
        
        return embeddings
        

    def _preprocess_input(self, data, separator=", "):
        """
        Preprocess input data into plain text strings.
        
        Args:
            data: Various input formats
            separator: String to separate prompt and response when concatenating
        
        Returns:
            List[str]: List of text strings
        """
        prompts = data['prompt']
        num_prompts = len(prompts)
        
        exist_completion = 'completions' in data
        if self.embedding_input_type in ['concat', 'template']:
            if not exist_completion:
                raise ValueError(f'For {self.embedding_input_type}, completions should be provided. Make sure to call generate_responses with decode_completion=True')
            
            completions = data['completions']
            if len(completions) != 2 * num_prompts:
                raise ValueError(f"Expected {2 * num_prompts} completions but got {len(completions)}")
            
            completions1, completions2 = chunk_list(completions, 2)
               
        texts = []
        for i in range(num_prompts):
            # 1)  Prompt
            if isinstance(prompts[i], str):                       # plain string
                text = prompts[i]
            else:                                                 # conversational list
                text = prompts[i][0]["content"]

            # 2)  Handle concat / template if requested
            if self.embedding_input_type == "concat":
                if isinstance(completions1[i], str):
                    comp0 = completions1[i]
                    comp1 = completions2[i]
                else:  # list-of-dict conversational
                    comp0 = completions1[i][0]["content"]
                    comp1 = completions2[i][0]["content"]

                text += (
                    separator + "Response0:" + comp0
                    + separator + "Response1:" + comp1
                )

            elif self.embedding_input_type == "template":
                if isinstance(completions1[i], str):
                    comp0 = completions1[i]
                    comp1 = completions2[i]
                else:
                    comp0 = completions1[i][0]["content"]
                    comp1 = completions2[i][0]["content"]

                # FIX: Use dataset-specific templates
                dataset_type = self._detect_dataset_type(data)
                template = self.dataset_templates.get(dataset_type, self.dataset_templates['default'])
                text = template.format(prompt=text, response0=comp0, response1=comp1)

            texts.append(text)

        return texts
    
    def _extract_embedings(self, batch_texts, model, tokenizer, pooling_strategy):
        if tokenizer is None:
            inputs = batch_texts
            with torch.inference_mode():
                embeddings = model.encode(batch_texts, convert_to_tensor=True, device=self.device)
        else:
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.inference_mode():
                outputs = model(**inputs, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]
            attention_masks = inputs['attention_mask']
        
            if pooling_strategy == "mean":
                embeddings = (last_hidden_state * attention_masks[:, :, None]).sum(dim=1) 
                embeddings /= attention_masks[:, :, None].sum(dim=1)
            elif pooling_strategy == "last":
                if tokenizer.padding_side == 'left':
                    embeddings = last_hidden_state[:, -1, :]
                else:
                    embeddings = []
                    for i, attention_mask in enumerate(attention_masks):
                        # Find last non-padding token
                        last_token_idx = attention_mask.sum() - 1
                        embeddings.append(last_hidden_state[i, last_token_idx, :])
                    embeddings = torch.stack(embeddings)
            elif pooling_strategy == "cls":
                assert "cls_token" in tokenizer.special_tokens, "for cls pooling strategy, we need cls_token. Only Bert-type of models have cls_token"
                embeddings = last_hidden_state[:, 0, :]
                
        return embeddings
    

    # def get_embeddings(self, data, extractor_type: Optional[str] = None, model_name: str = "answerdotai/ModernBERT-large", 
    #                    batch_size: int = 64, concat_mode: bool = False, pooling_strategy: str = "mask"):
    #     """
    #     Function that loads different embedding extractors with backward compatibility.
        
    #     Args:
    #         data: Input data (list of prompts/texts or generated responses)
    #         extractor_type: "sentence_transformer", "llm", "difference_vectors", "separate_concat", or specific model name
    #                        If None, uses default based on embedding_input_type
    #         model_name: Model name for BERT-based extractions
    #         batch_size: Batch size for processing
    #         concat_mode: For difference vectors, whether to concatenate [z0, z1, z0-z1]
    #         pooling_strategy: "mask" or "mean" for BERT-based methods
        
    #     Returns:
    #         torch.Tensor: Embeddings of shape (batch_size, embedding_dim)
    #     """
    #     # Backward compatibility: map embedding_input_type to extractor_type if not specified
    #     if extractor_type is None:
    #         if self.embedding_input_type == "difference_vectors":
    #             extractor_type = "difference_vectors"
    #         elif self.embedding_input_type == "separate_concat":
    #             extractor_type = "separate_concat"
    #         else:
    #             extractor_type = "sentence_transformer"

    #     # Handle new embedding methods
    #     if extractor_type == "difference_vectors":
    #         return self._extract_difference_vectors(data, model_name, batch_size, concat_mode, pooling_strategy)
    #     elif extractor_type == "separate_concat":
    #         return self._extract_separate_concat(data, model_name, batch_size, pooling_strategy)
    #     else:
    #         # Existing methods - use the original preprocessing logic
    #         # print(f'Extract preprocessed texts')
    #         preprocessed_texts = self._preprocess_input(data)
            
    #         if extractor_type == "sentence_transformer":
    #             return self._extract_sentence_transformer_embeddings(preprocessed_texts)
    #         elif extractor_type == "llm":
    #             return self._extract_llm_embeddings(preprocessed_texts)
    #         else:
    #             # Assume it's a specific sentence transformer model name
    #             return self._extract_sentence_transformer_embeddings(preprocessed_texts, model_name=extractor_type)

    # def _extract_difference_vectors(self, data, model_name="answerdotai/ModernBERT-large", batch_size=16, 
    #                                concat_mode=False, pooling_strategy="mask"):
    #     """
    #     Extract difference vectors: z0 - z1 where:
    #     z0 = embed("Post: {post} Summary: {summary0} [MASK]")
    #     z1 = embed("Post: {post} Summary: {summary1} [MASK]")
        
    #     Returns:
    #         torch.Tensor: Shape (num_prompts, embedding_dim) or (num_prompts, 3*embedding_dim) if concat_mode=True
    #     """
    #     # print(f"  Extracting difference vectors with {pooling_strategy} pooling...")
        
    #     # Ensure we have completions data
    #     if 'completions' not in data:
    #         raise ValueError("Completions required for difference vectors. Make sure to call generate_responses with decode_completion=True")
        
    #     prompts = data['prompt']
    #     num_prompts = len(prompts)
        
    #     # FIX: Correct completion splitting - completions come as [resp0_prompt1, resp1_prompt1, resp0_prompt2, resp1_prompt2, ...]
    #     completions = data['completions']
    #     if len(completions) != 2 * num_prompts:
    #         raise ValueError(f"Expected {2 * num_prompts} completions but got {len(completions)}")
        
    #     # Correctly pair completions: every 2 consecutive completions belong to the same prompt
    #     completions1, completions2 = chunk_list(completions, 2)
        
        
    #     # Detect dataset type for appropriate template
    #     dataset_type = self._detect_dataset_type(data)
    #     template = self.dataset_templates['single']
        
    #     summary0_texts = []
    #     summary1_texts = []
        
    #     for i in range(num_prompts):
    #         # Get prompt text
    #         if isinstance(prompts[i], str):
    #             prompt_text = prompts[i]
    #         else:
    #             prompt_text = prompts[i][0]["content"]
            
    #         # Get completion texts
    #         if isinstance(completions1[i], str):
    #             comp0 = completions1[i]
    #             comp1 = completions2[i]
    #         else:
    #             comp0 = completions1[i][0]["content"]
    #             comp1 = completions2[i][0]["content"]
            
    #         # Create texts with context using templates
    #         if pooling_strategy == "mask":
    #             summary0_text = template.format(prompt=prompt_text, response0=comp0) + " [MASK]"
    #             summary1_text = template.format(prompt=prompt_text, response0=comp1) + " [MASK]"
    #         else:
    #             summary0_text = template.format(prompt=prompt_text, response0=comp0)
    #             summary1_text = template.format(prompt=prompt_text, response0=comp1)
            
    #         summary0_texts.append(summary0_text)
    #         summary1_texts.append(summary1_text)
        
    #     # Extract embeddings
    #     z0_embeddings = self._extract_bert_embeddings(summary0_texts, model_name, batch_size, pooling_strategy)
    #     z1_embeddings = self._extract_bert_embeddings(summary1_texts, model_name, batch_size, pooling_strategy)
        
    #     # Compute difference vectors
    #     diff_vectors = z0_embeddings - z1_embeddings

    #     if concat_mode:
    #         combined = torch.cat([z0_embeddings, z1_embeddings, diff_vectors], dim=1)
    #         # print(f"  Created combined features ({pooling_strategy}): {combined.shape}")
    #         return combined
    #     else:
    #         # print(f"  Created difference vectors ({pooling_strategy}): {diff_vectors.shape}")
    #         return diff_vectors

    # def _extract_separate_concat(self, data, model_name="answerdotai/ModernBERT-large", batch_size=16, pooling_strategy="mean"):
    #     """
    #     Extract separate embeddings for prompt, summary0, summary1 then concatenate:
    #     p = embed("Post: {post}")
    #     s0 = embed("Summary: {summary0}")
    #     s1 = embed("Summary: {summary1}")
    #     feat = concat([p, s0, s1])
        
    #     Returns:
    #         torch.Tensor: Shape (num_prompts, 3*embedding_dim)
    #     """
    #     print(f"  Extracting separate concatenated embeddings with {pooling_strategy} pooling...")
        
    #     # Ensure we have completions data
    #     if 'completions' not in data:
    #         raise ValueError("Completions required for separate concatenation. Make sure to call generate_responses with decode_completion=True")
        
    #     prompts = data['prompt']
    #     num_prompts = len(prompts)
        
        
    #     completions = data['completions']
    #     if len(completions) != 2 * num_prompts:
    #         raise ValueError(f"Expected {2 * num_prompts} completions but got {len(completions)}")
        
    #     completions1, completions2 = chunk_list(completions, 2)
        
    #     prompt_texts = []
    #     summary0_texts = []
    #     summary1_texts = []
        
    #     for i in range(num_prompts):
    #         # Get prompt text
    #         if isinstance(prompts[i], str):
    #             prompt_text = prompts[i]
    #         else:
    #             prompt_text = prompts[i][0]["content"]
            
    #         # Get completion texts
    #         if isinstance(completions1[i], str):
    #             comp0 = completions1[i]
    #             comp1 = completions2[i]
    #         else:
    #             comp0 = completions1[i][0]["content"]
    #             comp1 = completions2[i][0]["content"]
            
    #         # Create separate texts for each component
    #         prompt_texts.append(f"Post: {prompt_text}")
    #         summary0_texts.append(f"Summary: {comp0}")
    #         summary1_texts.append(f"Summary: {comp1}")
        
    #     # Extract embeddings for each component separately
    #     print("  - Embedding prompts...")
    #     prompt_embeddings = self._extract_bert_embeddings(prompt_texts, model_name, batch_size, pooling_strategy)
        
    #     print("  - Embedding summary 0...")
    #     summary0_embeddings = self._extract_bert_embeddings(summary0_texts, model_name, batch_size, pooling_strategy)
        
    #     print("  - Embedding summary 1...")
    #     summary1_embeddings = self._extract_bert_embeddings(summary1_texts, model_name, batch_size, pooling_strategy)
        
    #     # Concatenate all three embeddings
    #     concatenated_embeddings = torch.cat([prompt_embeddings, summary0_embeddings, summary1_embeddings], dim=1)
        
    #     print(f"  Created concatenated embeddings ({pooling_strategy}): {concatenated_embeddings.shape}")
    #     print(f"    - Prompt embeddings: {prompt_embeddings.shape}")
    #     print(f"    - Summary0 embeddings: {summary0_embeddings.shape}")
    #     print(f"    - Summary1 embeddings: {summary1_embeddings.shape}")
        
    #     return concatenated_embeddings

    # def _extract_bert_embeddings(self, texts, model_name="answerdotai/ModernBERT-large", batch_size=16, pooling_strategy="mask"):
    #     """
    #     Extract embeddings from BERT-based models (e.g., ModernBERT) with specified pooling strategy.
        
    #     Returns:
    #         torch.Tensor: Shape (len(texts), embedding_dim) on self.device
    #     """
    #     try:
    #         from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM
    #     except ImportError:
    #         raise ImportError("transformers not installed. Install with: pip install transformers")
        
    #     device = self.device
        
    #     # Initialize model if not cached
    #     if model_name not in self._bert_models:
    #         print(f"Loading BERT model: {model_name}")
    #         tokenizer = AutoTokenizer.from_pretrained(model_name)
    #         from transformers import BitsAndBytesConfig
    #         bnb_config = BitsAndBytesConfig(
    #             load_in_4bit=True, 
    #             bnb_4bit_compute_dtype=torch.bfloat16,  # Computation still happens in fp16/bf16
    #             bnb_4bit_use_double_quant=True,        # Optional compression trick
    #             bnb_4bit_quant_type="nf4"              # "nf4" is the default in QLoRA papers
    #         )   
    #         if pooling_strategy == "mask":
    #             model = AutoModelForMaskedLM.from_pretrained(
    #                 model_name, quantization_config=bnb_config).to(device).eval()
    #         else:
    #             model = AutoModel.from_pretrained(
    #                 model_name, quantization_config=bnb_config).to(device).eval()

    #         self._bert_models[model_name] = {
    #             'tokenizer': tokenizer,
    #             'model': model
    #         }
        
    #     tokenizer = self._bert_models[model_name]['tokenizer']
    #     model = self._bert_models[model_name]['model']
        
    #     all_embeddings = []
        
    #     for i in range(0, len(texts), batch_size):
    #         batch_texts = texts[i:i+batch_size]
    #         if torch.cuda.is_available():
    #             torch.cuda.empty_cache()
            
    #         # Tokenize the batch
    #         inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    #         inputs = {k: v.to(device) for k, v in inputs.items()}
            
    #         with torch.no_grad():
    #             outputs = model(**inputs, output_hidden_states=True)
            
    #         if pooling_strategy == "mask":
    #             # Extract [MASK] token embeddings
    #             mask_embeddings = []
    #             for j, input_ids in enumerate(inputs['input_ids']):
    #                 mask_token_id = tokenizer.mask_token_id
    #                 mask_positions = (input_ids == mask_token_id).nonzero(as_tuple=True)[0]
                    
    #                 if len(mask_positions) > 0:
    #                     mask_pos = mask_positions[0]
    #                     mask_embedding = outputs.hidden_states[-1][j, mask_pos, :]
    #                     mask_embeddings.append(mask_embedding)
    #                 else:
    #                     print(f"Warning: No [MASK] token found in text: {batch_texts[j][:100]}...")
    #                     # Fallback to CLS token
    #                     mask_embeddings.append(outputs.hidden_states[-1][j, 0, :])
                
    #             batch_embeddings = torch.stack(mask_embeddings)
                
    #         elif pooling_strategy == "mean":
    #             # Average pooling over all tokens (excluding padding)
    #             attention_mask = inputs['attention_mask']
    #             hidden_states = outputs.hidden_states[-1]  # Last layer
                
    #             # Mask out padding tokens for proper averaging
    #             masked_hidden = hidden_states * attention_mask.unsqueeze(-1)
                
    #             # Sum over sequence length and divide by actual length (excluding padding)
    #             sum_embeddings = masked_hidden.sum(dim=1)
    #             lengths = attention_mask.sum(dim=1, keepdim=True).float()
    #             batch_embeddings = sum_embeddings / lengths
                
    #         else:
    #             raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")
            
            
    #         all_embeddings.append(batch_embeddings.to(device))
    #         # print(f"  Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size} (pooling: {pooling_strategy})")
        
    #     final_embeddings = torch.cat(all_embeddings, dim=0)
    #     return final_embeddings

    # def _preprocess_input(self, data, separator=", "):
    #     """
    #     Preprocess input data into plain text strings.
        
    #     Args:
    #         data: Various input formats
    #         separator: String to separate prompt and response when concatenating
        
    #     Returns:
    #         List[str]: List of text strings
    #     """
    #     prompts = data['prompt']
    #     num_prompts = len(prompts)
        
        exist_completion = 'completions' in data
        if self.embedding_input_type in ['concat', 'template', 'prefill', 'comparison']:
            if not exist_completion:
                raise ValueError(f'For {self.embedding_input_type}, completions should be provided. Make sure to call generate_responses with decode_completion=True')
            
    #         completions = data['completions']
    #         if len(completions) != 2 * num_prompts:
    #             raise ValueError(f"Expected {2 * num_prompts} completions but got {len(completions)}")
            
    #         completions1, completions2 = chunk_list(completions, 2)
               
    #     texts = []
    #     for i in range(num_prompts):
    #         # -------------------------------------------------------------
    #         # 1)  Prompt
    #         if isinstance(prompts[i], str):                       # plain string
    #             text = prompts[i]
    #         else:                                                 # conversational list
    #             text = prompts[i][0]["content"]

    #         # -------------------------------------------------------------
    #         # 2)  Handle concat / template if requested
    #         if self.embedding_input_type == "concat":
    #             if isinstance(completions1[i], str):
    #                 comp0 = completions1[i]
    #                 comp1 = completions2[i]
    #             else:  # list-of-dict conversational
    #                 comp0 = completions1[i][0]["content"]
    #                 comp1 = completions2[i][0]["content"]

    #             text += (
    #                 separator + "Response0:" + comp0
    #                 + separator + "Response1:" + comp1
    #             )
                text += (
                    separator + "Response0:" + comp0
                    + separator + "Response1:" + comp1
                )
                texts.append(text)

    #         elif self.embedding_input_type == "template":
    #             if isinstance(completions1[i], str):
    #                 comp0 = completions1[i]
    #                 comp1 = completions2[i]
    #             else:
    #                 comp0 = completions1[i][0]["content"]
    #                 comp1 = completions2[i][0]["content"]

                # FIX: Use dataset-specific templates
                dataset_type = self._detect_dataset_type(data)
                template = self.dataset_templates.get(dataset_type, self.dataset_templates['default'])
                text = template.format(prompt=text, response0=comp0, response1=comp1)
                texts.append(text)

            elif self.embedding_input_type == "prefill":
                if isinstance(completions1[i], str):
                    comp0 = completions1[i]
                    comp1 = completions2[i]
                else:
                    comp0 = completions1[i][0]["content"]
                    comp1 = completions2[i][0]["content"]

                # FIX: Use dataset-specific templates
                dataset_type = self._detect_dataset_type(data)
                template_a = self.dataset_templates.get(dataset_type, self.dataset_templates['prefill_a'])
                template_b = self.dataset_templates.get(dataset_type, self.dataset_templates['prefill_b'])
                text_a = template_a.format(prompt=text, response0=comp0, response1=comp1)
                text_b = template_b.format(prompt=text, response0=comp0, response1=comp1)
                texts.append(text_a)
                texts.append(text_b)

            elif self.embedding_input_type == "comparison":
                if isinstance(completions1[i], str):
                    comp0 = completions1[i]
                    comp1 = completions2[i]
                else:
                    comp0 = completions1[i][0]["content"]
                    comp1 = completions2[i][0]["content"]

                # FIX: Use single template for comparison
                template = self.dataset_templates['single']
                text_a = template.format(prompt=text, response0=comp0)
                text_b = template.format(prompt=text, response0=comp1)
                texts.append(text_a)
                texts.append(text_b)
            
            else:
                # Default case: prompt only
                texts.append(text)

        # Print one example AFTER embedding preprocessing
        # print("\n" + "="*50)
        # print("EXAMPLE AFTER EMBEDDING PREPROCESSING:")
        # print("="*50)
        # print(f"Number of processed texts: {len(texts)}")
        # if len(texts) > 0:
        #     print(f"First processed text: {str(texts[0])}")
        # print("="*50 + "\n")

        return texts
      
    def _extract_response_text(self, item):
        """
        Extract response/completion text from various formats.
        
    #     Args:
    #         item: Dictionary that might contain response data
            
    #     Returns:
    #         str or None: Extracted response text
    #     """
    #     response_fields = ["completion", "response", "generated_text", "completions", "output"]
        
    #     for field in response_fields:
    #         if field in item:
    #             response = item[field]
                
    #             if isinstance(response, list) and len(response) > 0:
    #                 # Handle nested list (from generate_responses conversational)
    #                 if isinstance(response[0], list) and len(response[0]) > 0:
    #                     if isinstance(response[0][0], dict) and "content" in response[0][0]:
    #                         return response[0][0]["content"]
                    
    #                 # Handle direct list of dicts
    #                 elif isinstance(response[0], dict) and "content" in response[0]:
    #                     return response[0]["content"]
                    
    #                 # Handle simple list of strings
    #                 else:
    #                     return str(response[0])
                
    #             elif isinstance(response, str):
    #                 return response
                
    #             elif field == "completion_ids" and hasattr(self.trainer, 'processing_class'):
    #                 try:
    #                     return self.trainer.processing_class.decode(response, skip_special_tokens=True)
    #                 except:
    #                     return str(response)
        
    #     return None

    # def _extract_sentence_transformer_embeddings(self, texts, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    #     """
    #     Extract embeddings using sentence transformers from preprocessed text.
        
    #     Args:
    #         texts: List of preprocessed text strings
    #         model_name: Sentence transformer model name
        
    #     Returns:
    #         torch.Tensor: Embeddings
    #     """
    #     try:
    #         from sentence_transformers import SentenceTransformer
    #     except ImportError:
    #         raise ImportError("sentence-transformers not installed. Install with: pip install sentence-transformers")
        
    #     # Initialize model if not cached
    #     if not hasattr(self, '_st_models'):
    #         self._st_models = {}
        
    #     if model_name not in self._st_models:
    #         self._st_models[model_name] = SentenceTransformer(model_name)
    #         if torch.cuda.is_available():
    #             self._st_models[model_name] = self._st_models[model_name].to(self.device)
    
    #     model = self._st_models[model_name]
        
    #     # Extract embeddings
    #     with torch.no_grad():
    #         embeddings = model.encode(texts, convert_to_tensor=True, device=self.device)
        
    #     return embeddings

    def _extract_llm_embeddings(self, texts, concat_mode=False):
        """
        Extract last token embeddings from the current LLM using preprocessed text.
        
        Args:
            texts: List of preprocessed text strings
        
        Returns:
            torch.Tensor: Embeddings
        """
        model = self.trainer.model
        model.eval()
        tokenizer = self.trainer.processing_class
        
        # Tokenize the preprocessed texts
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=True)
            
            # Extract last token embeddings
            hidden_states = outputs.hidden_states[-1]  # Last layer
            
            # For each sequence, get the embedding of the last non-padding token
            embeddings = []
            for i, attention_mask in enumerate(inputs["attention_mask"]):
                # Find last non-padding token
                last_token_idx = attention_mask.sum() - 1
                embeddings.append(hidden_states[i, last_token_idx])
            
            embeddings = torch.stack(embeddings)
        
        if self.embedding_input_type in ("prefill", "comparison"):
            z0 = embeddings[::2]  # Every 2nd element starting from 0
            z1 = embeddings[1::2]
            embeddings = z0 - z1  # Difference vectors
        
        return embeddings

def infinite_epoch_iterator(dataloader):
    epoch = 0
    while True:
        for batch in dataloader:
            yield epoch, batch
        epoch += 1

def multi_epoch_iterator(dataloader, num_epochs):
    for epoch in range(num_epochs):
        for batch in dataloader:
            yield batch
