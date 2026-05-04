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

import concurrent.futures
import logging
from abc import ABC, abstractmethod
from typing import Optional, Union, Any, Dict

import numpy as np
from accelerate import Accelerator
from huggingface_hub import InferenceClient
from transformers.utils import is_openai_available

from ..import_utils import is_llm_blender_available
from ..data_utils import apply_chat_template, is_conversational

import atexit
import torch
import torch.distributed.rpc as rpc
import torch.distributed as dist
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, DebertaV2ForSequenceClassification
from trl.scripts.utils import ScriptArguments
import time
from multiprocessing import Manager, Lock, Process, set_start_method
import gc

from .utils import get_reward, compute_reward_scores

if is_llm_blender_available():
    import llm_blender 

if is_openai_available():
    from openai import OpenAI


DEFAULT_PAIRWISE_SYSTEM_PROMPT = '''I require a leaderboard for various large language models. I'll provide you with prompts given to these models and their corresponding outputs. Your task is to assess these responses, and select the model that produces the best output from a human perspective.

## Instruction

{{
    "instruction": """{prompt}""",
}}

## Model Outputs

Here are the unordered outputs from the models. Each output is associated with a specific model, identified by a unique model identifier.

{{
    {{
        "model_identifier": "0",
        "output": """{response0}"""
    }},
    {{
        "model_identifier": "1",
        "output": """{response1}"""
    }}
}}

## Task

Evaluate the models on the basis of the quality and relevance of their results, and select the model that generated the best result. Reply with the identifier of the best model. Our evaluation will only take into account the first character of your answer, so make sure it contains only one of the identifiers and nothing else (no quotation marks, no spaces, no new lines, ...).
'''


DEFAULT_PAIRWISE_SYSTEM_PROMPT_FOR_LOCAL = '''
## Instruction

{{
    "instruction": """{prompt}""",
}}

## Model Outputs

Here are the unordered outputs from the models. Each output is associated with a specific model, identified by a unique model identifier.

{{
    {{
        "model_identifier": "0",
        "output": """{response0}"""
    }},
    {{
        "model_identifier": "1",
        "output": """{response1}"""
    }}
}}

## Task

Evaluate the models on the basis of the quality and relevance of their results, and select the model that generated the best result. Reply with only one character: `0` if the first output is better, `1` if the second is better. Just write **0** or **1**. No words, no punctuation, no explanation.
'''

class BaseJudge(ABC):
    """
    Base class for judges. The subclasses of this class should implement the `judge` method.
    """

    @abstractmethod
    def judge(self, prompts: list[str], completions: list[str], shuffle_order: bool = True) -> list:
        raise NotImplementedError("Judge subclasses must implement the `judge` method.")


class BaseRankJudge(ABC):
    """
    Base class for LLM ranking judges.

    **Example**:
    ```python
    class MyRankJudge(BaseRankJudge):
        def judge(self, prompts, completions, shuffle_order=True):
            return ...  # Your ranking logic here

    judge = MyRankJudge()
    judge.judge(
        prompts=["The capital of France is", "The capital of Germany is"],
        completions=[[" Paris", " Marseille", "Lyon"], [" Munich", " Berlin"]]
    )  # [[0, 1, 2], [1, 0]]
    ```
    """

    @abstractmethod
    def judge(self, prompts: list[str], completions: list[list[str]], shuffle_order: bool = True) -> list[list[int]]:
        """
        Judge the completion for the given prompts and return the ranks of each completion.

        Args:
            prompts (`list[str]`):
                List of prompts.
            completions (`list[list[str]]`):
                List of completions list, where each element is a list of completions for the corresponding prompt.
            shuffle_order (`bool`, *optional*, defaults to `True`):
                Whether to shuffle the order of the completions to avoid positional bias.

        Returns:
            `list[list[int]]`:
                List of lists of idxs, where each list contains the ranks of the completions for the corresponding
                prompt. E.g., `[1, 2, 0]` means that the second completion (`idx=1`) is the best, followed by the
                third, and then the first.
        """
        raise NotImplementedError("Judge subclasses must implement the `judge` method.")


class BasePairwiseJudge(BaseJudge):
    """
    Base class for pairwise judges.
    """

    @abstractmethod
    def judge(self, prompts: list[str], completions: list[list[str]], shuffle_order: bool = True) -> list[int]:
        """
        Judge the completion pairs for the given prompts.

        Args:
            prompts (`list[str]`):
                List of prompts.
            completions (`list[list[str]]`):
                List of completions pairs, where each element is a pair of completions for the corresponding prompt.
            shuffle_order (`bool`, *optional*, defaults to `True`):
                Whether to shuffle the order of the completions to avoid positional bias.

        Returns:
            `list[int]`:
                List of idxs, where each idx is the rank of the best completion for the corresponding prompt.
                E.g., `1` means that the second completion (`idx=1`) is the best.

        Note:
            If the judge returns `-1` for any prompt, it indicates that the inner process used to compute the
            preference has failed. For instance, this could occur if the underlying language model returned an invalid
            answer. In such cases, the caller should handle these invalid indices appropriately, possibly by
            implementing fallback logic or error handling.
        """
        raise NotImplementedError("Judge subclasses must implement the `judge` method.")


class BaseBinaryJudge(BaseJudge):
    """
    Base class for binary judges.
    """

    @abstractmethod
    def judge(
        self,
        prompts: list[str],
        completions: list[str],
        gold_completions: Optional[list[str]] = None,
        shuffle_order: bool = True,
    ) -> list[int]:
        """
        Judge the completion for a given prompt. Used to assess if a completion satisfies a constraint.

        This base class should be used to implement binary evaluations as done in section 4.1.4 of the
        [CGPO paper](https://huggingface.co/papers/2409.20370).
        It is relevant for assessing whether a prompt completion pair satisfies a specific contraint.

        Args:
            prompts (`list[str]`): List of prompts.
            completions (`list[str]`): List of completions.
            gold_completions (`list[str]`, `optional`): List of gold completions if it exists.
            shuffle_order (`bool`): Whether to shuffle the order of the completions to avoid positional bias.

        Returns:
            list[int]: A list of binary labels:
                - 1 indicates that the completion satisfies the evaluated constraint.
                - 0 indicates that the completion does not satisfy the evaluated constraint.

        Note:
            If the judge returns -1 for any prompt, it indicates that the inner process used to compute the preference has failed.
            For instance, this could occur if the underlying language model or rule based contraint returned an invalid answer.
            In such cases, the caller should handle these invalid indices appropriately, possibly by implementing fallback logic or error handling.
        """
        raise NotImplementedError("Judge subclasses must implement the `judge` method.")


class PairRMJudge(BasePairwiseJudge):
    """
    LLM judge based on the PairRM model from AllenAI.

    This judge uses the PairRM model to rank pairs of completions for given prompts. It's designed for pairwise
    comparison of language model outputs. The PairRM model is loaded using the llm-blender library and runs on the
    default Accelerator device.

    **Attributes**:

        blender (`llm_blender.Blender`):
            An instance of the Blender class from llm-blender.

    **Example**:
    ```python
    >>> pairrm_judge = PairRMJudge()
    >>> prompts = ["Translate 'hello' to French", "What's the capital of Japan?"]
    >>> completions = [["Bonjour", "Salut"], ["Kyoto", "Tokyo"]]
    >>> results = pairrm_judge.judge(prompts, completions)
    >>> print(results)  # [0, 1] (indicating the first completion is preferred for the first prompt and the second)
    ```

    <Tip>

    This class requires the llm-blender library to be installed. Install it with: `pip install llm-blender`.

    </Tip>
    """

    def __init__(self):
        if not is_llm_blender_available():
            raise ValueError("llm-blender is not installed. Please install it with `pip install llm-blender`.")
        self.blender = llm_blender.Blender()
        self.blender.loadranker("llm-blender/PairRM", device=Accelerator().device)

    def judge(
        self,
        prompts: list[str],
        completions: list[list[str]],
        shuffle_order: bool = True,
        return_scores: bool = False,
        temperature: float = 1.0,
    ) -> list[Union[int, float]]:
        """
        Judge the completion pairs for the given prompts using the PairRM model.

        Args:
            prompts (`list[str]`):
                List of prompts to judge.
            completions (`list[list[str]]`):
                List of completion pairs for each prompt.
            shuffle_order (`bool`, *optional*, defaults to `True`):
                Whether to shuffle the order of the completions to avoid positional bias.
            return_scores (`bool`, *optional*, defaults to `False`):
                If `True`, return probability scores of the first completion instead of ranks (i.e. a *soft-judge*).
            temperature (`float`, *optional*, defaults to `1.0`):
                Temperature for scaling logits if `return_scores` is True.

        Returns:
            `Union[list[int, float]]`:
                If `return_scores` is `False`, returns a list of ranks (`0` or `1`) for each prompt, indicating which
                completion is preferred.
                If `return_scores` is `True`, returns softmax probabilities for the first completion.

        Raises:
            `ValueError`:
                If the number of completions per prompt is not exactly 2.

        Note:
            Unlike llm-blender, ranks are 0-indexed (`0` means the first completion is preferred).
        """

        if len(completions[0]) != 2:
            raise ValueError("PairRM judge requires exactly 2 completions per prompt.")

        # Shuffle the order of the completions to avoid positional bias
        if shuffle_order:
            flip_mask = np.random.choice([True, False], size=len(prompts))
            completions = [pair[::-1] if flip else pair for flip, pair in zip(flip_mask, completions)]

        # Rank the completions
        ranks = self.blender.rank(prompts, completions, return_scores=return_scores, disable_tqdm=True)
        if not return_scores:
            ranks -= 1  # PairRM rank is 1-indexed, so we subtract 1 to make it 0-indexed
        else:
            # scale the logits by temperature
            ranks /= temperature

        # Flip back the ranks or scores to the original order if needed
        if shuffle_order:
            ranks[flip_mask] = ranks[flip_mask][:, ::-1]

        # Return the ranks or score probability
        if return_scores:
            logit_max = np.amax(ranks, axis=-1, keepdims=True)
            exp_logit_shifted = np.exp(ranks - logit_max)
            probs = exp_logit_shifted / np.sum(exp_logit_shifted, axis=-1, keepdims=True)
            return probs[:, 0].tolist()
        else:
            return ranks[:, 0].tolist()


class RMWrappedJudge(BasePairwiseJudge):
    
    def __init__(self, trainer):
        self.trainer = trainer
        self.device = self.trainer.accelerator.device
        
    def judge(
        self,
        prompts: list[str],
        completions: list[list[str]],
        shuffle_order: bool = True,
        return_scores: bool = False,
        temperature: float = 1.0,
    ):
        batch_step = self.trainer.args.per_device_eval_batch_size
        final_ranks = []
        for idx in range(0, len(prompts), batch_step):
            batch_prompts = prompts[idx : idx + batch_step]
            batch_completions = completions[idx : idx + batch_step]
            
            # Shuffle the order of the completions to avoid positional bias
            if shuffle_order:
                flip_mask = np.random.choice([True, False], size=len(batch_prompts))
                batch_completions = [pair[::-1] if flip else pair for flip, pair in zip(flip_mask, batch_completions)]
                
            flat_completions0 = []
            flat_completions1 = []
            for completion in batch_completions:
                flat_completions0.append(completion[0])
                flat_completions1.append(completion[1])
            flat_completions = flat_completions0 + flat_completions1
                
            batch_size = len(batch_prompts)
            batch_prompts = batch_prompts * 2
            # The reward model may not have the same chat template or tokenizer as the model, so we need to use the
            # raw data (string), apply the chat template (if needed), and tokenize it with the reward processing class.
            # prompts = 2 * prompts  # repeat the prompt: [prompt0, prompt1] -> [prompt0, prompt1, prompt0, prompt1]
            if is_conversational({"prompt": batch_prompts[0]}):
                examples = [{"prompt": p, "completion": c} for p, c in zip(batch_prompts, flat_completions)]
                examples = [apply_chat_template(example, self.trainer.reward_processing_class) for example in examples]
                batch_prompts = [example["prompt"] for example in examples]
                flat_completions = [example["completion"] for example in examples]

            reward_max_length = int(getattr(self.trainer.args, "reward_max_length", getattr(self.trainer, "max_length", 2048)))

            # Tokenize for the reward model.
            prev_trunc_side = getattr(self.trainer.reward_processing_class, "truncation_side", "right")
            try:
                self.trainer.reward_processing_class.truncation_side = "left"
                prompts_ids = self.trainer.reward_processing_class(
                    batch_prompts,
                    padding=True,
                    truncation=True,
                    max_length=reward_max_length,
                    return_tensors="pt",
                    padding_side="left",
                )["input_ids"].to(self.device)

                self.trainer.reward_processing_class.truncation_side = "right"
                completions_ids = self.trainer.reward_processing_class(
                    flat_completions,
                    padding=True,
                    truncation=True,
                    max_length=reward_max_length,
                    return_tensors="pt",
                    padding_side="right",
                )["input_ids"].to(self.device)
            finally:
                self.trainer.reward_processing_class.truncation_side = prev_trunc_side

            # Keep (prompt + completion) within the RM length budget.
            num_tokens_to_truncate = max(prompts_ids.size(1) + completions_ids.size(1) - reward_max_length, 0)
            if num_tokens_to_truncate > 0:
                prompts_ids = prompts_ids[:, num_tokens_to_truncate:]

            context_length = prompts_ids.shape[1]
            
            eos_token_id = self.trainer.processing_class.eos_token_id
            completion_id_list = [encoded for encoded in self.trainer.processing_class(flat_completions, add_special_tokens=False)['input_ids']]
            contain_eos_token = torch.tensor([eos_token_id in completion_ids for completion_ids in completion_id_list]).to(self.device)

            # Concatenate and score.
            prompt_completion_ids = torch.cat((prompts_ids, completions_ids), dim=1)
            with torch.inference_mode():
                scores = compute_reward_scores(
                    self.trainer.reward_model,
                    self.trainer.reward_processing_class,
                    prompts=batch_prompts,
                    completions=flat_completions,
                    device=self.device,
                    prompt_completion_ids=prompt_completion_ids,
                    context_length=context_length,
                    contain_eos_mask=contain_eos_token,
                    missing_eos_penalty=self.trainer.args.missing_eos_penalty,
                    reward_batch_size=max(1, int(getattr(self.trainer.args, "reward_batch_size", 2))),
                )

            # Split the scores in 2 (the prompts of the first half are the same as the second half)
            first_half, second_half = scores.split(batch_size)

            # when rank == 0, it means the first completion is the best
            # when rank == 1, it means the second completion is the best
            ranks = (first_half < second_half).int()

            if shuffle_order:
                ranks[flip_mask] = 1 - ranks[flip_mask]
            ranks = ranks.cpu().numpy().tolist()
            final_ranks += ranks
            
        return final_ranks
        


local_hf_model_registry = None
_local_registry = None
registry_lock = None

def cleanup():
    if dist.is_initialized():
        rank = dist.get_rank()
        dist.destroy_process_group()
    else:
        rank = -1
    if rpc._is_current_rpc_agent_set():
        if rank != 0:
            rpc.shutdown()

def ping():
    return "pong"

def get_rank(system_prompt, prompt, candidates):
    print("Running judge on prompt:", prompt)
    if not _local_registry:
        raise Exception('No model loaded in local registry.')
    
    _, model_dict = next(iter(_local_registry.items()))
    tokenizer = model_dict.get('tokenizer')
    device = model_dict.get('device')
    model = model_dict.get('model')
    input_text = system_prompt.format(prompt=prompt, response0=candidates[0], response1=candidates[1])
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    gen_token_ids = output[0][inputs['input_ids'].shape[1]:]
    generated_token = tokenizer.decode(gen_token_ids, skip_special_tokens=True)

    if generated_token in ["0", "1"]:
        return int(generated_token)
    else:
        logging.debug(f"Invalid response from the judge model: '{generated_token}'. Returning -1.")
        return -1

def load_model(model_name: str, device, cache_dir):
    if model_name in local_hf_model_registry:
        print(f"[{model_name}] already loaded.")
        return _local_registry[model_name]
    
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[{model_name}] loading...")
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True if os.environ['HF_HUB_OFFLINE'] == '1' else False).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True if os.environ['HF_HUB_OFFLINE'] == '1' else False)
    model_dict = {
        "model": model,
        "tokenizer": tokenizer,
        "device": device
    }
    print(f"[{model_name}] registered.")

    _local_registry[model_name] = model_dict
    local_hf_model_registry[model_name] = True

    return model_dict

class LocalHfPairwiseJudge(BasePairwiseJudge):    
    '''
        model_name : # Check if you need to change to [meta-llama/Meta-Llama-3-8B-Instruct] model
          - meta-llama/Llama-3.2-1B-Instruct
          - meta-llama/Llama-3.2-3B-Instruct
          - meta-llama/Meta-Llama-3-8B-Instruct
          - meta-llama/Llama-3.3-70B-Instruct    
    '''
    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3-8B", # "meta-llama/Llama-3.3-70B-Instruct", # Check if you need to change to [meta-llama/Meta-Llama-3-8B-Instruct] model
        script_args: Optional[ScriptArguments] = None,
        system_prompt: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: str = os.environ["HF_HUB_CACHE"]
    ):
        global local_hf_model_registry, _local_registry, registry_lock
        if registry_lock is None:
            set_start_method("spawn", force=True)
            _manager = Manager()
            local_hf_model_registry = _manager.dict()
            _local_registry = {}
            registry_lock = Lock()
            atexit.register(cleanup)

        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        self.is_rpc_client = not rank == 0
        options = rpc.TensorPipeRpcBackendOptions(init_method="env://")

        accelerator = Accelerator()
        if accelerator.is_main_process or not self.is_rpc_client:
            with registry_lock:
                if model_name not in local_hf_model_registry:
                    if script_args.judge_count > 0:
                        assert torch.cuda.is_available() and script_args.judge_index is not None, 'GPU Index for Judge is None.'

                        def get_least_used_gpu(target_indices):
                            least_used = None
                            min_mem = float('inf')
                            for idx in target_indices:
                                torch.cuda.set_device(idx)
                                torch.cuda.empty_cache()
                                mem = torch.cuda.memory_reserved(idx)
                                if mem < min_mem:
                                    min_mem = mem
                                    least_used = idx
                            return least_used

                        index = get_least_used_gpu(script_args.judge_index)
                        print(f"Judge model will use GPU: {index}")
                            
                        self.device = device or torch.device(f"cuda:{index}")
                        model_dict = load_model(model_name=model_name, cache_dir=cache_dir, device=self.device)
                        
                        rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size, rpc_backend_options=options)
                        rpc.functions.async_execution(ping)
                        rpc.functions.async_execution(get_rank)
                        print("[Rank 0] RPC server ready")
                    else:
                        device = device or accelerator.device
                        tokenizer = tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True if os.environ['HF_HUB_OFFLINE'] == '1' else False)
                        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True if os.environ['HF_HUB_OFFLINE'] == '1' else False)
                        model = accelerator.prepare(model)
                        model_dict = {
                            "model": model,
                            "tokenizer": tokenizer,
                            "device": device
                        }
                        _local_registry[model_name] = model_dict
                        local_hf_model_registry[model_name] = True
                
                model_dict = _local_registry.get(model_name)
                if model_dict is None:
                    raise RuntimeError(f"Model not found in _local_model_cache: {model_name}")
                self.tokenizer = model_dict.get('tokenizer')
                self.model = model_dict.get('model')
                self.device = model_dict.get('device')

                self.model.eval()
                print(f'[Rank{torch.distributed.get_rank()}] judge model eval - {model_name}')
        else:
            with registry_lock:
                if not rpc._is_current_rpc_agent_set():
                    rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size, rpc_backend_options=options)

                worker0_alive = False
                while not worker0_alive:
                    try:
                        response = rpc.rpc_sync("worker0", ping, args=(), timeout=2)
                        print(f"[Rank{torch.distributed.get_rank()}] worker0 judge model alive: {response}")
                        worker0_alive = True
                    except Exception as e:
                        print(f'[Rank{torch.distributed.get_rank()}] worker0 judge model wait - {model_name}')
                        time.sleep(1)  # 1초 대기 후 재시도
        
        self.system_prompt = system_prompt or DEFAULT_PAIRWISE_SYSTEM_PROMPT_FOR_LOCAL

    def judge(self, prompts: list[str], completions: list[list[str]], shuffle_order: bool = True) -> list[int]:
        if shuffle_order:
            flip_mask = np.random.choice([True, False], size=len(prompts))
            completions = [pair[::-1] if flip else pair for flip, pair in zip(flip_mask, completions)]
        else:
            flip_mask = [False] * len(prompts)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            if self.is_rpc_client:
                args_iter = [(prompt, candidates) for prompt, candidates in zip(prompts, completions)]
                ranks = list(executor.map(lambda args: rpc.rpc_sync("worker0", get_rank, args=(self.system_prompt, args[0], args[1])), args_iter))
            else:
                ranks = list(executor.map(lambda args: get_rank(self.system_prompt, args[0], args[1]), zip(prompts, completions)))

        # 복원 (if flipped)
        if shuffle_order:
            ranks = [ranks[i] if not flip else 1 - ranks[i] for i, flip in enumerate(flip_mask)]

        return ranks


class HfPairwiseJudge(BasePairwiseJudge):
    """
    Pairwise judge based on the Hugging Face API with chat completion.

    This judge is relevant for assessing the quality chat models, where the completion is a response to a given prompt.

    Args:
        model (`str`, *optional*, defaults to `"meta-llama/Meta-Llama-3-70B-Instruct"`):
            Model to use for the judge.
        token (`str`, *optional*):
            Hugging Face API token to use for the [`huggingface_hub.InferenceClient`].
        system_prompt (`str` or `None`, *optional*, defaults to `None`):
            The system prompt to be used for the judge. If not provided, a default prompt is used. Note that the system
            prompt should contain the following placeholders: `{prompt}`, `{response0}`, and `{response1}`. Also, the
            inference is called with `max_tokens=1`, consequently the system prompt should ask for a single token
            response.
    """

    def __init__(
        self,
        model="meta-llama/Meta-Llama-3-70B-Instruct",
        token: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ):
        self.client = InferenceClient(model=model, token=token)
        self.system_prompt = system_prompt or DEFAULT_PAIRWISE_SYSTEM_PROMPT

    def judge(self, prompts: list[str], completions: list[list[str]], shuffle_order: bool = True) -> list[int]:
        # Shuffle the order of the completions to avoid positional bias
        if shuffle_order:
            flip_mask = np.random.choice([True, False], size=len(prompts))
            completions = [pair[::-1] if flip else pair for flip, pair in zip(flip_mask, completions)]

        # Define a function to get the rank for a single prompt, will be called concurrently
        def get_rank(prompt, candidates):
            content = self.system_prompt.format(prompt=prompt, response0=candidates[0], response1=candidates[1])
            completion = self.client.chat_completion(messages=[{"role": "user", "content": content}], max_tokens=1)
            response = completion.choices[0].message.content
            if response in ["0", "1"]:
                return int(response)
            else:
                logging.debug(f"Invalid response from the judge model: '{response}'. Returning -1.")
                return -1

        # Call the completions concurrently
        with concurrent.futures.ThreadPoolExecutor() as executor:
            ranks = list(executor.map(get_rank, prompts, completions))

        # Flip back the ranks to the original order if needed
        if shuffle_order:
            ranks = [ranks[i] if not flip else 1 - ranks[i] for i, flip in enumerate(flip_mask)]

        # Return the ranks
        return ranks


class OpenAIPairwiseJudge(BasePairwiseJudge):
    """
    Judge based on the OpenAI API.

    This judge is relevant for assessing the quality chat models, where the completion is a response to a given prompt.

    Args:
        model (`str`, *optional*, defaults to `"gpt-4-turbo-preview"`):
            Model to use for the judge.
        system_prompt (`str` or `None`, *optional*, defaults to `None`):
            System prompt to be used for the judge. If not provided, a default prompt is used. Note that the system
            prompt should contain the following placeholders: `{prompt}`, `{response0}`, and `{response1}`. Also, the
            inference is called with `max_tokens=1`, consequently the system prompt should ask for a single token
            response.
        max_requests (`int` or `None`, *optional*, defaults to `1000`):
            Maximum number of requests to make to the OpenAI API. If set to `None`, there is no limit.
    """

    def __init__(
        self, model="gpt-4-turbo-preview", system_prompt: Optional[str] = None, max_requests: Union[int, None] = 1_000
    ):
        if not is_openai_available():
            raise ValueError("OpenAI client is not installed. Please install it with 'pip install openai'.")
        self.client = OpenAI()
        self.model = model
        self.system_prompt = system_prompt or DEFAULT_PAIRWISE_SYSTEM_PROMPT
        self.max_requests = max_requests
        self.num_requests = 0
        self._warned = False

    def judge(self, prompts: list[str], completions: list[list[str]], shuffle_order: bool = True) -> list[int]:
        # Check if the limit of requests is reached, if so, use random choice instead
        if self.max_requests is not None and self.num_requests >= self.max_requests:
            if not self._warned:  # Print the warning only once
                logging.warning(
                    f"Reached the maximum number of requests ({self.max_requests}). From now on, returning -1 instead. "
                    " To increase the limit, set `max_requests` to a higher value, or to `None` for no limit."
                )
                self._warned = True
            return [-1] * len(prompts)

        # Shuffle the order of the completions to avoid positional bias
        if shuffle_order:
            flip_mask = np.random.choice([True, False], size=len(prompts))
            completions = [pair[::-1] if flip else pair for flip, pair in zip(flip_mask, completions)]

        # Define a function to get the rank for a single prompt, will be called concurrently
        def get_rank(prompt, candidates):
            content = self.system_prompt.format(prompt=prompt, response0=candidates[0], response1=candidates[1])
            messages = [{"role": "user", "content": content}]
            completion = self.client.chat.completions.create(model=self.model, messages=messages, max_tokens=1)
            response = completion.choices[0].message.content
            if response in ["0", "1"]:
                return int(response)
            else:
                logging.debug(f"Invalid response from the judge model: '{response}'. Returning -1.")
                return -1

        # Call the completions concurrently
        with concurrent.futures.ThreadPoolExecutor() as executor:
            ranks = list(executor.map(get_rank, prompts, completions))

        # Flip back the ranks to the original order if needed
        if shuffle_order:
            ranks = [ranks[i] if not flip else 1 - ranks[i] for i, flip in enumerate(flip_mask)]

        # Update the number of requests
        self.num_requests += len(prompts)

        # Return the ranks
        return ranks


class AllTrueJudge(BaseBinaryJudge):
    """
    Unify the decision of multiple [`BaseBinaryJudge`] instances.

    Returns `1` only if all inner binary judges return `1`. If any judge returns `0`, it returns `0`.
    If any judge returns `-1`, indicating a failure in its process, this judge will also return `-1`.

    Implements the Mixture of Judges as described in the [CGPO paper](https://huggingface.co/papers/2409.20370).

    Args:
    judges (`list[BaseBinaryJudge]`): A list of [`BaseBinaryJudge`] instances whose decisions will be unified.
    """

    def __init__(self, judges: list[BaseBinaryJudge]):
        self.judges = judges

    def judge(
        self,
        prompts: list[str],
        completions: list[str],
        gold_completions: Optional[list[str]] = None,
        shuffle_order: bool = True,
    ) -> list[int]:
        all_binary_judgments = [
            judge.judge(prompts, completions, gold_completions, shuffle_order) for judge in self.judges
        ]
        output = []
        for binary_judgments in zip(*all_binary_judgments):
            # Check that all values are in {0, 1, -1}
            if any(binary_judgment not in {0, 1, -1} for binary_judgment in binary_judgments):
                raise ValueError(
                    f"Invalid binary judgment: {binary_judgments}, expected list of values in {{0, 1, -1}}."
                )

            # Unify the decision
            if -1 in binary_judgments:
                output.append(-1)
            elif all(binary_judgment == 1 for binary_judgment in binary_judgments):
                output.append(1)
            else:
                output.append(0)
        return output
