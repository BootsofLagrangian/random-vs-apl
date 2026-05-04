import torch
import numpy as np
import os
from sklearn.metrics import pairwise_distances
from scipy.stats import rv_discrete
from accelerate.utils import gather_object

from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerBase
from torch.distributed import all_gather_object, get_world_size, is_initialized

from .strategy import Strategy
from ..utils import (
    safe_gather, 
    merge_prompt_or_completion,
    broadcast_subset,
    gather_1d_once,
    gather_2d_once,
)


def pairwise_distances_batched(X, Y, batch_size=1024, metric='euclidean', n_jobs=-1):
    """
    Compute pairwise distances between two datasets in batches using scikit-learn.

    Args:
        X (np.ndarray): First dataset of shape (n_samples_X, n_features).
        Y (np.ndarray): Second dataset of shape (n_samples_Y, n_features).
        batch_size (int): Batch size for dividing the computation.
        metric (str): Distance metric (default: 'euclidean').
        n_jobs (int): Number of parallel jobs to run (default: -1 for all CPUs).

    Returns:
        np.ndarray: Pairwise distances of shape (n_samples_X, n_samples_Y).
    """
    n_samples_X = X.shape[0]
    distances = []
    for start_idx in range(0, n_samples_X, batch_size):
        end_idx = min(start_idx + batch_size, n_samples_X)
        batch_distances = pairwise_distances(X[start_idx:end_idx], Y, metric=metric, n_jobs=n_jobs)
        distances.append(batch_distances)
    return np.vstack(distances)


def init_centers(X, K, weight, gamma, batch_size=1024, metric='euclidean', n_jobs=-1):
    """
    K-means++ initialization using NumPy-based pairwise distance computation.

    Args:
        X (np.ndarray): Input data points of shape (n_samples, n_features).
        K (int): Number of clusters.
        weight (np.ndarray): Weights for the data points.
        gamma (float): Weight exponent.
        batch_size (int): Batch size for distance computation.
        metric (str): Distance metric (default: 'euclidean').
        n_jobs (int): Number of parallel jobs for distance computation.

    Returns:
        list: Indices of the selected centers.
    """
    embs = X
    ind = np.argmax(np.linalg.norm(embs, axis=1))
    mu = [embs[ind]]
    indsAll = [ind]
    centInds = [0.] * len(embs)
    cent = 0

    weight = np.array(weight)
    weight = weight / weight.sum() if weight.sum() > 0 else np.ones_like(weight) / len(weight)

    # print(weight)
    # print('#Samps\tTotal Distance')

    while len(mu) < K:
        if len(mu) == 1:
            D2 = pairwise_distances_batched(mu[-1].reshape(1, -1), embs, batch_size=batch_size, metric=metric, n_jobs=n_jobs).ravel()
        else:
            newD = pairwise_distances_batched(mu[-1].reshape(1, -1), embs, batch_size=batch_size, metric=metric, n_jobs=n_jobs).ravel()
            for i in range(len(embs)):
                if D2[i] > newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]

        # print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        if sum(D2) == 0.0:
            raise ValueError("Distance sum is zero, check the input data.")
        
        D2 = D2.ravel().astype(float)
        D2_w = D2 * (weight ** gamma)
        Ddist = (D2_w ** 2) / sum(D2_w ** 2)

        customDist = rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        while ind in indsAll:
            ind = customDist.rvs(size=1)[0]

        mu.append(embs[ind])
        indsAll.append(ind)
        cent += 1


    return indsAll


def init_centers_exist(X, mem, K, weight, gamma, batch_size=1024, metric='euclidean', n_jobs=-1):
    """
    K-means++ initialization using existing centers and NumPy-based distance computation.

    Args:
        X (np.ndarray): Input data points of shape (n_samples_X, n_features).
        mem (np.ndarray): Existing center points of shape (n_samples_mem, n_features).
        K (int): Number of clusters.
        weight (np.ndarray): Weights for the data points.
        gamma (float): Weight exponent.
        batch_size (int): Batch size for distance computation.
        metric (str): Distance metric (default: 'euclidean').
        n_jobs (int): Number of parallel jobs for distance computation.

    Returns:
        list: Indices of the selected centers.
    """
    embs = X
    mem = mem
    mu = []
    indsAll = []
    centInds = [0.] * len(embs)
    cent = 0

    weight = np.array(weight)
    weight = weight / weight.sum() if weight.sum() > 0 else np.ones_like(weight) / len(weight)

    # print(weight)
    # print('#Samps\tTotal Distance')


    while len(mu) < K:
        if len(mu) == 0:
            D2 = pairwise_distances_batched(mem, embs, batch_size=batch_size, metric=metric, n_jobs=n_jobs).min(0)
        else:
            newD = pairwise_distances_batched(mem, embs, batch_size=batch_size, metric=metric, n_jobs=n_jobs).min(0)
            for i in range(len(embs)):
                if D2[i] > newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]

        # print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        if sum(D2) == 0.0:
            raise ValueError("Distance sum is zero, check the input data.")

        D2 = D2.ravel().astype(float)
        D2_w = D2 * (weight ** gamma)
        Ddist = (D2_w ** 2) / sum(D2_w ** 2)

        customDist = rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        while ind in indsAll:
            ind = customDist.rvs(size=1)[0]


        mu.append(embs[ind])
        indsAll.append(ind)
        cent += 1

    return indsAll


import torch
import numpy as np
import os
from torch.distributed import all_gather_object
from accelerate.utils import gather_object

class DivReward(Strategy):
    def __init__(self, trainer, system_prompt, active_args):
        super(DivReward, self).__init__(trainer, system_prompt, active_args)
        self.tokenizer = trainer.tokenizer  
        self.device = trainer.accelerator.device
        self.rank = trainer.accelerator.process_index

        self.subset_factor = 16
        self.gamma = 0.5

    def make_response_dict_from_unlabeled(self, unlab_total_prompts, unlab_total_completion_ids, unlab_total_completion_mask):
        tokenizer: PreTrainedTokenizerBase = self.tokenizer
        device = self.trainer.accelerator.device

        # tokenize prompts
        max_prompt_length = getattr(tokenizer, "model_max_length", 1024)
        if not isinstance(max_prompt_length, int) or max_prompt_length > 4096:
            max_prompt_length = 1024

        tokenized_prompt = tokenizer(
            unlab_total_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_length
        )

        prompt_ids = tokenized_prompt["input_ids"]
        prompt_mask = tokenized_prompt["attention_mask"]

        valid_indices = torch.arange(prompt_ids.shape[0])

        completion_ids = torch.cat(unlab_total_completion_ids, dim=0)
        completion_mask = torch.cat(unlab_total_completion_mask, dim=0)

        assert completion_ids.shape[0] % prompt_ids.shape[0] == 0

        repeat_factor = completion_ids.shape[0] // prompt_ids.shape[0]
        prompt_ids = prompt_ids.repeat_interleave(repeat_factor, dim=0)
        prompt_mask = prompt_mask.repeat_interleave(repeat_factor, dim=0)

        # print(f'make_response_dict_from_unlabeled repeat_factor={repeat_factor}')
        # print(f'make_response_dict_from_unlabeled len(prompt_ids)={len(prompt_ids)}')
        # print(f'make_response_dict_from_unlabeled len(prompt_mask)={len(prompt_mask)}')
        # print(f'make_response_dict_from_unlabeled len(completion_ids)={len(completion_ids)}')
        # print(f'make_response_dict_from_unlabeled len(completion_mask)={len(completion_mask)}')

        response_dict = {
            "prompt_ids": prompt_ids.to(device),
            "prompt_mask": prompt_mask.to(device),
            "completion_ids": completion_ids.to(device),
            "completion_mask": completion_mask.to(device)
        }
        return response_dict, valid_indices


    def safe_all_gather(self, data, accelerator, max_retries=3, timeout_seconds=300):
        """Safe all_gather implementation - ensures data size consistency and timeout handling"""
        
        # Return data as-is for single process
        if accelerator.num_processes == 1:
            return data
        
        # Convert data to serializable format
        if isinstance(data, torch.Tensor):
            # For tensors, ensure all ranks have the same size
            max_size = data.numel()
            # Synchronize maximum size across all ranks
            max_size_tensor = torch.tensor([max_size], device=accelerator.device)
            gathered_max_sizes = [torch.zeros_like(max_size_tensor) for _ in range(accelerator.num_processes)]
            
            try:
                torch.distributed.all_gather(gathered_max_sizes, max_size_tensor, async_op=False)
                global_max_size = max([t.item() for t in gathered_max_sizes])
                
                # Pad current tensor to global_max_size
                if data.numel() < global_max_size:
                    padding_size = global_max_size - data.numel()
                    data = torch.cat([data.flatten(), torch.zeros(padding_size, device=data.device, dtype=data.dtype)])
                else:
                    data = data.flatten()[:global_max_size]
                
                # Now all ranks have tensors of the same size, safe to gather
                gathered_tensors = [torch.zeros_like(data) for _ in range(accelerator.num_processes)]
                torch.distributed.all_gather(gathered_tensors, data, async_op=False)
                
                return gathered_tensors
                
            except Exception as e:
                accelerator.print(f"Tensor all_gather failed: {e}, falling back to local data")
                return [data]
        
        # For general objects
        for attempt in range(max_retries):
            try:
                # Limit data size
                serialized_data = self._limit_data_size(data, max_size_mb=50)  # 50MB limit
                
                # Set timeout
                original_timeout = os.environ.get('TORCH_NCCL_BLOCKING_WAIT', '0')
                os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
                
                # Use accelerate's gather_object (more stable)
                gathered_data = gather_object(serialized_data)
                
                # Restore original setting
                os.environ['TORCH_NCCL_BLOCKING_WAIT'] = original_timeout
                
                return gathered_data
                
            except Exception as e:
                accelerator.print(f"All_gather attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    accelerator.print("All gather attempts failed, using local data only")
                    return [data] * accelerator.num_processes
                
                # Wait before next attempt
                import time
                time.sleep(1)
        
        return [data] * accelerator.num_processes

    def _limit_data_size(self, data, max_size_mb=50):
        """Limit data size to prevent memory overflow"""
        import pickle
        import sys
        
        max_size_bytes = max_size_mb * 1024 * 1024
        
        if isinstance(data, list):
            # For lists, check size and truncate if necessary
            try:
                current_size = sys.getsizeof(pickle.dumps(data))
                if current_size > max_size_bytes:
                    # Estimate individual item size
                    if len(data) > 0:
                        item_size = current_size // len(data)
                        max_items = max(1, max_size_bytes // item_size)
                        data = data[:max_items]
                        self.print_rank0(f"Data truncated to {max_items} items due to size limit")
            except:
                # If pickle fails, use simpler limit
                if len(data) > 1000:
                    data = data[:1000]
                    self.print_rank0("Data truncated to 1000 items")
        
        elif isinstance(data, str):
            if len(data) > max_size_bytes // 4:  # Rough string size estimate
                data = data[:max_size_bytes // 4]
                self.print_rank0("String data truncated due to size limit")
        
        return data

    def sync_distributed_state(self, accelerator):
        """Synchronize all ranks in distributed environment"""
        if accelerator.num_processes > 1:
            try:
                # Simple synchronization signal
                sync_tensor = torch.tensor([1.0], device=accelerator.device)
                torch.distributed.all_reduce(sync_tensor, op=torch.distributed.ReduceOp.SUM)
                accelerator.wait_for_everyone()
                return True
            except Exception as e:
                accelerator.print(f"Distributed sync failed: {e}")
                return False
        return True

    def query(self, n):
        accelerator = self.trainer.accelerator
        
        # Synchronize distributed state
        if not self.sync_distributed_state(accelerator):
            accelerator.print("Warning: Distributed sync failed, proceeding with caution")

        # Load labeled dataloader
        labeled_indices = np.array(list(self.trainer.replay_buffer.keys()))
        labeled_dataloader = self.trainer.get_indexed_dataloader(labeled_indices, self.active_args.optimal_batch_size)

        unlabeled_indices = self.trainer.extract_unlabeled_indices()
        unlabeled_indices = np.setdiff1d(unlabeled_indices, labeled_indices)
        accelerator.print(f'DivReward len(unlabeled_indices) = {len(unlabeled_indices)}')

        subset_size = min(int(n) * int(self.subset_factor), len(unlabeled_indices))
        assert subset_size >= n, f'DivReward unlabeled set size should be larger than query size n'
        per_device_bs = self.query_batch_size or self.active_args.optimal_batch_size or self.trainer.args.per_device_train_batch_size
        subset_unlabeled_indices = broadcast_subset(
            unlabeled_indices,
            desired_size=subset_size,
            accelerator=accelerator,
            per_device_bs=per_device_bs,
            k=1,
            seed=getattr(self.trainer.args, "seed", 42),
            enforce_multiple=False,
        ).cpu().numpy()

        unlabeled_dataloader = self.trainer.get_unlabeled_dataloader(subset_unlabeled_indices, per_device_bs)
        model = self.prepare_model(unlabeled_dataloader)

        unlab_total_prompts, unlab_total_completion_ids, unlab_total_completion_mask, unlab_embs = self.get_unlabeled_embeddings(unlabeled_dataloader, model)
        accelerator.print(f'DivReward len(unlab_total_prompts) = {len(unlab_total_prompts)}')

        unlab_response_dict, valid_indices = self.make_response_dict_from_unlabeled(unlab_total_prompts, unlab_total_completion_ids, unlab_total_completion_mask)
        unlab_rewards = self.batched_compute_reward_margin(model, unlab_response_dict)
        valid_unlab_embs = unlab_embs[valid_indices]

        lab_embs = None
        if len(labeled_dataloader) != 0:
            # Must run on all ranks since `get_labeled_embeddings` uses distributed gathers internally.
            lab_embs = self.get_labeled_embeddings(labeled_dataloader)

        selected_indices = None
        if accelerator.is_main_process:
            if len(labeled_dataloader) == 0:
                accelerator.print('DivReward no labeled data')
                selected_positions = init_centers(valid_unlab_embs, n, unlab_rewards.cpu(), self.gamma)
            else:
                accelerator.print('DivReward with labeled data')
                selected_positions = init_centers_exist(
                    valid_unlab_embs, lab_embs[: len(unlab_rewards)], n, unlab_rewards.cpu(), self.gamma
                )

            selected_positions = np.array(selected_positions, dtype=int)
            selected_indices = subset_unlabeled_indices[selected_positions]
            selected_indices = np.setdiff1d(selected_indices, self.trainer.prev_labeled_indices.data)
            accelerator.print(f'DivReward len(selected_indices)={len(selected_indices)}')

        from accelerate.utils import broadcast_object_list
        payload = [selected_indices.tolist() if selected_indices is not None else None]
        broadcast_object_list(payload)
        selected_indices = np.array(payload[0] or [], dtype=int)
        assert len(selected_indices) > 0, 'DivReward selected_indices is empty!'

        total_prompts = []
        total_reward_margins_local = []
        total_completion_ids_local = []
        total_completion_mask_local = []
        cand_dataloader = self.trainer.get_unlabeled_dataloader(selected_indices, per_device_bs)

        model = self.prepare_model(cand_dataloader)

        for batch_idx, inputs in enumerate(cand_dataloader):
            if inputs is None:
                continue

            with torch.inference_mode():
                response_dict = self.generate_responses(model, inputs, n=2)
                reward_margins = self.compute_reward_margin(model, response_dict)
            
            total_prompts += inputs['prompt']
            total_completion_ids_local.append(response_dict['completion_ids'].detach().cpu())
            total_completion_mask_local.append(response_dict['completion_mask'].detach().cpu())
            total_reward_margins_local.append(reward_margins.detach().cpu())

            # Progress reporting
            if batch_idx % 5 == 0:
                accelerator.print(f'Processing batch {batch_idx}, prompts collected: {len(total_prompts)}')

        # Gather once across processes
        total_reward_margins, total_batch_size = gather_1d_once(
            total_reward_margins_local, accelerator, pad_index=float('-inf')
        )
        total_prompts = gather_object(total_prompts)[: total_batch_size]
        total_completion_ids_g, _ = gather_2d_once(
            total_completion_ids_local, accelerator, pad_token_id=self.trainer.processing_class.pad_token_id
        )
        total_completion_mask_g, _ = gather_2d_once(total_completion_mask_local, accelerator, pad_token_id=0)

        # Ensure completion rows are consistent with prompt count (2 completions per prompt)
        if total_completion_ids_g.shape[0] < 2 * total_batch_size:
            new_bs = int(total_completion_ids_g.shape[0] // 2)
            accelerator.print(
                f"DivReward warning: completion rows {int(total_completion_ids_g.shape[0])} < 2*{int(total_batch_size)}; "
                f"using batch_size={new_bs}"
            )
            total_batch_size = new_bs
            total_reward_margins = total_reward_margins[:new_bs]
            total_prompts = total_prompts[:new_bs]
            selected_indices = selected_indices[:new_bs]

        len_total_reward_margins = int(total_reward_margins.shape[0])
        max_k = min(len_total_reward_margins, int(n), len(selected_indices))
        accelerator.print(f"DivReward len(total_reward_margins)={len_total_reward_margins}, n={n} max_k = {max_k}")

        final_topk_indices = torch.topk(total_reward_margins, k=max_k, largest=True).indices.cpu()
        final_selected_indices = selected_indices[final_topk_indices.cpu().numpy()]

        # Select prompts
        selected_prompts = np.array(total_prompts)[final_topk_indices.cpu().numpy()].tolist()

        # Concatenate completions
        total_completion_ids = merge_prompt_or_completion([total_completion_ids_g], accelerator.num_processes, total_batch_size)
        total_completion_mask = merge_prompt_or_completion([total_completion_mask_g], accelerator.num_processes, total_batch_size)
        
        args_to_update = {
            'topk_indices': final_topk_indices,
            'selected_indices': final_selected_indices,
            'prompt': selected_prompts,
            'completion_ids': total_completion_ids,
            'completion_mask': total_completion_mask,
        }

        args_to_update = self._prepare_for_data_update(args_to_update, total_batch_size)

        return args_to_update

    def print_rank0(self, msg):
        if self.rank == 0:
            print(f'[rank{self.rank}] {msg}')
            
    def _flatten_prompt(self, item):
        if isinstance(item, str):
            return item
        if isinstance(item, list) and item:
            first = item[0]
            if isinstance(first, dict) and "content" in first:
                return first["content"]
        return str(item)
