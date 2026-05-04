import numpy as np
import torch
from trl.scripts.utils import find_optimal_batch_size
from trl.trainer.utils import pad
from typing import Dict, List, Tuple, Optional
from accelerate.utils import broadcast_object_list

def split_list(lst, split_size):
    """
    Splits a list into chunks of size `split_size`, similar to torch.split.

    Args:
        lst (list): The list to split.
        split_size (int): The number of elements in each chunk.

    Returns:
        List[List]: A list of chunks (sublists).
    """
    return [lst[i:i + split_size] for i in range(0, len(lst), split_size)]

def chunk_list(lst, chunks):
    """
    Splits a list into chunks of size `chunks`, similar to torch.chunk.
    """
    k, m = divmod(len(lst), chunks)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(chunks)]

def merge_prompt_or_completion(list_ids_or_mask, num_processes, total_batch_size):
    total_ids_or_mask1 = []
    total_ids_or_mask2 = []
    for ids_or_mask in list_ids_or_mask:
        all_device_ids_or_mask = ids_or_mask.chunk(num_processes)
        for per_device_ids_or_mask in all_device_ids_or_mask:
            ids_or_mask1, ids_or_mask2 = per_device_ids_or_mask.chunk(2, dim=0)

            total_ids_or_mask1.append(ids_or_mask1)
            total_ids_or_mask2.append(ids_or_mask2)
    
    total_ids_or_mask1 = pad(total_ids_or_mask1)
    completion_max_token_size = total_ids_or_mask1.shape[-1]
    total_ids_or_mask1 = total_ids_or_mask1.reshape(-1, completion_max_token_size)[:total_batch_size]
    
    total_ids_or_mask2 = pad(total_ids_or_mask2)
    completion_max_token_size = total_ids_or_mask2.shape[-1]
    total_ids_or_mask2 = total_ids_or_mask2.reshape(-1, completion_max_token_size)[:total_batch_size]
    
    total_ids_or_mask = torch.cat([total_ids_or_mask1, total_ids_or_mask2], dim=0)
    return total_ids_or_mask

def reorder_prompt_or_completion(ids_or_mask, num_processes):
    per_device_ids_or_mask = ids_or_mask.chunk(num_processes, dim=0)
            
    total_first_half_ids_or_mask = []
    total_second_half_ids_or_mask = []
    for ids_or_mask in per_device_ids_or_mask:
        first_half_ids_or_mask, second_half_ids_or_mask = ids_or_mask.chunk(2, dim=0)
        total_first_half_ids_or_mask.append(first_half_ids_or_mask)
        total_second_half_ids_or_mask.append(second_half_ids_or_mask)
                
    total_first_half_ids_or_mask = torch.cat(total_first_half_ids_or_mask, dim=0)
    total_second_half_ids_or_mask = torch.cat(total_second_half_ids_or_mask, dim=0)
            
    return torch.cat([total_first_half_ids_or_mask, total_second_half_ids_or_mask], dim=0)

def safe_gather(tensor, accelerator):
    """Safely gather tensors across processes.

    Prefer `accelerator.gather_for_metrics`, which tolerates different
    per-rank batch sizes by padding under the hood. Using `accelerator.gather`
    can deadlock when the last batch sizes differ across ranks.
    """
    try:
        gathered = accelerator.gather_for_metrics(tensor)
        return gathered
    except Exception as e:
        print(f"[Rank {accelerator.process_index}] gather_for_metrics failed: {e}")
        try:
            # Best-effort fallback; may still fail if shapes differ
            return accelerator.gather(tensor)
        except Exception as e2:
            print(f"[Rank {accelerator.process_index}] Gather fallback failed: {e2}")
            return None
    
def safe_gather_to_cpu(tensor, accelerator):
    gathered = safe_gather(tensor, accelerator)
    return gathered.cpu() if gathered is not None else None


def broadcast_subset(
    unlabeled_indices: List[int],
    desired_size: int,
    accelerator,
    per_device_bs: int,
    k: int = 1,
    seed: Optional[int] = 42,
    enforce_multiple: bool = True,
) -> torch.Tensor:
    """Select a subset on rank0 and broadcast to all ranks.

    Optionally round down to a multiple of global batch size times k to keep
    dataloader steps aligned across ranks.
    """
    world = accelerator.num_processes
    global_bs = max(1, per_device_bs) * max(1, world)
    size = int(desired_size)
    if size <= 0:
        size = global_bs * max(1, k)
    if enforce_multiple:
        multiple = max(1, global_bs * max(1, k))
        size = max(multiple, size - (size % multiple))
        size = min(size, len(unlabeled_indices))

    if accelerator.is_main_process:
        rng = np.random.default_rng(int(seed or 42))
        chosen = rng.choice(unlabeled_indices, size=size, replace=False).tolist()
    else:
        chosen = None
    payload = [chosen]
    broadcast_object_list(payload)
    return torch.tensor(payload[0], dtype=torch.long)


def gather_1d_once(local_tensors: List[torch.Tensor], accelerator, pad_index: float = float("-inf")) -> Tuple[torch.Tensor, int]:
    """Concatenate local 1D tensors, then pad and gather once across ranks.

    Returns gathered 1D tensor and the true total length to slice to.
    """
    if len(local_tensors) > 0:
        local = torch.cat([t.detach().to(accelerator.device) for t in local_tensors], dim=0)
    else:
        local = torch.empty((0,), device=accelerator.device)
    local_len = torch.tensor([local.shape[0]], device=accelerator.device)
    total_len = int(accelerator.gather_for_metrics(local_len).sum().item())
    padded = accelerator.pad_across_processes(local, dim=0, pad_index=pad_index)
    gathered = accelerator.gather_for_metrics(padded)
    return gathered[: total_len].cpu(), total_len


def gather_2d_once(local_2d_list: List[torch.Tensor], accelerator, pad_token_id: int = 0) -> Tuple[torch.Tensor, int]:
    """Pad 2D tensors by width, stack locally, then pad rows and gather once.

    Returns gathered 2D tensor and the total row count.
    """
    if len(local_2d_list) > 0:
        # Use repo's pad util to align width
        local_3d = pad(local_2d_list)  # shape: [n_steps, B, L] or [n_steps, *, L]
        L = local_3d.shape[-1]
        local_2d = local_3d.reshape(-1, L).to(accelerator.device)
    else:
        local_2d = torch.empty((0, 1), dtype=torch.long, device=accelerator.device)
    local_rows = torch.tensor([local_2d.shape[0]], device=accelerator.device)
    total_rows = int(accelerator.gather_for_metrics(local_rows).sum().item())
    padded_rows = accelerator.pad_across_processes(local_2d, dim=0, pad_index=pad_token_id)
    gathered = accelerator.gather_for_metrics(padded_rows)
    return gathered[: total_rows].cpu(), total_rows


class LinearKernel(object):
    def compute_kernel(self, x1, x2, h=1.0, batch_size=512):
        k = torch.matmul(x1, x2.T)
        return k

class RBFKernel(object):
    def compute_kernel(self, x1, x2, h=1.0, batch_size=512):
        dist = compute_dist(x1, x2, batch_size=batch_size)
        k = torch.exp(-1.0 * (dist / h) ** 2)
        return k

class TopHatKernel(object):
    def compute_kernel(self, x1, x2, h=0.5, batch_size=512):
        dist = compute_dist(x1, x2, batch_size=batch_size)
        k = (dist < h)
        return k

def compute_dist(x1, x2, batch_size=512):
    x1, x2 = x1.unsqueeze(0), x2.unsqueeze(0) # 1 x n x d, 1 x n' x d
    dist_matrix = []
    batch_round = x2.shape[1] // batch_size + int(x2.shape[1] % batch_size > 0)
    for i in range(batch_round):
        x2_subset = x2[:, i * batch_size: (i + 1) * batch_size]
        dist = torch.cdist(x1, x2_subset, p=2.0)
        dist_matrix.append(dist)

    dist_matrix = torch.cat(dist_matrix, dim=-1).squeeze(0)
    return dist_matrix 


def slice_and_move_batch_for_device(batch: Dict, rank: int, world_size: int, device: str) -> Dict:
    """Slice a batch into chunks, and move each chunk to the specified device."""
    chunk_size = len(list(batch.values())[0]) // world_size
    start = chunk_size * rank
    end = chunk_size * (rank + 1)
    sliced = {k: v[start:end] for k, v in batch.items()}
    on_device = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in sliced.items()}
    return on_device


def get_optimal_batch_size(model, tokenizer, dataset, training_args, active_args):
    ################
    # Optimal Batch Size
    ################
    batch_size = training_args.per_device_train_batch_size
    if int(active_args.optimal_batch) > 0:
        prompt_samples = [ex["prompt"] for ex in dataset.shuffle().select(range(500))]
        batch_size = find_optimal_batch_size(
                                                model, tokenizer, prompt_texts=prompt_samples, 
                                                length_percentile=0.9, max_batch_size=16, device='cuda', 
                                                use_amp=training_args.fp16
                                            )
    
    return batch_size
