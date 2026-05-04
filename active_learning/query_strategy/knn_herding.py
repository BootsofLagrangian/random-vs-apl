import time
import numpy as np
import torch
from tqdm import tqdm

from accelerate.utils import gather_object
from .strategy import Strategy
from ..utils import (
    merge_prompt_or_completion, LinearKernel, RBFKernel, TopHatKernel, broadcast_subset)

def split_indices_among_processes(indices, num_processes):
    # Split into roughly equal chunks (first ones get one more if not divisible)
    chunk_sizes = [len(indices) // num_processes] * num_processes
    for i in range(len(indices) % num_processes):
        chunk_sizes[i] += 1

    splits = []
    start = 0
    for size in chunk_sizes:
        splits.append(indices[start:start+size])
        start += size
    return splits

class KNNHerding(Strategy):
    def __init__(self, trainer, system_prompt, config):
        super(KNNHerding, self).__init__(trainer, system_prompt, config)
        self.kernel = self.construct_kernel_fn('rbf')
        self.h = 1.0
        self.k = 3
        self.subset_factor = 64

    def construct_kernel_fn(self, kernel_name):
        if kernel_name == "rbf":
            kernel = RBFKernel()
        elif kernel_name == "liear":
            kernel = LinearKernel()
        elif kernel_name == "tophat":
            kernel = TopHatKernel()
        else:
            raise NotImplementedError(f"{kernel_name} not implemented")
        print(f'Constructed kernel: {kernel_name}')
        return kernel
    
    def query(self, n):
        start_time = time.time()
        # load a labeled dataloader
        labeled_indices = np.array(list(self.trainer.replay_buffer.keys()))
        labeled_dataloader = self.trainer.get_indexed_dataloader(
            labeled_indices, batch_size=self.active_args.optimal_batch_size)
        
        # load a unlabeled dataloader
        unlabeled_indices = self.trainer.extract_unlabeled_indices()
        unlabeled_indices = np.setdiff1d(unlabeled_indices, labeled_indices)
        
        subset_size = min(n * self.subset_factor, len(unlabeled_indices))
        assert subset_size >= n, f"unlabeled set size should be larger than query size n"
        accelerator = self.trainer.accelerator
        per_device_bs = self.active_args.optimal_batch_size
        subset_unlabeled_indices = broadcast_subset(
            unlabeled_indices,
            desired_size=int(subset_size),
            accelerator=accelerator,
            per_device_bs=per_device_bs,
            k=1,
            seed=getattr(self.trainer.args, "seed", 42),
            enforce_multiple=False,
        ).cpu().numpy()
        
        unlabeled_dataloader = self.trainer.get_unlabeled_dataloader(
            subset_unlabeled_indices, batch_size=self.active_args.optimal_batch_size)
        
        num_unlabeled = len(subset_unlabeled_indices)
        num_labeled = len(labeled_indices)
        
        # set up a model line 
        accelerator = self.trainer.accelerator
        device = accelerator.device
        model = self.prepare_model(unlabeled_dataloader)
        print(f'Prepared data and model: {time.time() - start_time} sec')
        start_time = time.time()

        (total_prompts, total_completion_ids,
         total_completion_mask, unlabeled_embeddings) = self.get_unlabeled_embeddings(unlabeled_dataloader, model)

        if len(total_prompts) > num_unlabeled:
            total_prompts = total_prompts[:num_unlabeled]
            unlabeled_embeddings = unlabeled_embeddings[:num_unlabeled]
        print(f'Prepared unlabeled data: {time.time() - start_time} sec')
        start_time = time.time()
            
        # obtain embeddings for labeled data points
        if num_labeled > 0:
            labeled_embeddings = self.get_labeled_embeddings(labeled_dataloader)
            if labeled_embeddings.shape[0] > num_labeled:
                labeled_embeddings = labeled_embeddings[:num_labeled]
        
            # merge embeddings
            total_embeddings = torch.cat([labeled_embeddings, unlabeled_embeddings])
        else:
            total_embeddings = unlabeled_embeddings
            
        # normalize embeddings
        total_embeddings = total_embeddings.to(accelerator.device)
        total_embeddings = (total_embeddings / torch.norm(total_embeddings, dim=-1, keepdim=True)) # N x d
        print(f'Prepared labeled data: {time.time() - start_time} sec')
        start_time = time.time()
        
        # Each process will compute part of the kernel (a block of rows)
        with accelerator.split_between_processes(total_embeddings) as local_rows:
            kernel_block = self.kernel.compute_kernel(local_rows, total_embeddings, self.h)  # shape: N_local x N
        kernel_all = accelerator.gather(kernel_block) # N x N
        N = kernel_all.shape[0]
        print(f'Prepared kernel: {time.time() - start_time} sec')
        start_time = time.time()
        
        # # keep track of labeled data
        # is_labeled = torch.zeros(num_labeled + num_unlabeled).bool().to(accelerator.device) # N
        # is_labeled[:num_labeled] = True
        
        # TODO: 1. use faiss to find knn fast
        # 2. update uncertainty only for those data points
        
        inner_lSet = torch.arange(num_labeled).to(device)
        fixed_inner_uSet = torch.arange(N)[len(inner_lSet):].to(device)
        inner_uSet_bool = torch.ones_like(fixed_inner_uSet).bool().to(device)
        inner_uSet = fixed_inner_uSet[inner_uSet_bool]
        
        cand_batch_size = N // 10
        for i in tqdm(range(n), desc=f'{self.k}-NN Herding select {n} out of {num_unlabeled}'):
            num_lSet = len(inner_lSet)
            num_uSet = len(inner_uSet)
            init_k = min(self.k, num_lSet)
                                    
            if i == 0:
                kernel_la = kernel_all[:,inner_lSet].unsqueeze(0).expand(cand_batch_size, -1, -1) # N x L -> C x N x L 
                kernel_la_topk = torch.topk(kernel_la, k=init_k, dim=-1, largest=True) # C x N x L -> C x N x k
                inner_lSet_topk = inner_lSet[kernel_la_topk.indices] # C x N x k
            else:
                try:
                    kernel_la = torch.cat((kernel_la_topk.values, kernel_all[:,[prev_selected_index]].unsqueeze(0).expand(cand_batch_size, -1, -1)), dim=-1) # C x N x (k+1)
                    kernel_la_topk = torch.topk(kernel_la, k=k, dim=-1, largest=True) # C x N x (k+1) -> C x N x k
                except:
                    if accelerator.is_main_process:
                        import IPython; IPython.embed()
                    accelerator.wait_for_everyone()
                
                inner_lSet_topk_plus = torch.cat([inner_lSet_topk, (torch.ones(cand_batch_size, N, 1) * prev_selected_index).int().to(self.device)], dim=-1) # C x N x (k+1)
                inner_lSet_topk = inner_lSet_topk_plus.gather(2, kernel_la_topk.indices) # C x N x k
           
            split_indices = split_indices_among_processes(list(range(0, num_uSet, cand_batch_size)), accelerator.num_processes)
            with accelerator.split_between_processes(split_indices) as start_indices:
                start_indices = np.array(start_indices).reshape(-1)
            
                vars = []
                for start_idx in start_indices:
                    end_idx = min(start_idx + cand_batch_size, num_uSet)
                    candidate_indices = inner_uSet[start_idx:end_idx] # C
                    candidate = kernel_all[candidate_indices].float() # C x N
                    num_candidate = candidate.shape[0]
                    
                    if num_candidate < cand_batch_size:
                        candidate = torch.cat((kernel_la_topk.values[:num_candidate], candidate.unsqueeze(-1)), dim=-1) # C x N x (k+1)
                        exp_candidate_indices = torch.cat((inner_lSet_topk[:num_candidate], candidate_indices.reshape(num_candidate, 1, 1).expand(-1, N, -1)), dim=-1)  # C x N x k, C -> C x N x k, C x N x 1 -> C x N x (k+1)
                    else:
                        candidate = torch.cat((kernel_la_topk.values, candidate.unsqueeze(-1)), dim=-1) # C x N x (k+1)
                        exp_candidate_indices = torch.cat((inner_lSet_topk, candidate_indices.reshape(num_candidate, 1, 1).expand(-1, N, -1)), dim=-1)  #  N x k, C -> C x N x k, C x N x 1 -> C x N x (k+1)
                
                    # obtain new topk using previous topk + candidates    
                    k = min(self.k, num_lSet+1)
                    topk = torch.topk(candidate, k, dim=-1, largest=True) # C x N x (k+1) -> C x N x k
                    topk_values = topk.values # C x N x k 
                    topk_indices = topk.indices # C x N x k
                            
                    # obtain topk indices in range of [0, N-1]
                    topk_candidate_indices = exp_candidate_indices.gather(2, topk_indices) # C x N x k
                    
                    # Method1: original
                    row_indices = topk_candidate_indices.unsqueeze(-1).expand(-1, -1, -1, k)
                    col_indices = topk_candidate_indices.unsqueeze(-2).expand(-1, -1, k, -1)
                    topk_matrix = kernel_all[row_indices, col_indices]

                    # compute variance
                    LU, pivots = torch.linalg.lu_factor((topk_matrix + 1e-6 * torch.eye(k).reshape(1, 1, k, k).repeat(*topk_matrix.shape[:2], 1, 1).to(accelerator.device)).double())
                    X = torch.linalg.lu_solve(LU, pivots, topk_values.unsqueeze(-1).double()).float()

                    var = (topk_values.unsqueeze(2) @ X.float()).squeeze(-1).squeeze(-1) ** 0.5 # C x N x 1 x 1 -> C x N
                    vars.append(var)
                    
                vars = torch.cat(vars)
                list_vars = gather_object(vars)
            
            total_vars = torch.stack([var.cpu() for var in list_vars]) # U x U
                    
            mean_one_minus_vars = total_vars.mean(dim=-1) # U
            selected_index = torch.argmax(mean_one_minus_vars) # 1
            selected_index = inner_uSet[selected_index]
            prev_selected_index = selected_index.item()
            
            # # update lSet and uSet
            inner_lSet = torch.cat((inner_lSet, selected_index.view(-1)))
            inner_uSet_bool[selected_index - num_labeled] = False
            inner_uSet = fixed_inner_uSet[inner_uSet_bool]
            
            # # update lSet and uSet
            # assert not is_labeled[selected_index].item(), "Selected index was previously selected for MaxHerding"
            # is_labeled[selected_index] = True
        
        # if accelerator.is_main_process:
        #     import IPython; IPython.embed()
        # accelerator.wait_for_everyone()
        print(f'Selection took: {time.time() - start_time} sec')
                
        selected_indices = torch.sort(inner_lSet[num_labeled:])[0]
        assert len(selected_indices) == n, "The number of selected indices does not match with a budget"
        
         # select topk indices    
        final_topk_indices = selected_indices.cpu() - num_labeled
        final_selected_indices = subset_unlabeled_indices[final_topk_indices]
        
        # select prompts
        selected_prompts = np.array(total_prompts)[final_topk_indices].tolist()
        
        # concat completions
        total_batch_size = unlabeled_embeddings.shape[0]
        total_completion_ids = merge_prompt_or_completion(total_completion_ids, accelerator.num_processes, total_batch_size) # (B x 2) x L
        total_completion_mask = merge_prompt_or_completion(total_completion_mask, accelerator.num_processes, total_batch_size) # (B x 2) x L
        
        args_to_update = {
            'topk_indices': final_topk_indices,
            'selected_indices': final_selected_indices,
            'prompt': selected_prompts,
            'completion_ids': total_completion_ids,
            'completion_mask': total_completion_mask,
        }
        
        args_to_update = self._prepare_for_data_update(args_to_update, total_batch_size)        
        
        return args_to_update            
            
