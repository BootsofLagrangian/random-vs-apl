import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from .strategy import Strategy
from ..utils import safe_gather, merge_prompt_or_completion, broadcast_subset
import time

from accelerate.utils import gather_object

class UHerding(Strategy):
    def __init__(self, trainer, system_prompt, config):
        super(UHerding, self).__init__(trainer, system_prompt, config)
        self.kernel = self.construct_kernel_fn('rbf')
        self.h = 1.0 
        self.n = 2 # original APL: 20 but it takes too long
        self.k = 2
        self.subset_factor = self.subset_factor // 4
        
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
        print(f'Start querying using uherding...')  
        
        # load a labeled dataloader
        labeled_indices = np.array(list(self.trainer.replay_buffer.keys()))
        labeled_dataloader = self.trainer.get_indexed_dataloader(
            labeled_indices, batch_size=self.active_args.optimal_batch_size)
        
        # load a unlabeled dataloader
        unlabeled_indices = self.trainer.extract_unlabeled_indices()
        unlabeled_indices = np.setdiff1d(unlabeled_indices, labeled_indices)
        subset_size = int(n * self.subset_factor * self.k)
        accelerator = self.trainer.accelerator
        base_bs = self.query_batch_size or self.trainer.args.per_device_train_batch_size
        per_device_bs = max(1, base_bs // self.n)
        subset_unlabeled_indices = broadcast_subset(
            unlabeled_indices,
            desired_size=subset_size,
            accelerator=accelerator,
            per_device_bs=per_device_bs,
            k=1,
            seed=getattr(self.trainer.args, "seed", 42),
            enforce_multiple=False,
        ).cpu().numpy()
        print(f'subset size is set to {len(subset_unlabeled_indices)}')
        unlabeled_dataloader = self.trainer.get_unlabeled_dataloader(
            subset_unlabeled_indices, batch_size=per_device_bs)
        
        num_unlabeled = len(subset_unlabeled_indices)
        num_labeled = len(labeled_indices)
        
        # set up a model line 
        accelerator = self.trainer.accelerator
        model = self.prepare_model(unlabeled_dataloader)
        
        start_time = time.time()
        (_, _, _, unlabeled_embeddings, total_entropies) = self.get_unlabeled_embeddings(
             unlabeled_dataloader, model, compute_log_probs=True)
        if len(total_entropies) > num_unlabeled:
            # total_prompts = total_prompts[:num_unlabeled]
            unlabeled_embeddings = unlabeled_embeddings[:num_unlabeled]
            total_entropies = total_entropies[:num_unlabeled]
        
        print(f'unlabeled embedding prep took {time.time()-start_time}sec')
        
        # obtain embeddings for labeled data points
        if num_labeled > 0:
            labeled_embeddings = self.get_labeled_embeddings(labeled_dataloader)
            if labeled_embeddings.shape[0] > num_labeled:
                labeled_embeddings = labeled_embeddings[:num_labeled]
        
            # merge embeddings
            total_embeddings = torch.cat([labeled_embeddings, unlabeled_embeddings])
            total_entropies = torch.cat([torch.zeros(num_labeled).to(total_entropies.device), total_entropies], dim=0).reshape(1, -1)
        else:
            total_embeddings = unlabeled_embeddings
            total_entropies = total_entropies.reshape(1, -1)
        total_entropies = total_entropies.to(accelerator.device)
        
        # normalize embeddings
        total_embeddings = total_embeddings.to(accelerator.device)
        total_embeddings = (total_embeddings / torch.norm(total_embeddings, dim=-1, keepdim=True)) # N x d
        
        # Each process will compute part of the kernel (a block of rows)
        with accelerator.split_between_processes(total_embeddings) as local_rows:
            kernel_block = self.kernel.compute_kernel(local_rows, total_embeddings, self.h)  # shape: N_local x N
        kernel_all = accelerator.gather(kernel_block) # N x N
        
        # keep track of labeled data
        is_labeled = torch.zeros(num_labeled + num_unlabeled).bool().to(accelerator.device) # N
        is_labeled[:num_labeled] = True
        
        # construct max embedding
        if num_labeled > 0:
            kernel_la = kernel_all[:num_labeled] # L x N
            max_embedding = kernel_la.max(dim=0, keepdim=True).values # 1 x N
        else:
            max_embedding = torch.zeros(1, num_labeled + num_unlabeled).to(accelerator.device) # 1 x N
            
        
        # Select n * self.subset_factor using entropy herding
        start_time = time.time()
        for _ in range(int(n * self.subset_factor)):
            updated_max_embedding = (kernel_all - max_embedding) # N x N
            updated_max_embedding[updated_max_embedding < 0] = 0.
            
            mean_max_embedding = (total_entropies * updated_max_embedding).mean(dim=-1) # N
            
            # select a point from u
            mean_max_embedding[is_labeled] = -np.inf # N
            selected_index = torch.argmax(mean_max_embedding)

            # update lSet and uSet
            assert not is_labeled[selected_index].item(), "Selected index was previously selected for MaxHerding"
            is_labeled[selected_index] = True
                
        indices = torch.where(is_labeled)[0][num_labeled:].cpu().numpy() - num_labeled
        selected_indices = subset_unlabeled_indices[indices]

        assert len(selected_indices) == n * self.subset_factor, "The number of selected indices does not match with a budget"
        print(f'{n} Query selection took {time.time()-start_time}sec')
        
        # 2. Select a subset using max margin
        total_prompts = []
        total_reward_margins = []
        total_completion_ids = []
        total_completion_mask = []
        cand_base_bs = self.query_batch_size or self.trainer.args.per_device_train_batch_size
        cand_dataloader = self.trainer.get_unlabeled_dataloader(selected_indices, max(1, cand_base_bs // 2))
        for inputs in cand_dataloader:
            with torch.inference_mode():
                response_dict = self.generate_responses(model, inputs, n=2, decode_completion=True) # estimation for entropy per prompt (refer to APL); shape: B
                reward_margins = self.compute_reward_margin(model, response_dict, compute_abs=False)  
            
            gathered_prompts = gather_object(inputs['prompt'])
            total_prompts += gathered_prompts
            
            completion_ids = response_dict['completion_ids']
            gathered_completion_ids = safe_gather(completion_ids, accelerator).cpu()
            total_completion_ids.append(gathered_completion_ids)
            
            completion_mask = response_dict['completion_mask']
            gathered_completion_mask = safe_gather(completion_mask, accelerator).cpu()
            total_completion_mask.append(gathered_completion_mask)
            
            gathered_reward_margins = safe_gather(reward_margins, accelerator).cpu()
            total_reward_margins.append(gathered_reward_margins)
         
        # select topk indices  
        total_reward_margins = torch.cat(total_reward_margins, dim=0)  
        total_abs_reward_margins = torch.abs(total_reward_margins)
        final_topk_indices = torch.topk(total_abs_reward_margins, k=n, largest=True).indices.cpu()
        final_selected_indices = selected_indices[final_topk_indices]
        
        # import IPython; IPython.embed()
        #======================================
        
        
        #  select topk indices    
        # final_topk_indices = selected_indices.cpu() - num_labeled
        # final_selected_indices = subset_unlabeled_indices[final_topk_indices]
        
        # select prompts
        selected_prompts = np.array(total_prompts)[final_topk_indices].tolist()
        
        # concat completions
        total_batch_size = total_abs_reward_margins.shape[0]
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
             
        print(f'{n} Total query selection took {time.time()-start_time}sec')
        return args_to_update


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

def compute_dist(x1, x2, batch_size=512, eps=1e-6):
    x1 = x1.unsqueeze(0)  # Shape: [1, n, d]
    x2 = x2.unsqueeze(0)  # Shape: [1, n', d]
    dist_matrix = []
    
    batch_round = x2.shape[1] // batch_size + int(x2.shape[1] % batch_size > 0)
    
    for i in range(batch_round):
        x2_subset = x2[:, i * batch_size: (i + 1) * batch_size]  # [1, b, d]
        
        # Broadcasting: (1, n, d) - (1, b, d) -> (n, b, d)
        diff = x1.transpose(0, 1) - x2_subset  # [n, b, d]
        dist = torch.sqrt(torch.sum(diff ** 2, dim=-1) + eps)  # [n, b]
        dist_matrix.append(dist)
    
    dist_matrix = torch.cat(dist_matrix, dim=1)  # [n, n']
    return dist_matrix

   
