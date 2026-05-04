import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from .strategy import Strategy
from ..utils import merge_prompt_or_completion, broadcast_subset
from trl.data_utils import is_conversational
class KCenterGreedy(Strategy):
    def __init__(self, trainer, system_prompt, config):
        super(KCenterGreedy, self).__init__(trainer, system_prompt, config)
            
    def query(self, n):
        # load a labeled dataloader
        labeled_indices = np.array(list(self.trainer.replay_buffer.keys()))
        labeled_dataloader = self.trainer.get_indexed_dataloader(labeled_indices)
        num_labeled = len(labeled_indices)
        
        # load a unlabeled dataloader
        unlabeled_indices = self.trainer.extract_unlabeled_indices()
        unlabeled_indices = np.setdiff1d(unlabeled_indices, labeled_indices)
        accelerator = self.trainer.accelerator
        per_device_bs = self.active_args.optimal_batch_size
        subset_unlabeled_indices = broadcast_subset(
            unlabeled_indices,
            desired_size=int(n * self.subset_factor),
            accelerator=accelerator,
            per_device_bs=per_device_bs,
            k=1,
            seed=getattr(self.trainer.args, "seed", 42),
            enforce_multiple=False,
        ).cpu().numpy()
        unlabeled_dataloader = self.trainer.get_unlabeled_dataloader(subset_unlabeled_indices, per_device_bs)
        
        # set up a model line 
        accelerator = self.trainer.accelerator
        model = self.prepare_model(unlabeled_dataloader)

        (total_prompts, total_completion_ids,
         total_completion_mask, unlabeled_embeddings) = self.get_unlabeled_embeddings(unlabeled_dataloader, model)
        
        if num_labeled > 0:
            labeled_embeddings = self.get_labeled_embeddings(labeled_dataloader)
                    
            # compute min distance
            min_distances = torch.cdist(unlabeled_embeddings, labeled_embeddings).min(dim=1).values
    
            final_topk_indices = []
            num_selection = n
        else:            
            # if there is no labeled data, randomly select the first index among unlabeled data
            init_idx = np.random.choice(unlabeled_embeddings.shape[0], size=1, replace=False)
            final_topk_indices = [init_idx.item()]
            min_distances = F.pairwise_distance(unlabeled_embeddings, unlabeled_embeddings[init_idx].unsqueeze(0))

            num_selection = n - 1

        # k-center greey selection
        
        for _ in range(num_selection):
            # Select the farthest point from current labeled set
            farthest_idx = torch.argmax(min_distances).item()
            final_topk_indices.append(farthest_idx)

            # Update distances: check if new point is closer
            new_dist = F.pairwise_distance(unlabeled_embeddings, unlabeled_embeddings[farthest_idx].unsqueeze(0))
            min_distances = torch.minimum(min_distances, new_dist)
            
        
        # select topk indices    
        final_topk_indices = torch.tensor(final_topk_indices)
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
     
