import torch
import numpy as np
from tqdm import tqdm

from .strategy import Strategy
from ..utils import safe_gather, merge_prompt_or_completion
from accelerate.utils import gather_object

def pearson_corr(x, y):
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    return (x_centered * y_centered).sum() / (x_centered.norm() * y_centered.norm())

class APLTest(Strategy):
    def __init__(self, trainer, system_prompt, config):
        super(APLTest, self).__init__(trainer, system_prompt, config)
        self.n = 2 # original APL: 20 but it takes too long
        self.k = 2
        self.subset_factor = self.subset_factor // 4

    def query(self, n):          
        # load a unlabeled dataloader
        unlabeled_indices = self.trainer.extract_unlabeled_indices()
        subset_unlabeled_indices = np.random.choice(unlabeled_indices, size=int(n * self.subset_factor * self.k), replace=False)
        unlabeled_dataloader = self.trainer.get_unlabeled_dataloader(
            subset_unlabeled_indices, batch_size=self.trainer.args.per_device_train_batch_size // self.n) # self.active_args.optimal_batch_size // (4 * self.n))
        
        # set up a model line 
        accelerator = self.trainer.accelerator
        model = self.prepare_model(unlabeled_dataloader)
        model.eval()

        total_reward_margins = []
        total_entropies = []
        for inputs in unlabeled_dataloader:
            with torch.inference_mode():
                response_dict = self.generate_responses(
                    model, inputs, n=self.n) # estimation for entropy per prompt (refer to APL); shape: B
                reward_margins, mean_logprobs = self.compute_reward_margin(model, response_dict,  n=self.n, compute_log_probs=True)    
                # mean_logprobs = self.compute_log_probs(model, response_dict, n=self.n)
            
            gathered_reward_margins = safe_gather(reward_margins, accelerator).cpu()
            gathered_entropies = safe_gather(-1.0 * mean_logprobs, accelerator).cpu()
            total_reward_margins.append(gathered_reward_margins)
            total_entropies.append(gathered_entropies)
        
        total_reward_margins = torch.cat(total_reward_margins, dim=0)
        total_entropies = torch.cat(total_entropies, dim=0)
        assert total_entropies.shape[0] == len(subset_unlabeled_indices), f"subset_unlabeled_indices shape is not equal to total_entropies shape"
        topk_indices = torch.topk(total_entropies, k=len(subset_unlabeled_indices) // self.k, largest=True).indices.cpu()
        
        corr = pearson_corr(total_reward_margins, total_entropies)
        print(f"========Correlation: {corr.item()}=========")

        # select indices
        selected_indices = subset_unlabeled_indices[topk_indices]
        
        total_prompts = []
        total_reward_margins = []
        total_completion_ids = []
        total_completion_mask = []
        cand_base_bs = self.query_batch_size or self.trainer.args.per_device_train_batch_size
        cand_dataloader = self.trainer.get_unlabeled_dataloader(selected_indices, max(1, cand_base_bs // 2))
        for inputs in cand_dataloader:
            with torch.inference_mode():
                response_dict = self.generate_responses(model, inputs, n=2) # estimation for entropy per prompt (refer to APL); shape: B
                reward_margins = self.compute_reward_margin(model, response_dict)                
            
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
        final_topk_indices = torch.topk(total_reward_margins, k=n, largest=True).indices.cpu()
        final_selected_indices = selected_indices[final_topk_indices]
        
        # select prompts
        selected_prompts = np.array(total_prompts)[final_topk_indices].tolist()

        # concat completions
        total_batch_size = total_reward_margins.shape[0]
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
