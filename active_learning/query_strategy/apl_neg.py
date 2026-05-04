import torch
import numpy as np
from tqdm import tqdm

from .strategy import Strategy
from ..utils import (
    safe_gather,
    merge_prompt_or_completion,
    broadcast_subset,
    gather_1d_once,
    gather_2d_once,
)
from accelerate.utils import gather_object

class APLNeg(Strategy):
    def __init__(self, trainer, system_prompt, config):
        super(APLNeg, self).__init__(trainer, system_prompt, config)
        self.n = 2 # original APL: 20 but it takes too long
        self.k = 2
        self.subset_factor = self.subset_factor // 4

    def query(self, n):        
        print(f'Start querying using apl_neg...')  
        # load a unlabeled dataloader
        unlabeled_indices = self.trainer.extract_unlabeled_indices()
        accelerator = self.trainer.accelerator
        base_bs = self.query_batch_size or self.trainer.args.per_device_train_batch_size
        per_device_bs = max(1, base_bs // self.n)
        desired = int(n * self.subset_factor * self.k)
        subset_unlabeled_indices = broadcast_subset(
            unlabeled_indices,
            desired_size=desired,
            accelerator=accelerator,
            per_device_bs=per_device_bs,
            k=self.k,
            seed=getattr(self.trainer.args, "seed", 42),
            enforce_multiple=False,
        ).cpu().numpy()
        unlabeled_dataloader = self.trainer.get_unlabeled_dataloader(
            subset_unlabeled_indices, batch_size=per_device_bs)
        
        # set up a model line 
        accelerator = self.trainer.accelerator
        model = self.prepare_model(unlabeled_dataloader)
        model.eval()

        local_entropies = []
        for inputs in unlabeled_dataloader:
            with torch.inference_mode():
                response_dict = self.generate_responses(
                    model, inputs, n=self.n) # estimation for entropy per prompt (refer to APL); shape: B
                mean_logprobs = self.compute_log_probs(model, response_dict, n=self.n)
            local_entropies.append(mean_logprobs.detach().cpu())

        total_entropies, total_len = gather_1d_once(local_entropies, accelerator, pad_index=float('-inf'))
        assert total_entropies.shape[0] == len(subset_unlabeled_indices), f"subset_unlabeled_indices shape is not equal to total_entropies shape"
        topk_indices = torch.topk(total_entropies, k=len(subset_unlabeled_indices) // self.k, largest=True).indices.cpu()
        
        # select indices
        selected_indices = subset_unlabeled_indices[topk_indices]
        
        total_prompts_local = []
        total_reward_margins_local = []
        total_completion_ids_local = []
        total_completion_mask_local = []
        cand_base_bs = self.query_batch_size or self.trainer.args.per_device_train_batch_size
        cand_dataloader = self.trainer.get_unlabeled_dataloader(selected_indices, max(1, cand_base_bs // 2))
        for inputs in cand_dataloader:
            with torch.inference_mode():
                response_dict = self.generate_responses(model, inputs, n=2, decode_completion=True) # estimation for entropy per prompt (refer to APL); shape: B
                reward_margins = self.compute_reward_margin(model, response_dict, compute_abs=False)  
            
            total_prompts_local += inputs['prompt']
            
            completion_ids = response_dict['completion_ids']
            total_completion_ids_local.append(completion_ids.detach().cpu())
            
            completion_mask = response_dict['completion_mask']
            total_completion_mask_local.append(completion_mask.detach().cpu())
            
            total_reward_margins_local.append(reward_margins.detach().cpu())
         
                        
        # select topk indices  
        total_reward_margins, total_batch_size = gather_1d_once(total_reward_margins_local, accelerator, pad_index=float('-inf'))
        total_prompts = gather_object(total_prompts_local)[: total_batch_size]
        total_completion_ids_g, _ = gather_2d_once(total_completion_ids_local, accelerator, pad_token_id=self.trainer.processing_class.pad_token_id)
        total_completion_mask_g, _ = gather_2d_once(total_completion_mask_local, accelerator, pad_token_id=0)
        total_abs_reward_margins = torch.abs(total_reward_margins)
        final_topk_indices = torch.topk(total_abs_reward_margins, k=n, largest=True).indices.cpu()
        final_selected_indices = selected_indices[final_topk_indices]
            
        # select prompts
        selected_prompts = np.array(total_prompts)[final_topk_indices].tolist()

        # concat completions
        total_completion_ids = merge_prompt_or_completion([total_completion_ids_g], accelerator.num_processes, total_batch_size) # (B x 2) x L
        total_completion_mask = merge_prompt_or_completion([total_completion_mask_g], accelerator.num_processes, total_batch_size) # (B x 2) x L
 
        args_to_update = {
            'topk_indices': final_topk_indices,
            'selected_indices': final_selected_indices,
            'prompt': selected_prompts,
            'completion_ids': total_completion_ids,
            'completion_mask': total_completion_mask,
        }
        
        args_to_update = self._prepare_for_data_update(args_to_update, total_batch_size)        
        
        return args_to_update
