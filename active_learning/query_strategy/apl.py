import torch
import numpy as np
from tqdm import tqdm

from .strategy import Strategy
from ..utils import safe_gather, merge_prompt_or_completion
from trl.trainer.utils import pad
from accelerate.utils import gather_object, broadcast_object_list

class APL(Strategy):
    def __init__(self, trainer, system_prompt, config):
        super(APL, self).__init__(trainer, system_prompt, config)
        # Original APL paper uses n=20 samples per prompt, which is very
        # expensive in our setting. Make this configurable via `apl_n`
        # and default to a smaller, more practical value.
        self.n = getattr(config, "apl_n", 2)
        self.k = 2
        self.subset_factor = self.subset_factor // 4

    def query(self, n):          
        # load a unlabeled dataloader
        unlabeled_indices = self.trainer.extract_unlabeled_indices()

        # Ensure all ranks use the IDENTICAL subset and the size divides evenly across
        # global batch to avoid mismatched dataloader steps and collective calls.
        accelerator = self.trainer.accelerator
        if getattr(self, "query_batch_size", 0) > 0:
            per_device_bs = self.query_batch_size
        else:
            per_device_bs = min(self.trainer.args.per_device_train_batch_size, 4)
        global_bs = max(1, per_device_bs) * accelerator.num_processes
        # We will later pick topk with factor `self.k`, ensure divisibility for both stages
        desired = int(n * self.subset_factor * self.k)
        if desired <= 0:
            desired = global_bs * self.k
        # Round down to a multiple of global_bs * self.k (so selected_indices size is also multiple of global_bs)
        multiple = max(1, global_bs * self.k)
        subset_size = max(multiple, desired - (desired % multiple))
        subset_size = min(subset_size, len(unlabeled_indices))

        if accelerator.is_main_process:
            # Use a process-agnostic RNG so every run picks the same subset across ranks
            rng = np.random.default_rng(int(getattr(self.trainer.args, "seed", 42)))
            chosen = rng.choice(unlabeled_indices, size=subset_size, replace=False).tolist()
        else:
            chosen = None
        payload = [chosen]
        broadcast_object_list(payload)
        subset_unlabeled_indices = np.array(payload[0], dtype=int)
        unlabeled_dataloader = self.trainer.get_unlabeled_dataloader(
            subset_unlabeled_indices, batch_size=per_device_bs) # self.active_args.optimal_batch_size // (4 * self.n))
        
        # set up a model line 
        accelerator = self.trainer.accelerator
        model = self.prepare_model(unlabeled_dataloader)
        model.eval()

        # Accumulate locally, then gather once (avoids per-step collective mismatches)
        local_entropies = []
        for inputs in tqdm(unlabeled_dataloader, desc="APL querying", leave=False):
            with torch.inference_mode():
                response_dict = self.generate_responses(
                    model, inputs, n=self.n) # estimation for entropy per prompt (refer to APL); shape: B
                mean_logprobs = -1.0 * self.compute_log_probs(model, response_dict, n=self.n)
            local_entropies.append(mean_logprobs.detach().cpu())
        torch.cuda.empty_cache()

        if len(local_entropies) > 0:
            local_entropies = torch.cat(local_entropies, dim=0).to(accelerator.device)
        else:
            local_entropies = torch.empty((0,), device=accelerator.device)
        local_len = torch.tensor([local_entropies.shape[0]], device=accelerator.device)
        all_lens = accelerator.gather_for_metrics(local_len)
        total_len = int(all_lens.sum().item())
        padded_entropies = accelerator.pad_across_processes(local_entropies, dim=0, pad_index=float('-inf'))
        total_entropies = accelerator.gather_for_metrics(padded_entropies).cpu()
        total_entropies = total_entropies[: total_len]
        assert total_entropies.shape[0] == len(subset_unlabeled_indices), (
            f"subset_unlabeled_indices shape is not equal to total_entropies shape: "
            f"{total_entropies.shape[0]} vs {len(subset_unlabeled_indices)}"
        )
        topk_indices = torch.topk(total_entropies, k=len(subset_unlabeled_indices) // self.k, largest=True).indices.cpu()
        
        # select indices
        selected_indices = subset_unlabeled_indices[topk_indices]
        
        total_prompts_local = []
        total_reward_margins_local = []
        total_completion_ids_local = []
        total_completion_mask_local = []
        # Second pass: ensure selected_indices length is divisible by global batch for the candidate dataloader
        base_bs = self.query_batch_size or self.trainer.args.per_device_train_batch_size
        cand_per_device_bs = max(1, base_bs // 2)
        cand_global_bs = cand_per_device_bs * accelerator.num_processes
        if len(selected_indices) % cand_global_bs != 0:
            # Trim to nearest lower multiple to keep steps aligned across ranks
            trim = len(selected_indices) - (len(selected_indices) // cand_global_bs) * cand_global_bs
            if trim > 0:
                prev_selected_indices = selected_indices
                selected_indices = selected_indices[:-trim]
                if len(selected_indices) < n:
                    # Do not compromise the final top-n guarantee
                    selected_indices = prev_selected_indices
                total_reward_margins = []  # fresh containers even if unchanged
                total_completion_ids = []
                total_completion_mask = []
                total_prompts = []

        cand_dataloader = self.trainer.get_unlabeled_dataloader(selected_indices, cand_per_device_bs)
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


        # Gather once across processes
        if len(total_reward_margins_local) > 0:
            local_margins = torch.cat(total_reward_margins_local, dim=0).to(accelerator.device)
        else:
            local_margins = torch.empty((0,), device=accelerator.device)
        local_len = torch.tensor([local_margins.shape[0]], device=accelerator.device)
        all_lens = accelerator.gather_for_metrics(local_len)
        total_batch_size = int(all_lens.sum().item())
        padded_margins = accelerator.pad_across_processes(local_margins, dim=0, pad_index=float('-inf'))
        total_reward_margins = accelerator.gather_for_metrics(padded_margins).cpu()
        total_reward_margins = total_reward_margins[: total_batch_size]

        # Combine prompts: one gather at the end
        total_prompts = gather_object(total_prompts_local)

        # Combine completion ids/masks: pad and gather once, then merge
        if len(total_completion_ids_local) > 0:
            # Pad along both batch and sequence dims locally, then flatten batch dims
            local_ids_3d = pad(total_completion_ids_local)
            local_mask_3d = pad(total_completion_mask_local)
            max_L = local_ids_3d.shape[-1]
            local_completion_ids = local_ids_3d.reshape(-1, max_L).to(accelerator.device)
            local_completion_mask = local_mask_3d.reshape(-1, max_L).to(accelerator.device)
        else:
            local_completion_ids = torch.empty((0, 1), dtype=torch.long, device=accelerator.device)
            local_completion_mask = torch.empty((0, 1), dtype=torch.long, device=accelerator.device)
        padded_completion_ids = accelerator.pad_across_processes(local_completion_ids, dim=0, pad_index=self.trainer.processing_class.pad_token_id)
        padded_completion_mask = accelerator.pad_across_processes(local_completion_mask, dim=0, pad_index=0)
        total_completion_ids_gathered = accelerator.gather_for_metrics(padded_completion_ids).cpu()
        total_completion_mask_gathered = accelerator.gather_for_metrics(padded_completion_mask).cpu()
        total_abs_reward_margins = torch.abs(total_reward_margins)
        final_topk_indices = torch.topk(total_abs_reward_margins, k=n, largest=True).indices.cpu()
        final_selected_indices = selected_indices[final_topk_indices]
        
        # select prompts
        selected_prompts = np.array(total_prompts)[: total_batch_size][final_topk_indices].tolist()

        # concat completions
        total_completion_ids = merge_prompt_or_completion([total_completion_ids_gathered], accelerator.num_processes, total_batch_size) # (B x 2) x L
        total_completion_mask = merge_prompt_or_completion([total_completion_mask_gathered], accelerator.num_processes, total_batch_size) # (B x 2) x L
 
        args_to_update = {
            'topk_indices': final_topk_indices,
            'selected_indices': final_selected_indices,
            'prompt': selected_prompts,
            'completion_ids': total_completion_ids,
            'completion_mask': total_completion_mask,
        }
        
        args_to_update = self._prepare_for_data_update(args_to_update, total_batch_size)        
        
        return args_to_update
