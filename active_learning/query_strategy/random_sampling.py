import os
import torch
from tqdm import tqdm
import numpy as np
from .strategy import Strategy
from ..utils import safe_gather, merge_prompt_or_completion, broadcast_subset, gather_2d_once, gather_1d_once
from accelerate.utils import gather_object

class RandomSampling(Strategy):
    def __init__(self, trainer, system_prompt, active_args):
        super(RandomSampling, self).__init__(trainer, system_prompt, active_args)

    def query(self, n, save_metrics=False):
        # load an unlabeled dataloader
        unlabeled_indices = self.trainer.extract_unlabeled_indices()
        assert len(unlabeled_indices) >= n, f"the number of unlabeled indices: {len(unlabeled_indices)} is less than query number: {n} in random sampling"
        accelerator = self.trainer.accelerator
        per_device_bs = self.active_args.optimal_batch_size
        final_selected_indices = broadcast_subset(
            unlabeled_indices,
            desired_size=int(n),
            accelerator=accelerator,
            per_device_bs=per_device_bs,
            k=1,
            seed=getattr(self.trainer.args, "seed", 42),
            enforce_multiple=False,
        ).cpu().numpy()
        num_unlabeled = len(final_selected_indices)
        unlabeled_dataloader = self.trainer.get_unlabeled_dataloader(final_selected_indices, per_device_bs)
            
        # set up a model
        accelerator = self.trainer.accelerator
        model = self.trainer.model_wrapped

        total_prompts_local = []
        total_completion_ids_local = []
        total_completion_mask_local = []
        total_reward_margins_local = []
        total_entropies_local = []
        total_labels_local = []
        for inputs in unlabeled_dataloader:
            with torch.no_grad():
                response_dict = self.generate_responses(model, inputs, n=2, decode_completion=True)
                if save_metrics:
                    reward_margins, mean_logprobs = self.compute_reward_margin(
                        model, response_dict,  n=2, compute_abs=False, compute_log_probs=True)    
                    labels = self.judge(response_dict)
                    
                    total_reward_margins_local.append(reward_margins.detach().cpu())
                    total_entropies_local.append(mean_logprobs.detach().cpu())
                    total_labels_local.append(labels.detach().cpu())
        
            total_prompts_local += inputs['prompt']
            
            completion_ids = response_dict['completion_ids']
            total_completion_ids_local.append(completion_ids.detach().cpu())
            
            completion_mask = response_dict['completion_mask']
            total_completion_mask_local.append(completion_mask.detach().cpu())
        
        # select prompts
        total_prompts = gather_object(total_prompts_local)
        if len(total_prompts) > num_unlabeled:
            total_prompts = total_prompts[:num_unlabeled]
        total_batch_size = len(final_selected_indices)
        final_topk_indices = np.arange(total_batch_size)
        selected_prompts = np.array(total_prompts)[final_topk_indices].tolist()
        
        # concat completions
        total_completion_ids_g, _ = gather_2d_once(total_completion_ids_local, accelerator, pad_token_id=self.trainer.processing_class.pad_token_id)
        total_completion_mask_g, _ = gather_2d_once(total_completion_mask_local, accelerator, pad_token_id=0)
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
        
        if save_metrics:
            total_labels = torch.cat(total_labels_local, dim=0).int().to(accelerator.device)
            total_labels = accelerator.gather_for_metrics(total_labels).cpu().numpy()
            total_reward_margins = torch.cat(total_reward_margins_local, dim=0).float().to(accelerator.device)
            total_reward_margins = accelerator.gather_for_metrics(total_reward_margins).cpu()
            total_predictions = (total_reward_margins > 0).int().numpy()

            total_abs_reward_margins = torch.abs(total_reward_margins).numpy()
            total_entropies = torch.cat(total_entropies_local, dim=0).float().to(accelerator.device)
            total_entropies = accelerator.gather_for_metrics(total_entropies).cpu().numpy()
            
            entropy_path = os.path.join(self.trainer.args.output_dir, f"entropy_step{self.trainer.state.global_step}.npy")
            reward_margin_path = os.path.join(self.trainer.args.output_dir, f"reward_margin_step{self.trainer.state.global_step}.npy")
            label_path = os.path.join(self.trainer.args.output_dir, f"label_step{self.trainer.state.global_step}.npy")
            prediction_path = os.path.join(self.trainer.args.output_dir, f"prediction_step{self.trainer.state.global_step}.npy")

            np.save(entropy_path, total_entropies)
            np.save(reward_margin_path, total_abs_reward_margins)
            np.save(label_path, total_labels)
            np.save(prediction_path, total_predictions)
            
            print(f'Alignment accuracy between SFT and Judge: {np.mean((total_labels == total_predictions)).item()}')
            
        return args_to_update
    
