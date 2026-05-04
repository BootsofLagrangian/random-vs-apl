import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from .strategy import Strategy
from ..utils import merge_prompt_or_completion


class ADPO(Strategy):
    def __init__(self, trainer, system_prompt, active_args):
        super(ADPO, self).__init__(trainer, system_prompt, active_args)
        
        # ADPO-specific hyperparameters
        self.epsilon = getattr(active_args, 'epsilon', 1e-3)
        self.linear_pool = getattr(active_args, 'linear_pool', 'mean')
        self.fitting_epochs = getattr(active_args, 'fitting_epochs', 10)
        
        # Will be set during query()
        self.linearized_policy = None
        
    def query(self, n):
        """
        Main ADPO query function following Algorithm 1.
        
        Args:
            n (int): Number of data points to select
            
        Returns:
            dict: Selected indices and their data for update_data()
        """
        # Step 1: Sample candidate subset
        unlabeled_indices = self.trainer.extract_unlabeled_indices()
        subset_size = min(n * self.subset_factor, len(unlabeled_indices))
        
        # assert subset_size > n, f"unlabeled set size should be larger than query size n"
        # subset_unlabeled_indices = np.random.choice(unlabeled_indices, size=subset_size, replace=False)
        
        # unlabeled_dataloader = self.trainer.get_unlabeled_dataloader(
        #     subset_unlabeled_indices, batch_size=self.active_args.optimal_batch_size)
        
        # num_unlabeled = len(subset_unlabeled_indices)
        
        
        if len(unlabeled_indices) < subset_size:
            subset_indices = unlabeled_indices
        else:
            subset_indices = np.random.choice(
                unlabeled_indices, 
                size=subset_size, 
                replace=False
            )
        
        # Step 2: Create dataloader for candidates
        subset_dataset = self.trainer.train_dataset.select(subset_indices)
        subset_dataloader = DataLoader(
            subset_dataset,
            batch_size=self.trainer.args.per_device_eval_batch_size,
            collate_fn=self.trainer.data_collator,
            shuffle=False
        )
        
        # Step 3: Build and fit linearized policy
        self._build_linearized_policy()
        self._fit_linearized_policy()
        
        # Step 4: Extract features for all candidates
        features = self._extract_features(subset_dataloader).to(self.device)
        
        # Step 5: Run greedy D-optimal selection
        selected_local_indices = self._greedy_d_optimal_selection(features, n)
        selected_global_indices = subset_indices[selected_local_indices]
        
        # Step 6: Generate completions for selected items
        selected_dataset = self.trainer.train_dataset.select(selected_global_indices)
        selected_dataloader = DataLoader(
            selected_dataset,
            batch_size=self.trainer.args.per_device_eval_batch_size,
            collate_fn=self.trainer.data_collator,
            shuffle=False
        )
        
        # Generate responses for selected items
        total_completion_ids = []
        total_completion_mask = []
        all_prompts = []
        
        model = self.prepare_model(selected_dataloader)
        for inputs in tqdm(selected_dataloader, desc='ADPO: Generating responses for selected items'):
            with torch.no_grad():
                response_dict = self.generate_responses(model, inputs, n=2)
                
            all_prompts.extend(response_dict['prompt'])
            total_completion_ids.append(response_dict['completion_ids'])
            total_completion_mask.append(response_dict['completion_mask'])
        
        # Step 7: Prepare return data
        total_batch_size = len(selected_global_indices)
        total_completion_ids = merge_prompt_or_completion(total_completion_ids, self.trainer.accelerator.num_processes, total_batch_size)
        total_completion_mask = merge_prompt_or_completion(total_completion_mask, self.trainer.accelerator.num_processes, total_batch_size)
        
        args_to_update = {
            'topk_indices': np.arange(total_batch_size),
            'selected_indices': selected_global_indices,
            'prompt': all_prompts,
            'completion_ids': total_completion_ids,
            'completion_mask': total_completion_mask,
        }
        
        args_to_update = self._prepare_for_data_update(args_to_update, total_batch_size)
        
        # Step 8: Clean up linearized policy
        self._cleanup_linearized_policy()

        return args_to_update
    
    def _build_linearized_policy(self):
        """Build the temporary linearized policy for feature extraction."""
        # Clone the backbone model
        backbone = self.trainer.model
        hidden_size = backbone.config.hidden_size
        
        # Create linearized policy wrapper
        class LinearizedPolicy(nn.Module):
            def __init__(self, backbone_model, pool_method="mean"):
                super().__init__()
                self.backbone = backbone_model
                self.pool = pool_method
                
                # Freeze all backbone parameters
                for param in self.backbone.parameters():
                    param.requires_grad = False
                
                # Add trainable linear head with same dtype as backbone
                self.logit_head = nn.Linear(hidden_size, 1, bias=True)
                # Convert to same dtype as backbone model
                self.logit_head = self.logit_head.to(dtype=next(backbone_model.parameters()).dtype)
            
            def forward(self, input_ids, attention_mask, **kwargs):
                # Get hidden states from backbone
                with torch.no_grad():
                    outputs = self.backbone(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        **kwargs
                    )
                
                last_hidden = outputs.hidden_states[-1]  # (B, L, H)
                
                # Pool over sequence
                if self.pool == "mean":
                    # Mean pooling over valid tokens
                    pooled = (last_hidden * attention_mask.unsqueeze(-1)).sum(1)
                    pooled = pooled / attention_mask.sum(1, keepdim=True)
                elif self.pool == "cls":
                    pooled = last_hidden[:, 0]  # CLS token
                else:
                    raise ValueError(f"Unknown pooling method: {self.pool}")
                
                # Get logits and return both logits and features
                logits = self.logit_head(pooled).squeeze(-1)
                return logits, pooled
        
        # Create the linearized policy
        self.linearized_policy = LinearizedPolicy(backbone, self.linear_pool)
        self.linearized_policy = self.linearized_policy.to(self.device)
        
        # Create optimizer for only the linear head
        trainable_params = [p for p in self.linearized_policy.parameters() if p.requires_grad]
        self.linear_optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)

    def _fit_linearized_policy(self):
        """Optionally fit the linear head on current labeled data."""
        if len(self.trainer.labeled_indices) == 0:
            # No labeled data to fit on
            return
        
        # Get a small sample of labeled data for fitting
        labeled_dataset = self.trainer.train_dataset.select(self.trainer.labeled_indices)
        fit_dataloader = DataLoader(
            labeled_dataset,
            batch_size=min(8, len(labeled_dataset)),
            collate_fn=self.trainer.data_collator,
            shuffle=True
        )
        
        
        # fit_dataloader = self.trainer.get_active_dataloader(self.trainer.prev_labeled_indices)
        # prompts = [self.index_to_responses[int(idx)]['prompt'] for idx in selected_indices]
        #     prompt_ids, prompt_mask = self._generate_inputs(prompts)
                        
        #     completion_ids = torch.stack([self.index_to_responses[int(idx)]['completion_ids'] for idx in selected_indices]) # B x 2 x L
        #     completion_mask = torch.stack([self.index_to_responses[int(idx)]['completion_mask'] for idx in selected_indices]) # B x 2 x L
                        
        #     completion_ids = torch.cat([completion_ids[:,0], completion_ids[:,1]], dim=0).to(device) # (B x 2) x L
        #     completion_mask = torch.cat([completion_mask[:,0], completion_mask[:,1]], dim=0).to(device) # (B x 2) x L
            
        self.linearized_policy.train()
        
        for step in range(self.fitting_epochs):
            total_loss = 0
            for batch in fit_dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Get prompt and completion inputs
                if "prompt_input_ids" in batch and "chosen_input_ids" in batch and "rejected_input_ids" in batch:
                    prompt_ids = batch["prompt_input_ids"]
                    chosen_ids = batch["chosen_input_ids"] 
                    rejected_ids = batch["rejected_input_ids"]
                    
                    # Concatenate prompt + completion
                    chosen_full = torch.cat([prompt_ids, chosen_ids], dim=1)
                    rejected_full = torch.cat([prompt_ids, rejected_ids], dim=1)
                    
                    chosen_mask = torch.ones_like(chosen_full)
                    rejected_mask = torch.ones_like(rejected_full)
                    
                    # Forward pass
                    chosen_logits, _ = self.linearized_policy(chosen_full, chosen_mask)
                    rejected_logits, _ = self.linearized_policy(rejected_full, rejected_mask)
                    
                    # Simple preference loss (DPO-style)
                    loss = -torch.log(torch.sigmoid(chosen_logits - rejected_logits)).mean()
                    
                    self.linear_optimizer.zero_grad()
                    loss.backward()
                    self.linear_optimizer.step()
                    
                    total_loss += loss.item()
            
            if step % 5 == 0:
                print(f"Fitting step {step}: loss = {total_loss:.4f}")
        
        self.linearized_policy.eval()

    def _extract_features(self, dataloader):
        """Extract features for D-optimal computation."""
        all_features = []
        
        self.linearized_policy.eval()
        
        for batch in tqdm(dataloader, desc="ADPO: Extracting features"):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            with torch.no_grad():
                # Generate two responses for each prompt
                response_dict = self.generate_responses(
                    self.trainer.model, batch, n=2, decode_completion=False
                )
                
                # Get features for each response pair
                batch_features = self._get_pair_features(response_dict)
                all_features.append(batch_features.cpu())
        
        # Concatenate all features
        features = torch.cat(all_features, dim=0)  # (subset_size, hidden_size)
        return features

    def _get_pair_features(self, response_dict):
        """Get difference features for response pairs."""
        prompt_ids = response_dict["prompt_ids"]
        prompt_mask = response_dict["prompt_mask"] 
        completion_ids = response_dict["completion_ids"]
        completion_mask = response_dict["completion_mask"]
        
        batch_size = prompt_ids.size(0) // 2
        
        # Split into two responses per prompt
        comp_a, comp_b = completion_ids.chunk(2, dim=0)
        mask_a, mask_b = completion_mask.chunk(2, dim=0)
        prompt_a = prompt_ids[:batch_size]
        prompt_b = prompt_ids[batch_size:]
        pmask_a = prompt_mask[:batch_size] 
        pmask_b = prompt_mask[batch_size:]
        
        # Concatenate prompt + completion
        full_a = torch.cat([prompt_a, comp_a], dim=1)
        full_b = torch.cat([prompt_b, comp_b], dim=1)
        full_mask_a = torch.cat([pmask_a, mask_a], dim=1)
        full_mask_b = torch.cat([pmask_b, mask_b], dim=1)
        
        # Get features from linearized policy
        with torch.no_grad():
            _, feat_a = self.linearized_policy(full_a, full_mask_a)
            _, feat_b = self.linearized_policy(full_b, full_mask_b)
        
        # Compute difference features (following Algorithm 1)
        diff_features = feat_a - feat_b
        
        # Optional: scale by beta (DPO parameter)
        if hasattr(self.trainer, 'beta'):
            diff_features = diff_features * self.trainer.beta
        
        return diff_features

    def _greedy_d_optimal_selection(self, features, n):
        """Run greedy D-optimal selection algorithm."""
        m, d = features.shape  # m candidates, d-dimensional features
        
        # Initialize inverse covariance matrix with same dtype as features
        C_inv = torch.eye(d, device=features.device, dtype=features.dtype) / self.epsilon
        
        selected_indices = []
        available_mask = torch.ones(m, dtype=torch.bool, device=features.device)
        
        for i in tqdm(range(min(n, m)), desc='D-optimal selection'):
            # Compute scores for all available candidates
            # Score = f_i^T C^{-1} f_i (gain in log-determinant)
            scores = torch.zeros(m, device=features.device, dtype=features.dtype)
            scores.fill_(-float('inf'))  # Mask unavailable candidates
            
            # for j in range(m):
            #     if available_mask[j]:
            #         f_j = features[j:j+1]  # (1, d)
            #         score = (f_j @ C_inv @ f_j.T).item()
            #         scores[j] = score
            active_features = features[available_mask]  # shape (m', d)
            quad_scores = torch.einsum('nd,dk,nk->n', active_features, C_inv, active_features)  # shape (m',)

            # fill the scores
            scores[available_mask] = quad_scores

            # Select candidate with highest score
            best_idx = torch.argmax(scores).item()
            selected_indices.append(best_idx)
            available_mask[best_idx] = False
            
            # Update C^{-1} using Sherman-Morrison formula
            # C_new^{-1} = C^{-1} - (C^{-1} f f^T C^{-1}) / (1 + f^T C^{-1} f)
            f_best = features[best_idx:best_idx+1]  # (1, d)
            Cf = C_inv @ f_best.T  # (d, 1)
            denominator = 1 + (f_best @ Cf).item()
            
            if denominator > 1e-12:  # Avoid numerical issues
                C_inv = C_inv - (Cf @ Cf.T) / denominator
            
            if (i+1) % 10 == 0:
                print(f"ADPO selection: {i+1}/{n} selected")
        
        return selected_indices
    
    def _cleanup_linearized_policy(self):
        """Clean up the temporary linearized policy."""
        if self.linearized_policy is not None:
            del self.linearized_policy
            self.linearized_policy = None
        torch.cuda.empty_cache()
