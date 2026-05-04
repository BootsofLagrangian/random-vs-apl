import torch
import torch.optim.lr_scheduler
import numpy as np
import einops
import torch.nn.functional as F
import random
import jinja2

from .strategy import Strategy
from ..utils import merge_prompt_or_completion, safe_gather
from accelerate.utils import gather_object
from ..utils import broadcast_subset, gather_2d_once
from torch.utils.data import DataLoader
from transformers.training_args import OptimizerNames
from transformers import is_apex_available, AutoTokenizer
from trl.data_utils import is_conversational, apply_chat_template
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE, get_reward, truncate_right
from trl.trainer.erm_trainer import EnsembleRewardModel, DebertaV2PairRM
from transformers import TrainerCallback

if is_apex_available():
    from apex import amp

class SimpleDataset:
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

class RewardModelTrainingCallback(TrainerCallback):
    """Callback to train the reward model after each training step."""
    
    def __init__(self, sea_strategy):
        self.sea_strategy = sea_strategy
    
    def on_step_end(self, args, state, control, **kwargs):
        """Called after each training step."""
        # Train reward model on main process only, then sync weights across all processes
        self.sea_strategy.train_reward_model()

class HybridJudge:
    """A judge that combines an external labeler (judge or reward model) with SEA's internal reward model."""
    
    def __init__(self, external_judge=None, external_reward_model=None, external_reward_processing_class=None, erm=None, processing_class=None):
        if external_judge is not None and external_reward_model is not None:
            raise ValueError("Cannot provide both external_judge and external_reward_model. Choose one.")
        if external_judge is None and external_reward_model is None:
            raise ValueError("Must provide either external_judge or external_reward_model.")
        
        self.external_judge = external_judge
        self.external_reward_model = external_reward_model
        self.external_reward_processing_class = external_reward_processing_class
        self.erm = erm
        self.processing_class = processing_class
        self.gamma = 1.0
    
        
    def judge(self, prompts, completions):
        """Judge using hybrid approach: some samples from external labeler, others from internal reward model."""
        batch_size = len(prompts)
        results = []
        judge_labeled_data = []  
        num_external_samples = int(batch_size * self.gamma)
        external_indices = set(np.random.choice(batch_size, size=num_external_samples, replace=False))
                
        for i, (prompt, (comp1, comp2)) in enumerate(zip(prompts, completions)):
            if i in external_indices:
                # Use external labeler for this sample
                if self.external_judge is not None:
                    judge_result = self.external_judge.judge([prompt], [(comp1, comp2)])
                    label = judge_result[0]
                    results.append(label)
                else:
                    label = self._judge_with_reward_model(prompt, comp1, comp2)
                    results.append(label)
                
                judge_labeled_data.append({
                    'prompt': prompt,
                    'completion_1': comp1,
                    'completion_2': comp2,
                    'label': label
                })
            else:                
                self.erm.eval()
                
                with torch.no_grad():
                    features = self.erm.get_features([prompt, prompt], [comp1, comp2])
                    features = features.reshape(2, 1, -1)
                    rewards = self.erm.get_rewards(features).flatten()
                
                label = 0 if rewards[0] > rewards[1] else 1
                results.append(label)

        return results, judge_labeled_data
    
    def _judge_with_reward_model(self, prompt, comp1, comp2):
        """Use external reward model to judge between two completions."""
        device = next(self.external_reward_model.parameters()).device
        
        prompts = [prompt, prompt]
        completions = [comp1, comp2]
        
        if is_conversational({"prompt": prompt}):
            examples = [{"prompt": p, "completion": c} for p, c in zip(prompts, completions)]
            examples = [apply_chat_template(example, self.external_reward_processing_class) for example in examples]
            prompts = [example["prompt"] for example in examples]
            completions = [example["completion"] for example in examples]

        prompts_ids = self.external_reward_processing_class(
            prompts, padding=True, return_tensors="pt", padding_side="left"
        )["input_ids"].to(device)
        context_length = prompts_ids.shape[1]

        completions_ids = self.external_reward_processing_class(
            completions, padding=True, return_tensors="pt", padding_side="right"
        )["input_ids"].to(device)

        prompt_completion_ids = torch.cat((prompts_ids, completions_ids), dim=1)
        
        with torch.inference_mode():
            _, scores, _ = get_reward(
                self.external_reward_model, prompt_completion_ids, 
                self.external_reward_processing_class.pad_token_id, context_length
            )
        return 0 if scores[0] >= scores[1] else 1

class SEA(Strategy):
    def __init__(self, trainer, system_prompt, config):
        super(SEA, self).__init__(trainer, system_prompt, config)
        self.m = 20
        assert(self.m % 2 == 0)

        # Set config attributes with defaults
        self.max_buffer_size = 50000
        self.max_reward_iterations = 5
        self.num_ensemble = 10
        self.reg_lambda = 0.5
        self.scheduler_t_max = 5000
        
        self.num_encountered_examples = 0

        self.build_reward_model()

        # Initialize judge dataset as internal attributes
        self.judge_data = []
        self.judge_tokenizer = self.erm.tokenizer
        
        if hasattr(self.trainer, 'judge') and self.trainer.judge is not None:
            print("SEA: Auto-initializing hybrid labeling with external judge")
            print(f"SEA: Judge type: {type(self.trainer.judge).__name__}")
            
            # Create hybrid judge with external judge
            self.hybrid_judge = HybridJudge(
                external_judge=self.trainer.judge,
                erm=self.erm,
                processing_class=self.trainer.processing_class,
            )
         
        elif hasattr(self.trainer, 'reward_model') and self.trainer.reward_model is not None:
            print("SEA: Auto-initializing hybrid labeling with external reward model")
            print(f"SEA: Reward model type: {type(self.trainer.reward_model).__name__}")
            
            # Create hybrid judge with external reward model
            self.hybrid_judge = HybridJudge(
                external_reward_model=self.trainer.reward_model,
                external_reward_processing_class=self.trainer.reward_processing_class,
                erm=self.erm,
                processing_class=self.trainer.processing_class,
            )
            
        else:
            raise ValueError("SEA requires either a judge or reward_model in the trainer for hybrid labeling")

        self.trainer.training_step = self.training_step
        self.trainer.add_callback(RewardModelTrainingCallback(self))

    def _process_judge_buffer_items(self, buffer_items):
        """Convert buffer items to dataset format."""
        for item in buffer_items:
            prompt = item['prompt']
            comp1 = item['completion_1']
            comp2 = item['completion_2']
            label = item['label']

            if label == 0:  # First completion is better
                self.judge_data.append({"prompt": prompt, "chosen": comp1, "rejected": comp2})
            else:  # Second completion is better
                self.judge_data.append({"prompt": prompt, "chosen": comp2, "rejected": comp1})

    def _update_judge_dataset(self, new_buffer_items):
        """Update the dataset with new buffer items."""
        if not new_buffer_items:
            return
            
        self._process_judge_buffer_items(new_buffer_items)
        
        if len(self.judge_data) > self.max_buffer_size:
            items_to_remove = len(self.judge_data) - self.max_buffer_size
            removed_items = self.judge_data[:items_to_remove]
            self.judge_data = self.judge_data[items_to_remove:]
            del removed_items

    def _get_judge_dataset_len(self):
        return len(self.judge_data)

    def _get_judge_dataset_item(self, i):
        return self.judge_data[i]

    def _append_to_stats(self, key, value, max_length=100):
        """Helper method to append to stats with automatic cleanup to prevent memory leaks."""
        if key not in self.trainer.stats:
            self.trainer.stats[key] = []
        self.trainer.stats[key].append(value)
        if len(self.trainer.stats[key]) > max_length:
            self.trainer.stats[key] = self.trainer.stats[key][-max_length:]

    def build_reward_model(self):
        reward_model = DebertaV2PairRM.from_pretrained("llm-blender/PairRM-hf")
        reward_tokenizer = AutoTokenizer.from_pretrained('llm-blender/PairRM-hf')
        
        if reward_tokenizer.pad_token is None:
            if reward_tokenizer.eos_token is not None:
                reward_tokenizer.pad_token = reward_tokenizer.eos_token
            else:
                reward_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                reward_model.resize_token_embeddings(len(reward_tokenizer))
        
        if hasattr(reward_model, 'config'):
            reward_model.config.pad_token_id = reward_tokenizer.pad_token_id
        
        self.erm = EnsembleRewardModel(
            reward_model, 
            reward_tokenizer, 
            batch_size=self.trainer.args.per_device_train_batch_size,
            num_ensemble=self.num_ensemble,
        ).to(self.trainer.model.device)

        trainable_params = [p for p in self.erm.parameters() if p.requires_grad]
        # Store initial parameters for regularization
        self.init_params = {}
        for name, param in self.erm.named_parameters():
            if param.requires_grad:
                self.init_params[name] = param.clone().detach()
        
        self.erm_optimizer = torch.optim.AdamW(trainable_params, lr=1e-5)
        self.erm_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.erm_optimizer, 
            T_max=self.scheduler_t_max,
            eta_min=1e-7
        )
    
    def regularization(self):
        """Prior towards independent initialization."""
        reg_loss = 0.0
        count = 0
        
        for name, param in self.erm.named_parameters():
            if param.requires_grad and name in self.init_params:
                # Ensure both tensors are on same device
                init_param = self.init_params[name].to(param.device)
                param_diff = (param - init_param) ** 2
                param_reg_loss = param_diff.sum()
                reg_loss += param_reg_loss
                count += param.numel()
        
        if count > 0:
            return reg_loss / count  # Mean squared deviation
        else:
            return torch.tensor(0.0, device=next(self.erm.parameters()).device, requires_grad=True)
    
    def train_reward_model(self):
        # Only main process does the actual training
        if self.trainer.accelerator.is_main_process:
            self.erm.reward_heads.train()
           
            total_loss_sum = 0
            total_rew_loss_sum = 0
            total_reg_loss_sum = 0
            num_batches = 0
            total_chosen_rewards = 0
            total_rejected_rewards = 0
            total_margin = 0
            total_correct_predictions = 0
            total_samples = 0
            
            if len(self.judge_data) > self.trainer.args.per_device_train_batch_size * self.trainer.accelerator.num_processes:
                total_batch_size = self.trainer.args.per_device_train_batch_size * self.trainer.accelerator.num_processes
                                
                judge_dataset = SimpleDataset(self.judge_data)
                judge_dataloader = DataLoader(judge_dataset, batch_size=total_batch_size, shuffle=True)
                
                for batch_idx, batch in enumerate(judge_dataloader):
                    if batch_idx >= self.max_reward_iterations:
                        break
                        
                    batch = {k: v.to(self.trainer.model.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}

                    with torch.no_grad():
                        chosen_features = self.erm.get_features(batch["prompt"], batch["chosen"])
                        chosen_features = chosen_features.reshape(len(batch["prompt"]), 1, -1)
                        rejected_features = self.erm.get_features(batch["prompt"], batch["rejected"])
                        rejected_features = rejected_features.reshape(len(batch["prompt"]), 1, -1)
                    chosen_rewards = self.erm.get_rewards(chosen_features)
                    rejected_rewards = self.erm.get_rewards(rejected_features)

                    loss_rew = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
                    
                    # Add regularization loss
                    loss_reg = (
                        self.reg_lambda
                        * total_batch_size
                        / max(self.num_encountered_examples, 1)
                        * self.regularization()
                    )
                    
                    total_loss = loss_rew + loss_reg
                    
                    self.erm_optimizer.zero_grad()
                    total_loss.backward()
                    
                    self.erm_optimizer.step()
                    self.erm_scheduler.step()

                    chosen_rewards_mean = chosen_rewards.mean().item()
                    rejected_rewards_mean = rejected_rewards.mean().item()
                    reward_margin = chosen_rewards_mean - rejected_rewards_mean
                    
                    # Accumulate accuracy statistics for this batch
                    batch_correct_predictions = (chosen_rewards > rejected_rewards).float().sum().item()
                    batch_samples = chosen_rewards.size(0)
                    
                    total_loss_sum += total_loss.item()
                    total_rew_loss_sum += loss_rew.item()
                    total_reg_loss_sum += loss_reg.item()
                    total_chosen_rewards += chosen_rewards_mean
                    total_rejected_rewards += rejected_rewards_mean
                    total_margin += reward_margin
                    total_correct_predictions += batch_correct_predictions
                    total_samples += batch_samples
                    num_batches += 1

                if num_batches > 0:
                    avg_total_loss = total_loss_sum / num_batches
                    avg_rew_loss = total_rew_loss_sum / num_batches
                    avg_reg_loss = total_reg_loss_sum / num_batches
                    avg_chosen_rewards = total_chosen_rewards / num_batches
                    avg_rejected_rewards = total_rejected_rewards / num_batches
                    avg_margin = total_margin / num_batches
                    
                    # Calculate accuracy across all batches
                    overall_accuracy = total_correct_predictions / total_samples if total_samples > 0 else 0.0
                    
                    self._append_to_stats("epistemic_reward_model/total_loss", avg_total_loss)
                    self._append_to_stats("epistemic_reward_model/reward_loss", avg_rew_loss)
                    self._append_to_stats("epistemic_reward_model/regularization_loss", avg_reg_loss)
                    self._append_to_stats("epistemic_reward_model/chosen_rewards", avg_chosen_rewards)
                    self._append_to_stats("epistemic_reward_model/rejected_rewards", avg_rejected_rewards)
                    self._append_to_stats("epistemic_reward_model/margin", avg_margin)
                    self._append_to_stats("epistemic_reward_model/num_batches", num_batches)
                    self._append_to_stats("epistemic_reward_model/dataset_size", len(self.judge_data))
                    self._append_to_stats("epistemic_reward_model/accuracy", overall_accuracy)
                                        
                self.erm.reward_heads.eval()
            else:
                self._append_to_stats("epistemic_reward_model/skipped", 1)
                self._append_to_stats("epistemic_reward_model/dataset_size", len(self.judge_data))

        # ALL processes (including non-main) call synchronization
        self._synchronize_reward_model_weights()

    def _synchronize_reward_model_weights(self):
        """Synchronize reward head weights from main process to all other processes."""
        self.trainer.accelerator.wait_for_everyone()
        
        if self.trainer.accelerator.num_processes > 1:
            from accelerate.utils import broadcast_object_list
            
            if self.trainer.accelerator.is_main_process:
                # Only broadcast the trained reward heads
                reward_heads_state = self.erm.reward_heads.state_dict()
                object_list = [reward_heads_state]
            else:
                object_list = [None]
            
            broadcast_object_list(object_list, from_process=0)
            
            if not self.trainer.accelerator.is_main_process:
                received_state = object_list[0]
                if received_state is not None:
                    self.erm.reward_heads.load_state_dict(received_state)
        
        self.trainer.accelerator.wait_for_everyone()

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Custom training step that supports hybrid preference labeling."""
        model.train()
        prompts = inputs["prompt"]
        batch_size = len(prompts)
        
        if "completion_ids" in self.trainer.train_dataset.column_names and "completion_mask" in self.trainer.train_dataset.column_names:
            prompts, prompt_ids, prompt_mask, completion_ids, completion_mask = self.trainer._load_generated_inputs(inputs)
        else:
            if self.trainer.args.use_vllm:
                prompt_ids, prompt_mask, completion_ids, completion_mask = self.trainer._generate_vllm(model, prompts)
            else:
                prompt_ids, prompt_mask, completion_ids, completion_mask = self.trainer._generate(model, prompts)

        contain_eos_token = torch.any(completion_ids == self.trainer.processing_class.eos_token_id, dim=-1)

        logprobs = self.trainer._forward(model, prompt_ids, prompt_mask, completion_ids, completion_mask)
        with torch.no_grad():
            if self.trainer.ref_model is not None:
                ref_logprobs = self.trainer._forward(self.trainer.ref_model, prompt_ids, prompt_mask, completion_ids, completion_mask)
            else:  # peft case: we just need to disable the adapter
                with self.trainer.model.disable_adapter():
                    ref_logprobs = self.trainer._forward(self.trainer.model, prompt_ids, prompt_mask, completion_ids, completion_mask)

        device = logprobs.device
        completions = self.trainer.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational({"prompt": prompts[0]}):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]
        
        judge_prompts = prompts.copy()
        judge_completions = list(zip(completions[:batch_size], completions[batch_size:]))

        if is_conversational({"prompt": prompts[0]}):
            environment = jinja2.Environment()
            template = environment.from_string(SIMPLE_CHAT_TEMPLATE)
            judge_prompts = [template.render(messages=prompt) for prompt in judge_prompts]
            judge_completions = [
                (template.render(messages=comp1), template.render(messages=comp2))
                for comp1, comp2 in judge_completions
            ]
        
        ranks_of_first_completion, judge_labeled_data = self.hybrid_judge.judge(judge_prompts, judge_completions)
        
        # Convert ranks to mask
        mask = torch.tensor([rank == 0 for rank in ranks_of_first_completion], device=device)

        # ===== CONTINUE WITH STANDARD DPO LOSS COMPUTATION =====
        batch_range = torch.arange(batch_size, device=device)
        chosen_indices = batch_range + (~mask * batch_size)
        rejected_indices = batch_range + (mask * batch_size)

        cr_indices = torch.cat((chosen_indices, rejected_indices), dim=0)
        cr_logprobs = logprobs[cr_indices]
        cr_ref_logprobs = ref_logprobs[cr_indices]

        padding_mask = ~completion_mask.bool()
        cr_padding_mask = padding_mask[cr_indices]

        cr_logprobs_sum = (cr_logprobs * ~cr_padding_mask).sum(1)
        cr_ref_logprobs_sum = (cr_ref_logprobs * ~cr_padding_mask).sum(1)

        chosen_logprobs_sum, rejected_logprobs_sum = torch.split(cr_logprobs_sum, batch_size)
        chosen_ref_logprobs_sum, rejected_ref_logprobs_sum = torch.split(cr_ref_logprobs_sum, batch_size)
        pi_logratios = chosen_logprobs_sum - rejected_logprobs_sum
        ref_logratios = chosen_ref_logprobs_sum - rejected_ref_logprobs_sum

        logits = pi_logratios - ref_logratios

        if self.trainer.args.loss_type == "sigmoid":
            losses = -F.logsigmoid(self.trainer.beta * logits)
        elif self.trainer.args.loss_type == "ipo":
            losses = (logits - 1 / (2 * self.trainer.beta)) ** 2
        else:
            raise NotImplementedError(f"invalid loss type {self.trainer.args.loss_type}")

        loss = losses.mean()

        # Logging
        self._append_to_stats("val/contain_eos_token", contain_eos_token.float().mean().item())
        self._append_to_stats("logps/chosen", 
            self.trainer.accelerator.gather_for_metrics(chosen_logprobs_sum).mean().item()
        )
        self._append_to_stats("logps/rejected",
            self.trainer.accelerator.gather_for_metrics(rejected_logprobs_sum).mean().item()
        )

        kl = logprobs - ref_logprobs
        mean_kl = kl.sum(1).mean()
        self._append_to_stats("objective/kl",
            self.trainer.accelerator.gather_for_metrics(mean_kl).mean().item()
        )
        non_score_reward = (-self.trainer.beta * kl).sum(1)
        mean_non_score_reward = non_score_reward.mean()
        self._append_to_stats("objective/non_score_reward",
            self.trainer.accelerator.gather_for_metrics(mean_non_score_reward).mean().item()
        )
        
        mean_entropy = -logprobs.sum(1).mean()
        self._append_to_stats("objective/entropy",
            self.trainer.accelerator.gather_for_metrics(mean_entropy).mean().item()
        )
        
        chosen_rewards = self.trainer.beta * (chosen_logprobs_sum - chosen_ref_logprobs_sum)
        gathered_chosen_rewards = self.trainer.accelerator.gather_for_metrics(chosen_rewards)
        self._append_to_stats("rewards/chosen", gathered_chosen_rewards.mean().item())
        
        rejected_rewards = self.trainer.beta * (rejected_logprobs_sum - rejected_ref_logprobs_sum)
        gathered_rejected_rewards = self.trainer.accelerator.gather_for_metrics(rejected_rewards)
        self._append_to_stats("rewards/rejected", gathered_rejected_rewards.mean().item())
        
        margin = gathered_chosen_rewards - gathered_rejected_rewards
        self._append_to_stats("rewards/margins", margin.mean().item())
        accuracy = margin > 0
        self._append_to_stats("rewards/accuracies", accuracy.float().mean().item())
        
        self._append_to_stats("beta", self.trainer.beta)

        # Optimizer setup
        kwargs = {}
        if self.trainer.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self.trainer._get_learning_rate()

        if self.trainer.args.n_gpu > 1:
            loss = loss.mean()

        # Backward pass
        if self.trainer.use_apex:
            with amp.scale_loss(loss, self.trainer.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.trainer.accelerator.backward(loss, **kwargs)

        return loss.detach() / self.trainer.args.gradient_accumulation_steps
    
    def query(self, n):
        unlabeled_indices = self.trainer.extract_unlabeled_indices()
        accelerator = self.trainer.accelerator
        per_device_bs = self.active_args.optimal_batch_size
        subset_unlabeled_indices = broadcast_subset(
            unlabeled_indices,
            desired_size=int(n),
            accelerator=accelerator,
            per_device_bs=per_device_bs,
            k=1,
            seed=getattr(self.trainer.args, "seed", 42),
            enforce_multiple=False,
        ).cpu().numpy()
        unlabeled_dataloader = self.trainer.get_unlabeled_dataloader(subset_unlabeled_indices, per_device_bs)
        model = self.prepare_model(unlabeled_dataloader)
        
        total_prompts_local = []  
        total_completion_ids_local = []
        total_completion_mask_local = []
        for inputs in unlabeled_dataloader:
            with torch.no_grad():
                # Generate responses in batches of 2 to reduce VRAM usage
                batch_size = len(inputs["prompt"])
                all_completion_ids = []
                all_completion_mask = []
                all_completions = []
                
                # Generate self.m responses by generating 2 at a time
                for _ in range(self.m // 2):
                    response_dict = self.generate_responses(model, inputs, n=2)
                    
                    prompt_ids = response_dict["prompt_ids"]
                    prompt_mask = response_dict["prompt_mask"]
                    completion_ids = response_dict["completion_ids"]
                    completion_mask = response_dict["completion_mask"]
                    
                    eos_token_id = self.trainer.processing_class.eos_token_id
                    pad_token_id = self.trainer.processing_class.pad_token_id
                    completion_ids, completion_mask = truncate_right(completion_ids, eos_token_id, pad_token_id)
                    
                    if hasattr(self.trainer, 'max_length'):
                        max_length = self.trainer.max_length
                        num_tokens_to_truncate = max(prompt_ids.size(1) + completion_ids.size(1) - max_length, 0)
                        if num_tokens_to_truncate > 0:
                            # Truncate left to avoid oom
                            prompt_ids = prompt_ids[:, num_tokens_to_truncate:]
                            prompt_mask = prompt_mask[:, num_tokens_to_truncate:]
                    
                    # Decode completions for this batch
                    batch_completions = self.trainer.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
                    
                    # Reshape from (batch_size * 2, seq_len) to (batch_size, 2, seq_len)
                    completion_ids = completion_ids.reshape(batch_size, 2, -1)
                    completion_mask = completion_mask.reshape(batch_size, 2, -1)
                    
                    all_completion_ids.append(completion_ids)
                    all_completion_mask.append(completion_mask)
                    all_completions.extend(batch_completions)

                # Concatenate all completion batches: (batch_size, self.m, seq_len)
                completion_ids = torch.cat(all_completion_ids, dim=1)
                completion_mask = torch.cat(all_completion_mask, dim=1)
                completions = all_completions
                
                # Create prompts list for reward model features
                prompts = [prompt for prompt in inputs["prompt"] for _ in range(self.m)]
                
                features = self.erm.get_features(prompts, completions)
                features = features.reshape(batch_size, self.m, -1)

                rewards = self.erm.get_rewards(features, all_heads=True)
                E, M, _, _ = rewards.shape
                best_actions = rewards.argmax(dim=2)  # (E, M, 1)
                # sample without replacement
                s1 = list(range(E))
                random.shuffle(s1)
                first_actions = best_actions[s1[0]]

                pref_logits = rewards - einops.rearrange(
                    rewards, "e m n 1 -> e m 1 n"
                )  # (E, M, N, N')
                variances = pref_logits.var(dim=0)
                second_actions = torch.stack(
                    [variances[i][first_actions[i]].argmax() for i in range(M)], dim=0
                ).view(M, 1)

                selected_completion_ids_1 = completion_ids[torch.arange(batch_size), first_actions]
                selected_completion_mask_1 = completion_mask[torch.arange(batch_size), first_actions]
                selected_completion_ids_2 = completion_ids[torch.arange(batch_size), second_actions]
                selected_completion_mask_2 = completion_mask[torch.arange(batch_size), second_actions]

                merged_completion_ids = torch.stack([selected_completion_ids_1, selected_completion_ids_2], dim=1).reshape(-1, selected_completion_ids_1.shape[-1])
                merged_completion_mask = torch.stack([selected_completion_mask_1, selected_completion_mask_2], dim=1).reshape(-1, selected_completion_mask_1.shape[-1])

                total_prompts_local += inputs["prompt"]
                total_completion_ids_local.append(merged_completion_ids.detach().cpu())
                total_completion_mask_local.append(merged_completion_mask.detach().cpu())

        self.num_encountered_examples += n
        if self.num_encountered_examples >= 1000:
            self.hybrid_judge.gamma = 0.7
        
        # Concatenate completions via single gather across processes
        total_completion_ids_g, rows = gather_2d_once(total_completion_ids_local, accelerator, pad_token_id=self.trainer.processing_class.pad_token_id)
        total_completion_mask_g, _ = gather_2d_once(total_completion_mask_local, accelerator, pad_token_id=0)
        total_batch_size = len(subset_unlabeled_indices)
        total_prompts = gather_object(total_prompts_local)[: total_batch_size]
        total_completion_ids = merge_prompt_or_completion([total_completion_ids_g], self.trainer.accelerator.num_processes, total_batch_size=total_batch_size)
        total_completion_mask = merge_prompt_or_completion([total_completion_mask_g], self.trainer.accelerator.num_processes, total_batch_size=total_batch_size)
        
        # Update judge buffer with the selected completion pairs
        if self.trainer.accelerator.is_main_process and total_batch_size > 0:
            # Decode the selected completions for judge labeling
            selected_completions_1 = self.trainer.processing_class.batch_decode(
                total_completion_ids[::2], skip_special_tokens=True
            )
            selected_completions_2 = self.trainer.processing_class.batch_decode(
                total_completion_ids[1::2], skip_special_tokens=True
            )
            
            # Get original prompts (one per pair, not duplicated for each completion)
            judge_prompts = total_prompts
            judge_completions = list(zip(selected_completions_1, selected_completions_2))
            
            # Handle conversational format if needed
            if is_conversational({"prompt": judge_prompts[0]}) if judge_prompts else False:
                environment = jinja2.Environment()
                template = environment.from_string(SIMPLE_CHAT_TEMPLATE)
                judge_prompts = [template.render(messages=prompt) for prompt in judge_prompts]
                judge_completions = [
                    (template.render(messages=[{"role": "assistant", "content": comp1}]), 
                     template.render(messages=[{"role": "assistant", "content": comp2}]))
                    for comp1, comp2 in judge_completions
                ]
            
            # Get judge labels for the selected pairs
            _, judge_labeled_data = self.hybrid_judge.judge(judge_prompts, judge_completions)
            
            if judge_labeled_data:
                current_step = getattr(self.trainer.state, 'global_step', 0)
                for item in judge_labeled_data:
                    item['global_step'] = current_step
                    item['batch_count'] = current_step
                    item['labeling_type'] = 'query_step'
                
                # Update the judge dataset buffer
                self._update_judge_dataset(judge_labeled_data)
        
        args_to_update = {
            'topk_indices': torch.arange(total_batch_size),
            'completion_ids': total_completion_ids,
            'completion_mask': total_completion_mask,
            'prompt': total_prompts,
            'selected_indices': subset_unlabeled_indices,
        }
        
        args_to_update = self._prepare_for_data_update(args_to_update, total_batch_size)
            
        return args_to_update
