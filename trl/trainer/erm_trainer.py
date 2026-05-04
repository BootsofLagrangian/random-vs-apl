import torch
import torch.nn as nn
from transformers import Trainer
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2PreTrainedModel,
    DebertaV2Model,
    SequenceClassifierOutput
)
from typing import Optional, List, Tuple, Union
import torch.nn.functional as F
import einops

class DebertaV2PairRM(DebertaV2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        
        self.n_tasks = config.n_tasks
        self.drop_out = config.drop_out
        
        # LM
        self.pretrained_model = DebertaV2Model(config)
        self.hidden_size = config.hidden_size
        
        self.source_prefix_id = config.source_prefix_id
        self.cand_prefix_id = config.cand_prefix_id
        self.cand1_prefix_id = config.cand1_prefix_id
        self.cand2_prefix_id = config.cand2_prefix_id
        
        self.head_layer = nn.Sequential(
            nn.Dropout(self.drop_out),
            nn.Linear(2*self.hidden_size, 1*self.hidden_size),
            nn.Tanh(),
            nn.Dropout(self.drop_out),
            nn.Linear(1 * self.hidden_size, self.n_tasks),
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        #  <source_prefix_id>...<sep><cand1_prefix_id>...<sep><cand2_prefix_id> ... <sep>
        assert all([self.source_prefix_id in input_ids[i] for i in range(input_ids.shape[0])]), "<source> id not in input_ids"
        assert all([self.cand1_prefix_id in input_ids[i] for i in range(input_ids.shape[0])]), "<candidate1> id not in input_ids"
        assert all([self.cand2_prefix_id in input_ids[i] for i in range(input_ids.shape[0])]), "<candidate2> id not in input_ids"
        
        keep_column_mask = attention_mask.ne(0).any(dim=0)
        input_ids = input_ids[:, keep_column_mask]
        attention_mask = attention_mask[:, keep_column_mask]
        outputs = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=return_dict,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
        )
        encs = outputs.hidden_states[-1]
        source_idxs = torch.where(input_ids == self.source_prefix_id)
        source_encs = encs[source_idxs[0], source_idxs[1], :]
        cand1_idxs = torch.where(input_ids == self.cand1_prefix_id)
        cand1_encs = encs[cand1_idxs[0], cand1_idxs[1], :]
        cand2_idxs = torch.where(input_ids == self.cand2_prefix_id)
        cand2_encs = encs[cand2_idxs[0], cand2_idxs[1], :]
        
        # reduce
        source_cand1_encs = torch.cat([source_encs, cand1_encs], dim=-1)
        source_cand2_encs = torch.cat([source_encs, cand2_encs], dim=-1)
        left_pred_scores = self.head_layer(source_cand1_encs)
        right_pred_scores = self.head_layer(source_cand2_encs)

        loss = None
        if labels is not None:
            loss = self.compute_loss(left_pred_scores, right_pred_scores, labels)
        
        preds = (left_pred_scores - right_pred_scores).mean(dim=-1)
        return SequenceClassifierOutput(
            loss=loss, logits=preds, 
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions
        )

    def get_embs(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        
        #  <source_prefix_id>...<sep><cand_prefix_id>...<sep>
        assert all([self.source_prefix_id in input_ids[i] for i in range(input_ids.shape[0])]), "<source> id not in input_ids"
        assert all([self.cand_prefix_id in input_ids[i] for i in range(input_ids.shape[0])]), "<candidate> id not in input_ids"
        
        keep_column_mask = attention_mask.ne(0).any(dim=0)
        input_ids = input_ids[:, keep_column_mask]
        attention_mask = attention_mask[:, keep_column_mask]
        outputs = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
        )
        encs = outputs.hidden_states[-1]
        return encs
    
    def compute_loss(self, left_pred_scores, right_pred_scores, labels):
        """
        Args:
            left_pred_scores: [n_candidates, n_task]
            right_pred_scores: [n_candidates, n_task]
            labels: [n_candidates, n_task], 1/0/-1 for left/right/both is better
        """
        device = left_pred_scores.device
        loss = torch.tensor(0.0).to(left_pred_scores.device)
        
        dif_scores = labels
        left_pred_scores = left_pred_scores * dif_scores.sign()
        right_pred_scores = - right_pred_scores * dif_scores.sign()
        cls_loss = torch.tensor(0.0, device=device)
        cls_loss += - torch.log(torch.sigmoid(left_pred_scores+right_pred_scores)).mean()
        loss += cls_loss
        return loss

class EnsembleRewardModel(nn.Module):
    """Ensemble of reward models for uncertainty estimation."""

    def __init__(
        self,
        base_model,
        tokenizer,
        num_ensemble: int = 20,
        hidden_dim: int = 128,
        source_max_length: int = 128,
        max_length: int = 256,
        batch_size: int = 4,
    ):
        super().__init__()
        self.num_ensemble = num_ensemble
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.source_token = "<|source|>"
        self.candidate_token = "<|candidate|>"
        self.source_max_length = source_max_length
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Define token IDs for preprocessing
        self.source_prefix_id = tokenizer.convert_tokens_to_ids(self.source_token)
        self.cand_prefix_id = tokenizer.convert_tokens_to_ids(self.candidate_token)
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        self.reward_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        base_model.config.hidden_size*2, hidden_dim
                    ),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(num_ensemble)
            ]
        )
        
        # Match the dtype of the reward heads to the base model
        base_model_dtype = next(base_model.parameters()).dtype
        self.reward_heads = self.reward_heads.to(base_model_dtype)

    @property
    def device(self):
        """Return the device of the model."""
        return next(self.base_model.parameters()).device

    def preprocess(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        #  <source_prefix_id>...<sep><cand_prefix_id>...<sep>
        if self.source_prefix_id is not None:
            assert all(
                [
                    self.source_prefix_id in input_ids[i]
                    for i in range(input_ids.shape[0])
                ]
            ), "<source> id not in input_ids"
        if self.cand_prefix_id is not None:
            assert all(
                [self.cand_prefix_id in input_ids[i] for i in range(input_ids.shape[0])]
            ), "<candidate> id not in input_ids"

        keep_column_mask = attention_mask.ne(0).any(dim=0)
        input_ids = input_ids[:, keep_column_mask]
        attention_mask = attention_mask[:, keep_column_mask]
        return input_ids, attention_mask
    
    def postprocess(self, encs, input_ids: torch.Tensor):
        source_idxs = torch.where(input_ids == self.source_prefix_id)
        source_encs = encs[source_idxs[0], source_idxs[1], :]
        cand_idxs = torch.where(input_ids == self.cand_prefix_id)
        cand_encs = encs[cand_idxs[0], cand_idxs[1], :]

        # reduce
        source_cand_encs = torch.cat([source_encs, cand_encs], dim=-1)
        return source_cand_encs.detach()
        
    def tokenize_pair(self, prompt:str, candidate:str, source_max_length: int, max_length: int):
        source_ids = self.tokenizer.encode(
            self.source_token + prompt,
            max_length=source_max_length,
            truncation=True,
            add_special_tokens=False,
        )
        candidate_max_length = max_length - len(source_ids)
        candidate_ids = self.tokenizer.encode(
            self.candidate_token + candidate,
            max_length=candidate_max_length,
            truncation=True,
            add_special_tokens=False,
        )
        return source_ids + candidate_ids
    
    def get_feature(self, input_ids, attention_mask):

        input_ids, attention_mask = self.preprocess(input_ids, attention_mask)

        hidden_states = self.base_model.get_embs(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return self.postprocess(hidden_states, input_ids)
    
    def get_features(
        self,
        prompts: List[str],
        candidates: List[str],
    ):
        input_ids = []
        for p, c in zip(prompts, candidates):
            pair_ids = self.tokenize_pair(
                prompt=p,
                candidate=c,
                source_max_length=self.source_max_length,
                max_length=self.max_length,
            )
            input_ids.append(pair_ids)

        encodings = self.tokenizer.pad(
            {"input_ids": input_ids},
            return_tensors="pt",
        )

        features = []
        total_pairs = len(input_ids)  # Total number of prompt-candidate pairs
        for ndx in range(0, total_pairs, self.batch_size):
            batch_enc = {
                key: value[ndx : min(ndx + self.batch_size, total_pairs)].to(self.device)
                for key, value in encodings.items()
            }
            features.append(self.get_feature(**batch_enc))
        features = torch.cat(features, dim=0)  # (M*N, d)
        return features
    
    def get_rewards(self, features: torch.Tensor, all_heads: bool = False) -> torch.Tensor:
        M, N, _ = features.shape
        E = self.num_ensemble
        features = einops.rearrange(features, "m n d -> (m n) d")
        rewards = []
        for ndx in range(0, len(features), self.batch_size):
            batch_feat = features[ndx : min(ndx + self.batch_size, len(features))]
            batch_feat = batch_feat[None, :, :]
            r = []
            for head in self.reward_heads:
                r.append(head(batch_feat))
            r = torch.cat(r, dim=0)  # (E, M*N, 1)
            if all_heads:
                rewards.append(r)
            else:
                rewards.append(r.mean(dim=0))
        
        rewards = torch.cat(rewards, dim=1)  # (E, M*N, 1)
        if all_heads:
            rewards = rewards.view(E, M, N, 1)
        else:
            rewards = rewards.view(M, N, 1)
        return rewards
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        sample: bool = False,
        all_heads: bool = False,
    ) -> torch.Tensor:
        """Forward pass through ensemble reward model."""

        with torch.no_grad():
            hidden_states = self.base_model.get_embs(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

        # Get rewards from each ensemble member
        if sample:
            import random
            selected_head_idx = random.randint(0, self.num_ensemble - 1)
            selected_head = self.reward_heads[selected_head_idx]
            reward = selected_head(hidden_states[:, -1, :])
        else:
            rewards = []
            for head in self.reward_heads:
                reward = head(hidden_states[:, -1, :])  # Use last token representation
                rewards.append(reward)

            # Stack rewards from ensemble
            rewards = torch.stack(rewards, dim=0)  # [num_ensemble, batch_size, 1]

            if all_heads:
                return rewards
            # Compute mean
            reward = rewards.mean(dim=0)  # [batch_size, 1]

        return reward

class EnsembleRewardTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        chosen_rewards = model(
            input_ids=inputs["chosen_ids"],
            attention_mask=inputs["chosen_mask"],
            all_heads=True,
        )
        rejected_rewards = model(
            input_ids=inputs["rejected_ids"],
            attention_mask=inputs["rejected_mask"],
            all_heads=True,
        )

        loss = 0
        num_ensemble = chosen_rewards.shape[0]
        for i in range(num_ensemble):
            member_loss = -F.logsigmoid(chosen_rewards[i] - rejected_rewards[i]).mean()
            loss += member_loss

        return (
            (loss, {"chosen_rewards": chosen_rewards, "rejected_rewards": rejected_rewards})
            if return_outputs
            else loss
        )
