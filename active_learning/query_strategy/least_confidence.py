import torch
from tqdm import tqdm

from .strategy import Strategy
# from ..utils.utils import (
#     slice_and_move_batch_for_device,
# )

class LeastConfidence(Strategy):
    def __init__(self, trainer, config):
        super(LeastConfidence, self).__init__(trainer, config)
        assert self.trainer.config.loss.name in {'dpo', 'ipo'}

    def query(self, n, rd):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        inference_batches = self.predict_prob(unlabeled_data) 

        uncertainties = []
        for inference_batch in (tqdm(inference_batches, desc='Least Confidence selection') if self.trainer.rank == 0 else inference_batch):
            local_inf_batch = slice_and_move_batch_for_device(inference_batch, self.trainer.rank, self.trainer.world_size, self.trainer.rank)
            with torch.no_grad():
                pclogps, _, pclog, prlog, logps, ps = self.trainer.concatenated_forward_log(self.trainer.policy, local_inf_batch, avg_mode=True, all_token=True)
            
            topk_porbs = torch.topk(ps, dim=-1, k=2, largest=True).values.cpu() # B x L x V
            confidences = topk_porbs[:,:,0].mean(dim=-1) # B x L
            uncertainties.append(confidences)
            
        uncertainties = torch.cat(uncertainties, dim=0)
        uncertainties= uncertainties[:len(unlabeled_idxs)]
        
        return unlabeled_idxs[uncertainties.sort(descending=True)[1][:n]]
