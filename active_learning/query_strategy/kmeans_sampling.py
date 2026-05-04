import numpy as np
from .strategy import Strategy
from sklearn.cluster import KMeans

from tqdm import tqdm

import torch
import matplotlib.pyplot as plt
from .strategy import Strategy
from collections import defaultdict

# from ..utils.utils import (
#     slice_and_move_batch_for_device,
# )

from sentence_transformers import SentenceTransformer



class KMeansSampling(Strategy):
    def __init__(self, trainer, config):
        super(KMeansSampling, self).__init__(trainer, config)

    def query(self, n, rd):
        # unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        # embeddings = self.get_embeddings(unlabeled_data)
        # embeddings = embeddings.numpy()
        # cluster_learner = KMeans(n_clusters=n)
        # cluster_learner.fit(embeddings)
        
        # cluster_idxs = cluster_learner.predict(embeddings)
        # centers = cluster_learner.cluster_centers_[cluster_idxs]
        # dis = (embeddings - centers)**2
        # dis = dis.sum(axis=1)
        # q_idxs = np.array([np.arange(embeddings.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(n)])

        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        inference_batches = self.predict_prob(unlabeled_data)

        # embeddings= np.array([]) #unlabeled x class


        model_st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embeddings_list = []


        with torch.no_grad():
            for inference_batch in (tqdm(inference_batches, desc='kMeans selection') if self.trainer.rank == 0 else inference_batch):
                local_inf_batch = slice_and_move_batch_for_device(inference_batch, self.trainer.rank, self.trainer.world_size, self.trainer.rank)
                if self.trainer.config.loss.name in {'dpo', 'ipo'}:
                    # print(local_inf_batch)
                    pclogps, _, pclog, prlog, logps, log, emb = self.trainer.concatenated_forward_log(self.trainer.policy, local_inf_batch, avg_mode=True, all_token=True, output_hidden_state=True)
                    emb_c = model_st.encode(local_inf_batch['chosen'], show_progress_bar = False)
                    emb_r = model_st.encode(local_inf_batch['rejected'], show_progress_bar = False)

                    embeddings_list.append(np.concatenate([emb_c, emb_r], axis=1))
                    # embeddings = np.concatenate([embeddings, emb], axis=0)
                    
                else: raise ValueError('Not implemented for SFT!')
        
        # embeddings = embeddings.cpu().numpy()
        embeddings = np.vstack(embeddings_list)
        embeddings= embeddings[:len(unlabeled_idxs)]
        
        cluster_learner = KMeans(n_clusters=n)
        cluster_learner.fit(embeddings)
      
        cluster_idxs = cluster_learner.predict(embeddings)
        centers = cluster_learner.cluster_centers_[cluster_idxs]
        dis = (embeddings - centers)**2
        dis = dis.sum(axis=1)
        q_idxs = np.array([np.arange(embeddings.shape[0])[cluster_idxs==i][dis[cluster_idxs==i].argmin()] for i in range(n)])


        return unlabeled_idxs[q_idxs]
