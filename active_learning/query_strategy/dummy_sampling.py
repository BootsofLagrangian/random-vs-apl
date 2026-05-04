import torch
from .strategy import Strategy


class DummySampling(Strategy):
    def __init__(self, trainer, system_prompt, active_args):
        super(DummySampling, self).__init__(trainer, system_prompt, active_args)

    def query(self, n):
        return {}
    
