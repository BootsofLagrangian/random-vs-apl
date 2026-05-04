from .query_strategy import (DummySampling, RandomSampling, EntropySampling, EntropyNegSampling, MarginSampling, LeastConfidence, KCenterGreedy,
                             APL, APLNeg, APLBoth, APLTest, DivReward, KMeansSampling, RewardMargin, MaxHerding, KNNHerding, UHerding, SEA, XPO, ADPO)
from trl.build_utils import detect_dataset_type, DATASET_SPECIFIC_PROMPTS

def get_strategy(active_args, script_args, trainer):
    dataset_type = detect_dataset_type(script_args.dataset_name)
    system_prompt = DATASET_SPECIFIC_PROMPTS[dataset_type]
    
    query_strategy = active_args.query_strategy
    if query_strategy == 'dummy':
        return DummySampling(trainer, system_prompt, active_args)
    elif query_strategy == 'random':
        return RandomSampling(trainer, system_prompt, active_args)
    elif query_strategy == 'conf':
        return LeastConfidence(trainer, system_prompt, active_args)
    elif query_strategy in {'margin', 'rmargin'}:
        # Reward-margin sampling (difference between 2 sampled completions).
        return RewardMargin(trainer, system_prompt, active_args)
    elif query_strategy == 'entropy':
        return EntropySampling(trainer, system_prompt, active_args)
    elif query_strategy == 'entropy_neg':
        return EntropyNegSampling(trainer, system_prompt, active_args)
    elif query_strategy == 'apl':
        return APL(trainer, system_prompt, active_args)
    elif query_strategy == 'apl_neg':
        return APLNeg(trainer, system_prompt, active_args)
    elif query_strategy == 'apl_both':
        return APLBoth(trainer, system_prompt, active_args)
    elif query_strategy == 'apl_test':
        return APLTest(trainer, system_prompt, active_args)
    elif query_strategy == 'kmeans':
        return KMeansSampling(trainer, system_prompt, active_args)
    elif query_strategy == 'coreset':
        return KCenterGreedy(trainer, system_prompt, active_args)
    elif query_strategy == 'dreward':
        return DivReward(trainer, system_prompt, active_args)
    elif query_strategy == 'maxherding':
        return MaxHerding(trainer, system_prompt, active_args)
    elif query_strategy == 'kherding':
        return KNNHerding(trainer, system_prompt, active_args)
    elif query_strategy == 'uherding':
        return UHerding(trainer, system_prompt, active_args)
    elif query_strategy == 'sea':
        return SEA(trainer, system_prompt, active_args)
    elif query_strategy == 'xpo':
        return XPO(trainer, system_prompt, active_args)
    elif query_strategy == 'adpo':
        return ADPO(trainer, system_prompt, active_args)
    else:
        raise NotImplementedError
