"""ARFT reward adapter for step scores produced by the forecasting environment."""
import ray
import numpy as np


def compute_reward(data, reward_fn):
    if "rm_scores" in data.batch:
        extra = {name: np.asarray(data.non_tensor_batch[name]).tolist()
                 for name in ("MSE", "MAE") if name in data.non_tensor_batch}
        return data.batch["rm_scores"], extra
    if reward_fn is None:
        raise ValueError("An environment score or reward function is required")
    result = reward_fn(data, return_dict=True)
    return result["reward_tensor"], result.get("reward_extra_info", {})


@ray.remote(num_cpus=1)
def compute_reward_async(data, config, tokenizer):
    if "rm_scores" in data.batch:
        return compute_reward(data, None)
    from verl.trainer.ppo.reward import load_reward_manager
    return compute_reward(data, load_reward_manager(config, tokenizer, num_examine=0))
