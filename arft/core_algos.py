# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO-like algorithms.
"""

import torch
import numpy as np
from collections import defaultdict


def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    trajectory_uids: np.ndarray,
    step_indices: np.ndarray,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    import verl.utils.torch_functional as verl_F

    device = token_level_rewards.device

    with torch.no_grad():
        # Step-level reward: sum of token rewards inside the step.
        rewards = token_level_rewards.sum(dim=1)
        values = values.gather(1, response_mask.sum(1).unsqueeze(1) - 1).squeeze(1)

        # Map trajectories to contiguous ids for compact padding.
        # Use numpy's unique to handle both object and numeric types
        unique_traj_np, traj_inv_np = np.unique(trajectory_uids, return_inverse=True)
        num_traj = len(unique_traj_np)
        traj_inv = torch.as_tensor(traj_inv_np, dtype=torch.long, device=device)
        step_ids = torch.as_tensor(step_indices, device=device)
        max_step = int(step_ids.max().item()) + 1

        # reshape to (num_traj, max_step).
        # Use the same dtype as rewards and values to avoid type mismatch
        rewards_map = torch.zeros((num_traj, max_step), dtype=rewards.dtype, device=device)
        values_map = torch.zeros((num_traj, max_step), dtype=values.dtype, device=device)

        rewards_map[traj_inv, step_ids] = rewards
        values_map[traj_inv, step_ids] = values

        lastgaelam = 0
        advantages_reversed = []

        for t in reversed(range(max_step)):
            nextvalues = values_map[:, t + 1] if t < max_step - 1 else 0.0
            delta = rewards_map[:, t] + gamma * nextvalues - values_map[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages_map = torch.stack(advantages_reversed[::-1], dim=1)

        # Map back to batch rows and then to token level.
        advantages = advantages_map[traj_inv, step_ids]
        returns = advantages + values

        advantages = advantages.unsqueeze(1) * response_mask
        returns = returns.unsqueeze(1) * response_mask
        advantages = verl_F.masked_whiten(advantages, response_mask)

    return advantages, returns

def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    trajectory_uids: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    step_indices: np.ndarray | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        index: `(np.ndarray)`
            index array for grouping
        epsilon: `(float)`
            small value to avoid division by zero
        norm_adv_by_std_in_grpo: `(bool)`
            whether to scale the GRPO advantage

    Note:
        If norm_adv_by_std_in_grpo is True, the advantage is scaled by the std, as in the original GRPO.
        If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    # Batch balancing can reorder turns. Select the terminal turn by its explicit
    # step index, then broadcast ONE outcome advantage to every turn's policy tokens.
    row_scores = token_level_rewards.sum(dim=-1)
    if epsilon <= 0:
        raise ValueError("GRPO requires a positive numerical stabilizer")
    if step_indices is None:
        if len(set(trajectory_uids)) != len(trajectory_uids):
            raise ValueError("Multi-turn GRPO requires step_indices to locate terminal rewards")
        step_indices = np.zeros(len(trajectory_uids), dtype=int)
    if not (len(index) == len(trajectory_uids) == len(step_indices) == len(row_scores)):
        raise ValueError("GRPO metadata must have one entry per response row")

    with torch.no_grad():
        terminal_rows = {}
        trajectory_groups = {}
        active_trajectories = {trajectory for row, trajectory in enumerate(trajectory_uids)
                               if response_mask[row].any()}
        for row, (trajectory, group, step) in enumerate(zip(trajectory_uids, index, step_indices)):
            if trajectory not in active_trajectories:
                continue
            if trajectory in trajectory_groups and trajectory_groups[trajectory] != group:
                raise ValueError("A trajectory cannot belong to multiple input groups")
            trajectory_groups[trajectory] = group
            previous = terminal_rows.get(trajectory)
            if previous is None or step > step_indices[previous]:
                terminal_rows[trajectory] = row

        groups = defaultdict(list)
        for trajectory, row in terminal_rows.items():
            groups[trajectory_groups[trajectory]].append(trajectory)
        trajectory_advantages = {}
        for trajectories in groups.values():
            outcomes = torch.stack([row_scores[terminal_rows[t]] for t in trajectories])
            mean = outcomes.mean()
            std = outcomes.std() if len(trajectories) > 1 else outcomes.new_tensor(0.0)
            centered = outcomes - mean
            if norm_adv_by_std_in_grpo:
                centered = centered / (std + epsilon)
            trajectory_advantages.update(zip(trajectories, centered))

        advantages = torch.zeros_like(row_scores)
        for row, trajectory in enumerate(trajectory_uids):
            if trajectory in trajectory_advantages:
                advantages[row] = trajectory_advantages[trajectory]
        advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages
