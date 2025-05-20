from legged_gym.rsl_rl.modules import ActorCritic
import torch
import torch.nn as nn

class ActorCriticWithCost(ActorCritic):
    def __init__(self, actor: nn.Module, critic: nn.Module, cost_critic: nn.Module, num_actions: int, init_noise_std=1.0, **kwargs):
        super().__init__(actor, critic, num_actions, init_noise_std, **kwargs)
        self.cost_critic = cost_critic

    def evaluate_cost(self, cost_critic_observations: torch.Tensor):
        return self.cost_critic(cost_critic_observations)