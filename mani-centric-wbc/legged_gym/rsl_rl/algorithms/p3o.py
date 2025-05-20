from legged_gym.rsl_rl.algorithms.ppo import PPO
from typing import Dict, Tuple, Union
from git import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from legged_gym.rsl_rl.modules import ActorCritic
from legged_gym.rsl_rl.storage import RolloutStorage
from legged_gym.rsl_rl.modules import ActorCriticWithCost

class P3O(PPO):
    def __init__(
            self, 
            *args, 
            cost_limit=25.0, 
            kappa=1.0, 
            actor_critic_with_cost:ActorCriticWithCost, 
            **kwargs):
        
        super().__init__(*args, **kwargs)
        self.cost_limit = cost_limit
        self.kappa = kappa
        self.actor_critic_with_cost = actor_critic_with_cost

    def update_per_batch(
        self,
        obs: torch.Tensor,
        critic_obs: torch.Tensor,
        action: torch.Tensor,
        target_value: torch.Tensor,
        advantage: torch.Tensor,
        returns: torch.Tensor,
        old_action_log_prob: torch.Tensor,
        old_mu: torch.Tensor,
        old_sigma: torch.Tensor,
        hid_state: Tuple[Optional[torch.Tensor], Optional[torch.Tensor]],
        mask: Optional[torch.Tensor],
        learning_iter: int,
        cost_advantage: torch.Tensor,
        cost_return: torch.Tensor,
        ep_cost=None
    ):
        stats = {}

        # Standard reward critic
        value = self.actor_critic_with_cost.evaluate(critic_obs)
        actions_log_prob, mu_batch, sigma_batch, entropy_batch = (
            self.get_ppo_actor_actions(obs, critic_obs, action)
        )
        # KL
        if self.desired_kl != None and self.schedule == "adaptive":
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma_batch / old_sigma + 1.0e-5)
                    + (torch.square(old_sigma) + torch.square(old_mu - mu_batch))
                    / (2.0 * torch.square(sigma_batch))
                    - 0.5,
                    axis=-1,
                )
                kl_mean = torch.mean(kl)

                if kl_mean > self.desired_kl * 2.0:
                    self.learning_rate = max(self.min_lr, self.learning_rate / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.learning_rate = min(self.max_lr, self.learning_rate * 1.5)

                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.learning_rate

        # Standard PPO surrogate loss
        ratio = torch.exp(actions_log_prob - torch.squeeze(old_action_log_prob))
        stats["mean_raw_ratios"] = ratio.mean().item()
        stats["mean_clipped_ratio"] = (
            ((ratio < 1.0 - self.clip_param) | (ratio > 1.0 + self.clip_param))
            .float()
            .mean()
            .item()
        )
        surrogate = -advantage.squeeze() * ratio
        surrogate_clipped = -advantage.squeeze() * torch.clamp(
            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        # Value loss
        if self.use_clipped_value_loss:
            value_clipped = target_value + (value - target_value).clamp(
                -self.clip_param, self.clip_param
            )
            value_losses = (value - returns).pow(2)
            value_losses_clipped = (value_clipped - returns).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (returns - value).pow(2).mean()

        loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

        stats["mean_surrogate_loss"] = surrogate_loss.item()
        stats["mean_value_loss"] = value_loss.item()

        # Add cost loss if provided
        if cost_advantage is not None and cost_return is not None and ep_cost is not None:
            value_cost = self.actor_critic_with_cost.evaluate_cost(critic_obs)
            cost_value_loss = (value_cost - cost_return).pow(2).mean()

            ratio_cost = torch.exp(actions_log_prob - old_action_log_prob.squeeze())
            surrogate_cost = (ratio_cost * cost_advantage).mean()
            Jc = ep_cost - self.cost_limit
            loss_cost = self.kappa * F.relu(surrogate_cost + Jc)

            loss += loss_cost + self.value_loss_coef * cost_value_loss

            stats["mean_cost_surrogate"] = surrogate_cost.item()
            stats["mean_cost_value_loss"] = cost_value_loss.item()
            stats["mean_cost_penalty"] = loss_cost.item()

        return loss, stats


    def update(self, learning_iter: int, ep_cost: float) -> Dict[str, float]:
        update_stats = {}
        if self.actor_critic_with_cost.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

        for (
            obs_batch,
            critic_obs,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            cost_advantage_batch,
            cost_return_batch,
        ) in generator:
            loss, stats = self.update_per_batch(
                obs=obs_batch,
                critic_obs=critic_obs,
                action=actions_batch,
                target_value=target_values_batch,
                advantage=advantages_batch,
                returns=returns_batch,
                old_action_log_prob=old_actions_log_prob_batch,
                old_mu=old_mu_batch,
                old_sigma=old_sigma_batch,
                hid_state=hid_states_batch,
                mask=masks_batch,
                learning_iter=learning_iter,
                cost_advantage=cost_advantage_batch,
                cost_return=cost_return_batch,
                ep_cost=ep_cost,
            )

            for k, v in stats.items():
                if k not in update_stats:
                    update_stats[k] = 0.0
                update_stats[k] += v

            self.optimizer.zero_grad()
            loss.backward()
            nn_utils.clip_grad_norm_(self.actor_critic_with_cost.parameters(), self.max_grad_norm)
            self.optimizer.step()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        return {
            **{k: v / num_updates for k, v in update_stats.items()},
            **{
                "learning_rate": self.learning_rate,
                "action_std": self.actor_critic_with_cost.std.mean().item(),
            },
        }
