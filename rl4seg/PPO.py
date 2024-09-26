from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from RLmodule import RLmodule


class PPO(RLmodule):

    def __init__(self,
                 clip_value: float = 0.2,
                 k_steps_per_batch: int = 5,
                 entropy_coeff: float = 0.0,
                 divergence_coeff: float = 0.0,
                 *args: Any,
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.clip_value = clip_value
        self.k_steps = k_steps_per_batch
        self.entropy_coeff = entropy_coeff
        self.divergence_coeff = divergence_coeff

        # since optimization is done manually, this flag needs to be set
        self.automatic_optimization = False

    def training_step(self, batch: dict[str, Tensor], nb_batch):
        """
            Defines PPO training steo
            Get actions, log_probs and rewards from current policy
            Calculate and backprop losses for actor and critic K times in loop over same batch
        Args:
            batch: current mini batch of replay data
            nb_batch: batch number

        Returns:
            Training loss and log metrics or None
        """
        opt_net, opt_critic = self.optimizers()

        b_img, b_gt, b_approx_gt, b_use_gt = batch['img'], batch['gt'], batch['approx_gt'], batch['use_gt']
        if self.trainer.datamodule.hparams.supervised:
            b_approx_gt = b_gt

        # get actions, log_probs, rewards, etc from pi (stays constant for all steps k)
        prev_actions, prev_log_probs, prev_rewards = self.rollout(b_img, b_approx_gt, b_use_gt)

        # iterate with pi prime k times
        for k in range(self.k_steps):
            # calculates training loss
            loss, critic_loss, metrics_dict = self.compute_policy_loss((b_img, prev_actions, prev_rewards,
                                                                        prev_log_probs, b_approx_gt, b_use_gt))

            opt_net.zero_grad()
            self.manual_backward(loss, retain_graph=True)
            nn.utils.clip_grad_norm_(self.actor.parameters(),
                                     0.5)
            opt_net.step()

            # TODO: should this be outside the loop? According to real algo...
            opt_critic.zero_grad()
            self.manual_backward(critic_loss)
            opt_critic.step()

            logs = {**metrics_dict,
                    **{'loss': loss,
                       'critic_loss': critic_loss,
                       }
                    }

            self.log_dict(logs, prog_bar=True)

    def compute_policy_loss(self, batch, **kwargs):
        """
            Compute unsupervised loss to maximise reward using PPO method.
        Args:
            batch: batch of images, actions, log_probs, rewards and ground truth
            sample: whether to sample from distribution or deterministic approach (mainly for val, test steps)

        Returns:
            mean loss(es) for the batch, metrics dictionary
        """
        b_img, b_actions, b_rewards, b_log_probs, b_gt, b_use_gt = batch

        _, logits, log_probs, entropy, v, old_log_probs = self.actor.evaluate(b_img, b_actions)

        log_pi_ratio = (log_probs - old_log_probs)
        with torch.no_grad():
            total_reward = b_rewards - (self.divergence_coeff * log_pi_ratio)
            # ignore divergence if using ground truth
            total_reward[b_use_gt, ...] = torch.ones_like(b_rewards)[b_use_gt, ...]

        # assert b_rewards.shape == v.shape
        adv = total_reward - v

        # PPO loss
        # importance ratio
        assert b_log_probs.shape == log_probs.shape
        ratio = (log_probs - b_log_probs).exp()

        # clamp with epsilon value
        clipped = ratio.clamp(1 - self.clip_value, 1 + self.clip_value)
        surr_loss = torch.min(adv * ratio, adv * clipped)
        # surr_loss[b_use_gt, ...] = (adv * ratio)[b_use_gt, ...]

        # min trick
        loss = -surr_loss.mean() + (-self.entropy_coeff * entropy.mean())

        # Critic loss
        if b_rewards.shape != v.shape:  # if critic is resnet, use reward mean instead of pixel-wise
            b_rewards = b_rewards.mean(dim=(1, 2), keepdim=True)
        critic_loss = nn.MSELoss()(v, b_rewards)

        # metrics dict
        metrics = {
                'v': v.mean(),
                'advantage': adv.mean(),
                'reward': b_rewards.mean(),
                'log_probs': log_probs.mean(),
                'ratio': ratio.mean(),
                'approx_kl_div': log_pi_ratio.mean(),
        }

        return loss, critic_loss, metrics

