from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


class SoftmaxAgent(nn.Module):
    def __init__(self, actor, critic, discretizer_actor=None, discretizer_critic=None, betha = 1,device = torch.device("cpu")) -> None:
        super(SoftmaxAgent, self).__init__()

        self.actor = actor
        self.critic = critic

        self.discretizer_actor = discretizer_actor
        self.discretizer_critic = discretizer_critic
        self.betha = betha
        self.device = device

    def pi(self, state: np.ndarray) -> torch.distributions.Normal:
        state = torch.as_tensor(state, device=self.device).double()

        # Parameters
        if self.discretizer_actor:
            state = state.numpy().reshape(-1, len(self.discretizer_actor.buckets))
            indices = self.discretizer_actor.get_index(state)
            logits = self.betha * self.actor(indices).squeeze()
        else:
            logits = self.betha * self.actor(state).squeeze()
        #print(logits)
        # Distribution
        pi = torch.distributions.categorical.Categorical(logits=logits)
        return pi

    def evaluate_logprob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # Actor
        dist = self.pi(state)
        action_logprob = dist.log_prob(action)
        if len(action_logprob.shape) > 1:
            return torch.sum(action_logprob, dim=1)
        else:
            return action_logprob

    def evaluate_value(self, state: torch.Tensor) -> torch.Tensor:
        # Critic
        if self.discretizer_critic:
            state = state.numpy().reshape(-1, len(self.discretizer_actor.buckets))
            indices = self.discretizer_critic.get_index(state)
            value = self.critic(indices)
            return value.squeeze()
        value = self.critic(state)
        return value.squeeze()

    def act(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dist = self.pi(state)
        action = dist.sample()
        action_logprob = torch.sum(dist.log_prob(action))
        return action.detach().flatten(), action_logprob.detach().flatten()
    
class GaussianAgent(nn.Module):
    def __init__(self, actor, critic, discretizer_actor=None, discretizer_critic=None,device = torch.device("cpu")) -> None:
        super(GaussianAgent, self).__init__()

        self.actor = actor
        self.critic = critic

        self.discretizer_actor = discretizer_actor
        self.discretizer_critic = discretizer_critic

        self.device = device

    def pi(self, state: np.ndarray) -> torch.distributions.Normal:
        state = torch.as_tensor(state, device=self.device).double()

        # Parameters
        if self.discretizer_actor:
            state = state.numpy().reshape(-1, len(self.discretizer_actor.buckets))
            indices = self.discretizer_actor.get_index(state)
            mu, log_sigma = self.actor(indices)
        else:
            mu, log_sigma = self.actor(state)
        sigma = log_sigma.exp()

        # Distribution
        pi = torch.distributions.Normal(mu.squeeze(), sigma.squeeze())
        return pi

    def evaluate_logprob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # Actor
        dist = self.pi(state)
        action_logprob = dist.log_prob(action)
        return action_logprob.squeeze()

    def evaluate_value(self, state: torch.Tensor) -> torch.Tensor:
        # Critic
        if self.discretizer_critic:
            state = state.numpy().reshape(-1, len(self.discretizer_actor.buckets))
            indices = self.discretizer_critic.get_index(state)
            value = self.critic(indices)
            return value.squeeze()
        value = self.critic(state)
        return value.squeeze()

    def act(self, state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dist = self.pi(state)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach().flatten(), action_logprob.detach().flatten()