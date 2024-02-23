from typing import Tuple, List
from copy import deepcopy

import numpy as np
import torch

from algorithms.utils import Buffer
from algorithms.agents.agents import SoftmaxAgent

class PPOSoftmaxNN:
    def __init__(
        self,
        actor,
        critic,
        discretizer_actor=None,
        discretizer_critic=None,
        lr_actor: float=1e-3,
        gamma: float=0.99,
        tau: float=0.97,
        epochs: int=1000,
        eps_clip: float=0.2,
    ) -> None:

        self.gamma = gamma
        self.tau = tau
        self.eps_clip = eps_clip
        self.epochs = epochs

        self.buffer = Buffer()

        actor_old = deepcopy(actor)
        critic_old = deepcopy(critic)

        self.policy = SoftmaxAgent(actor, critic, discretizer_actor, discretizer_critic)
        self.policy_old = SoftmaxAgent(actor_old, critic_old, discretizer_actor, discretizer_critic)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.opt_actor = torch.optim.Adam(self.policy.actor.parameters(), lr_actor)

        self.opt_critic = torch.optim.LBFGS(
            self.policy.critic.parameters(),
            history_size=100,
            max_iter=25,
            line_search_fn='strong_wolfe',
        )

    def select_action(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            state = torch.as_tensor(state).double()
            action, action_logprob = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        return action.item()

    def calculate_returns(self, values) -> List[float]:
        returns = []
        advantages=[]

        prev_return = 0
        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(len(self.buffer.rewards))):
            reward = self.buffer.rewards[i]
            mask = 1 - self.buffer.terminals[i]

            actual_return = reward + self.gamma*prev_return*mask
            actual_delta = reward + self.gamma*prev_value*mask - values[i]
            actual_advantage = actual_delta + self.gamma*self.tau*prev_advantage*mask        

            returns.insert(0, actual_return)
            advantages.insert(0, actual_advantage)

            prev_return = actual_return
            prev_value = values[i]
            prev_advantage = actual_advantage

        returns = torch.as_tensor(returns).double().detach().squeeze()
        advantages = torch.as_tensor(advantages).double().detach().squeeze()
        advantages = (advantages - advantages.mean())/advantages.std()

        return returns, advantages

    def zero_grad(self, model, idx=None):
        if idx is None:
            return

        for i, param in enumerate(model.parameters()):
            if i != idx:
                param.grad.zero_()

    def update_critic(self, idx=None):
        states = torch.stack(self.buffer.states, dim=0).detach()

        # GAE estimation
        values = self.policy.evaluate_value(states)
        rewards, advantages = self.calculate_returns(values.data.numpy())

        # LBFGS training
        def closure():
            self.opt_critic.zero_grad()
            values = self.policy.evaluate_value(states)
            loss = (values - rewards).pow(2).mean()
            loss.backward()
            self.zero_grad(self.policy.critic, idx)
            return loss
        self.opt_critic.step(closure)

        return advantages

    def update_actor(self, advantages, idx=None):
        old_states = torch.stack(self.buffer.states, dim=0).detach()
        old_actions = torch.stack(self.buffer.actions, dim=0).detach().squeeze()
        old_logprobs = torch.stack(self.buffer.logprobs, dim=0).detach().squeeze()

        # Stochastic Gradient Ascent
        for _ in range(self.epochs):
            logprobs = self.policy.evaluate_logprob(old_states, old_actions)
            ratio = torch.exp(logprobs - old_logprobs)
            ratio_clamped = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip)

            minorizer_raw = ratio * advantages
            minorizer_clamped = ratio_clamped * advantages

            loss_actor = -torch.min(minorizer_raw, minorizer_clamped)

            self.opt_actor.zero_grad()
            loss_actor.mean().backward()
            self.zero_grad(self.policy.actor, idx)
            self.opt_actor.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
