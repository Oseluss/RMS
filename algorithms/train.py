from typing import Tuple
import torch
from .utils import sim_trayectorias

class Trainer:
    def __init__(self, actor_opt, critic_opt):
        self.actor_opt = actor_opt
        self.critic_opt = critic_opt

    def _update(self, agent):
        if self.actor_opt == 'bcd':
            n_params_critic = len(list(agent.policy.critic.parameters()))
            for i in range(n_params_critic):
                advantages = agent.update_critic(i)
        else:
            advantages = agent.update_critic()

        if self.critic_opt == 'bcd':
            n_params_actor = len(list(agent.policy.actor.parameters()))
            for i in range(n_params_actor):
                agent.update_actor(advantages, i)
        else:
            agent.update_actor(advantages)

        agent.buffer.clear()

    def train(
        self,
        env,
        agent,
        epochs: int,
        max_steps: int,
        update_freq: int,
        initial_offset: int,
    ):
        returns = []
        timesteps = []

        Rs = []
        muestreador = 100
        num_sim = 10

        for epoch in range(epochs):
            state, _ = env.reset()
            cum_reward = 0

            for t in range(max_steps):
                action = agent.select_action(state)
                state_next, reward, done, _, _ = env.step(action.tolist())

                if t + 1 == max_steps:
                    done = True

                agent.buffer.rewards.append(reward)
                agent.buffer.terminals.append(done)
                cum_reward += reward

                if len(agent.buffer) >= update_freq and epoch > initial_offset:
                    self._update(agent)

                if done:
                    break

                state = state_next
            returns.append(cum_reward)
            timesteps.append(t)

            #Probar como de bien funciona sin modificar
            if epoch % muestreador == 0:
                Rsamplig, _, _, _ = sim_trayectorias(env,agent,num_sim,max_steps,0,model = "PG")
                Rs.append(Rsamplig)

            timesteps.append(t)
            print(f'{epoch}/{epochs}: {sum(Rs[-1])/(len(Rs[-1]))} \r', end='')

        return agent, Rs,timesteps
