from typing import Tuple
import torch
#from .utils import sim_trayectorias
from .utils import Buffer
import multiprocessing as mp
from functools import partial
import numpy as np
import time

def worker(
        i: int,
        env,
        agent,
        epochs: int,
        max_steps: int,
        torch_threads: int,
):
    n = semilla_aleatoria = np.random.randint(0, 10000) + i
    np.random.seed(n)
    torch.manual_seed(n)
    torch.set_num_threads(torch_threads)
    buffer = [[],[],[],[],[]]
    returns = []

    for epoch in range(epochs):
        state, _ = env.set_initial(s = [0]*env.I)
        cum_reward = 0

        for t in range(max_steps):
            action, action_logprob = agent.select_action2(state)
            buffer[0].append(np.array(action.cpu()))
            buffer[1].append(np.array(action_logprob.cpu()))
            action = action.cpu()
            state_next, reward, done, _, _ = env.step(action)

            if t + 1 == max_steps:
                done = True

            buffer[2].append(reward)
            buffer[3].append(done)
            #buffer.states.append(state)
            buffer[4].append(state)

            cum_reward += reward

            if done:
                break

            state = state_next
        returns.append(cum_reward)
    return buffer, returns

def worker2(
        i: int,
        env,
        agent,
        epochs: int,
        max_steps: int,
        queue,
): 
    n = semilla_aleatoria = np.random.randint(0, 10000) + i
    np.random.seed(n)
    torch.manual_seed(n)
    torch.set_num_threads(1)
    buffer = Buffer()
    returns = []

    for epoch in range(epochs):
        state, _ = env.set_initial(s = [0]*env.I)
        cum_reward = 0

        for t in range(max_steps):
            action, action_logprob = agent.select_action2(state)
            buffer.actions.append(np.array(action))
            buffer.logprobs.append(np.array(action_logprob))
            action = action.cpu()
            state_next, reward, done, _, _ = env.step(action)

            if t + 1 == max_steps:
                done = True

            buffer.rewards.append(reward)
            buffer.terminals.append(done)
            #buffer.states.append(state)
            buffer.states.append(torch.as_tensor(state, device=agent.device).double())

            cum_reward += reward

            if done:
                break

            state = state_next
        returns.append(cum_reward)
    queue.put((buffer, returns))


def worker3(
        i: int,
        env,
        agent,
        epochs: int,
        max_steps: int,
        lock,
        result_list,
        torch_threads: int,
): 
    n = semilla_aleatoria = np.random.randint(0, 10000) + i
    np.random.seed(n)
    torch.manual_seed(n)
    torch.set_num_threads(torch_threads)
    buffer = [[],[],[],[],[]]
    returns = []

    for epoch in range(epochs):
        state, _ = env.set_initial(s = [0]*env.I)
        cum_reward = 0

        for t in range(max_steps):
            action, action_logprob = agent.select_action2(state)
            buffer[0].append(np.array(action.cpu()))
            buffer[1].append(np.array(action_logprob.cpu()))
            action = action.cpu()
            state_next, reward, done, _, _ = env.step(action)

            if t + 1 == max_steps:
                done = True

            buffer[2].append(reward)
            buffer[3].append(done)
            #buffer.states.append(state)
            buffer[4].append(state)

            cum_reward += reward

            if done:
                break

            state = state_next
        returns.append(cum_reward)
    lock.acquire()
    try:
        # Operación con el valor compartido
        result_list.append((buffer, returns))
    finally:
        # Liberar el bloqueo
        lock.release()


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
        mu_list = []
        sigma_list = []

        Rs = []
        muestreador = 1
        num_sim = 1
        #start_time = time.time()
        last_epoc = 0
        for epoch in range(epochs):
            state, _ = env.set_initial(s = [0]*env.I)
            cum_reward = 0

            for t in range(max_steps):
                action = agent.select_action(state)
                action = action.cpu()
                state_next, reward, done, _, _ = env.step(action)

                if t + 1 == max_steps:
                    done = True

                agent.buffer.rewards.append(reward)
                agent.buffer.terminals.append(done)

                mu, log_sigma = agent.policy_old.actor(torch.as_tensor(state).double())
                mu_list.append(mu.detach().numpy())
                sigma_list.append(log_sigma.detach().numpy())

                cum_reward += reward

                if len(agent.buffer) >= update_freq and epoch > initial_offset:
                    """end_time = time.time()
                    print(f'Timepo de muestreo {end_time-start_time} de {epoch-last_epoc}')
                    last_epoc = epoch
                    start_time = time.time()"""
                    self._update(agent)
                    """end_time = time.time()
                    print(f'Timepo de optimización {end_time-start_time}')
                    start_time = time.time()"""

                if done:
                    break

                state = state_next
            returns.append(cum_reward)

            #Probar como de bien funciona sin modificar
            """if epoch % muestreador == 0:
                Rsamplig, _, _, _ = sim_trayectorias(env,agent,num_sim,max_steps,0,model = "PG")
                Rs.append(Rsamplig) """

            timesteps.append(t)
            print(f'{epoch}/{epochs}: {returns[-1]} \r', end='')

        return agent, returns,timesteps, mu_list, sigma_list


    def train2(
            self,
            env,
            agent,
            epochs: int,
            max_steps: int,
            num_updates: int,
            num_sim: int,
            initial_offset: int,
        ):
            cuda_device = torch.device("cuda")
            cpu_device = torch.device("cpu")
            returns = []
            timesteps = []

            for u in range(num_updates):
                if agent.device == cuda_device:
                    agent.policy_old.actor.to(cpu_device)

                #start_time = time.time()
                # Crear una lista para mantener los procesos
                procesos = []
                manager = mp.Manager()
                lista_compartida = manager.list()
                lock = mp.Lock()

                # Crear los procesos solicitados por el usuario
                for i in range(num_sim):
                    proceso = mp.Process(target=worker3, args=(i, env, agent, epochs, max_steps,lock,lista_compartida,1))
                    procesos.append(proceso)
                    proceso.start()

                # Esperar a que todos los procesos terminen
                for proceso in procesos:
                    proceso.join()

                if agent.device == cuda_device:
                    agent.policy_old.actor.to(cuda_device)
                """end_time = time.time()
                print(f"Tiempo de muestreo{end_time-start_time}")
                start_time = time.time()"""
                for result in lista_compartida:
                    agent.buffer.actions.extend(torch.as_tensor(action, device=agent.device).double() for action in result[0][0])
                    agent.buffer.logprobs.extend(torch.as_tensor(logprob, device=agent.device).double() for logprob in result[0][1])
                    agent.buffer.rewards.extend(result[0][2])
                    agent.buffer.terminals.extend(result[0][3])
                    agent.buffer.states.extend(torch.as_tensor(state, device=agent.device).double() for state in result[0][4])
                    returns.extend(result[1])
                """end_time = time.time()
                print(f"Tiempo de guardado de datos {end_time-start_time}")
                start_time = time.time()"""
                self._update(agent)
                """end_time = time.time()
                print(f"Tiempo optimización {end_time-start_time}")"""
                    
                print(f'{u}/{num_updates}: {returns[-1]} \r', end='')

            return agent, returns,timesteps
