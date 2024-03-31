import numpy as np
import torch
from itertools import product
import matplotlib.pyplot as plt
from dataclasses import dataclass
from matplotlib.animation import FuncAnimation
import random
from enviroments import env_red, env_red_p2p, env_red_toy1, env_hubs0, env_hubs1, env_hubs3
import pickle
from algorithms.Action_Gen import Action_generation
from algorithms.RDQL import action_space_generation, ini_action_list, caculate_op_cost,update_action_space
from algorithms.DQL import DQL_algorithm, select_action
from algorithms.QL_LowRank import select_action as select_action_LR
import time
import algorithms.models as models
from algorithms.agents import reinforce, trpo, ppo
import algorithms.train as train
import multiprocessing as mp
from functools import partial
from algorithms.utils import Buffer

device = torch.device("cpu")

env = env_hubs3()
M = 6
F = 2

#Dimensiónes del espacio de estados
dims_state = list(env.C + 1)
dims_state.append(env.T+1)

#Dimensiones del espacio de acciones
dims_action = [M, F]

actor = models.PolicyNetwork(
    num_inputs=len(dims_state), 
    num_hiddens=[128, 128], 
    num_outputs=[M,F], 
    model="Softmax"
).double().to(device)

critc = models.ValueNetwork(
    num_inputs=len(dims_state), 
    num_hiddens=[128, 128], 
    num_outputs=1, 
).double().to(device)

Trainer = train.Trainer("sgd", "sgd")

ppo_agent = ppo.PPOSoftmaxNN(actor, critc,gamma=1, tau=1, lr_actor=1e-3, epochs=50, betha=1e-3,eps_clip=0.1,device=device)

num_sim = 10
epochs = 1
max_steps = 2000
# Crear un objeto de bloqueo para hilos
print(f"Número de subprocesos {num_sim}, numero de epochs por thread: {epochs}")
print()

for torch_threads in range(1,2):
    returns = []

    with mp.Pool(num_sim) as pool:
        start_time = time.time()
        partial_worker = partial(
            train.worker, env=env, agent=ppo_agent, epochs=epochs, max_steps=max_steps, torch_threads = torch_threads,
        )      
        results = pool.map(partial_worker, range(num_sim))
        #end_time = time.time()
        #print(f"El tiempo de ejecución con pool_multithreding y torch_threads = {torch_threads}: {end_time-start_time}")
        #start_time = time.time()                
        for result in results:
            ppo_agent.buffer.actions.extend(torch.as_tensor(action, device=ppo_agent.device).double() for action in result[0][0])
            ppo_agent.buffer.logprobs.extend(torch.as_tensor(logprob, device=ppo_agent.device).double() for logprob in result[0][1])
            ppo_agent.buffer.rewards.extend(result[0][2])
            ppo_agent.buffer.terminals.extend(result[0][3])
            ppo_agent.buffer.states.extend(torch.as_tensor(state, device=ppo_agent.device).double() for state in result[0][4])
            
            returns.extend(result[1])
        end_time = time.time()

    #print(f"El tiempo de merge de datos {end_time-start_time}")
    print(f"Tiempo de muestreo {end_time-start_time}")
    print(len(returns))

    ppo_agent.buffer.clear()

"""
returns = []
torch_threads = 1

start_time = time.time()
result = train.worker(i = 0, env=env, agent=ppo_agent, epochs=epochs*num_sim, max_steps=max_steps, torch_threads=torch_threads, num_CPUs=num_cpu_sim)
    
end_time = time.time()
print(f"El tiempo de ejecución sin multithreding y torch_threads = {torch_threads}: {end_time-start_time}")
start_time = time.time()                

ppo_agent.buffer.actions.extend(torch.as_tensor(action, device=ppo_agent.device).double() for action in result[0][0])
ppo_agent.buffer.logprobs.extend(torch.as_tensor(logprob, device=ppo_agent.device).double() for logprob in result[0][1])
ppo_agent.buffer.rewards.extend(result[0][2])
ppo_agent.buffer.terminals.extend(result[0][3])
ppo_agent.buffer.states.extend(torch.as_tensor(state, device=ppo_agent.device).double() for state in result[0][4])
returns.extend(result[1])
end_time = time.time()

print(f"El tiempo de merge de datos {end_time-start_time}")
print()
ppo_agent.buffer.clear()
"""
returns = []
torch_threads = num_sim
start_time = time.time()
result = train.worker(i = 0, env=env, agent=ppo_agent, epochs=epochs*num_sim, max_steps=max_steps, torch_threads=torch_threads)
    
end_time = time.time()
print(f"El tiempo de ejecución sin multithreding y torch_threads = {torch_threads}: {end_time-start_time}")
start_time = time.time()                

ppo_agent.buffer.actions.extend(torch.as_tensor(action, device=ppo_agent.device).double() for action in result[0][0])
ppo_agent.buffer.logprobs.extend(torch.as_tensor(logprob, device=ppo_agent.device).double() for logprob in result[0][1])
ppo_agent.buffer.rewards.extend(result[0][2])
ppo_agent.buffer.terminals.extend(result[0][3])
ppo_agent.buffer.states.extend(torch.as_tensor(state, device=ppo_agent.device).double() for state in result[0][4])
returns.extend(result[1])
end_time = time.time()

print(f"El tiempo de merge de datos {end_time-start_time}")
print()
ppo_agent.buffer.clear()

for torch_threads in range(1,2):
    
    # Crear un objeto de bloqueo para hilos
    returns = []
    start_time = time.time()
    # Crear una lista para mantener los procesos
    procesos = []
    manager = mp.Manager()
    lista_compartida = manager.list()
    lock = mp.Lock()

    # Crear los procesos solicitados por el usuario
    for i in range(num_sim):
        proceso = mp.Process(target=train.worker3, args=(i, env, ppo_agent, epochs, max_steps,lock,lista_compartida,torch_threads))
        procesos.append(proceso)
        proceso.start()

    # Esperar a que todos los procesos terminen
    for proceso in procesos:
        proceso.join()

    #end_time = time.time()

    #print(f"El tiempo de ejecución con process_multithreding y torch_threads = {torch_threads}: {end_time-start_time}")
    #start_time = time.time()                
    for result in lista_compartida:
        ppo_agent.buffer.actions.extend(torch.as_tensor(action, device=ppo_agent.device).double() for action in result[0][0])
        ppo_agent.buffer.logprobs.extend(torch.as_tensor(logprob, device=ppo_agent.device).double() for logprob in result[0][1])
        ppo_agent.buffer.rewards.extend(result[0][2])
        ppo_agent.buffer.terminals.extend(result[0][3])
        ppo_agent.buffer.states.extend(torch.as_tensor(state, device=ppo_agent.device).double() for state in result[0][4])
        returns.extend(result[1])
    end_time = time.time()

    #print(f"El tiempo de merge de datos {end_time-start_time}")
    print(f"Tiempo de muestreo {end_time-start_time}")
    print()

    ppo_agent.buffer.clear()

device = torch.device("cpu")

env = env_hubs3()
M = 6
F = 2

#Dimensiónes del espacio de estados
dims_state = list(env.C + 1)
dims_state.append(env.T+1)

#Dimensiones del espacio de acciones
dims_action = [M, F]

actor = models.PolicyNetwork(
    num_inputs=len(dims_state), 
    num_hiddens=[128, 128], 
    num_outputs=[M,F], 
    model="Softmax"
).double().to(device)

critc = models.ValueNetwork(
    num_inputs=len(dims_state), 
    num_hiddens=[128, 128], 
    num_outputs=1, 
).double().to(device)

Trainer = train.Trainer("sgd", "sgd")

ppo_agent = ppo.PPOSoftmaxNN(actor, critc,gamma=1, tau=1, lr_actor=1e-3, epochs=50, betha=1e-3,eps_clip=0.1,device=device)

returns = []
torch_threads = num_sim
torch_threads = num_sim
start_time = time.time()
result = train.worker(i = 0, env=env, agent=ppo_agent, epochs=epochs*num_sim, max_steps=max_steps, torch_threads=torch_threads)
    
end_time = time.time()
print(f"El tiempo de ejecución sin multithreding y GPU {end_time-start_time}")
start_time = time.time()                

ppo_agent.buffer.actions.extend(torch.as_tensor(action, device=ppo_agent.device).double() for action in result[0][0])
ppo_agent.buffer.logprobs.extend(torch.as_tensor(logprob, device=ppo_agent.device).double() for logprob in result[0][1])
ppo_agent.buffer.rewards.extend(result[0][2])
ppo_agent.buffer.terminals.extend(result[0][3])
ppo_agent.buffer.states.extend(torch.as_tensor(state, device=ppo_agent.device).double() for state in result[0][4])
returns.extend(result[1])
end_time = time.time()

print(f"El tiempo de merge de datos {end_time-start_time}")
print()
ppo_agent.buffer.clear()
