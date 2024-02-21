import algorithms.train as train
import algorithms.utils as utils
import algorithms.models as models
from enviroments import env_red, env_red_p2p, env_red_toy1, env_hubs1
from algorithms.agents import reinforce
import matplotlib.pyplot as plt
import numpy as np
import torch
import gym
import time
import pickle

name_exp = "Exp6"
Red_name = "hub1"
Demand_Model = "Exp" #Puede ser EXP/MNL
Qfun_model = "LR-PG" #Puede ser LR/NN

env = env_hubs1(model="Exp", T=2000)
M = 33
F = 2

#Dimensiónes del espacio de estados
dims_state = list(env.C + 1)
dims_state.append(env.T+1)

#Dimensiones del espacio de acciones
dims_action = [M, F]

dimensions_actor = dims_state + dims_action
dimensions_critic = dims_state

freq = 5000

lr_actor = 1e-5

k = 8

actor = models.PolicyPARAFAC(dimensions_actor, k=k, model= "SoftMax", scale = 1)
critc = models.ValuePARAFAC(dimensions_critic, k=k, scale = 1)

agent = reinforce.ReinforceSoftmaxNN(actor, critc, gamma=.99, tau=.99, lr_actor= 1e-6)
Trainer = train.Trainer("sgd", "sgd")

R_exp = []
time_exp = []
qfun_exp = []

start_time = time.time()
agent, totals,_ = Trainer.train(env, agent, epochs=5000, max_steps=2000, update_freq=8000, initial_offset=0)
end_time = time.time()
execution_time = end_time - start_time

R_exp.append(totals)
time_exp.append(execution_time)
qfun_exp.append(agent)

import os
# Asegurarse de que el directorio exista, si no existe, créalo
directory = "results/" + name_exp
if not os.path.exists(directory):
    os.makedirs(directory)

exp = {}
exp["R_exp"] = R_exp
exp["Time_exp"] = time_exp
exp["qfun_exp"] = qfun_exp

with open("results/"+ name_exp +"/" + Demand_Model + "_" + Qfun_model +"_" + Red_name +".pickle", 'wb') as f:
    pickle.dump(exp, f)

name_exp = "Exp6"
Red_name = "hub1"
Demand_Model = "Exp" #Puede ser EXP/MNL
Qfun_model = "NN-PG" #Puede ser LR/NN

env = env_hubs1(model="Exp", T=2000)
M = 33
F = 2

#Dimensiónes del espacio de estados
dims_state = list(env.C + 1)
dims_state.append(env.T+1)

#Dimensiones del espacio de acciones
dims_action = [M, F]

actor = models.PolicyNetwork(
    num_inputs=len(dims_state), 
    num_hiddens=[64, 64], 
    num_outputs=[M,F], 
    model="Softmax"
).double()

critc = models.ValueNetwork(
    num_inputs=len(dims_state), 
    num_hiddens=[64, 64], 
    num_outputs=1, 
).double()

agent = reinforce.ReinforceSoftmaxNN(actor, critc, gamma=.99, tau=.99, lr_actor= 1e-6)
Trainer = train.Trainer("sgd", "sgd")

R_exp = []
time_exp = []
qfun_exp = []

start_time = time.time()
agent, totals,_ = Trainer.train(env, agent, epochs=5000, max_steps=2000, update_freq=8000, initial_offset=0)
end_time = time.time()
execution_time = end_time - start_time

R_exp.append(totals)
time_exp.append(execution_time)
qfun_exp.append(agent)

import os
# Asegurarse de que el directorio exista, si no existe, créalo
directory = "results/" + name_exp
if not os.path.exists(directory):
    os.makedirs(directory)

exp = {}
exp["R_exp"] = R_exp
exp["Time_exp"] = time_exp
exp["qfun_exp"] = qfun_exp

with open("results/"+ name_exp +"/" + Demand_Model + "_" + Qfun_model +"_" + Red_name +".pickle", 'wb') as f:
    pickle.dump(exp, f)

from algorithms.utils import generate_random_colors

colors = generate_random_colors(1)

Rs = np.array(totals).ravel()

mean = np.array([np.mean(Rs[i:i+100]) for i in range(len(Rs) - 10)])
std = np.array([np.std(Rs[i:i+100]) for i in range(len(Rs) - 10)])
time = np.arange(mean.size)
plt.plot(mean,color='b')
plt.fill_between(time, mean - std, mean + std, color='b', alpha=0.2)

plt.xlim(0, len(Rs))
plt.grid()
plt.ylabel("Return")
plt.legend()
plt.title("LRQL-Returns de 10 trayectorias cada 100 episodios")
    
plt.savefig("results/"+ name_exp +"/Returns_" +Demand_Model + "_" + Qfun_model +"_" + Red_name +".png")