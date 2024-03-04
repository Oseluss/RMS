import algorithms.train as train
import algorithms.utils as utils
import algorithms.models as models
from enviroments import env_red, env_red_p2p, env_red_toy1, env_hubs1, env_hubs2
from algorithms.agents import reinforce, trpo, ppo
import matplotlib.pyplot as plt
import numpy as np
import torch
import gym
import time
import pickle

name_exp = "Exp15"
Red_name = "hub2"
Demand_Model = "Exp" #Puede ser EXP/MNL
Qfun_model = "NN-PG" #Puede ser LR/NN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

env = env_hubs2(model="Exp", T=2000)
M = 13
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

re_agent = reinforce.ReinforceSoftmaxNN(actor, critc, gamma=.99, tau=.99, lr_actor= 1e-5,device=device)
trpo_agent = trpo.TRPOSoftmaxNN(actor, critc, gamma=.999, tau=.999, delta=.01, cg_dampening=0.3, cg_tolerance=1e-10, cg_iteration=15,device=device)
ppo_agent = ppo.PPOSoftmaxNN(actor, critc,gamma=.999, tau=.999, lr_actor=1e-4, epochs=1000, eps_clip=0.005,device=device)

R_exp = []
time_exp = []
qfun_exp = []
pg_model = []

PG_MODEL = "PPO"
epochs=20000
max_steps=2000
update_freq=8000
initial_offset=0

if PG_MODEL == "PPO":
    start_time = time.time()
    agent, totals,_ = Trainer.train(env, re_agent, epochs, max_steps, update_freq, initial_offset)
    end_time = time.time()
    execution_time = end_time - start_time

elif PG_MODEL == "TRPO":
    start_time = time.time()
    agent, totals,_ = Trainer.train(env, trpo_agent, epochs, max_steps, update_freq, initial_offset)
    end_time = time.time()
    execution_time = end_time - start_time

elif PG_MODEL == "PPO":
    start_time = time.time()
    agent, totals,_ = Trainer.train(env, ppo_agent, epochs, max_steps, update_freq, initial_offset)
    end_time = time.time()
    execution_time = end_time - start_time

pg_model.append(PG_MODEL)
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
exp["PG_model"] = pg_model

with open("results/"+ name_exp +"/" + Demand_Model + "_" + Qfun_model +"_" + Red_name +".pickle", 'wb') as f:
    pickle.dump(exp, f)

from algorithms.utils import generate_random_colors

colors = generate_random_colors(1)
totals = R_exp[0]
Rs = np.array(totals).ravel()

mean = np.array([np.mean(Rs[i:i+100]) for i in range(len(Rs) - 10)])
std = np.array([np.std(Rs[i:i+100]) for i in range(len(Rs) - 10)])
time = np.arange(mean.size)
plt.plot(mean,color='b')
#plt.plot(Rsim, color = 'r')
plt.fill_between(time, mean - std, mean + std, color='b', alpha=0.2)

plt.xlim(0, len(Rs))
plt.grid()
plt.ylabel("Return")
plt.legend()
plt.title("LRQL-Returns de 10 trayectorias cada 100 episodios")
    
plt.savefig("results/"+ name_exp +"/Returns_" +Demand_Model + "_" + Qfun_model +"_" + Red_name +".png")