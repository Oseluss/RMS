import torch
from enviroments import env_hubs3_mult
import time
import algorithms.models as models
from algorithms.agents import ppo
import algorithms.train as train
import pickle

name_exp = "Exp26"
Red_name = "hub3_mult"
Demand_Model = "Exp" #Puede ser EXP/MNL
Qfun_model = "NN-PG7" #Puede ser LR/NN

device = torch.device("cuda")

env = env_hubs3_mult()
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

R_exp = []
time_exp = []
qfun_exp = []
pg_model = []

PG_MODEL = "PPO"
epochs=1
max_steps=2000
num_updates=12000
initial_offset=0
num_sim = 25


if PG_MODEL == "PPO":
    start_time = time.time()
    agent, totals,_ = Trainer.train2(env, ppo_agent, epochs, max_steps, num_updates,num_sim, initial_offset)
    #agent, totals,_ = Trainer.train(env, ppo_agent,  num_updates*num_sim, max_steps, env.T*num_sim, initial_offset)
    end_time = time.time()
    execution_time = end_time - start_time

pg_model.append(PG_MODEL)
R_exp.append(totals)
time_exp.append(execution_time)
qfun_exp.append(agent)

print(f"Tiempo de ejecución {execution_time}")

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