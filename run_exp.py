import numpy as np
import torch
from itertools import product
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
import random
from enviroments import env_hubs0
import pickle
from algorithms.Action_Gen import Action_generation
from algorithms.RDQL import RDQL_algorithm
import time
from algorithms.models import QfunNN, QfunPARAFAC
from algorithms.DQL import DQL_algorithm
from algorithms.QL_LowRank import LRQL_algorithm

NUMERO_DE_NUCLEOS = 64
torch.set_num_threads(NUMERO_DE_NUCLEOS)

name_exp = "Exp6"
Red_name = "hubs0"
Demand_Model = "Exp" #Puede ser EXP/MNL
Qfun_model = "LR" #Puede ser LR/NN

#Tamaño del run
n = 1

T = 2000
env = env_hubs0(Demand_Model,T)

#Hiperarametros del algoritmo
gamma = 1
alpha = 1e-3
eps = 1.0
eps_decay = 0.999997**(1/n*(env.T/50))
batch_size = 200

#Tamaño de la ejecución
max_episodes = 70000*n
max_steps = T

#Hiperparametros del modelo
if Qfun_model == "NN":
    num_inputs = env.I + 1
    exp_layers =[
        [128,128,128]
    ]
    num_outputs = len(env.action_space)
    num_exp = len(exp_layers)

elif Qfun_model == "AG":
    nA_initial = 5
    exp_layers =[
        [64,64],
        [128],
        [64],
        [32]
    ]
    num_exp = len(exp_layers)

elif Qfun_model == "LR":
    exp_ranks = [5,15,20]
    num_exp = len(exp_ranks)
    tensor_dims = list(env.C + 1)
    tensor_dims.append(env.T+1)
    tensor_dims.append(2**(env.J))


R_exp = []
time_exp = []
qfun_exp = []
a_space_exp = [] 

for i in range(num_exp):

    print(f"Experimento {i}:")
    if Qfun_model == "NN":
        qfun = QfunNN(num_inputs, exp_layers[i], num_outputs).double()
        opt = torch.optim.Adam(qfun.parameters(), lr=alpha)
        
        start_time = time.time()
        qfun, Rs, eps = DQL_algorithm(max_episodes,max_steps,50_000, env, qfun,eps_decay,opt,gamma,batch_size,num_outputs)
        end_time = time.time()
        execution_time = end_time - start_time

    elif Qfun_model == "AG":
        start_time = time.time()
        qfun, env, Rs = RDQL_algorithm(max_episodes,max_steps, env,eps_decay,gamma,batch_size,alpha, nA_initial,exp_layers[i])
        end_time = time.time()
        execution_time = end_time - start_time

    elif Qfun_model == "LR":
        qfun = QfunPARAFAC(dims= tensor_dims, k = exp_ranks[i], scale= 0.1)
        opt = torch.optim.Adam(qfun.parameters(), lr=alpha)

        start_time = time.time()
        qfun, Rs, eps = LRQL_algorithm(max_episodes,max_steps,10_000, env, qfun ,eps_decay,opt,gamma,batch_size,2**(env.J))
        end_time = time.time()
        execution_time = end_time - start_time

    R_exp.append(Rs)
    time_exp.append(execution_time)
    qfun_exp.append(qfun)
    a_space_exp.append(env.action_space)


import os
# Asegurarse de que el directorio exista, si no existe, créalo
directory = "results/" + name_exp
if not os.path.exists(directory):
    os.makedirs(directory)
exp = {}
exp["R_exp"] = R_exp
exp["Time_exp"] = time_exp
exp["qfun_exp"] = qfun_exp
exp["a_space_exp"] = a_space_exp

with open("results/"+ name_exp +"/" + Demand_Model + "_" + Qfun_model +"_" + Red_name +".pickle", 'wb') as f:
    pickle.dump(exp, f)