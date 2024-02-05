
from .DQL import DQL_algorithm
from .Action_Gen import Action_generation, caculate_op_cost
from itertools import product
import random
import torch
import numpy as np
from .models import QfunNN

def action_space_generation(J,ini_num_actions):
    return random.sample(list(product([0,1], repeat=J)), ini_num_actions)

def ini_action_list(action_space):
    action_list = []
    for action in action_space:
        a = []
        for j, a_j in enumerate(action):
            if a_j == 1:
                a.append(j)
        action_list.append(set(a))
    return action_list
        

def update_action_space(action, env):
    new_action = [0] * env.J
    for a in action:
        new_action[a] = 1
    env.action_space.append(tuple(new_action))

    
def RDQL_algorithm(max_episodes,max_steps, buffer_size, env,eps_decay,gamma,batch_size,lr_alpha, nA_initial,num_hiddens):

    nA = nA_initial
    nS = env.I + 1

    #Generate random action sapce
    a_space = action_space_generation(env.J,nA)

    #Intalice action spaces in the enviroment
    env.action_space = a_space

    #Generate an action_list to the A_gen algorithm
    action_list = ini_action_list(a_space)

    R_over = []

    end = False

    while not end:
        
        qnet = QfunNN(nS, num_hiddens, nA).double()

        opt = torch.optim.Adam(qnet.parameters(), lr=lr_alpha)

        #Con el primer espacio de acciones comupatamos la función Q usando DQN
        qnet, R, eps = DQL_algorithm(max_episodes,max_steps,buffer_size, env, qnet,eps_decay,opt,gamma,batch_size,nA)

        R_over.append(np.array(R).ravel())
        
        #Calculamos los op_costs/bid prices
        op_cost = caculate_op_cost(env.I,qnet)
        
        #Con action gen añadimos una nueva acción
        end, S = Action_generation(env,op_cost,action_list)
        if not end:
            nA += 1
            action_list.append(S.copy())
            update_action_space(S.copy(), env)

    return qnet, env, R_over