
from .DQL import DQL_algorithm
from .Action_Gen import Action_generation, caculate_op_cost
from itertools import product
import random
import torch
import numpy as np
from .models import QfunNN

def integer_to_binary_tuple(integer, word_size):
    # Obtener la representación binaria del número entero sin el prefijo '0b'
    binary_str = bin(integer)[2:]

    # Asegurarse de que la cadena binaria tenga el tamaño deseado llenando con ceros a la izquierda si es necesario
    binary_str = binary_str.zfill(word_size)

    # Crear una tupla con cada bit del número binario
    binary_tuple = tuple(int(bit) for bit in binary_str)

    return binary_tuple

def action_space_generation(J, ini_num_actions):
    action_space = list()  # Usamos un conjunto para asegurarnos de que no haya duplicados

    # Generamos ini_num_actions muestras únicas
    while len(action_space) < ini_num_actions:
        action = random.randint(0, 2**J-1)  # Genera un entero aleatorio de J bits
        action_space.append(integer_to_binary_tuple(action, J))  # Agrega la acción al conjunto

    return action_space

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