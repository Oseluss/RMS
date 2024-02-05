import numpy as np
from enviroments import P_j, P_j_dist
import torch

def caculate_op_cost(I,qnet):
    x = np.zeros(I)
    t = 0
    s_array = np.append(x, t)
    op_cost = []

    for i in range(I):
        e_i = np.zeros(I+1)
        e_i[i] = 1
        e_i[I] = t+1
        V_x = torch.max(qnet(torch.tensor(s_array, dtype=torch.double)))
        V_x_ei = torch.max(qnet(torch.tensor(s_array+e_i, dtype=torch.double)))
        op_cost.append((V_x-V_x_ei).detach().item())
    return op_cost

def phi(S,J,L,j,v_lj,p_l,w_j,model):
    a = np.zeros(J)
    for t in S:
        a[t] = 1
    list = []
    for j, a_j in enumerate(a):
        if a_j == 1:
            list.append(P_j(a,L,j,v_lj,p_l,model)*w_j[j])
    return sum(list)

def Action_generation(env,op_cost,action_list):
    Action_gen_ends = False

    #Potential new offersets
    Psi = []

    #New potential offerset
    S = set()

    #The produt j is in the new offerset (S) or not
    y_j = np.zeros(env.J) 

    #Set of unassigned y_jnew_offerset
    S_p = set()

    # Diference between price and bid price

    w_j = np.array([env.r_j[j] - sum(env.A_ij[:,j]*op_cost) for j in range(env.J)])

    for j, w in enumerate(w_j):
        if w <= 0:
            y_j[j] = 0
        else:
            S_p.add(j)

    #Mirar esta linea --> Actualizarla
    dist = P_j_dist(np.ones(env.J),env.L,env.v_lj,env.p_l,env.lambd,env.model)
    j_1 = np.argmax(w_j*dist[1:])

    y_j[j_1] = 0
    S.add(j_1)
    S_p.remove(j_1)

    parada = True
    while parada:
        list = []
        phi_new = 0
        j_new = None
        for j in S_p:
            phi_j = phi(S.union({j}),env.J,env.L,j,env.v_lj,env.p_l,w_j,env.model)
            if phi_new < phi_j:
                phi_new = phi_j
                j_new = j
            list.append([j,phi_j])
        if phi(S.union({j_new}),env.J,env.L,j,env.v_lj,env.p_l,w_j,env.model) > phi(S,env.J,env.L,j,env.v_lj,env.p_l,w_j,env.model):
            S.add(j_new)
            S_p.remove(j_new)
            Psi.append(S.copy())
        else:
            parada = False
    
    pop_list = []
    for i, s in enumerate(Psi):
        if s in action_list:
            pop_list.append(s)

    for s in pop_list:
        Psi.remove(s)


    if len(Psi) == 0:
        Action_gen_ends = True
    else:
        S = Psi[-1]

    return Action_gen_ends, S