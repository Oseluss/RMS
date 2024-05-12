from itertools import combinations, product
import numpy as np
import pickle
import time
import torch

class env_red_count:

    def __init__(self,L,c_j,C,T,I,A_ij,J,lambd_m,betha) -> None:

        #Max time steps
        self.T = T
        #Capacity vector
        self.C = C
        #Numer of fligth
        self.I = I
        #Numer of products
        self.J = J
        #Fare per product
        self.c_j = c_j
        #Numer of type of client
        self.L = L
        #Adjacenci fligth-product
        self.A_ij = A_ij
        #MNL filtes
        self.filter = np.ones(J)
        #Probabilidad de cliente
        #self.p_l = p_l
        #Self lambda_m
        self.lambd_m = lambd_m
        #Self Wilness to Pay
        self.betha = betha
    
    #Distribución de compra de un cliente l siendo el primer elemento la probabilidad de no comprar
    #Se asume con esta función que un cliente l, solo está interesado en la compra del producto l-ésimo
    def P_j_dist_dado_m(self,a,m):
        a_filter = a[m]*self.filter[m]
        betha = self.betha[m]
        c = self.c_j[m]

        P_j = np.zeros(self.J+1)
        P_j[m+1] = np.exp(-betha*a_filter)/(np.exp(-betha*a_filter)+ np.exp(-betha*c))
        P_j[0] = 1 - np.sum(P_j[1:])
        return P_j
        
        
    def fligths_full(self):
        return np.linalg.norm(self.x - self.C) == 0

    #El filtro se usa para evitar la compra de un ticket cuyos asientos esten vendidos
    def checkfilter(self):
        for i in range(len(self.C)):
            if self.x[i] == self.C[i]:
                for j in range(len(self.filter)):
                    if self.A_ij[i,j] == 1:
                        self.filter[j] = 0
        return self.filter

    def step(self,u):
        x = self.x
        t = self.t
        
        a = np.array(u)

        reward = 0
        for m, lamb_m in enumerate(self.lambd_m):
            if np.random.choice([0,1],p=[1-lamb_m, lamb_m]) == 1:
                dist = self.P_j_dist_dado_m(a,m)
                sample = np.random.choice(len(dist), p=dist)
            else:
                sample = 0
            
            if sample == 0:
                delta_reward = 0
            else:
                product = sample - 1
                delta_reward = a[product]
                delta_x = np.squeeze(np.transpose(self.A_ij[:,product]))
                for i in range(self.I):
                    if x[i] == self.C[i]:
                        if self.A_ij[i,product] == 1:
                            delta_x = np.zeros(self.I)
                            delta_reward = 0
                x = x + delta_x
                self.filter = self.checkfilter()
            reward += delta_reward   
        self.x = x
        self.t = t+1
        done =  True if self.fligths_full() or (self.t >= (self.T-1)) else False
        return self._get_obs(), reward, done, {}, {}

    def reset(self):
        self.filter = np.ones(self.J)
        self.x = np.array([np.random.choice(c) for c in self.C])
        self.t = np.random.choice(self.T-1)
        return self._get_obs(), {}

    def set_initial(self, s):
        self.filter = np.ones(self.J)
        self.x = s
        self.t = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.append(self.x, self.t)
    


def env_hubs_cont_1(model="MNL", T=300):

    time = np.arange(1, T) #TIme index
    #Mercados
    M = 6
    lambd_m = np.array([1]*M)

    #Tarifas por mercado
    F = 1
    #Set of products
    J = int(M) #Num of products

    #Precio de la competencia
    c_j = np.array([500,700,500,
                    800,750,
                    1100]) #Price for products

    #Set of resoruces
    I = 3 #Soruce/Fly legs

    #Explore diferent initials capacity
    alpha = 1
    c_i = alpha*np.array([30,50,40]) #Capacity for fly leg

    #Customer segment per market
    L = 1 #Types of customer

    #Adjacency matrix Fligt products
    A_ij = np.array([
        [1, 0, 0, 1, 0, 1],
        [0, 1, 0, 1, 1, 1],
        [0, 0, 1, 0, 1, 1],
    ])

    betha = [0.01]*M

    env = env_red_count(L,c_j,c_i,T,I,A_ij,J,lambd_m,betha)
    
    return env