from itertools import combinations, product
import numpy as np
import pickle

def pacum2pproduct(p_acumulada):
    p_j = []
    p_j.append(p_acumulada[0])
    for i in range(1,len(p_acumulada)):
        diff = -p_acumulada[i-1]+p_acumulada[i]
        p_j.append(diff)
    return p_j

def combinaciones_de_n(n):
    conjunto_N = set(range(0, n))  # Crear el conjunto N

    todas_combinaciones = []
    todas_combinaciones.append(())
    # Generar todas las combinaciones posibles del conjunto
    for r in range(1, n + 1):
        combinaciones_de_r = combinations(conjunto_N, r)
        todas_combinaciones.extend(list(combinaciones_de_r))

    return [list(tupla) for tupla in todas_combinaciones]

class env_p2p():

    def __init__(self, lambd, beta, C, T, f,model) -> None:

        self.lambd = lambd
        self.beta = beta
        self.C = C
        self.T = T
        self.f = f
        self.action_space = combinaciones_de_n(len(f))

        states_map = {}
        seats = np.arange(0, C + 1)
        time = np.arange(0, T)
        for idx, (x, t) in enumerate(product(seats, time)):
            states_map[(x, t)] = idx
        
        self.states_map = states_map
        self.model = model

    def step(self, a):
        x, t = self.state
        u = self.action_space[a]
        if u == []:
            reward = 0
            self.state = np.array([x , t+1])
        else:
            p_j_acumulada = np.exp(-self.beta*(self.f/np.min(self.f)-1))
            p_j = np.array(pacum2pproduct(p_j_acumulada))
            logits = np.hstack(((self.lambd)*p_j, (1-self.lambd)))
            sample = np.random.choice(len(logits), p=logits)
            
            if sample == len(self.f) or sample not in u:
                reward = 0
                self.state = np.array([x , t+1])
            else:
                reward = self.f[sample]
                self.state = np.array([x+1 , t+1])
        
        x, t = self.state
        done =  True if x == self.C or t == self.T-1 else False
        return self._get_obs(), reward, done, {}, {}

    def reset(self):
        self.state = [np.random.choice(self.C), np.random.choice(self.T-1)]
        return self._get_obs(), {}
    
    def set_initial(self, s):
        self.state = s
        return self._get_obs(), {}


    def get_state(self):
        return self.state


    def _get_obs(self):
        x, t = self.state
        if self.model == "QL":
            return self.states_map[x,t]
        if self.model == "DQL":
            return np.array([x , t])

def P_lj(S,l,j,v_lj,model):
    if model == "Exp":
        return v_lj[l,j+1]/(v_lj[l,0]+(v_lj[l,1:]).sum())
    if model == "MNL":
        return v_lj[l,j+1]/(v_lj[l,0]+(S*v_lj[l,1:]).sum())


#Probaiblidad de que se selecione un producto j, por cualquier tipo de cliente
def P_j(S,L,j,v_lj,p_l,model):
    return sum([p_l[l]*P_lj(S,l,j,v_lj,model) for l in range(L)])

#Distribución de reserva de los productos
def P_j_dist(a,L,v_lj,p_l,lambd,model):
    list = []
    for j, a_j in enumerate(a):
        if a_j == 1:
            list.append(lambd*P_j(a,L,j,v_lj,p_l,model))
        else:
            list.append(0)
    P_reserva = sum(list)
    list.insert(0, 1-P_reserva)
    return list

def fligths_full(x,C):
    return np.linalg.norm(x-C) == 0

def checkfilter(actual_sell,C,A_ij,filter):
    for i in range(len(C)):
        if actual_sell[i] == C[i]:
            for j in range(len(filter)):
                if A_ij[i,j] == 1:
                    filter[j] = 0
    return filter

def integer_to_binary_tuple(integer, word_size):
    # Obtener la representación binaria del número entero sin el prefijo '0b'
    binary_str = bin(integer)[2:]

    # Asegurarse de que la cadena binaria tenga el tamaño deseado llenando con ceros a la izquierda si es necesario
    binary_str = binary_str.zfill(word_size)

    # Crear una tupla con cada bit del número binario
    binary_tuple = tuple(int(bit) for bit in binary_str)

    return binary_tuple

class env_red:

    def __init__(self,L,v_lj,r_j,C,T,I,A_ij,J,p_l,lambd,demand_model = "Exp") -> None:

        #Max time steps
        self.T = T
        #Capacity vector
        self.C = C
        #Numer of fligth
        self.I = I
        #Numer of products
        self.J = J
        #Fare per product
        self.r_j = r_j
        #Numer of type of client
        self.L = L
        #Adjacenci fligth-product
        self.A_ij = A_ij
        #MNL logits
        self.v_lj = v_lj
        #MNL filtes
        self.filter = np.ones(J)
        #Probabilidad de cliente
        self.p_l = p_l
        #Self lambda
        self.lambd = lambd
        #Tipe of demand model
        self.model = demand_model
        
        #Action space
        self.action_space = []
        
    def step(self,u):
        x = self.x
        t = self.t
        if len(self.action_space) == 0:
            a = integer_to_binary_tuple(u, self.J)
        else:
            a = self.action_space[u]
        if self.model == "MNL":
            a = a*self.filter
        dist = P_j_dist(a,self.L,self.v_lj,self.p_l,self.lambd,self.model)
        sample = np.random.choice(len(dist), p=dist)
        if sample == 0:
            reward = 0
            self.t = t+1
        else:
            product = sample - 1
            reward = self.r_j[product]
            delta_x = np.squeeze(np.transpose(self.A_ij[:,product]))
            for i in range(self.I):
                if self.x[i] == self.C[i]:
                    if self.A_ij[i,product] == 1:
                        delta_x = np.zeros(self.I)
                        reward = 0
            self.x = x + delta_x
            if self.model == "MNL":
                self.filter = checkfilter(self.x,self.C,self.A_ij,self.filter)
            self.t = t+1
        done =  True if fligths_full(self.x, self.C) or (self.t >= (self.T-1)) else False
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
    

def env_red_p2p():
    
    # Max time for departure
    T = 50 #Max time
    time = np.arange(1, T) #TIme index

    #Set of products
    J = 5 #Num of products
    r_j = [100,200,300,400,500] #Price for products

    #Set of resoruces
    I = 1 #Soruce/Fly legs
    c_i = np.array([50]) #Capacity for fly leg

    #Customer segment
    L = 1 #Types of customer
    p_l = np.array([1]) #Probabilidad de pertenecer a un segmento
    lambd = 0.3 #PRobabilidad de llegada de un cliente
    lambd_l = lambd*p_l

    #Adjacency matrix Fligt products
    A_ij = np.array([
        [1, 1, 1, 1, 1]
    ])

    with open('results/p_j.pickle', 'rb') as archivo:
        p_j = np.flip(pickle.load(archivo))
        
    #Weigjt of MNL
    v_lj = np.array([
        np.insert(p_j, 0, 0)
    ])
    env = env_red(L,v_lj,r_j,c_i,T,I,A_ij,J,p_l,lambd,model = "MNL")

    return env
    
def env_red0():

    # Max time for departure
    T = 50 #Max time
    time = np.arange(0, T) #TIme index

    #Set of products
    J = 6 #Num of products
    r_j = [100,500,150,550,220,1000] #Price for products

    #Set of resoruces
    I = 2 #Soruce/Fly legs

    #Explore diferent initials capacity
    alpha = 1
    c_i = alpha*np.array([50,50]) #Capacity for fly leg

    #Customer segment
    L = 4 #Types of customer
    p_l = np.array([0.1, 0.15, 0.2, 0.05]) #Probabilidad de pertenecer a un segmento
    lambd = 1 #PRobabilidad de llegada de un cliente
    lambd_l = lambd*p_l

    #Adjacency matrix Fligt products
    A_ij = np.array([
        [1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1]
    ])

    #Weigjt of MNL
    v_0 = [1,5,5,1]
    v_lj = np.array([
        [v_0[0],  5,  10,  5, 10,  5, 10], 
        [v_0[1],  5,  0,  1,  0, 10, 0],
        [v_0[2], 10,  8,  6,  4,  3, 1],
        [v_0[3],  8, 10,  4,  6,  1, 3],
    ])

    env = env_red(L,v_lj,r_j,c_i,T,I,A_ij,J,p_l,lambd)

    return env

def env_red_toy(model):
    # Max time for departure
    T = 50 #Max time
    time = np.arange(1, T) #TIme index

    #Set of products
    J = 6 #Num of products
    r_j = [400,800,500,1000,300,600] #Price for products

    #Set of resoruces
    I = 2 #Soruce/Fly legs

    #Explore diferent initials capacity
    alpha = 1
    c_i = alpha*np.array([50,50]) #Capacity for fly leg

    #Customer segment
    L = 4 #Types of customer
    p_l = np.array([0.1, 0.15, 0.2, 0.05]) #Probabilidad de pertenecer a un segmento
    lambd = 1 #PRobabilidad de llegada de un cliente
    lambd_l = lambd*p_l

    #Adjacency matrix Fligt products
    A_ij = np.array([
        [1, 1, 0, 0, 1, 1],
        [0, 0, 1, 1, 1, 1]
    ])

    #Weigjt of MNL
    v_0 = [1,5,5,1]
    v_lj = np.array([
        [v_0[0],  0,  5,  0, 10,  0, 1],
        [v_0[1],  5,  0,  1,  0, 10, 0],
        [v_0[2], 10,  8,  6,  4,  3, 1],
        [v_0[3],  8, 10,  4,  6,  1, 3],
    ])

    env = env_red(L,v_lj,r_j,c_i,T,I,A_ij,J,p_l,lambd,demand_model=model)

    return env

def env_red_toy1(model,T):
    # Max time for departure
    #T = 300 #Max time
    time = np.arange(1, T) #TIme index

    #Set of products
    J = 10 #Num of products
    r_j = [400,800,500,1000,200,600,700,1500,500,1100] #Price for products

    #Set of resoruces
    I = 3 #Soruce/Fly legs

    #Explore diferent initials capacity
    alpha = 1
    c_i = alpha*np.array([50,50,50]) #Capacity for fly leg

    #Customer segment
    L = 4 #Types of customer
    p_l = np.array([0.1, 0.15, 0.2, 0.05]) #Probabilidad de pertenecer a un segmento
    lambd = 1 #PRobabilidad de llegada de un cliente
    lambd_l = lambd*p_l

    #Adjacency matrix Fligt products
    A_ij = np.array([
        [1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 0, 0, 1, 1]
    ])

    #Weigjt of MNL
    v_0 = [1,5,5,1]
    v_lj = np.array([
        [v_0[0],  0,  5,  0, 10, 0, 7, 0, 1, 0, 1],
        [v_0[1],  5,  0,  1,  0, 4, 0, 3, 0, 2, 0],
        [v_0[2], 10,  8,  6,  4, 1, 0, 3, 1, 2, 1],
        [v_0[3],  8, 10,  4,  6, 0, 1, 1, 3, 1, 2],
    ])

    env = env_red(L,v_lj,r_j,c_i,T,I,A_ij,J,p_l,lambd,demand_model=model)

    return env

def env_red1():

    # Max time for departure
    T = 1000 #Max time
    time = np.arange(1, T) #TIme index

    #Set of products
    J = 22 #Num of products
    r_j = [1000,400,400,300,300,500,500,600,600,700,700,
        500,200,200,150,150,250,250,300,300,350,350] #Price for products

    #Set of resoruces
    I = 7 #Soruce/Fly legs

    #Explore diferent initials capacity
    alpha = 1
    c_i = alpha*np.array([100,150,150,150,80,150,80]) #Capacity for fly leg

    #Customer segment
    L = 10 #Types of customer
    p_l = np.array([0.08, 0.2, 0.05, 0.2, 0.1, 0.15, 0.02, 0.05, 0.02, 0.04]) #Probabilidad de pertenecer a un segmento
    lambd = 1 #PRobabilidad de llegada de un cliente
    lambd_l = lambd*p_l

    #Adjacency matrix Fligt products
    A_ij = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    ])

    #Weigjt of MNL
    v_0 = [5,10]
    v_lj = np.array([
        [v_0[0], 10,  0,  0,  0,  0,  0, 0, 8, 8,  0, 0, 6,  0,  0,  0, 0,  0, 0,  4,  4,  0,  0],
        [v_0[1],  1,  0,  0,  0,  0,  0, 0, 2, 2,  0, 0, 8,  0,  0,  0, 0,  0, 0, 10, 10,  0,  0],
        [v_0[0],  0, 10, 10,  0,  0,  0, 0, 0, 0,  0, 0, 0,  5,  5,  0, 0,  0, 0,  0,  0,  0,  0],
        [v_0[1],  0,  2,  2,  0,  0,  0, 0, 0, 0,  0, 0, 0, 10, 10,  0, 0,  0, 0,  0,  0,  0,  0],
        [v_0[0],  0,  0,  0, 10, 10,  0, 0, 0, 0,  0, 0, 0,  0,  0,  5, 5,  0, 0,  0,  0,  0,  0],
        [v_0[1],  0,  0,  0,  2,  2,  0, 0, 0, 0,  0, 0, 0,  0,  0, 10, 8,  0, 0,  0,  0,  0,  0],
        [v_0[0],  0,  0,  0,  0,  0, 10, 8, 0, 0,  0, 0, 0,  0,  0,  0, 0,  5, 5,  0,  0,  0,  0],
        [v_0[1],  0,  0,  0,  0,  0,  2, 2, 0, 0,  0, 0, 0,  0,  0,  0, 0, 10, 8,  0,  0,  0,  0],
        [v_0[0],  0,  0,  0,  0,  0,  0, 0, 0, 0, 10, 8, 0,  0,  0,  0, 0,  0, 0,  0,  0,  5,  5],
        [v_0[1],  0,  0,  0,  0,  0,  0, 0, 0, 0,  2, 2, 0,  0,  0,  0, 0,  0, 0,  0,  0, 10, 10]
    ])

    env = env_red(L,v_lj,r_j,c_i,T,I,A_ij,J,p_l,lambd)

    return env

def env_hubs0():

    # Max time for departure
    T = 1000 #Max time
    time = np.arange(1, T) #TIme index

    #Set of products
    J = 66 #Num of products
    r_j = np.array([300,400,350,300,500,300,400,350,300,
           750,800,800,700,    750,800,800,700,
           1050,1100,1100,1050,
           1150,1200,1200,1150,
           1100,1150,1150,1100,
           1050,1100,1100,1050,]) #Price for products

    r_j = np.concatenate((r_j, 1.5*r_j), axis=0)

    #Set of resoruces
    I = 9 #Soruce/Fly legs

    #Explore diferent initials capacity
    alpha = 1
    c_i = alpha*np.array([100,80,100,100,150,100,80,100,100]) #Capacity for fly leg

    #Customer segment
    L = 6 #Types of customer
    p_l = np.array([0.08, 0.2, 0.05, 0.2, 0.1, 0.15, 0.02, 0.05, 0.02, 0.04]) #Probabilidad de pertenecer a un segmento
    lambd = 1 #PRobabilidad de llegada de un cliente
    lambd_l = lambd*p_l

    #Adjacency matrix Fligt products
    A_ij = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    ])

    A_ij = np.concatenate((A_ij, A_ij), axis=0)

    #Weigjt of MNL
    v_0 = [5,10]
    v_j1 = [20,10]
    v_j2 = [4,20]

    print("Hola")
    def compute_v_lj(v_0, v_j1, v_j2):
        v_lj = np.zeros((2*J,2*J+1))
        for j in range(J):
            v_lj[2*j,0] = v_0[0]
            v_lj[2*j+1,0] = v_0[1]

            v_lj[2*j,j+1] = v_j1[0]
            v_lj[2*j,j+1+J] = v_j1[1]
            
            v_lj[2*j+1,j+1] = v_j2[0]
            v_lj[2*j+1,j+1+J] = v_j2[1]
        return v_lj

    v_lj = compute_v_lj(v_0, v_j1, v_j2)

    env = env_red(L,v_lj,r_j,c_i,T,I,A_ij,J,p_l,lambd)
    
    return env