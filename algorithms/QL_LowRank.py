import random
from dataclasses import dataclass
import torch
import numpy as np
from .utils import sim_trayectorias

@dataclass
class Transition:
    state: int
    action: int
    next_state: int
    reward: float
    done: bool

class ReplayBuffer(object):
    def __init__(self, size):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def push(self, *args):
        data = Transition(*args)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def sample_batch(self, batch_size):
        return [self._storage[random.randint(0, len(self._storage) - 1)] for _ in range(batch_size)]

#Cambiar
def select_action(qtensor, s, epsilon, nA):
    if np.random.rand() < epsilon:
        return np.random.choice(nA)
    with torch.no_grad():
        return qtensor(torch.tensor(np.array(s),dtype=torch.long).view(1, -1)).argmax().item()
    

def train(qtensor, buffer, opt, gamma,batch_size):
    sample = buffer.sample_batch(batch_size)
    s = torch.tensor(np.array([t.state for t in sample]), dtype=torch.double)
    s_prime = torch.tensor(np.array([t.next_state for t in sample]), dtype=torch.double)
    a = torch.tensor([t.action for t in sample])
    r = torch.tensor([t.reward for t in sample])
    d = 1 - 1*torch.tensor([t.done for t in sample])

    def closure():
        opt.zero_grad()
        q = qtensor(torch.cat((s, a.unsqueeze(1)), dim=1).long())
        q_expected = r + gamma * d * qtensor(s_prime.long()).max(dim=1).values
        criterion = torch.nn.MSELoss()
        loss = criterion(q, q_expected)
        loss.backward()
        return loss
    opt.step(closure)

def LRQL_algorithm(max_episodes,max_steps,buffer_size, env, qtensor,eps_decay,opt,gamma,batch_size,nA,eps= 1):

    Rs = []
    Eps = []
    buffer = ReplayBuffer(buffer_size)

    # Cada cuanto se muestrean las trayectorias
    muestreador = 100
    #Numero de simulacionas para muestrear el comportamiento 
    num_sim = 10

    for episode in range(max_episodes):
        s, _ = env.reset()
        R = 0
        for step in range(max_steps):
            a = select_action(qtensor, s, eps,nA)
            s_prime, r, done, _, _ = env.step(a)
            buffer.push(s, a, s_prime, r, done)
            R += r
        
            if done:
                break
            s = s_prime
            eps *= eps_decay
        train(qtensor, buffer, opt, gamma, batch_size)

        #Probar como de bien funciona sin modificar
        if episode % muestreador == 0:
            Rsamplig, _, _, _ = sim_trayectorias(env,qtensor,num_sim,max_steps,nA,model = "LR")
            Rs.append(Rsamplig)

        #Rs.append(R)
        Eps.append(eps)
        print(f'{episode}/{max_episodes}: {sum(Rs[-1])/(len(Rs[-1]))} \r', end='')

    return qtensor, Rs, Eps