import numpy as np

def select_action(Q, s, epsilon, nA, action_space):
    if np.random.rand() < epsilon:
        return np.random.choice(nA)
    else:
        s= action_space[tuple(s)]
        a_t = np.argmax(Q[s, :])
        return a_t

def QL_algorithm(Q,env,max_episodes,max_steps,gamma,alpha, eps_decay, nS, nA ,action_space ,eps = 1.0):

    Rs = []
    Eps = []

    for episode in range(max_episodes):
        s, _ = env.reset()
        R = 0
        for step in range(max_steps): 
            a = select_action(Q, s, eps, nA, action_space)
            s_prime, r, done, _, _ = env.step(a)
            
            R += r
            Q[s, a] += alpha*(r + gamma*np.max(Q[s_prime, :]) - Q[s, a])

            if done:
                break
                
            s = s_prime
            eps *= eps_decay

        Rs.append(R)
        Eps.append(eps)
        print(f'{episode}/{max_episodes}: {Rs[-1]} \r', end='')

    return Q, Rs, eps