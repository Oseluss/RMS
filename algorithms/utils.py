import torch
from itertools import product
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.colors as mcolors

def select_action(net, s, epsilon, nA):
    if np.random.rand() < epsilon:
        return np.random.choice(nA)
    with torch.no_grad():
        return net(torch.tensor(np.array(s),dtype=torch.double)).argmax().item()
    
def select_action_LR(qtensor, s, epsilon, nA):
    if np.random.rand() < epsilon:
        return np.random.choice(nA)
    with torch.no_grad():
        return qtensor(torch.tensor(np.array(s),dtype=torch.long).view(1, -1)).argmax().item()

def sim_trayectorias(env,Qfun,num_sim,max_steps,nA,model = "NN"):
    
    Rs = []
    r_t= []
    s_t = []
    a_t = []

    for episode in range(num_sim):
        s, _ = env.set_initial(s = [0]*env.I)
        R = 0
        for step in range(max_steps):

            if model == "NN":
                a = select_action(Qfun, s, 0, nA)
            if model == "LR":
                a = select_action_LR(Qfun, s, 0, len(env.action_space))

            s_prime, r, done, _, _ = env.step(a)
            
            r_t.append(r)
            s_t.append(s)
            a_t.append(a) 

            R += r

            if done:
                break
                
            s = s_prime
        Rs.append(R)
    return Rs, r_t, s_t, a_t

def get_maximun(env, Qfun_list, model, dims):
    max_list = []
    C_1 = range(0, env.C[dims[0]])
    C_2 = range(0, env.C[dims[1]])

    for Qfun in Qfun_list:
        max_i = float('-inf')  # Inicializar el máximo como negativo infinito
        for (x1, x2, t_i) in (product(C_1, C_2, range(env.T-1))):
            s_list = [0] * env.I
            s_list[dims[0]] = x1
            s_list[dims[1]] = x2
            s_list.append(t_i)
            s = torch.tensor(s_list, dtype=torch.double)

            if model == "NN":
                 max_val = torch.max(Qfun(s)).item()
            elif model == "LR":
                max_val = torch.max(Qfun(torch.tensor(s_list, dtype=torch.long).view(1, -1))).item()

            max_i = max(max_i, max_val)

        max_list.append(max_i)

    return max_list

def saveGIFT(env, Qfun_list, model, dims, name):
    num_plots = len(Qfun_list)

    # Función para generar datos de la imagen
    def generate_data(frame, Qfun_list):
        t = frame
        all_data = []
        for Qfun in Qfun_list:
            data = np.zeros((len(C_1), len(C_2)))
            for (x1, x2) in (list(product(C_1, C_2))):
                s_list = [0] * (env.I)
                s_list[dims[0]] = x1
                s_list[dims[1]] = x2
                s_list.append(t)
                s = torch.tensor(s_list, dtype=torch.double)
                if model == "NN":
                    data[x1, x2] = torch.max(Qfun(s)).item()
                elif model == "LR":
                    data[x1, x2] = torch.max(Qfun(torch.tensor(np.array(s), dtype=torch.long).view(1, -1))).item()
            all_data.append(data)
        return all_data

    # Función para la animación
    def animate(frame, Qfun_list,max_list):
        plt.clf()  # Limpiar el gráfico en cada frame
        all_data = generate_data(frame, Qfun_list)
        for i, data in enumerate(all_data):
            plt.subplot(1, num_plots, i + 1)
            plt.imshow(data, cmap='inferno', interpolation='nearest', vmin=0,vmax=max_list[i])
            plt.colorbar()  # Agregar barra de colores para referencia
            plt.title(f'V(s)-{model}_toy, Frame {frame}')
            plt.xlabel(f'Asientos vendidos, x{dims[0]+1}')
            plt.ylabel(f'Asientos vendidos, x{dims[1]+1}')

    # Dims muestra las dimensiones de los asientos que se indiquen en el array
    C_1 = range(0, env.C[dims[0]])
    C_2 = range(0, env.C[dims[1]])

    # Crear la figura
    fig, ax = plt.subplots(1, num_plots, figsize=(5*num_plots, 5))

    # Crear la animación
    max_list =  get_maximun(env, Qfun_list, model, dims)
    animation = FuncAnimation(fig, animate, frames=env.T, interval=200, fargs=(Qfun_list,max_list,))
    
    # Guardar la animación en un archivo GIF
    animation.save(name, writer='imagemagick', fps=5)

def generate_random_colors(num_colors):
    # Generar valores de matiz (H)
    hue_values = np.linspace(0, 1, num_colors, endpoint=False)

    # Fijar saturación (S) y valor (V) para obtener colores brillantes
    saturation = 0.8
    value = 0.8

    # Convertir de HSV a RGB
    hsv_colors = np.ones((num_colors, 3))
    hsv_colors[:, 0] = hue_values
    hsv_colors[:, 1] = saturation
    hsv_colors[:, 2] = value
    rgb_colors = mcolors.hsv_to_rgb(hsv_colors)

    return rgb_colors

def compare_Qfun(env, Qfun1, Qfun2, model1, model2):

    s_vecors = [range(0,c) for c in env.C]
    s_vecors.append(range(env.T-1))
    dif = 0  # Inicializar el máximo como negativo infinito
    
    for s_i in (product(*s_vecors)):

        s = torch.tensor(s_i, dtype=torch.double)

        if model1 == "NN":
            val1 = torch.max(Qfun1(s)).item()
        elif model1 == "LR":
            val1 = torch.max(Qfun1(torch.tensor(s_i, dtype=torch.long).view(1, -1))).item()

        if model2 == "NN":
            val2 = torch.max(Qfun2(s)).item()
        elif model2 == "LR":
            val2 = torch.max(Qfun2(torch.tensor(s_i, dtype=torch.long).view(1, -1))).item()

        dif += abs(val1-val2)

    return dif

def compare_Qfun_exp(env,Qfun_list, model,name):
    data = np.zeros((len(Qfun_list), len(Qfun_list)))
    for q1 in range(len(Qfun_list)):
        for q2 in range(q1+1,len(Qfun_list)):
            diff = compare_Qfun(env, Qfun_list[q1], Qfun_list[q2], model, model)
            data[q1, q2] = diff
            data[q2, q1] = diff
    plt.imshow(np.array(data))
    plt.title("Diferencia en las funciones de valor")
    plt.xlabel("Exp1")
    plt.ylabel("Exp2")
    plt.colorbar()
    plt.savefig(name)