import Covid19_model as Cm
from Reinforce_learn import *
import time
from itertools import product
import pandas as pd
from save_training import * 
from load_initialize import *

def offline_stage(n_site, budget, iter_num):
    # Set model parameters
    istrain = 1
    D = 1e2
    pop = np.array(pd.read_csv("population_"+str(n_site)+".csv",header=None))
    pop = np.reshape(np.ones((6, 1)) * pop, (6, n_site))
    init_state = np.array(pd.read_csv("state_"+str(n_site)+".csv",header=None))/pop
    init_state = init_state.tolist()
    density = np.array(pd.read_csv("density_"+str(n_site)+".csv",header=None))
    dij = np.array(pd.read_csv("distance_"+str(n_site)+".csv",header=None)) / D
    N = np.sum(init_state)
    init_state.append([1])
    init_state.append([0]*n_site)
    n_state = 7 * n_site + 1
    n_action = 5 * n_site
    density = np.reshape(density, (1, n_site))
    T = 90


    deter_para = dict(N=N, P=pop, B=budget, T=T, D=dij, site=n_site, density=density,
                      gamma=0.69, alpha=0.6, v_max = 0.2/14,
                      cost_ti=0.0977, cost_ta=0.02, cost_v=0.07, 
                      cost_poc_0=0.000369, cost_poc_1=0.001057, 
                      pid=1.10/1000, psr=0.7/3, pid_plus=0.1221/1000, pir=1/8, prs=1/180)

    in_deter_para = dict(beta2=[0.15204, 0.16287], beta1=[0.7602, 0.81435],
                         pei=[0.07143, 0.14286], per=[0.04000,0.05556])

    in_deter_truth = dict(beta2=0.15747, beta1=0.78735, pei=0.10714, per=0.04545)
    para_truth = {**deter_para, **in_deter_truth}

    # hyper-parameters
    noise = [round(0.1 * i, 1) for i in range(1, 4)]
    lr = [1e-4, 5e-4]
    units = [64, 128, 256]
    layer = [1, 2, 3]
    hyper = product(noise, lr, units, layer)
    if istrain == 0:
        agent = []
        for h in hyper:
            temp = Agent(state_size=n_state, action_size=n_action, initialize=0,
                         noise=h[0], lr=h[1], units=h[2], layer=h[3], site=n_site)
            agent.append(temp)
    else:
        agent = load('network_initialization/init_' + 'high', n_state, n_action, 54, n_site)
    env = Cm.Env_model(init_state, deter_para, in_deter_para, seed=0)
    env.set_para_truth(para_truth)
    tic = time.time()
    agent = train(agent, env, para_truth, iter_num)
    print(["computing time:", tic - time.time()])
    save_training(agent, 'result/transferRL_' + str(round(n_site)) + 'high')
