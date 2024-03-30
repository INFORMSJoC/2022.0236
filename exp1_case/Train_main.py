import Covid19_model as Cm
from Reinforce_learn import *
import time
from itertools import product
from save_training import *
from load_initialize import *
N = 100020
init_state = np.array([100000 / N, 20 / N, 0, 0, 0, 0, 1, 0])
n_state = 8
n_action = 5
T = 50
istrain = 1

def offline_stage(interval):
    """model parameters"""
    budget = 3.6e3
    iter_num = 200
    """system parameters"""
    deter_para = dict(N=N, B=budget, T=T, alpha=0.6, v_max=0.2/14,
                      cost_ti=0.0977, cost_ta=0.02, cost_v=0.07, cost_poc_0=0.000369,
                      cost_poc_1=0.001057, pid=1.10/1000, psr=0.7/3,
                      pid_plus=0.1221/1000, pir=1/8)

    in_deter_para = dict(beta2=[0.78735 * (1-interval/2), 0.78735 * (1+interval/2)],
                         beta1=[0.15747 * (1-interval/2), 0.15747 * (1+interval/2)],
                         pei=[0.10714 * (1-interval/2), 0.10714 * (1+interval/2)],
                         per=[0.04545 * (1-interval/2), 0.04545 * (1+interval/2)])
    
    in_deter_truth = dict(beta1=0.15747, beta2=0.78735, pei=0.10714, per=0.04545)
    para_truth = {**deter_para, **in_deter_truth}

    """hyper-parameters"""
    noise = [round(0.1 * i, 1) for i in range(1, 4)]
    lr = [1e-4, 5e-4]
    units = [64, 128, 256]
    layer = [1, 2, 3]
    hyper = product(noise, lr, units, layer)

    """Agent and Environment"""
    if istrain == 0:
        agent = []
        for h in hyper:
            temp = Agent(state_size=n_state, action_size=n_action, initialize=0,
                         noise=h[0], lr=h[1], units=h[2], layer=h[3])
            agent.append(temp)
    else:
        agent = load('network_initialization/init_' + str(round(interval * 100)), n_state, n_action, 54)
    
    env = Cm.Env_model(init_state, deter_para, in_deter_para, seed=0)
    env.set_para_truth(para_truth)
    # np.random.seed(0)
    tic = time.time()

    """Train"""
    agent = train(agent, env, iter_num)
    # print(["computing time:", tic - time.time()])
    save_training(agent, 'result/env_' + str(round(interval * 100)) + '_epd' + str(round(iter_num)))
