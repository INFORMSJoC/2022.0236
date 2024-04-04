import Covid19_model as Cm
from Reinforce_learn import *
import time
from itertools import product
import pandas as pd
from save_training import * 
from read_training import *

def offline_test(n_site, budget):
    # Set model parameters
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

    env = Cm.Env_model(init_state, deter_para, in_deter_para, seed=0)
    env.set_para_truth(para_truth)
    path = 'result/transferRL_' + str(round(n_site)) + 'patch'
    agent = read_training(path, n_state, n_action, env, 54, 1)
    
    tic = time.time()
    state = env.init_state.copy()
    observation = state[:6] * pop
    state_buffer = np.zeros([T, 6])
    new_buffer = np.zeros([T, 4])
    budget, score, ni, ne = env.B, 0, 0, 0 
    done = False
    n_select = 1
    epd = 1
    p = np.zeros((1, n_site))
    action_buffer = []
    reward_buffer = []
    index = np.array(range(n_select))
    actor = agent[0]
    for t in range(T):
        p_set = env.para_sampling(100)
        state_nn = state[0]+state[1]+state[2]+state[3]+state[4]+state[5]+state[6]+state[7]
        if t > start_time:
            act = actor.act_target(np.array(state_nn))[0]
            action = np.reshape(act, (5, n_site))   
        else:
            action = np.zeros((5, n_site))
        next_state, obs, reward, cost, new_i, new_e, ck, done = env.step_online(t, state, observation,
                                                                                 action, p_set,
                                                                                 1, 1)
        score += reward
        ni += new_i
        ne += new_e
        action = np.reshape(action, (5, n_site))
        state = state[:6] * env.pop
        new_buffer[t, 0], new_buffer[t, 1], new_buffer[t, 2], new_buffer[t, 3] = new_i, ni, new_e, ne
        print('\r', 'Time: {} | S: {:.1f}, E: {:.1f}, A: {:.1f}, I: {:.1f}, D: {:.1f}, R: {:.1f}, Budget: {:.2f}, reward:{:.2f}'
              .format(t, np.sum(observation[0]), np.sum(observation[1]), np.sum(observation[2]), 
                      np.sum(observation[3]), np.sum(observation[4]), np.sum(observation[5]), 
                      budget, score))
        print('\r', 'Time: {} | S: {:.1f}, E: {:.1f}, A: {:.1f}, I: {:.1f}, D: {:.1f}, R: {:.1f}, Budget: {:.2f}, reward:{:.2f}'
              .format(t, np.sum(state[0]), np.sum(state[1]), np.sum(state[2]), np.sum(state[3]), 
                      np.sum(state[4]), np.sum(state[5]), budget, score))
        print('\r', 'Time: {} | new_e:{:.1f}, cum_e:{:.1f}, new_i:{:.1f}, cum_i:{:.1f}'.format(t, new_e, ne, new_i, ni))
        print(action)
        budget -= cost
        state_buffer[t, :] = sum(observation.T)
        state = next_state.copy()
        observation = obs.copy()
    print(time.time()-tic)
