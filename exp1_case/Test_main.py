import Covid19_model as Cm
import time
from read_training import *
import pickle
from Reinforce_learn import *
N = 100020
T = 50


def online_test(interval):
    """model parameters"""
    budget = 3.6e3
    init_state = np.array([100000 / N, 20 / N, 0, 0, 0, 0, 1, 0])
    n_select = 10
    n_state = 8
    n_action = 5
    epd = 1
    iter_num = 200
    """system parameters"""
    deter_para = dict(N=N, B=budget, T=T, alpha=0.6, v_max=0.2 / 14,
                      cost_ti=0.0977, cost_ta=0.02, cost_v=0.07, cost_poc_0=0.000369,
                      cost_poc_1=0.001057, pid=1.10 / 1000, psr=0.7 / 3,
                      pid_plus=0.1221 / 1000, pir=1 / 8)

    in_deter_para = dict(beta2=[0.78735 * (1-interval/2), 0.78735 * (1+interval/2)],
                         beta1=[0.15747 * (1-interval/2), 0.15747 * (1+interval/2)],
                         pei=[0.10714 * (1-interval/2), 0.10714 * (1+interval/2)],
                         per=[0.04545 * (1-interval/2), 0.04545 * (1+interval/2)])
    result = np.zeros(10)

    for s in range(10):
        env = Cm.Env_model(init_state, deter_para, in_deter_para, 0)
        agent = read_training('result/env_' + str(round(interval * 100)) + '_epd' + str(round(iter_num)), n_state,
                              n_action,
                              env, 54, 54)
        para_truth = pickle.load(open('para_truth' + str(round(interval * 100)), 'rb'))[s]
        env.set_para_truth(para_truth)
        tic = time.time()
        state = init_state
        observation = np.array([100000, 20, 0, 0, 0, 0])
        budget, score, = env.B, 0
        index = np.array(range(n_select))

        for t in range(T):
            p_set = env.para_sampling(100)
            # agent_copy = copy_agent(agent, n_state, n_action)
            agent, select_id = train_online(agent, env, index, epd, p_set)
            actor = agent[select_id]
            action = actor.act_target(state)[0]
            next_state, obs, reward, cost, new_i, new_e, ck, done = env.step_online(t, state, observation, action,
                                                                                    n_select, 100, 1)
            score += reward
            '''
            if not done:
                env.space_update(ck, n_select)
                
            '''
            env.space_update(ck, n_select)
            index = ck.argsort()[0][:n_select]
            budget -= cost
            state = next_state.copy()
            observation = obs.copy()
            print(['time:', t, 'state:', state])
            print(['action:', action])
            # print('\r', 'processing: ' + str(10 * (s * T + t) / T) + '%', end='', flush=True)
        result[s] = score
        print('\n', score)
        # print(time.time() - tic)
    print(result)
    print(np.mean(result))

    
def online_test_out(interval, out):
    """model parameters"""
    budget = 3.6e3
    init_state = np.array([100000 / N, 20 / N, 0, 0, 0, 0, 1, 0])
    n_select = 10
    n_state = 8
    n_action = 5
    epd = 1
    iter_num = 200
    """system parameters"""
    deter_para = dict(N=N, B=budget, T=T, alpha=0.6, v_max=0.2 / 14,
                      cost_ti=0.0977, cost_ta=0.02, cost_v=0.07, cost_poc_0=0.000369,
                      cost_poc_1=0.001057, pid=1.10 / 1000, psr=0.7 / 3,
                      pid_plus=0.1221 / 1000, pir=1 / 8)

    in_deter_para = dict(beta2=[0.78735 * (1-interval/2), 0.78735 * (1+interval/2)],
                         beta1=[0.15747 * (1-interval/2), 0.15747 * (1+interval/2)],
                         pei=[0.10714 * (1-interval/2), 0.10714 * (1+interval/2)],
                         per=[0.04545 * (1-interval/2), 0.04545 * (1+interval/2)])
    result = np.zeros(10)

    for s in range(10):
        env = Cm.Env_model(init_state, deter_para, in_deter_para, 0)
        agent = read_training('result/env_' + str(round(interval * 100)) + '_epd' + str(round(iter_num)), n_state,
                              n_action,
                              env, 54, 54)
        para_truth = pickle.load(open('out_of_sample_' + str(round(interval * 100)) + '_' + str(round(out * 100)), 'rb'))[s]
        env.set_para_truth(para_truth)
        tic = time.time()
        state = init_state
        observation = np.array([100000, 20, 0, 0, 0, 0])
        budget, score, = env.B, 0
        index = np.array(range(n_select))

        for t in range(T):
            p_set = env.para_sampling(100)
            # agent_copy = copy_agent(agent, n_state, n_action)
            agent, select_id = train_online(agent, env, index, epd, p_set)
            actor = agent[select_id]
            action = actor.act_target(state)[0]
            next_state, obs, reward, cost, new_i, new_e, ck, done = env.step_online(t, state, observation, action,
                                                                                    n_select, 100, 1)
            score += reward
            '''
            if not done:
                env.space_update(ck, n_select)
                
            '''
            env.space_update(ck, n_select)
            index = ck.argsort()[0][:n_select]
            budget -= cost
            state = next_state.copy()
            observation = obs.copy()
            print(['time:', t, 'state:', state])
            print(['action:', action])
            # print('\r', 'processing: ' + str(10 * (s * T + t) / T) + '%', end='', flush=True)
        result[s] = score
        print('\n', score)

    print(result)
    print(np.mean(result))
