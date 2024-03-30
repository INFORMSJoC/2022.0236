import Covid19_model as CM
from read_training import *
import pickle

def para_generator(seed, interval):
    
    budget = 3.6e3
    N = 100020
    init_state = np.array([100000/N, 20/N, 0, 0, 0, 0, 1, 0])
    n_state = 8
    n_action = 5
    T = 50
    
    deter_para = dict(N=N, B=budget, T=T, alpha=0.6, v_max=0.2/14,
                      cost_ti=0.0977, cost_ta=0.02, cost_v=0.07, cost_poc_0=0.000369,
                      cost_poc_1=0.001057, pid=1.10/1000, psr=0.7/3,
                      pid_plus=0.1221/1000, pir=1/8)

    in_deter_para = dict(beta2=[0.78735 * (1-interval/2), 0.78735 * (1+interval/2)],
                         beta1=[0.15747 * (1-interval/2), 0.15747 * (1+interval/2)],
                         pei=[0.10714 * (1-interval/2), 0.10714 * (1+interval/2)],
                         per=[0.04545 * (1-interval/2), 0.04545 * (1+interval/2)])

    in_deter_truth = dict(beta1=0.15747, beta2=0.78735, pei=0.10714, per=0.04545)

    para_gen = []
    env = CM.Env_model(init_state, deter_para, in_deter_para, 0)
    np.random.seed(seed)
    for i in range(10):

        para_gen.append(env.para_sampling(1)[0])
    pickle.dump(para_gen, open('para_truth' + str(round(interval * 100)), 'wb'))
