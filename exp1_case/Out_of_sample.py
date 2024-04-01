import Covid19_model as CM
from read_training import *
import pickle

def para_out_sample(seed, interval):
    
    budget = 3.6e3
    N = 100020
    init_state = np.array([100000/N, 20/N, 0, 0, 0, 0, 1, 0])
    n_state = 8
    n_action = 5
    T = 50
    out = 0.1
    deter_para = dict(N=N, B=budget, T=T, alpha=0.6, v_max=0.2/14,
                      cost_ti=0.0977, cost_ta=0.02, cost_v=0.07, cost_poc_0=0.000369,
                      cost_poc_1=0.001057, pid=1.10/1000, psr=0.7/3,
                      pid_plus=0.1221/1000, pir=1/8)

    in_deter_para = dict(beta2=[0.78735 * (interval-out), 0.78735 * (2 - interval+out)],
                         beta1=[0.15747 * (interval-out), 0.15747 * (2 - interval+out)],
                         pei=[0.10714 * (interval-out), 0.10714 * (2 - interval+out)],
                         per=[0.04545 * (interval-out), 0.04545 * (2 - interval+out)])
    
    
    in_deter_truth = dict(beta1=0.15747, beta2=0.78735, pei=0.10714, per=0.04545)
    para_gen = []
    env = CM.Env_model(init_state, deter_para, in_deter_para, 0)
    np.random.seed(seed)
    
    while len(para_gen) < 10:
        temp1 = 0
        temp2 = 0
        para = env.para_sampling(1)[0]
        if para['beta1'] < 0.15747 * interval:
            temp1 += 1
            
        if para['beta2'] < 0.78735 * interval:
            temp1 += 1
            
        if para['pei'] < 0.10714 * interval:
            temp1 += 1
            
        if para['per'] < 0.04545 * interval:
            temp1 += 1
            
            
        if para['beta1'] > 0.15747 * (2-interval):
            temp2 += 1
            
        if para['beta2'] > 0.78735 * (2-interval):
            temp2 += 1
            
        if para['pei'] > 0.10714 * (2-interval):
            temp2 += 1
            
        if para['per'] > 0.04545 * (2-interval):
            temp2 += 1
            
        if temp1 + temp2 == 4:
            para_gen.append(para)
            print([para['beta1'], 0.15747 * interval])
            print([para['beta1'],0.15747 * (2-interval)])
            print([para['beta2'], 0.78735 * interval])
            print([para['beta2'],0.78735 * (2-interval)])
            print([para['pei'],0.10714 * interval])
            print([para['pei'],0.10714 * (2-interval)])
            print([para['per'],0.04545 * interval] )
            print([para['per'],0.04545 * (2-interval)])
    pickle.dump(para_gen, open('out_of_sample' + str(round(interval * 100)), 'wb'))
