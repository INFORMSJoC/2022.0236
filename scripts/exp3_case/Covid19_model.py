import numpy as np
import pickle
import random
import lhsmdu

class Env_model:
    def __init__(self, init_state, deter_para, in_deter_para, seed):
        self.T = deter_para['T']
        self.B = deter_para['B'] # total budget
        self.N = deter_para['N']
        self.pop = deter_para['P']
        self.dij = deter_para['D']
        self.d_para = deter_para
        self.ind_para = in_deter_para
        self.para_truth = []
        self.para = {**deter_para, **in_deter_para}
        self.init_state = init_state
        self.para_set = []
        self.site = deter_para['site']
        self.seed = seed
           
    def set_para_truth(self, para_truth):
        self.para_truth = para_truth
        
    def para_sampling(self, num_samples):
        para_set = []
        lhs = np.array(lhsmdu.sample(len(self.ind_para), num_samples)).astype(np.float32)
        for i in range(num_samples):
            sample_para = {}
            j = 0
            for key in self.ind_para:
                sample_para[key] = np.float(self.ind_para[key][0] + lhs[j][i] * 
                                            (self.ind_para[key][1] - self.ind_para[key][0]))
                j += 1
            para_set.append({**self.d_para, **sample_para})
        self.para_set = para_set
        return para_set
    
    
    def cost_function(self, state, action):
        params = self.d_para
        st, et, at, it, dt, rt = np.array(state[0]), np.array(state[1]), np.array(state[2]), np.array(state[3]), np.array(state[4]), np.array(state[5])
        u_sv, u_it, u_aq, u_iq, w = action[0], action[1], action[2], action[3], action[4]
        cost_v = np.sum(st * u_sv * params['cost_v'] * params['v_max'])     
        cost_it = np.sum(it * u_it * params['cost_ti'])
        cost_q = np.sum((u_aq * at + u_iq * it) * params['cost_ta'])
        per_poc = (np.sum((st + rt) * w) / np.sum(state)) ** 2 * params['cost_poc_0'] + params['cost_poc_1'] 
        cost_poc = np.sum((st + rt) * w) * per_poc
        return cost_v + cost_poc + cost_it + cost_q
    
    def cost_specific(self, state, action):
        params = self.d_para
        st, et, at, it, dt, rt = np.array(state[0]), np.array(state[1]), np.array(state[2]), np.array(state[3]), np.array(state[4]), np.array(state[5])
        u_sv, u_it, u_aq, u_iq, w = action[0], action[1], action[2], action[3], action[4]
        cost_v = np.sum(st * u_sv * params['cost_v'] * params['v_max'])     
        cost_it = np.sum(it * u_it * params['cost_ti'])
        cost_q = np.sum((u_aq * at + u_iq * it) * params['cost_ta'])
        per_poc = (np.sum((st + rt) * w) / np.sum(state)) ** 2 * params['cost_poc_0'] + params['cost_poc_1'] 
        cost_poc = np.sum((st + rt) * w) * per_poc
        return [cost_v, cost_poc, cost_it, cost_q]
        
        
    def transition(self, state, action, params):
        st, et, at, it, dt, rt = np.array(state[0]), np.array(state[1]), np.array(state[2]), np.array(state[3]), np.array(state[4]), np.array(state[5])
        u_sv, u_it, u_aq, u_iq, w = action[0], action[1], action[2], action[3], action[4]
        ht = st + et + at + it + rt
        
        '''transition rate'''
        qij = params['density'] * np.exp(-self.dij) / sum(params['density'] * np.exp(-self.dij))
        theta_e = (1 - params['gamma']) * (et + at * (1 - u_aq)) / ht + params['gamma'] * np.sum(qij * (et + at * (1 - u_aq)), 1) / ht
        theta_i = (1 - params['gamma']) * it * (1 - u_iq) / ht + params['gamma'] * np.sum(qij * it * (1 - u_iq), 1) / ht
        pse = params['alpha'] * (params['beta1'] * theta_e + params['beta2'] * theta_i)
        psi = (1-params['alpha']) * (params['beta1'] * theta_e + params['beta2'] * theta_i)
        
        st_out_e = st * ((1 - u_sv * params['v_max']) + u_sv * params['v_max'] * (1- params['psr'])) * pse
        st_out_i = st * ((1 - u_sv * params['v_max']) + u_sv * params['v_max'] * (1- params['psr'])) * psi
        st_out_r = u_sv * params['v_max'] * st * params['psr']
        et_out_r = et * params['per']
        et_out_i = et * params['pei']
        et_out_a = et * w * (1 - params['per'] - params['pei'])
        at_out_r = at * params['per']
        at_out_i = at * params['pei']

        it_out_d = it * params['pid'] * (1 - u_it) + it * u_it * params['pid_plus']
        it_out_r = it * params['pir'] * u_it
        rt_out_s = rt * params['prs']

        stn = st + rt_out_s - st_out_r - st_out_i - st_out_e
        etn = et - et_out_r - et_out_i - et_out_a + st_out_e
        atn = at - at_out_r - at_out_i + et_out_a
        itn = it - it_out_d - it_out_r + st_out_i + et_out_i + at_out_i
        dtn = dt + it_out_d
        rtn = rt + st_out_r + et_out_r + at_out_r + it_out_r - rt_out_s
       
        reward = -np.sum(st_out_e + st_out_i) * 2.6 - 60 * np.sum(it_out_d)
        new_i = np.sum(st_out_i + et_out_i + at_out_i)
        new_e = np.sum(st_out_e)
        cost = self.cost_function(state, action)
        next_state = np.array([stn, etn, atn, itn, dtn, rtn]).tolist()
        return next_state, reward, new_i, new_e, cost
    
    
    def step(self, t, state, ac, para):
        s, p, b = state[:6] * self.pop, np.array(state[-1]), state[-2][0] * self.B
        cost = self.cost_function(s, ac)
        b -= cost
        action = ac.copy() if b > 0 else np.zeros((5, self.site))
        next_state, reward, _, _, _ = self.transition(s, action, para)
        p[action[-1]>0.5] = 0
        p[action[-1]<=0.5] += 1
        next_state /= self.pop
        next_state = next_state.tolist()
        next_state.append([b/self.B])
        next_state.append(p.tolist())
        done = True if t == para['T'] - 1 or b<0 else False 
        return next_state, reward, done
    
    
    def step_online(self, t, state, obs, ac, p_set, num_select, num_samples):
        s, p, b = state[:6] * self.pop, np.array(state[-1]), state[-2][0] * self.B
        cost = self.cost_function(s, ac)
        b -= cost
        action = ac.copy() if b > 0 else np.zeros((5, self.site))
        next_state, obs, reward, c, new_i, new_e, ck = self.online_state_estimate(s, obs, action, p_set, num_select, num_samples)
        p[action[-1]>0.5] = 0
        p[action[-1]<=0.5] += 1
        next_state /= self.pop
        next_state = next_state.tolist()
        next_state.append([b/self.B])
        next_state.append(p.tolist())
        done = True if t == self.T - 1 or b<0 else False      
        return next_state, obs, reward, c, new_i, new_e, ck, done
  

    def online_state_estimate(self, state, obs, action, para_set, num_select, num_samples):
        
        ck, state_buffer = np.zeros([1, num_samples]), np.zeros([num_samples, len(state[0])*6])
        obs_, r, new_i, new_e, c = self.transition(obs, action, self.para_truth)
        obs_ = np.array(obs_)
        '''use all ck to estimate system state'''
        
        for n in range(num_samples):
            state_, reward, _, _, _ = self.transition(state.copy(), action, para_set[n])
            
            # est error of I, D
            state_ = np.array(state_)
            ck[0][n] = np.mean((state_[2:5] - obs_[2:5]) ** 2)
            state_buffer[n][:] = np.reshape(state_, [1, len(state[0])*6])  # record the est of state

        '''keep the lowest 10 ck for state estimation'''
        index = ck.argsort()[0][num_select:]
        ck_sub = 1 / (np.delete(ck, index, axis=1) + 1e-6)
        ck_sub /= np.sum(ck_sub)
        state_sub = np.delete(state_buffer, index, axis=0)

        ''' state update'''
        state = np.round(np.sum(ck_sub.T * state_sub, axis=0))
        state = np.reshape(state, [6, len(obs[0])])
        state[2:5] = obs_[2:5]

        obs = obs_
        return state, obs, r, c, new_i, new_e, ck
    
    def space_update(self, ck, para_set, num_select):
        index = ck.argsort()[0][:num_select]
        for key in self.ind_para:
            bound = [para_set[i][key] for i in index.tolist()]
            self.ind_para[key] = [np.min(bound), np.max(bound)]
            print([np.min(bound), np.max(bound)])
