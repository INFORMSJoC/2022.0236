import numpy as np
import lhsmdu

class Env_model:
    def __init__(self, init_state, deter_para, in_deter_para, seed):
        self.B = deter_para['B']  # total budget
        self.N = deter_para['N']
        self.d_para = deter_para
        self.ind_para = in_deter_para
        self.para_truth = []
        self.para = {**deter_para, **in_deter_para}
        self.init_state = init_state
        self.para_set = []
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
        st, et, at, it, dt, rt = state[0], state[1], state[2], state[3], state[4], state[5]
        u_sv, u_it, u_aq, u_iq, w = action[0], action[1], action[2], action[3], action[4]
        cost_it = it * u_it * params['cost_ti']
        cost_v = st * u_sv * params['cost_v'] * params['v_max']
        cost_q = (u_aq * at + u_iq * it) * params['cost_ta']
        per_poc = ((st + et + at + rt) * w / self.N) ** 2 * params['cost_poc_0'] + params['cost_poc_1']
        cost_poc = (st + et + rt + at) * w * per_poc
        return cost_poc + cost_it + cost_v + cost_q

    def transition(self, state, action, params):
        st, et, at, it, dt, rt = state[0], state[1], state[2], state[3], state[4], state[5]
        u_sv, u_it, u_aq, u_iq, w = action[0], action[1], action[2], action[3], action[4]
        ht = st + et + at + it + rt

        st_out = (1 - u_sv * params['v_max']) * st * (
                    it * (1 - u_iq) * params['beta1'] + (et + at * (1 - u_aq)) * params['beta2']) / ht
        st_out_e = st_out * params['alpha']
        st_out_i = st_out * (1 - params['alpha'])
        st_out_r = u_sv * params['v_max'] * st * params['psr']
        et_out_r = et * params['per']
        et_out_i = et * params['pei']
        et_out_a = et * w * (1 - params['per'] - params['pei'])
        at_out_i = at * params['pei']
        at_out_r = at * params['per']
        it_out_r = it * params['pir'] * u_it
        it_out_d = it * params['pid'] * (1 - u_it) + it * u_it * params['pid_plus']

        stn = st - st_out_e - st_out_i - st_out_r
        etn = et - et_out_r - et_out_i - et_out_a + st_out_e
        atn = at - at_out_r - at_out_i + et_out_a
        itn = it - it_out_r - it_out_d + st_out_i + et_out_i + at_out_i
        dtn = dt + it_out_d
        rtn = rt + et_out_r + at_out_r + it_out_r + st_out_r

        reward = -2.6 * st_out - 60 * it_out_d
        new_i = st_out_i + et_out_i + at_out_i
        new_e = st_out_e
        next_state = np.array([stn, etn, atn, itn, dtn, rtn])
        cost = self.cost_function(state, action)
        return next_state, reward, new_i, new_e, cost

    def step(self, t, state, action, para):
        s, p, b = state[:-2] * self.N, state[-1], state[-2] * self.B
        # u_sv, u_it, u_aq, u_iq, w = action[0], action[1], action[2], action[3], action[4]
        # action[4] = 0 if action[4]<0.5 else 1  # w = 0, 1
        cost = self.cost_function(s, action)
        b -= cost
        act = action if b > 0 else np.zeros((5))
        next_state, reward, _, _, _ = self.transition(s, act, para)
        if act[-1] > 0.5:
            p = 0
        else:
            p += 1
        next_state = np.append(next_state / self.N, b / self.B)
        next_state = np.append(next_state, p)
        done = True if t == para['T'] - 1 or b < 0 else False
        return next_state, reward, done

    def step_online(self, t, state, obs, action, num_select, num_samples, flag):
        s, p, b = state[:-2] * self.N, state[-1], state[-2] * self.B
        # u_sv, u_it, u_aq, u_iq, w = action[0], action[1], action[2], action[3], action[4]
        # action[4] = 0 if action[4]<0.5 else 1  # w = 0, 1、
        cost = self.cost_function(s, action)
        b -= cost
        act = action if b > 0 else np.zeros((5))
        next_state, obs, reward, c, new_i, new_e, ck = self.online_state_estimate(s, obs, act, num_select, num_samples,
                                                                                  flag)
        if act[-1] > 0.5:
            p = 0
        else:
            p += 1
        next_state = np.append(next_state / self.N, b / self.B)
        next_state = np.append(next_state, p)
        done = True if b < 0 else False
        return next_state, obs, reward, c, new_i, new_e, ck, done

    def online_state_estimate(self, state, obs, action, num_select, num_samples, flag):
        ck, state_buffer = np.zeros([1, num_samples]), np.zeros([num_samples, 6])
        obs_, r, new_i, new_e, c = self.transition(obs, action, self.para_truth)

        '''use all ck to estimate system state'''
        # np.random.seed(self.seed)
        for n in range(num_samples):
            state_noise = state.copy()
            # if state[1] == 0: state_noise[1] += np.random.rand(1)

            state_, _, _, _, _ = self.transition(state, action, self.para_set[n])
            if flag == 1:
                ck[0][n] = np.mean((state_[2:5] - obs_[2:5]) ** 2)  # unknown S E R
            elif flag == 2:
                ck[0][n] = np.mean((state_[0] - obs_[0] + state_[2] - obs_[2] + state_[3] - obs_[3] + state_[4] - obs_[
                    4]) ** 2)  # unknown E R
            elif flag == 3:
                ck[0][n] = np.mean((state_[2:] - obs_[2:]) ** 2)  # unknown S E
            elif flag == 4:
                ck[0][n] = np.mean((state_[0] - obs_[0] + state_[2] - obs_[2] + state_[3] - obs_[3] + state_[4] - obs_[
                    4] + state_[5] - obs_[5]) ** 2)  # unknown E
            elif flag == 5:
                ck[0][n] = np.mean((state_ - obs_) ** 2)  # known
            state_buffer[n][:] = np.reshape(state_, [1, 6])  # record the est of state

        '''keep the lowest 10 ck for state estimation'''
        index = ck.argsort()[0][num_select:]
        ck_sub = 1 / (np.delete(ck, index, axis=1) + 1e-6)
        ck_sub /= np.sum(ck_sub)
        state_sub = np.delete(state_buffer, index, axis=0)

        ''' state update'''
        state = np.round(np.sum(ck_sub.T * state_sub, axis=0))

        if flag == 1:
            state[2:5] = obs_[2:5].copy()
        elif flag == 2:
            state[[0, 2, 3, 4]] = obs_[[0, 2, 3, 4]].copy()
        elif flag == 3:
            state[2:] = obs_[2:].copy()
        elif flag == 4:
            state[[0, 2, 3, 4, 5]] = obs_[[0, 2, 3, 4, 5]].copy()
        elif flag == 5:
            state = obs_.copy()

        obs = obs_
        return state, obs, r, c, new_i, new_e, ck

    def space_update(self, ck, num_select):
        index = ck.argsort()[0][:num_select]
        for key in self.ind_para:
            bound = [self.para_set[i][key] for i in index.tolist()]
            #             self.ind_para[key] = [np.mean(bound)-2*np.std(bound), np.mean(bound)+2*np.std(bound)]
            self.ind_para[key] = [np.min(bound), np.max(bound)]
            # print([np.min(bound), np.max(bound)])
