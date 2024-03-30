from ddpg_agent import *

start_time = 6
gamma = 1
BATCH_SIZE = 2 ** 8

def reward_shaping(value):
    return np.exp(10 + value / 1e7)


def rollout(agent, env, t, state, r, para):
    s = state.copy()
    q_value = 0
    for i in range(t+1, para['T']):
        q_value += r 
        s_nn = s[0]+s[1]+s[2]+s[3]+s[4]+s[5]+s[6]+s[7]
        act = agent.act_target(np.array(s_nn))[0]
        action = np.reshape(act, (5, para['site']))
        ns, r, _ = env.step(i, s, action, para) 
        s = ns.copy()
    return reward_shaping(q_value) 


def train(agent, env, para, iter_num):
    time_len = para['T']
    n_state = 7 * para['site'] + 1
    n_action = 5 * para['site']
    qu = [[] for i in range(len(agent))]
    qu_time = [0] * len(agent)
    qu_step = [0] * len(agent)
    for epd in range(iter_num):
        para = env.para_sampling(1)[0]
        for i in range(len(agent)):
            if qu_time[i] > 50:
                result = []
                for j in range(len(agent)):
                    result.append(np.mean(qu[j][-10:]))
                    if np.mean(qu[j][-10:]) > -1000:
                        qu_step[j] = 1
                if result[i] < result[np.argsort(result)[10]]:
                    best = agent[np.argsort(result)[-random.randint(1, 10)]]
                    agent[i] = Agent(state_size=n_state, action_size=n_action, initialize=0,
                                     noise=best.noise, lr=best.LR_ACTOR,
                                     units=best.units, layer=best.layer, site=para['site'])
                    agent[i].actor_local.load_state_dict(best.actor_local.state_dict())
                    agent[i].actor_target.load_state_dict(best.actor_target.state_dict())
                    agent[i].critic_local.load_state_dict(best.critic_local.state_dict())
                    agent[i].critic_local.load_state_dict(best.critic_local.state_dict())
                    agent[i].memory = best.memory
                    agent[i].noise = best.noise * random.uniform(0.9, 1.1)
                    agent[i].LR_ACTOR = best.LR_ACTOR * random.uniform(0.9, 1.1)
                    agent[i].LR_CRITIC = agent[i].LR_ACTOR * 10
                    qu_time[i] = 0
                    qu_step[i] = 0
            if qu_step[i] == 0:
                done, budget, p = False, env.B, 0
                state = env.init_state.copy()
                t = 0
                qu_time[i] += 1
                while not done:
                    state_nn = state[0]+state[1]+state[2]+state[3]+state[4]+state[5]+state[6]+state[7]
                    if t > start_time:
                        act = agent[i].act(np.array(state_nn))[0]
                        action = np.reshape(act, (5, para['site']))
                    else:
                        action = np.zeros((5, para['site']))          
                    next_state, reward, done = env.step(t, state, action, para)
                    if t > start_time:
                        q_values = rollout(agent[i], env, t, next_state, reward, para)
                        agent[i].step(state_nn, act, q_values)
                        agent[i].train()
                    state = next_state.copy()
                    t += 1
                score, e, d = eval_policy(agent[i], env, para)
                qu[i].append(score)
                
        for i in range(len(agent)):
            score, e, d = eval_policy(agent[i], env, env.para_truth)
            print([epd, i, score, e, d])
    return agent


def ddpg(agent, env, episodes, bar, noise):        
    for i in range(bar):
        with tqdm(total=int(episodes/bar), desc='E%d' % i) as pbar:
            for j in range(int(episodes/bar)):
                done, budget, T, p = False, env.B, env.para_truth['T'], 0
                para = env.para_sampling(1)[0]
                state = env.init_state.copy()
                t = 0
                while not done:
                    state_nn = state[0]+state[1]+state[2]+state[3]+state[4]+state[5]+state[6]+state[7]
                    if t > start_time:
                        act = agent.act(np.array(state_nn), noise)[0]
                        action = np.reshape(act, (5, para['site']))
                    else:
                        action = np.zeros((5, para['site']))
                    next_state, reward, done = env.step(t, state, action, para)
                    if t > start_time:
                        q_values = rollout(agent, env, t, next_state, reward, para)
                        agent.step(state_nn, act, q_values)
                        agent.train()  
                    state = next_state.copy()
                    t += 1
                if (j+1) % bar == 0:   
                    score, e, d = eval_policy(agent, env)
                    pbar.set_postfix({'e': '%.1f' % score,'d': '%.1f' % d, 't':'%.1f'%t})
                pbar.update(1)        


def eval_policy(agent, env, para):
    score = 0
    state = env.init_state.copy()
    for t in range(para['T']):
        state_nn = state[0]+state[1]+state[2]+state[3]+state[4]+state[5]+state[6]+state[7]
        if t > start_time:
            act = agent.act_target(np.array(state_nn))[0]
            action = np.reshape(act, (5, para['site']))
        else:
            action = np.zeros((5, para['site']))
        next_state, r, done = env.step(t, state, action, para)  
        score += r     
        state = next_state.copy()
        output = state.copy()[:6] * env.pop
        e = np.sum(output[1])
        death = np.sum(output[4])
    return score, e, death


def output(agent, env):
    done, budget, score = False, env.B, 0
    para = env.para_truth
    state = env.init_state.copy()
    action_buffer = []
    period = 0
    for t in range(para['T']):
        state_nn = state[0]+state[1]+state[2]+state[3]+state[4]+state[5]+state[6]+state[7]
        if t > start_time:
            act = agent.act_target(np.array(state_nn))[0]
            action = np.reshape(act, (5, para['site']))
        else:
            action = np.zeros((5, para['site']))
        action_buffer.append(action)
        next_state, r, done = env.step(t, state, action, para)  
        score += r     
        
        state = next_state.copy()
        if not done: period += 1
        output = state.copy()[:6] * env.pop
        print(action)
        print('\r', 'Time: {} | S: {:.1f}, E: {:.1f}, A: {:.1f}, I: {:.1f}, D: {:.1f}, R: {:.1f}, score:{:3f}'.format(period, np.sum(output[0]), np.sum(output[1]), np.sum(output[2]), np.sum(output[3]), np.sum(output[4]), np.sum(output[5]), score))    
    return action_buffer


def ddpg_online(agent, env, index, epd, p_set):
    for i in range(epd):
        for j in range(len(index)):
            for k in range(len(agent)):
                done, budget, p = False, env.B, 0
                para = p_set[j]
                # print([para['beta1'], para['beta2'], para['pei'], para['per']])
                state = env.init_state.copy()
                t = 0
                while not done:
                    state_nn = state[0]+state[1]+state[2]+state[3]+state[4]+state[5]+state[6]+state[7]
                    if t > start_time:
                        act = agent[k].act(np.array(state_nn))[0]
                        action = np.reshape(act, (5, para['site']))
                    else:
                        action = np.zeros((5, para['site']))

                    next_state, reward, done = env.step(t, state, action, para)
                    if t > start_time:
                        q_values = rollout(agent[k], env, t, next_state, reward, para)
                        agent[k].step(state_nn, act, q_values)
                        agent[k].train()
                    state = next_state.copy()
                    t += 1

    qu = [[] * i for i in range(len(agent))]
    for j in range(len(index)):
        para = p_set[j]
        for i in range(len(agent)):
            actor = agent[i]
            score, _, _ = eval_policy(actor, env, para)
            qu[i].append(score)
    result = []
    for j in range(len(agent)):
        result.append(np.mean(qu[j]))
    select_id = np.argsort(result)[-1]
    print('\n', [select_id, eval_policy(agent[select_id], env, env.para_truth)])
    return agent, select_id 


def copy_agent(agent, env):
    agent_copy = []
    n_site = env.para_truth['site']
    n_state = 7 * n_site + 1
    n_action = 5 * n_site
    for i in range(len(agent)):
        temp = Agent(state_size=n_state, action_size=n_action, initialize=0,
                     noise=agent[i].noise, lr=agent[i].LR_ACTOR,
                     units=agent[i].units, layer=agent[i].layer, site=n_site)
        temp.actor_local.load_state_dict(agent[i].actor_local.state_dict())
        temp.actor_target.load_state_dict(agent[i].actor_target.state_dict())
        temp.critic_local.load_state_dict(agent[i].critic_local.state_dict())
        temp.critic_local.load_state_dict(agent[i].critic_local.state_dict())
        temp.memory = agent[i].memory
        temp.noise = agent[i].noise
        temp.LR_ACTOR = agent[i].LR_ACTOR
        temp.LR_CRITIC = agent[i].LR_ACTOR * 10
        agent_copy.append(temp)
    return agent_copy


def agent_eval(agent, env, p_set, n_select):
    qu = [[] * i for i in range(len(agent))]
    for j in range(n_select):
        para = p_set[j]
        for i in range(len(agent)):
            actor = agent[i]
            score, _, _ = eval_policy(actor, env, para)
            qu[i].append(score)
    result = []
    for j in range(len(agent)):
        result.append(np.mean(qu[j]))
    return result


def train_online(agent, env, index, epd, p_set):
    n_site = env.para_truth['site']
    n_state = 7 * n_site + 1
    n_action = 5 * n_site
    for e in range(epd):
        para = p_set[e]
        result = []
        for i in range(len(agent)):
            done, budget, p, t = False, env.B, 0, 0
            state = env.init_state.copy()
            while not done:
                state_nn = state[0]+state[1]+state[2]+state[3]+state[4]+state[5]+state[6]+state[7]
                if t > start_time:
                    act = agent[i].act(np.array(state_nn))[0]
                    action = np.reshape(act, (5, para['site']))
                else:
                    action = np.zeros((5, para['site']))          
                next_state, reward, done = env.step(t, state, action, para)
                if t > start_time:
                    q_values = rollout(agent[i], env, t, next_state, reward, para)
                    agent[i].step(state_nn, act, q_values)
                    agent[i].train()
                state = next_state.copy()
                t += 1
            score, e, d = eval_policy(agent[i], env, para)
            result.append(score)
        for i in range(len(agent)): 
            if result[i] < result[np.argsort(result)[4]]:
                best = agent[np.argsort(result)[-random.randint(1, 4)]]
                agent[i] = Agent(state_size=n_state, action_size=n_action, initialize=0,
                                 noise=best.noise, lr=best.LR_ACTOR,
                                 units=best.units, layer=best.layer, site=para['site'])
                agent[i].actor_local.load_state_dict(best.actor_local.state_dict())
                agent[i].actor_target.load_state_dict(best.actor_target.state_dict())
                agent[i].critic_local.load_state_dict(best.critic_local.state_dict())
                agent[i].critic_local.load_state_dict(best.critic_local.state_dict())
                agent[i].memory = best.memory
                agent[i].noise = best.noise * random.uniform(0.9, 1.1)
                agent[i].LR_ACTOR = best.LR_ACTOR * random.uniform(0.9, 1.1)
                agent[i].LR_CRITIC = agent[i].LR_ACTOR * 10
    best_id = np.argsort(result)[-1]
    return agent, best_id