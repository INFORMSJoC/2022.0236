from ddpg_agent import *

start_time = 0
gamma = 1
n_state = 8
n_action = 5
BATCH_SIZE = 2 ** 8


def reward_shaping(value):
    return np.exp((5e5 + value) / 5e4)


def rollout(agent, env, t, state, r, para):
    s = state.copy()
    q_value = 0
    for i in range(t+1, para['T']):
        q_value += gamma ** (i - t - 1) * r 
        act = agent.act_target(s)[0]
        ns, r, _ = env.step(i, s, act, para) 
        s = ns.copy()
    # print(q_value)
    return reward_shaping(q_value)


def train(agent, env, iter_num):
    time_len = 50
    qu = [[] for i in range(len(agent))]
    qu_time = [0] * len(agent)
    qu_step = [0] * len(agent)
    for epd in range(iter_num):
        para = env.para_sampling(1)[0]
        for i in range(len(agent)):
            if qu_time[i] > time_len:
                result = []
                for j in range(len(agent)):
                    result.append(np.mean(qu[j][-10:]))
                    if np.mean(qu[j][-10:]) > -200:
                        qu_step[j] = 1
                if result[i] < result[np.argsort(result)[10]]:
                    best = agent[np.argsort(result)[-random.randint(1, 10)]]
                    agent[i] = Agent(state_size=n_state, action_size=n_action, initialize=0,
                                     noise=best.noise, lr=best.LR_ACTOR,
                                     units=best.units, layer=best.layer)
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
                    action = agent[i].act(state)[0]
                    next_state, reward, done = env.step(t, state, action, para)
                    q_values = rollout(agent[i], env, t, next_state, reward, para)
                    agent[i].step(state, action, q_values)
                    agent[i].train()
                    state = next_state.copy()
                    t += 1
                score = eval_policy(agent[i], env, para)
                qu[i].append(score)

        for i in range(len(agent)):
            score = eval_policy(agent[i], env, env.para_truth)
            print([epd, i, score])
    return agent


def copy_agent(agent, n_state, n_action):
    agent_copy = []
    for i in range(len(agent)):
        temp = Agent(state_size=n_state, action_size=n_action, initialize=0, 
                     noise=agent[i].noise, lr=agent[i].LR_ACTOR,
                     units=agent[i].units, layer=agent[i].layer)
        temp.actor_local.load_state_dict(agent[i].actor_local.state_dict())
        temp.actor_target.load_state_dict(agent[i].actor_target.state_dict())
        temp.critic_local.load_state_dict(agent[i].critic_local.state_dict())
        temp.critic_local.load_state_dict(agent[i].critic_local.state_dict())
        temp.memory = agent[i].memory
        temp.noise = agent[i].noise
        temp.LR_ACTOR = agent[i].LR_ACTOR
        temp.LR_CRITIC = temp.LR_ACTOR * 10
        agent_copy.append(temp)
    return agent_copy


def train_online(agent, env, index, epd, p_set):

    for e in range(epd):
        para = p_set[e]
        result = []
        for i in range(len(agent)):
            done, budget, p, t = False, env.B, 0, 0
            state = env.init_state.copy()
            while not done:
                action = agent[i].act(state)[0]
                next_state, reward, done = env.step(t, state, action, para)
                q_values = rollout(agent[i], env, t, next_state, reward, para)
                agent[i].step(state, action, q_values)
                agent[i].train()
                state = next_state.copy()
                t += 1
            score = eval_policy(agent[i], env, para)
            result.append(score)
        for i in range(len(agent)): 
            if result[i] < result[np.argsort(result)[10]]:
                best = agent[np.argsort(result)[-random.randint(1, 10)]]
                agent[i] = Agent(state_size=n_state, action_size=n_action, initialize=0,
                                 noise=best.noise, lr=best.LR_ACTOR,
                                 units=best.units, layer=best.layer)
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


def eval_policy(agent, env, para):
    score = 0
    #para = env.para_truth
    state = env.init_state.copy()
    for t in range(para['T']):
        action = agent.act_target(state)[0] 
        next_state, r, done = env.step(t, state, action, para)  
        score += r     
        state = next_state.copy()
    return score
            
    
def output(agent, env):
    done, budget, score = False, env.B, 0
    para = env.para_truth
    state = env.init_state.copy()
    action_buffer = []
    period = 0
    for t in range(para['T']):
        action = agent.act_target(state)[0] 
        next_state, r, done = env.step(t, state, action, para)  
        score += r     
        output = state.copy() * para['N']
        state = next_state.copy()
        if not done: period += 1
    state = output.copy()
    print('\r', 'Time: {} | S: {:.1f}, E: {:.1f}, A: {:.1f}, I: {:.1f}, D: {:.1f}, R: {:.1f}, score:{:3f}'.format(period, state[0], state[1], state[2], state[3], state[4], state[5], score))      

    
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
                    action = agent[k].act(state)[0]
                    next_state, reward, done = env.step(t, state, action, para)
                    q_values = rollout(agent[k], env, t, next_state, reward, para)
                    agent[k].step(state, action, q_values)
                    agent[k].train()
                    state = next_state.copy()
                    t += 1
    '''            
    actor = Agent(state_size=n_state, action_size=n_action, initialize=0,
                  noise=agent.noise, lr=agent.LR_ACTOR,
                  units=agent.units, layer=agent.layer)
    actor.actor_local.load_state_dict(agent.actor_local.state_dict())
    actor.actor_target.load_state_dict(agent.actor_target.state_dict())
    actor.critic_local.load_state_dict(agent.critic_local.state_dict())
    actor.critic_local.load_state_dict(agent.critic_local.state_dict())
    actor.memory = agent.memory
    for i in range(epd):
        for j in range(len(index)):
            done, budget, p = False, env.B, 0
            para = p_set[j]
            print(para['beta2'])
            state = env.init_state.copy()
            t = 0
            while not done:
                action = actor.act(state)[0]
                next_state, reward, done = env.step(t, state, action, para)
                q_values = rollout(actor, env, t, next_state, reward, para)
                actor.step(state, action, q_values)
                actor.train()
                state = next_state.copy()
                t += 1
            score = eval_policy(actor, env, env.para_truth)
            print(score)
    '''
    qu = [[] * i for i in range(len(agent))]
    for j in range(len(index)):
        para = p_set[j]
        for i in range(len(agent)):
            actor = agent[i]
            score = eval_policy(actor, env, para)
            qu[i].append(score)
    result = []
    for j in range(len(agent)):
        result.append(np.mean(qu[j]))
    select_id = np.argsort(result)[-1]
    print('\n', [select_id, eval_policy(agent[select_id], env, env.para_truth)])

    return agent, select_id 
