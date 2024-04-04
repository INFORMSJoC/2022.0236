import torch
from ddpg_agent_online import Agent
import numpy as np


def read_training(path, n_state, n_action, env, num, select):
    agent_save = []
    for i in range(num):
        actor_path1 = path + '_actor' + str(i) + '.pth'
        actor_path2 = path + '_actor_target' + str(i) + '.pth'
        critic_path1 = path + '_critic' + str(i) + '.pth'
        critic_path2 = path + '_critic_target' + str(i) + '.pth'
        memory_path = path + '_memory' + str(i) + '.pth'
        lr_path = path + '_lr' + str(i) + '.pth'
        noise_path = path + '_noise' + str(i) + '.pth'
        units_path = path + '_units' + str(i) + '.pth'
        layer_path = path + '_layer_path' + str(i) + '.pth'

        temp = Agent(state_size=n_state, action_size=n_action, initialize=0,
                     noise=torch.load(noise_path, map_location="cpu"),
                     lr=torch.load(lr_path, map_location="cpu"),
                     units=torch.load(units_path, map_location="cpu"),
                     layer=torch.load(layer_path, map_location="cpu"))

        temp.actor_local.load_state_dict(torch.load(actor_path1, map_location="cpu"))
        temp.critic_local.load_state_dict(torch.load(critic_path1, map_location="cpu"))
        temp.actor_target.load_state_dict(torch.load(actor_path2, map_location="cpu"))
        temp.critic_target.load_state_dict(torch.load(critic_path2, map_location="cpu"))
        temp.noise = torch.load(noise_path, map_location="cpu")
        temp.LR_ACTOR = torch.load(lr_path, map_location="cpu")
        temp.LR_CRITIC = temp.LR_ACTOR * 10
        memory_save = torch.load(memory_path, map_location="cpu")

        for j in range(len(memory_save)):
            temp.memory.add(memory_save[j][0], memory_save[j][1], memory_save[j][2])

        agent_save.append(temp)

    qu = [[] * i for i in range(len(agent_save))]
    for j in range(10):
        para = env.para_sampling(1)[0]
        for i in range(len(agent_save)):
            actor = agent_save[i]
            score = eval_policy(actor, env, para)
            qu[i].append(score)

    result = []
    for j in range(len(agent_save)):
        # print([j, np.mean(qu[j])])
        result.append(np.mean(qu[j]))

    agent_select = []
    for i in range(1, select + 1):
        agent_select.append(agent_save[np.argsort(result)[-i]])
    print('model have been loaded')
    return agent_select


def eval_policy(agent, env, para):
    score = 0
    # para = env.para_truth
    state = env.init_state.copy()
    for t in range(para['T']):
        action = agent.act_target(state)[0]
        next_state, r, done = env.step(t, state, action, para)
        score += r
        state = next_state.copy()
    return score
