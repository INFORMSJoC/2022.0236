import torch
from ddpg_agent import Agent
import numpy as np


def load(path, n_state, n_action, num, n_site):
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
        init = torch.load(path + '_init.pth')
        temp = Agent(state_size=n_state, action_size=n_action, initialize=init,
                     noise=torch.load(noise_path, map_location="cpu"),
                     lr=torch.load(lr_path, map_location="cpu"),
                     units=torch.load(units_path, map_location="cpu"),
                     layer=torch.load(layer_path, map_location="cpu"),
                     site = n_site)

        temp.actor_local.load_state_dict(torch.load(actor_path1, map_location="cpu"))
        temp.critic_local.load_state_dict(torch.load(critic_path1, map_location="cpu"))
        temp.actor_target.load_state_dict(torch.load(actor_path2, map_location="cpu"))
        temp.critic_target.load_state_dict(torch.load(critic_path2, map_location="cpu"))
        temp.noise = torch.load(noise_path, map_location="cpu")
        temp.LR_ACTOR = torch.load(lr_path, map_location="cpu")
        temp.LR_CRITIC = temp.LR_ACTOR * 10
        
        agent_save.append(temp)
    print('model have been loaded')
    return agent_save

