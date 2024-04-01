import pickle
import torch


def save_training(agent, path):
    for i in range(len(agent)):
        actor_path1 = path + '_actor' + str(i) + '.pth'
        actor_path2 = path + '_actor_target' + str(i) + '.pth'
        critic_path1 = path + '_critic' + str(i) + '.pth'
        critic_path2 = path + '_critic_target' + str(i) + '.pth'
        memory_path = path + '_memory' + str(i) + '.pth'
        lr_path = path + '_lr' + str(i) + '.pth'
        noise_path = path + '_noise' + str(i) + '.pth'
        units_path = path + '_units' + str(i) + '.pth'
        layer_path = path + '_layer_path' + str(i) + '.pth'
        memory = []
        for j in range(len(agent[i].memory.memory)):
            memory.append([agent[i].memory.memory[j].state, agent[i].memory.memory[j].action,
                           agent[i].memory.memory[j].q_value])
        torch.save(agent[i].actor_local.state_dict(), actor_path1)
        torch.save(agent[i].critic_local.state_dict(), critic_path1)
        torch.save(agent[i].actor_target.state_dict(), actor_path2)
        torch.save(agent[i].critic_target.state_dict(), critic_path2)
        torch.save(memory, memory_path)
        torch.save(agent[i].LR_ACTOR, lr_path)
        torch.save(agent[i].noise, noise_path)
        torch.save(agent[i].actor_local.fc1.out_features, units_path)
        torch.save(agent[i].actor_local.layer, layer_path)
    print("training result saved")
