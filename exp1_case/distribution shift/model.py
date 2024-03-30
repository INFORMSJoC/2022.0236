import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed, units, layer=1):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size - 1, units)
        self.layer = layer
        if self.layer > 1:
            self.fc2 = nn.Linear(units, units)
        if self.layer > 2:
            self.fc3 = nn.Linear(units, units)
        self.fc_last = nn.Linear(units, action_size)      
        self.fcw1 = nn.Linear(2, units)
        self.fcw2 = nn.Linear(units, 1)
        self.state_size = state_size - 1
        self.action_size = action_size - 1
        
    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = torch.tanh((self.fc1(state[:, :self.state_size])))
        if self.layer > 1:
            x = torch.tanh((self.fc2(x)))
        if self.layer > 2:
            x = torch.tanh((self.fc3(x)))
        x = torch.sigmoid(self.fc_last(x))
        aw = torch.cat([state[:, self.state_size:], x[:, self.action_size:]], 1)
        aw = torch.tanh((self.fcw1(aw)))
        aw = torch.sigmoid(self.fcw2(aw))
        act = torch.cat([x[:, :self.action_size], aw], 1) 
        
        return act 


class Critic(nn.Module):

    def __init__(self, state_size, action_size, seed, units, layer=1):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size + action_size, units)
        self.layer = layer
        if self.layer > 1:
            self.fcs2 = nn.Linear(units, units)
        if self.layer > 2:
            self.fcs3 = nn.Linear(units, units)
        self.fc1 = nn.Linear(units, 1)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = torch.cat([state, action], 1)
        xs = torch.relu((self.fcs1(xs)))
        if self.layer > 1:
            xs = torch.relu((self.fcs2(xs)))
        if self.layer > 2:
            xs = torch.relu((self.fcs3(xs)))
        return self.fc1(xs)
