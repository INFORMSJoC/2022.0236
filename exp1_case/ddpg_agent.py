import numpy as np
import random
from collections import namedtuple, deque
from model import Actor, Critic
import torch
import torch.nn.functional as F
import torch.optim as optim


BUFFER_SIZE = 2 ** 16        # replay buffer size (14-17)
BATCH_SIZE = 2 ** 8           # minibatch size (128, 256，512)
tau = 1e-2
WEIGHT_DECAY = 1e-3          # L2 weight decay
device = torch.device("cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, initialize, noise, lr, units, layer):
        """Initialize an Agent object."""
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(initialize)
        self.noise = noise
        self.LR_ACTOR = lr           # learning rate of the actor
        self.LR_CRITIC = lr * 10     # learning rate of the critic
        self.units = units
        self.layer = layer
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, initialize, self.units, self.layer).to(device)
        self.actor_target = Actor(state_size, action_size, initialize, self.units, self.layer).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.LR_ACTOR)
        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, initialize, self.units, self.layer).to(device)
        self.critic_target = Critic(state_size, action_size, initialize, self.units, self.layer).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.LR_CRITIC, weight_decay=WEIGHT_DECAY)
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, initialize)
        np.random.seed(0)
        
       
    def step(self, state, action, q_value):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, q_value)
    
    def train(self):
        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences)
            
    def act(self, state):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        if self.noise > 0:
            action = (action + np.random.normal(0, self.noise, size=self.action_size)).clip(float(0), float(1))
        return action
    
    def act_target(self, state):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.actor_target.eval()
        with torch.no_grad():
            action = self.actor_target(state).cpu().data.numpy()
        return action
    
    def act_local(self, state):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        return action
    
    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples. """
        states, actions, q_values = experiences

        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, q_values)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)                     

    def soft_update(self, local_model, target_model):
        """Soft update model parameters. """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "q_value"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, q_value):
        """Add a new experience to memory."""
        e = self.experience(state, action, q_value)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        q_values = torch.from_numpy(np.vstack([e.q_value for e in experiences if e is not None])).float().to(device)

        return states, actions, q_values

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)