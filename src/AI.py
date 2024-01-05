import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Import environment and controls
from Classes.game_class import Snake_Game, Data

# setting up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()


# Replay Memory
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward')) # encapsulate a single transition tuple of state, action, next_state, and reward.

class ReplayMemory(object): # Defines a memory buffer that stores transitions collected from the environment. It can store a maximum number of transitions defined by capacity.

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args): # Method to add a transition to the memory.
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size): # Method to randomly sample a batch of transitions from the memory.
        return random.sample(self.memory, batch_size)

    def __len__(self): # Returns the current size of the memory.
        return len(self.memory)

# Deep Q Network
class DQN(nn.Module): # Inherits from nn.Module. Represents the neural network that approximates the Q-value function.

    def __init__(self, n_observations, n_actions): # Sets up three linear layers to form a simple feed-forward neural network.
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x): # Defines the forward pass through the network with ReLU activations for the first two layers.
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    

# Hyperparameters and utilities (Definitions of various constants used in training the network.)
BATCH_SIZE = 128 # he number of transitions sampled from the replay buffer
GAMMA = 0.99 # the discount factor
EPS_START = 0.9 # the starting value of epsilon
EPS_END = 0.05 # the final value of epsilon
EPS_DECAY = 1000 # controls the rate of exponential decay of epsilon, higher means a slower decay
TAU = 0.005 # the update rate of the target network
LR = 1e-4 # LR is the learning rate of the ``AdamW`` optimizer

# Get number of actions from gym action space ##############################
n_actions = env.action_space.n
# Get the number of state observations
n_observations = len(con.s1)
# (Determines the size of the input and output layers of the network based on the environment and control setup.)


policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
# (Two instances of the DQN network; one for the current policy and one for the target.)



optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True) # Defines the optimization algorithm used to update the weights of the policy network.

memory = ReplayMemory(10000)


steps_done = 0 # A counter for the number of steps taken (actions selected).


def select_action(state): # Function that chooses an action using epsilon-greedy policy. It either selects a random action or the best action according to the policy network.
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = [] # A list to keep track of the duration of each episode.


def plot_durations(show_result=False): # A function for plotting the durations of episodes. It shows how long each episode lasted and optionally the average over the last 100 episodes.
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


# Training loop
            
def optimize_model(): # he main training loop function. It samples a batch of transitions, computes the loss using Huber loss, and updates the weights of the policy network using backpropagation.
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                        batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


