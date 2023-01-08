from pz_battlesnake.env import solo_v0
import pettingzoo
import gymnasium as gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time

env = solo_v0.env(width=7, height=7) # create a 7x7 solo enviorment

plt.ion()


device = torch.device("cpu") #running on laptop lmao
Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward')) #saving the result of taking action a in state s, we progress to the next state and observe a reward
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128) #input layer, takes in n observations about the state -> feeds into layer with 128 neurons
        self.layer2 = nn.Linear(128, 128) #hidden layer
        self.layer3 = nn.Linear(128, n_actions) #output layer

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the AdamW optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.001 #in long games each action is really important, so we want to be greedy after lots of training
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4   

# 4 actions, left, right, up, down
n_actions = 4
# Get the number of state observations
env.reset()

observation, reward, termination, truncation, info = env.last()
#print(observation)
'''example board

{'height': 15, 'width': 15, 'snakes': [{'id': 'agent_0', 'name': 'agent_0', 'latency': '0', 'health': 99, 'body': [{'x': 8, 'y': 3}, {'x': 7, 'y': 3}, {'x': 7, 'y': 3}], 'head': {'x': 8, 'y': 3}, 'length': 3, 'shout': '', 'squad': '', 'customizations': {'color': '#00FF00', 'head': '', 'tail': ''}}], 'food': [{'x': 13, 'y': 13}, {'x': 12, 'y': 10}], 'hazards': []}
'''


'''turn the observation dictionary we get from the environment into a matrix of values
we get:
The snakes health
Where our snakes head is
Where its body segments are
Where the food is
'''
def observation_to_values(observation):
    board = observation['board']
    health = 100
    head_matrix = [] #0 unless the head on that cell, then 1
    body_matrix = [] #0 unless a body segment on that cell, then 1
    food_matrix = [] #0 unless food on that cell, then 1
    #iterate over the grid
    for x in range(0, board["height"]):
        #fill the current row with 0s
        head_matrix.append([0 for i in range(board["width"])])
        body_matrix.append([0 for i in range(board["width"])])
        food_matrix.append([0 for i in range(board["width"])]) 
        for y in range(0, board["width"]):
            for snake in board["snakes"]:
                health = snake["health"]
                #if the head is on this cell, set the head matrix to 1
                if snake["head"]["x"] == x and snake["head"]["y"] == y:
                    head_matrix[x][y] = 1
                #if a body segment is on this cell, set the body matrix to 1
                for body in snake["body"]:
                    if body["x"] == x and body["y"] == y:
                        body_matrix[x][y] = 1
            #if food is on this cell, set the food matrix to 1
            for food in board["food"]:
                if food["x"] == x and food["y"] == y:
                    food_matrix[x][y] = 1

    #flatten the matrices into a single vector
    values = []
    for x in head_matrix:
        for y in x:
            values.append(y)
    for x in body_matrix:
        for y in x:
            values.append(y)
    for x in food_matrix:
        for y in x:
            values.append(y)
    values.append(health) 
    return values

#get the observation vector
state = observation_to_values(observation["observation"])
n_observations = len(state) #note the length of the vector

#print("size of obs vector: ", n_observations)

#initialize the networks
policy_net = DQN(n_observations, n_actions).to(device) 
target_net = DQN(n_observations, n_actions).to(device) 
target_net.load_state_dict(policy_net.state_dict()) 

#initialize the optimizer
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

#initialize the replay memory
memory = ReplayMemory(10000)


steps_done = 0

'''Select an action using the policy network, or a random action with probability epsilon'''
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space(env.agents[0]).sample()]], device=device, dtype=torch.long)


# number of turns the snake survives in each episode
episode_durations = []

'''interactive plotting'''
def plot_durations(show_result=False):
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



'''Optimize our Q function approximator using the replay memory
Mostly pulled from the pytorch DQN tutorial
'''
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    #print("memory", memory)
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
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
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

#in test reaches about 250 turns on average in 2000 episodes
num_episodes = 2000

for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    t = 0
    print("Episode: ", i_episode)
    env.reset()
    observation, reward, termination, truncation, info = env.last()
    state = observation_to_values(observation["observation"])
    #print("state: ", state)
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    done = False

    while not done:
        agent = env.agents[0]
        action = select_action(state)
        env.step(action.item())
        observation, reward, terminated, truncated, _ = env.last()
        done = terminated or truncated
        if i_episode % 100 == 0:
            time.sleep(0.1)
            env.render()
        if terminated:
            reward = 0
            next_state = None
        else:
            reward = 1
            #print(observation)
            next_state = torch.tensor(observation_to_values(observation), dtype=torch.float32, device=device).unsqueeze(0)
        t += reward
        reward = torch.tensor([reward], device=device)
        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            if i_episode % 100 == 0 and i_episode != 0: #only plotting every 100 eps to avoid the annoying popups
                plot_durations()
            break

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()