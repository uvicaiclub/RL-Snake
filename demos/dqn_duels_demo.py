import sys
import math
import random
import time
from collections import deque, namedtuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append('..')  # Janky fix for package not properly installing on remote
from pz_battlesnake.env import duels_v0


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions, hidden_dim):
        super(DQN, self).__init__()
        self.input_dim = n_observations
        self.output_dim = n_actions
        self.hidden_dim = hidden_dim
        current_dim = n_observations
        self.layers = nn.ModuleList()
        for hdim in hidden_dim:
            self.layers.append(nn.Linear(current_dim, hdim))
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim, n_actions))

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        out = self.layers[-1](x)
        return out


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        # Save a transition
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    # When an agent wins or loses we push a new transition to memory that associates
    # winning / losing move with a reward
    def add_transition_for_agent(self, agent, new_value):
        for i in range(len(self.memory)-1, -1, -1):
            if self.memory[i].agent == agent:
                new_transition = Transition(
                    self.memory[i].state,
                    self.memory[i].action,
                    self.memory[i].next_state,
                    torch.tensor([new_value], device=device),
                    self.memory[i].agent
                )
                self.push(*new_transition)
                break


'''turn the observation dictionary we get from the environment into a matrix of values
we get:
The snakes health
Where our snakes head is
Where its body segments are
Where the food is

example observation:
{
  'game': {'id': 'cb7e7773-03e7-43e4-afad-9da19c0ede0c', 'ruleset': {'name': 'standard', 'version': 'cli', 'settings': {'foodSpawnChance': 15, 'minimumFood': 1, 'hazardDamagePerTurn': 0, 'hazardMap': '','hazardMapAuthor': '', 'royale': {'shrinkEveryNTurns': 0},'squad': {'allowBodyCollisions': False, 'sharedElimination': False, 'sharedHealth': False, 'sharedLength': False}}}, map': 'standard', 'timeout': 0, 'source': ''},
  'turn': 0,
  'board': {'height': 11, 'width': 11, 'snakes': [
    {'id': 'agent_1', 'name': 'agent_1', 'latency': '0', 'health': 100, 'body': [{'x': 1, 'y': 1}, {'x': 1, 'y': 1}, {'x': 1, 'y': 1}], 'head': {'x': 1, 'y': 1}, 'length': 3, 'shout': '', 'squad': '', 'customizations': {'color': '#0000FF', 'head': '', 'tail': ''}}, 
    {'id': 'agent_0', 'name': 'agent_0', 'latency': '0', 'health': 100, 'body': [{'x': 9, 'y': 9}, {'x': 9, 'y': 9}, {'x': 9, 'y': 9}], 'head': {'x': 9, 'y': 9}, 'length': 3, 'shout': '', 'squad': '', 'customizations': {'color': '#00FF00', 'head': '', 'tail': ''}}], 'food': [{'x': 0, 'y': 2}, {'x': 10, 'y': 8}, {'x': 5, 'y': 5}], 'hazards': []}, 'you': {'id': 'agent_0', 'name': 'agent_0', 'latency': '0', 'health': 100, 'body': [{'x': 9, 'y': 9}, {'x': 9, 'y': 9}, {'x': 9, 'y': 9}], 'head': {'x': 9, 'y': 9}, 'length': 3, 'shout': '', 'squad': '', 'customizations': {'color': '#00FF00', 'head': '', 'tail': ''}
    }
  ]
}
'''
def observation_to_values(observation):
    try:
        board = observation['board']
    except:
        observation = observation['observation']
    # Init
    board = observation['board']
    health = 100
    n_channels = 6
    state_matrix = np.zeros((n_channels, board["height"], board["width"]))
    # Fill
    for _snake in board['snakes']:
        health = np.array(_snake['health'])
        # If us
        if _snake['id'] == observation['you']['id']:
            # Place head on channel 0
            state_matrix[0, _snake['head']['x'], _snake['head']['y']] = 1

            # Place body on channel 1
            for _body_segment in _snake['body']:
                state_matrix[1, _body_segment['x'], _body_segment['y']] = 1
        else:
            # Place head on channel 0
            state_matrix[2, _snake['head']['x'], _snake['head']['y']] = 1

            # Place body on channel 1
            for _body_segment in _snake['body']:
                state_matrix[3, _body_segment['x'], _body_segment['y']] = 1
    # Place food on channel 2
    for _food in board["food"]:
        state_matrix[4,_food['x'], _food['y']] = 1
    # Create health channel
    state_matrix[5] = np.full((board["height"], board["width"]), health)
    # Flatten
    state_matrix = state_matrix.reshape(-1, 1)

    state_matrix = np.concatenate([state_matrix, health.reshape(1, 1)], axis=0)
    return state_matrix.flatten()


# Select an action using the policy network, or a random action with probability epsilon
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + ((EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY))
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor(
            [[env.action_space(env.agents[0]).sample()]],
            device=device, dtype=torch.long
        )


# Interactive plotting
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


# Optimize our Q function approximator using the replay memory
# Mostly pulled from the pytorch DQN tutorial
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(
        tuple(map(
            lambda s: s is not None,
            batch.next_state
        )), device=device, dtype=torch.bool
    )

    non_final_next_states = torch.cat(
        [s for s in batch.next_state if s is not None]
    )
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


env = duels_v0.env()  # Create a default duels environment
plt.ion()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Saving the result of taking action a in state s,
# we progress to the next state and observe a reward
Transition = namedtuple(
    'Transition',
    ('state', 'action', 'next_state', 'reward', 'agent')
)
# BATCH_SIZE is the number of transitions sampled from the replay buffer
BATCH_SIZE = 128
# GAMMA is the discount factor as mentioned in the previous section
GAMMA = 0.99
# EPS_START is the starting value of epsilon
EPS_START = 0.1
# EPS_END is the final value of epsilon
EPS_END = 0  # In long games each action is really important, so
# we want to be greedy after lots of training
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
EPS_DECAY = 2000
# TAU is the update rate of the target network
TAU = 0.005
# LR is the learning rate of the AdamW optimizer
LR = 1e-4

# 4 actions, left, right, up, down
n_actions = 4
# Get the number of state observations
env.reset()
observation, reward, termination, truncation, info = env.last()
# Get the observation vector
state = observation_to_values(observation["observation"])
n_observations = len(state)  # Note the length of the vector

# Initialize the networks
num_hlayers = int(input("Number of hidden layers:   "))
width_hlayers = int(input("Width of hidden layers:   "))
hdims = [width_hlayers for i in range(0, num_hlayers)]

# Initialize the networks
policy_net = DQN(n_observations, n_actions, hdims).to(device)
target_net = DQN(n_observations, n_actions, hdims).to(device)
target_net.load_state_dict(policy_net.state_dict())

# Initialize the optimizer
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

# Initialize the replay memory
memory = ReplayMemory(10000)
steps_done = 0
# Number of turns the snake survives in each episode
episode_durations = []
# In test reaches about 250 turns on average in 2000 episodes
num_episodes = 500

for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    t = 0
    print("Episode: ", i_episode)
    env.reset()
    observation, reward, termination, truncation, info = env.last()
    state = observation_to_values(observation["observation"])
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    done = False

    while not done:
        for agent in env.agents:
            action = select_action(state)
            env.step(action.item())
            observation, reward, terminated, truncated, _ = env.last()
            done = terminated or truncated
            if i_episode % 50 == 0:
                time.sleep(0.1)
                env.render()
            if terminated:
                # Check which agent won, positive reward for winning, negative for losing
                # Go back into the memory and update the reward for the winning agent
                # Gets reward of 1 for making the winning move
                if len(observation["board"]["snakes"]) == 1:
                    winner = observation["board"]["snakes"][0]["id"]
                    loser = "agent_0" if winner == "agent_1" else "agent_1"
                    print("winner: ", winner, "loser: ", loser)
                    memory.add_transition_for_agent(winner, 1)
                    memory.add_transition_for_agent(loser, -1)
                else:
                    reward = 0
                next_state = None
            else:
                reward = 0 if t > 50 else 0.1
                next_state = torch.tensor(
                    observation_to_values(observation),
                    dtype=torch.float32, device=device
                ).unsqueeze(0)
            t += 0.5
            reward = torch.tensor([reward], device=device)
            # Store the transition in memory
            memory.push(state, action, next_state, reward, agent)

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
                if i_episode % 100 == 0 and i_episode != 0:
                    # Only plotting every 100 eps to avoid the annoying popups
                    plot_durations()
                break

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()

#torch.save(policy_net.state_dict(), f"./saved_models/policy_weights_{num_episodes}.pt")