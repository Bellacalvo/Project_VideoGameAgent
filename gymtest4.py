import numpy as np
import gymenv
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


env = gym.make('Sidescroller-v0', render_mode='human')
obs, info = env.reset()

num_bins = 10
num_actions = env.action_space.n
obs_low = env.observation_space.low  
obs_high = env.observation_space.high  

bins = [np.linspace(low, high, num_bins) for low, high in zip(obs_low, obs_high)]

Q_table = np.zeros([num_bins] * len(obs_low) + [num_actions])

def discretize_state(obs):
    discrete_state = []
    for i, value in enumerate(obs):
        discrete_state.append(np.digitize(value, bins[i]) - 1)  # Subtract 1 for zero-indexing
    return tuple(discrete_state)

alpha = 0.5 # learning rate (very high)
gamma = 0.9 # discount rate (mid-range)
epsilon = 1 # start at 100% exploration
epsilon_decay_rate = 0.0001

# Training loop
total_reward = 0
step_count = 0
done = False

while not done and step_count < 1000:
    if np.random.rand() < epsilon:
        action = env.action_space.sample()  # Exploration
    else:
        state_discrete = discretize_state(obs)
        action = np.argmax(Q_table[state_discrete])  # Exploitation
    
    # Take action and observe result
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    step_count += 1
    done = terminated or truncated
    env.render()


    state_discrete = discretize_state(obs)
    next_max = np.max(Q_table[state_discrete])  
    Q_table[state_discrete][action] += alpha * (reward + gamma * next_max - Q_table[state_discrete][action])
    
    # Log progress
    action_list = [ 'Run Right', 'Run Left', 'Jump', 
               'Jump Left', 'Jump Right', 'Shoot', 'Grenade' ]
    action = action_list[action]

    print(f"Step {step_count:5}: action={action:<10} "
          f"reward={reward:>6.2f} "
          f"health={info['player_health']:2} "
          f"position=({info['player_distance'][0]:4}, {info['player_distance'][0]:4}) "
          f"exit=({info['exit_distance'][0]:4}, {info['exit_distance'][0]:4}) "
          f"done={done}")

print(f"Total reward after {step_count} steps: {total_reward:.2f}")
env.close()
