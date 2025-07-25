import numpy as np
import matplotlib.pyplot as plt

GRID_SIZE = 4  # Should correspond to environment.py

def encode_state(state):
    """
    Encodes the environment state into a unique integer ID for tabular methods.
    State format: [[b1x, b1y], [b2x, b2y], [rx, ry], ball_holder]
    """
    b1x, b1y = state[0]
    b2x, b2y = state[1]
    rx, ry = state[2]
    ball_holder = state[3]
    return (
        b1x * GRID_SIZE**5 +
        b1y * GRID_SIZE**4 +
        b2x * GRID_SIZE**3 +
        b2y * GRID_SIZE**2 +
        rx * GRID_SIZE + 
        ry * 2 + 
        ball_holder
    )

def decode_state(state_id):
    """
    Decodes an integer state ID back into state components.
    """
    ball_holder = state_id % 2
    state_id //= 2
    ry = state_id % GRID_SIZE
    state_id //= GRID_SIZE
    rx = state_id % GRID_SIZE
    state_id //= GRID_SIZE
    b2y = state_id % GRID_SIZE
    state_id //= GRID_SIZE
    b2x = state_id % GRID_SIZE
    state_id //= GRID_SIZE
    b1y = state_id % GRID_SIZE
    state_id //= GRID_SIZE
    b1x = state_id % GRID_SIZE
    return [[b1x, b1y], [b2x, b2y], [rx, ry], ball_holder]

def plot_results(reward_history, window=100):
    """
    Plots a moving average of rewards over episodes.
    """
    plt.figure(figsize=(10,5))
    rewards = np.array(reward_history)
    mov_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.plot(mov_avg)
    plt.xlabel('Episode')
    plt.ylabel(f'Average Reward (per {window} episodes)')
    plt.title('Learning Curve')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def save_policy(policy, filename):
    """
    Saves a policy (array) to a file.
    """
    np.save(filename, policy)

def load_policy(filename):
    """
    Loads a policy (array) from a file.
    """
    return np.load(filename)

def state_to_str(state):
    """
    Returns a string representation of the state for debugging.
    """
    return f"B1:{state[0]}, B2:{state[1]}, R:{state[2]}, Ball:{'B1' if state[3]==0 else 'B2'}"

def action_to_str(action):
    """
    Returns an action’s string label.
    """
    action_labels = [
        "B1_Left", "B1_Right", "B1_Up", "B1_Down",
        "B2_Left", "B2_Right", "B2_Up", "B2_Down",
        "Pass", "Shoot"
    ]
    return action_labels[action] if 0 <= action < len(action_labels) else str(action)

import numpy as np
import matplotlib.pyplot as plt
import os

def plot_results(reward_history, window=100, save_path=None):
    """
    Plots and (optionally) saves a moving average of rewards.
    """
    plt.figure(figsize=(10,5))
    rewards = np.array(reward_history)
    mov_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.plot(mov_avg)
    plt.xlabel('Episode')
    plt.ylabel(f'Average Reward (per {window} episodes)')
    plt.title('Learning Curve')
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    plt.close()

def save_results(reward_history, filename):
    """
    Saves reward history or other experiment data to a CSV/NPY file.
    """
    np.save(filename, reward_history)
    print(f"Reward history saved to {filename}.npy")
