import numpy as np
import matplotlib.pyplot as plt
import copy
import os

GRID_SIZE = 4

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

def plot_results(reward_history, window=100, save_path=None):
    """
    Plots and (optionally) saves a moving average of rewards.
    """
    plt.figure(figsize=(10, 5))
    rewards = np.array(reward_history)
    if rewards.size < window:
        window = max(1, rewards.size)
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
    Saves reward history or other experiment data to a NPY file.
    """
    np.save(filename, reward_history)
    print(f"Reward history saved to {filename}.npy")

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

def animate_multiple_episodes(env, agent=None, num_episodes=10, max_steps=50, save_path=None):
    """
    Animates several episodes in sequence, showing how the agents move the ball over time.

    Args:
        env: The environment instance.
        agent: The agent; used for action selection if provided.
        num_episodes: How many episodes to animate in sequence.
        max_steps: Maximum steps per episode (for truncation).
        save_path: File name/path for saving the animation (optional).
    """
    import matplotlib.animation as animation
    fig, ax = plt.subplots(figsize=(5, 5))
    frames = []
    episode_ends = []
    color_ep = plt.cm.get_cmap('tab10')

    for ep in range(num_episodes):
        state = copy.deepcopy(env.reset())
        done = False
        step = 0
        while not done and step < max_steps:
            # Always deep copy state before appending for animation correctness
            if agent is not None:
                action = agent.select_action(state, explore=False)
            else:
                action = np.random.choice(10)
            frames.append((copy.deepcopy(state), action, ep))
            next_state, reward, done, info = env.step(action)
            state = copy.deepcopy(next_state)
            step += 1
        # Final state for episode marker
        frames.append((copy.deepcopy(state), None, ep))
        episode_ends.append(len(frames)-1)

    def draw_frame(frame_idx):
        ax.clear()
        state, action, ep = frames[frame_idx]
        b1, b2, r, holder = state
        ax.set_xlim(-0.5, GRID_SIZE - 0.5)
        ax.set_ylim(-0.5, GRID_SIZE - 0.5)
        ax.set_xticks(range(GRID_SIZE))
        ax.set_yticks(range(GRID_SIZE))
        ax.grid(True)
        def to_display(coord):
            return (coord[1], GRID_SIZE - 1 - coord[0])
        c = color_ep(ep % 10)
        ax.add_patch(plt.Circle(to_display(b1), 0.24, color=c, ec='blue', label='B1', zorder=2))
        ax.add_patch(plt.Circle(to_display(b2), 0.24, color=c, ec='green', label='B2', zorder=2))
        ax.add_patch(plt.Circle(to_display(r), 0.24, color='red', label='R', zorder=2))
        if holder == 0:
            ax.add_patch(plt.Circle(to_display(b1), 0.12, color='orange', zorder=3))
        else:
            ax.add_patch(plt.Circle(to_display(b2), 0.12, color='orange', zorder=3))
        ep_break = ""
        if frame_idx in episode_ends:
            ep_break = " (Episode End)"
        ax.set_title(f"Episode {ep+1}, Step {frame_idx if ep==0 else frame_idx-episode_ends[ep-1]}{ep_break}")
        ax.legend(['B1', 'B2', 'R'], loc="upper right")

    anim = animation.FuncAnimation(
        fig, draw_frame, frames=len(frames), interval=500, repeat=False
    )
    if save_path:
        anim.save(save_path, writer="imagemagick")
        print(f"Animation saved to {save_path}")
    else:
        plt.show()
