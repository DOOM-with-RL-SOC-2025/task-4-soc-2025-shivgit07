import numpy as np
import utils

class RLAgent:
    def __init__(self, state_space_size, action_space_size, gamma=1.0, epsilon=0.1, learning_rate=0.1):
        """
        Parameters:
        - state_space_size: total number of unique environment states (may need encoding).
        - action_space_size: total number of discrete actions allowed.
        - gamma: discount factor.
        - epsilon: exploration rate for epsilon-greedy policy.
        - learning_rate: step size for updates.
        """
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate

        # Tabular Q-values (state-action value estimates)
        self.q_table = np.zeros((state_space_size, action_space_size))

    def encode_state(self, state):
        return utils.encode_state(state)
    
    def select_action(self, state, explore=True):
        """
        Epsilon-greedy action selection.
        Returns the action integer.
        """
        state_id = self.encode_state(state)
        if explore and np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space_size)
        else:
            # Break ties randomly for max Q
            q_values = self.q_table[state_id]
            max_q = np.max(q_values)
            max_actions = np.where(q_values == max_q)[0]
            return np.random.choice(max_actions)

    def update(self, state, action, reward, next_state, done):
        """
        Q-learning update (off-policy TD control)
        """
        state_id = self.encode_state(state)
        next_state_id = self.encode_state(next_state)
        best_next = np.max(self.q_table[next_state_id])
        td_target = reward + self.gamma * best_next * (not done)
        td_error = td_target - self.q_table[state_id, action]
        self.q_table[state_id, action] += self.learning_rate * td_error

    def set_policy(self, q_table):
        """Set pre-computed action-value table externally (for Value Iteration, etc.)."""
        self.q_table = q_table.copy()

    def get_policy(self):
        """Returns a greedy policy based on current Q-values."""
        return np.argmax(self.q_table, axis=1)

    def save(self, filepath):
        """Save Q-table to file."""
        np.save(filepath, self.q_table)

    def load(self, filepath):
        """Load Q-table from file."""
        self.q_table = np.load(filepath)

