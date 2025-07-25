import numpy as np

class MDPSolver:
    def __init__(self, env, gamma=1.0, theta=1e-6):
        """
        env: an environment instance providing reset(), step(), and state/action spaces.
        gamma: discount factor for future rewards.
        theta: convergence threshold for Value Iteration.
        """
        self.env = env
        self.gamma = gamma
        self.theta = theta

        self.num_states = env.state_space_size
        self.num_actions = env.action_space_size
        self.V = np.zeros(self.num_states)   # State-value function
        self.policy = np.zeros(self.num_states, dtype=int)  # Deterministic optimal policy

    def one_step_lookahead(self, state_id):
        """
        Compute expected values for all actions in a given state
        under the current value function.
        """
        action_values = np.zeros(self.num_actions)

        for a in range(self.num_actions):
            total, transitions = 0.0, self.env.get_transition_probs(state_id, a)
            # 'transitions' should be iterable of (prob, next_state, reward, done)
            for prob, next_state, reward, done in transitions:
                total += prob * (reward + self.gamma * self.V[next_state] * (not done))
            action_values[a] = total
        return action_values

    def value_iteration(self, max_iterations=1000):
        """
        Performs classic value iteration to compute optimal values and policies.
        """
        for i in range(max_iterations):
            delta = 0.0
            for state_id in range(self.num_states):
                action_values = self.one_step_lookahead(state_id)
                best_value = np.max(action_values)
                delta = max(delta, abs(best_value - self.V[state_id]))
                self.V[state_id] = best_value
            if delta < self.theta:
                break

        # Extract policy
        for state_id in range(self.num_states):
            action_values = self.one_step_lookahead(state_id)
            self.policy[state_id] = np.argmax(action_values)

        return self.V, self.policy

    def policy_evaluation(self, policy, max_iterations=1000):
        """
        Evaluates a given policy (optional: if using policy iteration).
        """
        V = np.zeros(self.num_states)
        for i in range(max_iterations):
            delta = 0
            for state_id in range(self.num_states):
                a = policy[state_id]
                total = 0
                transitions = self.env.get_transition_probs(state_id, a)
                for prob, next_state, reward, done in transitions:
                    total += prob * (reward + self.gamma * V[next_state] * (not done))
                delta = max(delta, abs(V[state_id] - total))
                V[state_id] = total
            if delta < self.theta:
                break
        return V

    def policy_iteration(self, max_iterations=100):
        """
        (Optional) Policy iteration for policy/value optimization.
        """
        policy = np.random.choice(self.num_actions, size=self.num_states)
        for i in range(max_iterations):
            V = self.policy_evaluation(policy)
            policy_stable = True
            for state_id in range(self.num_states):
                old_action = policy[state_id]
                action_values = self.one_step_lookahead(state_id)
                policy[state_id] = np.argmax(action_values)
                if old_action != policy[state_id]:
                    policy_stable = False
            if policy_stable:
                break
        self.policy = policy
        self.V = self.policy_evaluation(policy)
        return self.V, self.policy
