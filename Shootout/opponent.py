import numpy as np

# Directions mapping (ensure consistent with environment.py)
DIRECTIONS = {
    "Left": (0, -1),
    "Right": (0, 1),
    "Up": (-1, 0),
    "Down": (1, 0),
    "Stay": (0, 0)
}
DIRECTION_LIST = ["Left", "Right", "Up", "Down", "Stay"]

class OpponentDefender:
    def __init__(self, policy_type='greedy'):
        """
        Initialize with a specified policy type:
        - 'greedy': Moves toward ball holder.
        - 'bus': Parks in front of goal (x=3).
        - 'random': Moves randomly.
        """
        self.policy_type = policy_type

    def get_action(self, state):
        """
        Returns the next move as a string: 'Left', 'Right', 'Up', 'Down', or 'Stay'.
        """
        b1, b2, r, ball_holder = state
        ball_pos = b1 if ball_holder == 0 else b2

        if self.policy_type == 'greedy':
            return self.greedy_policy(r, ball_pos)
        elif self.policy_type == 'bus':
            return self.park_the_bus_policy(r)
        elif self.policy_type == 'random':
            return np.random.choice(DIRECTION_LIST)
        else:
            raise ValueError("Unknown defender policy.")

    def greedy_policy(self, r, ball_pos):
        # Moves toward the ball holder—minimizes distance, prioritizing row(x) moves first
        move = 'Stay'
        min_dist = float('inf')
        r_x, r_y = r
        b_x, b_y = ball_pos

        # Try all moves and pick the one that gets closer
        for direction, (dx, dy) in DIRECTIONS.items():
            new_r = [r_x + dx, r_y + dy]
            dist = abs(new_r[0] - b_x) + abs(new_r[1] - b_y)
            if self._in_bounds(new_r) and dist < min_dist:
                min_dist = dist
                move = direction
        return move

    def park_the_bus_policy(self, r):
        # Always try to move to or stay at row 3, columns 1 or 2 (front of goal)
        targets = [[3,1], [3,2]]
        r_x, r_y = r

        if [r_x, r_y] in targets:
            return 'Stay'

        min_dist = float('inf')
        best_move = 'Stay'
        for target in targets:
            for direction, (dx, dy) in DIRECTIONS.items():
                new_r = [r_x + dx, r_y + dy]
                dist = abs(new_r[0] - target[0]) + abs(new_r[1] - target[1])
                if self._in_bounds(new_r) and dist < min_dist:
                    min_dist = dist
                    best_move = direction
        return best_move

    def _in_bounds(self, pos, grid_size=4):
        return 0 <= pos[0] < grid_size and 0 <= pos[1] < grid_size
