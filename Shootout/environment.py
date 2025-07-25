import numpy as np
import copy

GRID_SIZE = 4
ACTIONS = {
    0: "B1_Left", 1: "B1_Right", 2: "B1_Up", 3: "B1_Down",
    4: "B2_Left", 5: "B2_Right", 6: "B2_Up", 7: "B2_Down",
    8: "Pass", 9: "Shoot"
}
DIRECTIONS = {
    "Left": (0, -1), "Right": (0, 1), "Up": (-1, 0), "Down": (1, 0)
}

class FootballEnv:
    def __init__(self, p=0.2, q=0.8, defender_policy=None):
        self.p = p  # move failure probability
        self.q = q  # base pass/shoot probability
        self.defender_policy = defender_policy  # function handle
        self.reset()
    
    def reset(self, init_state=None):
        # State: ([b1x, b1y], [b2x, b2y], [rx, ry], ball_holder)
        if init_state:
            self.state = init_state
        else:
            self.state = [[3,1], [3,2], [1,1], 0]  # default initial positions, ball with B1
        self.terminated = False
        self.reward = 0
        return self.state

    def encode_state(self):
        return tuple(self.state[0] + self.state[1] + self.state[2] + [self.state[3]])

    def is_out_of_bounds(self, pos):
        return any([pos[0] < 0, pos[0] >= GRID_SIZE, pos[1] < 0, pos[1] >= GRID_SIZE])

    def move(self, pos, direction):
        return [pos[0] + direction[0], pos[1] + direction[1]]

    def step(self, action):

        if self.terminated:
            raise ValueError("Episode has terminated, reset to continue.")

        # Deep copy positions so changes do not leak between steps
        b1 = copy.deepcopy(self.state[0])
        b2 = copy.deepcopy(self.state[1])
        r = copy.deepcopy(self.state[2])
        who_has_ball = self.state[3]

        # Defender moves first
        r_action = self.defender_policy([copy.deepcopy(b1), copy.deepcopy(b2), copy.deepcopy(r), who_has_ball])
        if r_action in DIRECTIONS:
            r_new = self.move(r, DIRECTIONS[r_action])
            if self.is_out_of_bounds(r_new):
                r_new = r  # Defender stays if move is out of bounds
        else:
            r_new = r
        r = r_new

        # Update the environment's defender position before processing the agent's action
        self.state[2] = copy.deepcopy(r)

        # Agent action handling
        if action in [0, 1, 2, 3]:  # B1 moves
            return self._player_move(0, DIRECTIONS[list(DIRECTIONS.keys())[action % 4]], r)
        elif action in [4, 5, 6, 7]:  # B2 moves
            return self._player_move(1, DIRECTIONS[list(DIRECTIONS.keys())[action % 4]], r)
        elif action == 8:  # Pass
            return self._pass(r)
        elif action == 9:  # Shoot
            return self._shoot(r)
        else:
            raise ValueError("Invalid action.")


    def _player_move(self, player_idx, direction, r):
        b1, b2, _, who_has_ball = self.state
        positions = [b1, b2]
        current = positions[player_idx]
        new_pos = self.move(current, direction)
        # Out of bounds check
        if self.is_out_of_bounds(new_pos):
            self.terminated = True
            self.reward = 0
            return self.state, self.reward, self.terminated, {}
        
        # With ball vs. without ball
        probs = None
        if who_has_ball == player_idx:
            # With the ball: Success (1-2p), fail (2p: lose ball, end)
            probs = [1-2*self.p, 2*self.p]
            outcome = np.random.choice(['success','fail'], p=probs)
            if outcome == 'fail':
                self.terminated = True
                self.reward = 0
                return self.state, self.reward, self.terminated, {}
            # Handle defender collision
            if new_pos == r:
                tackle = np.random.choice(['keep','lose'], p=[0.5,0.5])
                if tackle == 'lose':
                    self.terminated = True
                    self.reward = 0
                    return self.state, self.reward, self.terminated, {}
            positions[player_idx] = new_pos
        else:
            # Without the ball: Success (1-p), fail (p)
            probs = [1-self.p, self.p]
            outcome = np.random.choice(['success','fail'], p=probs)
            if outcome == 'fail':
                self.terminated = True
                self.reward = 0
                return self.state, self.reward, self.terminated, {}
            positions[player_idx] = new_pos

        # Check defender tackle by swap (if both move adjacent/opposite)
        b1_new, b2_new = positions
        if (b1_new == r or b2_new == r):
            # Already handled above for with-ball, usually not double-checked
            pass
        self.state = [b1_new, b2_new, r, who_has_ball]
        return self.state, self.reward, self.terminated, {}

    def _pass(self, r):
        b1, b2, _, who_has_ball = self.state
        passer = b1 if who_has_ball == 0 else b2
        receiver = b2 if who_has_ball == 0 else b1
        dist = max(abs(passer[0]-receiver[0]), abs(passer[1]-receiver[1]))
        
        qpass = self.q - 0.1 * dist
        # If defender between (check if R aligns on path between -- simplified, assuming straight passing allowed)
        between = False
        if passer[0] == receiver[0] and r[0] == passer[0] and min(passer[1],receiver[1]) <= r[1] <= max(passer[1], receiver[1]):
            between = True
        elif passer[1] == receiver[1] and r[1] == passer[1] and min(passer[0],receiver[0]) <= r[0] <= max(passer[0], receiver[0]):
            between = True
        elif passer[0]+passer[1] == receiver[1]+receiver[1] and min(passer[0],receiver[0]) <= r[0] <= max(passer[0], receiver[0]):
            between = True
        if between:
            qpass *= 0.5
        outcome = np.random.choice(['success','fail'], p=[max(0,qpass), 1-max(0,qpass)])
        if outcome == 'fail':
            self.terminated = True
            self.reward = 0
            return self.state, self.reward, self.terminated, {}
        # Pass success: ball holder switches
        self.state[3] = 1 if who_has_ball == 0 else 0
        return self.state, self.reward, self.terminated, {}

    def _shoot(self, r):
        b1, b2, _, who_has_ball = self.state
        shooter = b1 if who_has_ball == 0 else b2
        x = shooter[0]
        # Defender blocks if in front (y == 1 or 2, x == 3)
        in_front = (r[0] == 3 and (r[1] == 1 or r[1] == 2))
        qgoal = self.q - 0.2*(3-x)
        if in_front:
            qgoal *= 0.5
        outcome = np.random.choice(['goal','miss'], p=[max(0,qgoal), 1-max(0,qgoal)])
        if outcome == 'goal':
            self.terminated = True
            self.reward = 1
        else:
            self.terminated = True
            self.reward = 0
        return self.state, self.reward, self.terminated, {}

    def render(self):
        grid = np.full((GRID_SIZE, GRID_SIZE), '_')
        b1, b2, r, who_has_ball = self.state
        grid[b1[0], b1[1]] = 'B1' if who_has_ball == 0 else 'b1'
        grid[b2[0], b2[1]] = 'B2' if who_has_ball == 1 else 'b2'
        grid[r[0], r[1]] = 'R'
        for row in grid:
            print(' '.join(row))
        print(f"Ball holder: {'B1' if who_has_ball == 0 else 'B2'}")
