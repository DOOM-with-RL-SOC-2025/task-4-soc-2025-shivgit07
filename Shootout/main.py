import argparse
import os
import numpy as np
from environment import FootballEnv
from agent import RLAgent
from opponent import OpponentDefender
from mdp_solver import MDPSolver
import utils

def parse_args():
    parser = argparse.ArgumentParser(description="2v1 Football Shootout RL")
    parser.add_argument('--episodes', type=int, default=1000, help="Number of training episodes")
    parser.add_argument('--p', type=float, default=0.1, help="Move failure probability")
    parser.add_argument('--q', type=float, default=0.9, help="Base probability for pass/shoot")
    parser.add_argument('--defender_policy', choices=['greedy', 'bus', 'random'], default='greedy', help="Defender policy")
    parser.add_argument('--gamma', type=float, default=1.0, help="Discount factor")
    parser.add_argument('--epsilon', type=float, default=0.1, help="Epsilon-greedy exploration rate")
    parser.add_argument('--lr', type=float, default=0.1, help="Learning rate")
    parser.add_argument('--seed', type=int, default=None, help="Random seed")
    parser.add_argument('--plan', action='store_true', help="Run DP Value Iteration (offline planning)")
    parser.add_argument('--render', action='store_true', help="Render environment each step")
    parser.add_argument('--save', action='store_true', help="Save learning curve and rewards to results/")
    parser.add_argument('--animate', action='store_true', help="Visualize an episode as a matplotlib animation")
    return parser.parse_args()

def main():
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    os.makedirs("results", exist_ok=True)

    # Opponent
    defender = OpponentDefender(policy_type=args.defender_policy)

    # Environment
    env = FootballEnv(p=args.p, q=args.q, defender_policy=defender.get_action)
    GRID_SIZE = 4
    state_space_size = GRID_SIZE ** 6 * 2  # b1x,b1y,b2x,b2y,rx,ry,ball_holder
    action_space_size = 10

    # Agent
    agent = RLAgent(state_space_size, action_space_size,
                    gamma=args.gamma, epsilon=args.epsilon, learning_rate=args.lr)

    # Optional: Planning with Value Iteration
    if args.plan:
        print("Running Value Iteration...")
        mdp_solver = MDPSolver(env, gamma=args.gamma)
        optimal_V, optimal_policy = mdp_solver.value_iteration()
        greedy_table = np.zeros((state_space_size, action_space_size))
        for s in range(state_space_size):
            a = optimal_policy[s]
            greedy_table[s, a] = 1.0
        agent.set_policy(greedy_table)
        print("Planning completed: Value Iteration Optimal Policy loaded.")

    reward_history = []

    print("Starting episodes:")
    for episode in range(args.episodes):
        state = env.reset()
        done = False
        total_reward = 0
        step = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            total_reward += reward

            if args.render:
                print(f"Episode {episode+1} - Step {step+1}: Action: {utils.action_to_str(action)}")
                env.render()
                print("-" * 30)

            state = next_state
            step += 1

        reward_history.append(total_reward)
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}: Total Reward={total_reward:.2f}")

    # Save rewards and learning curve if requested
    if args.save:
        reward_log_file = os.path.join("results", "reward_history.npy")
        plot_file = os.path.join("results", "learning_curve.png")
        utils.save_results(reward_history, reward_log_file)
        utils.plot_results(reward_history, window=100, save_path=plot_file)

    # Evaluation
    print("Evaluating learned policy...")
    wins, eval_episodes = 0, 100
    for _ in range(eval_episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state, explore=False)
            state, reward, done, info = env.step(action)
        wins += reward
    win_rate = wins / eval_episodes
    print(f"Win rate over {eval_episodes} evaluation episodes: {win_rate:.2%}")

    # Optional animation of a single episode
    if args.animate:
        try:
            print("Animating a single policy episode...")
            from utils import animate_episode
            animate_episode(env, agent)
        except ImportError:
            print("matplotlib.animation required for --animate. Please install it with 'pip install matplotlib'.")

if __name__ == "__main__":
    main()
