# 2v1 Football Shootout – Reinforcement Learning Project

A full RL project modeling a two-attacker vs. one-defender football (soccer) shootout as a Markov Decision Process (MDP), with modular code for training, planning, and evaluation.

## Project Structure

Shootout/</br>
├── environment.py  # Football MDP environment </br>
├── agent.py  # RL agent (Q-learning, policies)</br>
├── opponent.py  # Defender strategies</br>
├── mdp_solver.py   # Value Iteration / Planning</br>
├── main.py  # Runner script (train/evaluate)</br>
├── utils.py  # Helper functions (encoding, plotting, I/O)</br>
├── data/ # (Optional) Opponent/test states</br>
└── results/ # (Optional) Output graphs/logs</br>

## Overview

This project simulates a 2v1 half-field football shootout on a 4x4 grid. Two attackers (B1, B2) compete against one defender (R), modeled as an MDP for RL experimentation. You can train an RL agent, benchmark optimal value iteration policies, and experiment with different defender strategies.

## Setup

- **Python Requirements:**  
  - Python 3.7+
  - numpy
  - matplotlib

  Install required packages:

## Files Description

| File/Folder      | Description                                                                                    |
|------------------|------------------------------------------------------------------------------------------------|
| environment.py   | Environment class with MDP logic, transitions, rewards, and visualization                      |
| agent.py         | RL agent class (Q-learning, policy management, etc.)                                           |
| opponent.py      | Defender (R) policy logic: greedy, "bus", random                                               |
| mdp_solver.py    | Dynamic programming solvers: Value Iteration, Policy Iteration                                 |
| main.py          | Command-line runner: training, evaluation, planning, argument parsing                          |
| utils.py         | State encoding, plotting, saving/loading utilities         |
| data/            | *(Optional)* Store input data, pre-defined policies                                            |
| results/         | Output graphs, logs, experiment results                                           |

## Usage

Run from the command line:</br>
`python main.py`

### Key Arguments

| Argument             | Description                                       | Example                       |
|----------------------|---------------------------------------------------|-------------------------------|
| `--episodes`         | Number of training episodes                       | `--episodes 5000`             |
| `--p`                | Move failure probability                          | `--p 0.15`                    |
| `--q`                | Pass/shoot base probability                       | `--q 0.85`                    |
| `--defender_policy`  | Defender logic: `greedy`, `bus`, `random`         | `--defender_policy bus`       |
| `--gamma`            | Discount factor                                   | `--gamma 0.95`                |
| `--epsilon`          | Exploration rate (epsilon-greedy)                 | `--epsilon 0.2`               |
| `--lr`               | Learning rate                                     | `--lr 0.05`                   |
| `--seed`             | Random seed (for reproducibility)                 | `--seed 42`                   |
| `--plan`             | Run value iteration (offline planning)            | `--plan`                      |
| `--render`           | Show grid step-by-step during episodes            | `--render`                    |

View all options with:

## Example Commands

- **Train agent for 5,000 episodes with "bus" defense:**</br>
`python main.py --episodes 5000 --defender_policy bus`

- **Run Value Iteration only:**</br>
`python main.py --plan`

- **Visualize one episode:**</br>
`python main.py --render --episodes 1`


## Extending the Project
Since there are many policies and algos that could have been implemented and I tried only a few of them, the project can be extented in the following ways:
- Add new defender strategies by expanding `opponent.py`.
- Implement new RL algorithms (e.g., TD-Lambda) using the agent structure.
- Tune grid size, probabilities, or environment reward structure in `constants.py` or `environment.py`.
