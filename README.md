📌 Reinforcement Learning Playground

This repository contains my implementations and experiments with Reinforcement Learning (RL) algorithms using Gymnasium and Stable-Baselines3.

I am using this project to learn and compare different RL methods like:

Q-Learning (tabular method)
Deep Q-Network (DQN)
Proximal Policy Optimization (PPO)

...........................................
🧠 Algorithms Covered
Q-Learning
DQN (Deep Q-Network)
PPO (Proximal Policy Optimization)
...........................................
🎮 Environments Used
CartPole-v1
MountainCar-v0
FrozenLake-v1
(using Gymnasium)
...........................................
📁 Project Structure
rl-learning/
│
├── dqn/              # Deep Q-Network experiments
├── ppo/              # PPO experiments
├── q_learning/       # Tabular Q-learning
├── common/           # Helper functions (plots, utils)
├── results/          # Training results & graphs
├── requirements.txt
└── README.md

...........................................

⚙️ Installation

1. Clone the repository:
git clone https://github.com/AkArun12/rl-learning-summer-training.git
cd rl-learning-summer-training

2. Create virtual environment:
python -m venv rlvenv
source rlvenv/bin/activate   # Linux/Mac
# or
rlvenv\Scripts\activate      # Windows

3. Install dependencies:
pip install -r requirements.txt

...........................................

🚀 Running Examples
1. DQN on CartPole
python dqn/cartpole.py

2. PPO on CartPole
python ppo/cartpole.py

3. Q-Learning (FrozenLake)
python q_learning/frozenlake.py

...........................................
📊 Goals of this Project
Understand RL fundamentals
Compare Q-Learning vs Deep RL methods
Experiment with different environments
Track training performance and rewards

...........................................Author : Arun Kathariya