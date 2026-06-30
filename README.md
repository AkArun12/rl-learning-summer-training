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
LunarLender-v3

(using Gymnasium)
...........................................
📁 Project Structure
rl-learning/
│
├── cartpole_class_dqn/              # Deep Q-Network experiments
|__ cartpole_class_ppo/
├── lunarlander_class_dqn/              # DQN experiments
|__ lunarlander_class_ppo/
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

4. Running Examples

A. cartpole_class_dqn
python train.py
python evaluate.py
(we can run "python evaluate.py" directly, since the model is trained or we can retarain and then evaluate)

B. lunarlander_class_dqn
python train.py
python evaluate.py


...........................................

5. To view Tensorboard
Type in the cmd terminal:

tensorboard --logdir="./logs"

...........................................
📊 Goals of this Project
Understand RL fundamentals
Compare Q-Learning vs Deep RL methods
Experiment with different environments
Track training performance and rewards

...........................................
Author : Arun Kathariya
