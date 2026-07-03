# 📌 Reinforcement Learning Playground

This repository contains my implementations and experiments with Reinforcement Learning (RL) algorithms using **Gymnasium** and **Stable-Baselines3**.

I am using this project to learn and compare different RL methods such as:

* Q-Learning (tabular method)
* Deep Q-Network (DQN)
* Proximal Policy Optimization (PPO)

---

## 🧠 Algorithms Covered

* Q-Learning
* DQN (Deep Q-Network)
* PPO (Proximal Policy Optimization)

---

## 🎮 Environments Used

* CartPole-v1
* LunarLander-v3 *(Gymnasium)*

---

## 📁 Project Structure

```text
rl-learning/
│
├── cartpole_class_dqn/        # Deep Q-Network experiments
├── cartpole_class_ppo/        # PPO experiments
├── lunarlander_class_dqn/     # DQN experiments
├── lunarlander_class_ppo/     # PPO experiments
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/AkArun12/rl-learning-summer-training.git
cd rl-learning-summer-training
```

---

### 2. Create virtual environment

```bash
python -m venv rlvenv
```

Activate:

**Linux / Mac**

```bash
source rlvenv/bin/activate
```

**Windows**

```bash
rlvenv\Scripts\activate
```

---

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running Examples

### 🟢 CartPole (DQN)

```bash
python train.py
python evaluate.py
```

📌 Note:
You can run `evaluate.py` directly if a trained model already exists. Otherwise, train first.

---

### 🟢 LunarLander (DQN)

```bash
python train.py
python evaluate.py
```

---

## 📊 TensorBoard Visualization

To track training performance:

```bash
tensorboard --logdir=./logs
```

Then open in browser:

```
http://localhost:6006
```

---

## 🎯 Goals of This Project

* Understand Reinforcement Learning fundamentals
* Compare Q-Learning vs Deep RL vs PPO methods
* Experiment with different Gymnasium environments
* Track training performance and rewards

---

## 👨‍💻 Author

**Arun Kathariya**
GitHub: [https://github.com/AkArun12]

---
