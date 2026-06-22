import gymnasium as gym
from stable_baselines3 import DQN


class DQNAgent:
    def __init__(self, env_name="LunarLander-v3"):
        self.env_name = env_name
        self.env = gym.make(env_name)

        self.model = DQN(
            "MlpPolicy",
            self.env,
            learning_rate=1e-4,
            gamma=0.99,
            buffer_size=100000,
            batch_size=64,
            learning_starts=10000,
            target_update_interval=1000,
            verbose=1,
            tensorboard_log="./logs",
        )

    # Function to train the model:
    def train_agent(self, timesteps=200000):
        self.model.learn(timesteps)

    # Function to save the agent(model):
    def save_model(self, path):
        self.model.save(path)

    # Function to load the saved model:
    def load_model(self, path):
        self.model = DQN.load(path)
