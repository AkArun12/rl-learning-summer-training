import gymnasium as gym
from stable_baselines3 import DQN


class DQNAgent:
    def __init__(self, env_name="LunarLander-v3"):
        self.env_name = env_name
        self.env = gym.make(
            env_name,
            continuous=False,
            gravity=-10.0,
            enable_wind=False,
            wind_power=15.0,
            turbulence_power=1.5,
        )

        self.model = DQN(
            "MlpPolicy",
            self.env,
            learning_rate=1e-4,
            gamma=0.99,
            buffer_size=500_000,
            batch_size=128,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,
            max_grad_norm=10,
            exploration_fraction=0.2,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            learning_starts=50_000,
            verbose=1,
            tensorboard_log="./logs",
        )

    # Function to train the model:
    def train_agent(self, timesteps=2_000_000):
        self.model.learn(timesteps)

    # Function to save the agent(model):
    def save_model(self, path):
        self.model.save(path)

    # Function to load the saved model:
    def load_model(self, path):
        self.model = DQN.load(path, env=self.env)
