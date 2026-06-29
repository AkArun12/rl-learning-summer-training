from stable_baselines3 import PPO
import gymnasium as gym


class PPOAgent:
    def __init__(self, env_name="CartPole-v1"):
        self.env_name = env_name
        self.env = gym.make(env_name)

        self.model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=0.0003,
            gamma=0.99,
            batch_size=64,
            verbose=1,
            tensorboard_log="./logs",
        )

    # Function to train the model:
    def train_agent(self, timesteps=200000):
        self.model.learn(total_timesteps=timesteps)

    # Function to save the model:
    def save_model(self, path):
        self.model.save(path)

    # Function to load the model:

    def load_model(self, path):
        self.model = PPO.load(path, env=self.env)

    def close(self):
        self.env.close()
