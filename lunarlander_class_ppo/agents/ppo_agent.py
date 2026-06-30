from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
import gymnasium as gym

class PPOAgent:
    def __init__(self,env_name="LunarLander-v3"):
        self.env_name=env_name

        def make_env():
            return Monitor(gym.make(env_name))

        self.env=DummyVecEnv([make_env])
        self.env=VecNormalize(self.env, norm_obs=True, norm_reward=False)

        self.model=PPO(
            "MlpPolicy",
            self.env,
            learning_rate=3e-4,
            gamma=0.99,
            batch_size=64,
            verbose=1,
            gae_lambda=0.95,
            n_steps=2048,
            ent_coef=0.01,
            tensorboard_log="./logs"
        )

    #Function to train the model:

    def train_agent(self, timesteps=1_000_000):
        self.model.learn(total_timesteps=timesteps)
        self.env.save('vecnormalize.pkl')
        
    #Function to save the model:

    def save_model(self,path):
        self.model.save(path)
        
    #Function to load the model:

    def load_model(self,path):
        self.model=PPO.load(path, env=self.env)
        
    def close(self):
        self.env.close()
