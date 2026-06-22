import gymnasium as gym
from stable_baselines3 import DQN


def train():
    env = gym.make("CartPole-v1")

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        batch_size=64,
        buffer_size=10000,
        gamma=0.99,
        learning_starts=1000,
        target_update_interval=1000,
        verbose=1,
        tensorboard_log="./dqn_logs/",
    )

    model.learn(total_timesteps=200000)
    model.save("dqn_model/cartpole_dqn")

    env.close()


if __name__ == "__main__":
    train()
