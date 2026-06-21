import gymnasium as gym
from stable_baselines3 import DQN


def train():
    env = gym.make("CartPole-v1", render_mode="human")

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=10000,
        gamma=0.99,
        verbose=1,
    )

    model.learn(total_timesteps=50000)
    model.save("dqn_model/cartpole_dqn")

    env.close()


if __name__ == "__main__":
    train()
