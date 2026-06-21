import gymnasium as gym
from stable_baselines3 import DQN


def evaluate():
    env = gym.make("CartPole-v1", render_mode="human")

    model = DQN.load("dqn_model/cartpole_dqn")

    obs, info = env.reset()

    terminated = False
    truncated = False

    total_reward = 0

    while not (truncated or terminated):
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        print("Reward: ", reward)

    print("Total Reward: ", total_reward)

    env.close()


if __name__ == "__main__":
    evaluate()
