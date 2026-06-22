import gymnasium as gym
from stable_baselines3 import DQN


def evaluate():
    env = gym.make("CartPole-v1", render_mode="human")

    model = DQN.load("dqn_model/cartpole_dqn")

    scores = []

    for i in range(30):
        obs, info = env.reset()

        terminated = False
        truncated = False

        total_reward = 0

        while not (truncated or terminated):
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        scores.append(total_reward)

    print("Average Reward: ", sum(scores) / len(scores))
    print(f"Best: {max(scores)}")
    print(f"Worst: {min(scores)}")

    env.close()


if __name__ == "__main__":
    evaluate()
