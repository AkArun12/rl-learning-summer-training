import gymnasium as gym
from agents.dqn_agent import DQNAgent


def evaluate():
    agent = DQNAgent("CartPole-v1")

    agent.load_model("models/cartpole_dqn.zip")

    env = gym.make("CartPole-v1", render_mode="human")

    scores = []

    for i in range(30):
        obs, info = env.reset()

        terminated = False
        truncated = False

        total_rewards = 0

        while not (truncated or terminated):
            action, _ = agent.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            total_rewards += reward
        scores.append(total_rewards)

        print(f"Episode {i + 1}: {total_rewards}")

    print("Average Reward :", sum(scores) / len(scores))
    env.close()


if __name__ == "__main__":
    evaluate()
