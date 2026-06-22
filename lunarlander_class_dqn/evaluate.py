import gymnasium as gym

from agents.dqn_agent import DQNAgent


def main():
    agent = DQNAgent("LunarLander-v3")

    agent.load_model("models/lunarlander-v3_dqn")

    env = gym.make("LunarLander-v3", render_mode="human")

    scores = []

    for i in range(20):
        obs, item = env.reset()

        terminated = False
        truncated = False
        total_rewards = 0

        while not (truncated or terminated):
            action, _ = agent.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_rewards += reward

        scores.append(total_rewards)

        print(f" Episode {i + 1}: {total_rewards}")
    print(f"Average_reward: {sum(scores) / len(scores)}")

    env.close()


if __name__ == "__main__":
    main()
