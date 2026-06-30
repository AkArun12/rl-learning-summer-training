import gymnasium as gym

from agents.dqn_agent import DQNAgent


def main():
    agent = DQNAgent("LunarLander-v3")

    agent.load_model("models/lunarlander-v3_dqn")

    env = gym.make(
        "LunarLander-v3", 
        continuous=False,
        gravity=-10.0,
        enable_wind=False,
        wind_power=15.0,
        turbulence_power=1.5,
        render_mode="human")

    scores = []

    for episode in range(30):
        obs, info = env.reset()

        terminated = False
        truncated = False
        total_rewards = 0

        while not (truncated or terminated):
            action, _ = agent.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_rewards += reward

        scores.append(total_rewards)

        print(f" Episode {episode + 1}: {total_rewards}")
    
    print("\n--- Evaluation Results ---")
    print(f"\nAverage_reward: {sum(scores) / len(scores)}")
    print(f"Max reward: {max(scores):.2f}")
    print(f"Min reward: {min(scores):.2f}")
    env.close()


if __name__ == "__main__":
    main()
