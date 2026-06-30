import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from agents.ppo_agent import PPOAgent


def evaluate():
    agent = PPOAgent(env_name="LunarLander-v3")

    # Environment must match training setup
    env = DummyVecEnv([lambda: gym.make("LunarLander-v3", render_mode="human")])
    # Load normalization stats from training
    env = VecNormalize.load("vecnormalize.pkl", env)
    env.training = False  # No Training mode)
    env.norm_reward = False  # Don't normalize reward in evaluation

    # ===============================================

    # LOAD MODEL
    agent.load_model("models/lunarlander_ppo.zip")

    scores = []

    for episode in range(30):
        obs = env.reset()

        done = [False]

        episode_reward = 0

        while not done[0]:
            action, _ = agent.model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]

        scores.append(episode_reward)
        print(f"Episode {episode + 1}: Rewards: {episode_reward:.2f}")

    print(f"\nAvergae Reward: {sum(scores) / len(scores):.2f}")
    print(f"Max Reward: {max(scores):.2f}")
    print(f"Min Reward: {min(scores):.2f}")

    agent.close()


if __name__ == "__main__":
    evaluate()
