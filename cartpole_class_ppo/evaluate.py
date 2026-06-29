import gymnasium as gym
from agents.ppo_agent import PPOAgent

def evaluate():
    agent=PPOAgent("CartPole-v1")

    agent.load_model("models/cartpole_ppo.zip")

    env=gym.make("CartPole-v1", render_mode="human")
    scores=[]

    for i in range(30):
        obs, info= env.reset()
        terminated=False
        truncated=False

        total_rewards=0

        while not (terminated or truncated):
            action,_= agent.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_rewards+=reward
        scores.append(total_rewards)
        print(f"Episode {i+1} :  {total_rewards}")
    print(f"Average Reward: {sum(scores)/len(scores)}")
    print(f"Maximum Reward : {max(scores):.2f}")
    print(f"Minimum Reward : {min(scores):.2f}")
    env.close()

if __name__ == "__main__":
    evaluate()

