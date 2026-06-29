from agents.ppo_agent import PPOAgent


def main():
    agent = PPOAgent("CartPole-v1")

    agent.train_agent(200000)
    agent.save_model("models/cartpole_ppo")
    print("Agent trained and successfully saved!")


if __name__ == "__main__":
    main()
