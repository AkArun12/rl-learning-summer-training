from agents.dqn_agent import DQNAgent


def main():
    agent = DQNAgent("CartPole-v1")

    agent.train_agent(200000)

    agent.save_model("models/cartpole_dqn")
    print("Agent Trained and saved successfully !")


if __name__ == "__main__":
    main()
