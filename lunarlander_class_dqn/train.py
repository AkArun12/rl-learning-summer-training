from agents.dqn_agent import DQNAgent


def main():

    agent = DQNAgent("LunarLander-v3")

    agent.train_agent(timesteps=2_000_000)

    agent.save_model("models/lunarlander-v3_dqn")

    print(" Agent Trained Successfully and  model saved !")


if __name__ == "__main__":
    main()
