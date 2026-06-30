from agents.ppo_agent import PPOAgent

def main():
    agent=PPOAgent("LunarLander-v3")

    agent.train_agent(1_000_000)

    agent.save_model("models/lunarlander_ppo")

    agent.close()
    
    print("Model trained and saved successfully!")



if __name__ == "__main__":
    main()