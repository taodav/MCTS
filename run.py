from mcts import MonteCarloAgent
from game import GridWorld


if __name__ == "__main__":
    env = GridWorld()
    agent = MonteCarloAgent(env)

    action = agent.get_action()

    reward = env.act(action)
