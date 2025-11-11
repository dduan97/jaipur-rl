import numpy as np

from gymnasium import spaces
from pettingzoo.utils import AECEnv

from jaipur_engine import JaipurEngine

class JaipurEnv(AECEnv):
    def __init__(self):
        self.possible_agents = ["player_0", "player_1"]
        self.agents = self.possible_agents[:]
        self.agent_name_mapping = {name: i for i, name in enumerate(self.possible_agents)}

        # TODO: action spaces and observation spaces
        self.action_space = [spaces.Discrete(25469) for agent in self.agents]
        self.observation_spaces = {agent: spaces.Dict({
                'observation': spaces.Box(low=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]), high=np.array([7,7,7,7,7,7,5,5,5,5,5,5,5,11,221,11,5,5,5,7,7,9,7,6,5]), dtype=np.int16),
                'action_mask': spaces.Box(low=0, high=1, shape=(25469,), dtype=np.int8),
            }) for agent in self.agents
        }


    def reset(self):
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

    def observe(self, agent):
        pass
    def step(self, action):
        pass

    def _next_agent(self):
        pass
    def render(self):
        pass
    def close(self):
        pass