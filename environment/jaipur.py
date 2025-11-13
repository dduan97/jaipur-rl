import numpy as np

from gymnasium import spaces
from pettingzoo.utils import AECEnv

from environment.jaipur_engine import JaipurEngine


class JaipurEnv(AECEnv):
    def __init__(self):
        self.possible_agents = ["player_1", "player_2"]
        self.agents = self.possible_agents[:]
        self.agent_name_mapping = {
            name: i for i, name in enumerate(self.possible_agents)
        }

        # TODO: action spaces and observation spaces
        self.action_space = [spaces.Discrete(25469) for agent in self.agents]
        self.engine = JaipurEngine(self.agents)

        # Observations:
        # - marketplace (7 features, one for each goods type. Maxes are all 5)
        # - current hand (6 features, one for each goods type except camel. Maxes are in order, 6,6,6,8,8,10,11
        # - opponent hand size (1 feature, max 7)
        # - own herd size (1 feature, max 11)
        # - opponent herd size (1 feature, max 11)
        # - number of goods tokens left (6 features, one for each goods type except camel. Maxes are )
        # - number of bonus tokens left (3 features, one for each bonus token)
        # - whether or not there are cards left in the deck (1 feature, 0 or 1)
        # - cards in the discard (6 features, one for each goods type except camel)
        # Total: 32 features

        self.n_actions = len(self.engine.all_actions)
        self.observation_spaces = {
            agent: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=np.array([0 for _ in range(32)]),
                        high=np.array(
                            [
                                5,
                                5,
                                5,
                                5,
                                5,
                                5,
                                5,
                                6,
                                6,
                                6,
                                8,
                                8,
                                10,
                                7,
                                11,
                                11,
                                5,
                                5,
                                5,
                                7,
                                7,
                                9,
                                7,
                                6,
                                5,
                                1,
                                6,
                                6,
                                6,
                                8,
                                8,
                                10,
                            ]
                        ),
                        dtype=np.int8,
                    ),
                    "action_mask": spaces.Box(
                        low=0, high=1, shape=(self.n_actions,), dtype=np.int8
                    ),
                }
            )
            for agent in self.agents
        }
        self.reset()

    def reset(self):
        self.agents = self.possible_agents[:]
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.engine = JaipurEngine(self.agents)
        self.agent_selection = "player_1"
        self.iteration = 0

    def observe(self, agent: str) -> np.array:
        features = self.engine.get_observation(agent)
        return {
            "observation": np.array(features, dtype=np.int8),
            "action_mask": self.engine.get_masked_options(agent),
        }

    def step(self, action: int):
        rewards = {a: 0 for a in self.agents}
        if self.terminations[self.agent_selection]:
            self._was_dead_step(action)
            return rewards

        # Process the move
        self.engine.perform_action(player_name=self.agent_selection, action_idx=action)

        # Check for termination
        if self.engine.is_finished():
            print(f"Game is over after {self.iteration} steps")
            for agent in self.agents:
                self.terminations[agent] = True

            self.engine.finalize_round()
            for agent in self.agents:
                # And compute rewards
                rewards[agent] = self.engine.compute_score(agent)
            print('Final rewards', rewards)
        self.iteration += 1
        self.agent_selection = self._next_agent()
        return rewards

    def _next_agent(self) -> str:
        idx = self.agent_name_mapping[self.agent_selection]
        return self.agents[(idx + 1) % 2]

    def render(self):
        print(self.engine.game_state)

    def close(self):
        pass
