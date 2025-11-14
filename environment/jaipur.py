import numpy as np

from gymnasium import spaces
from pettingzoo.utils import AECEnv

from environment.jaipur_engine import JaipurEngine, ActionType


class JaipurEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "jaipur_v1"}

    def __init__(self, include_intermediate_rewards=False):
        self.include_intermediate_rewards = include_intermediate_rewards
        self.possible_agents = ["player_1", "player_2"]
        self.agents = self.possible_agents[:]
        self.agent_name_mapping = {
            name: i for i, name in enumerate(self.possible_agents)
        }

        self.engine = JaipurEngine(self.agents)
        self.n_actions = len(self.engine.all_actions)
        print('Number of actions:', self.n_actions)
        self.action_spaces = {
            agent: spaces.Discrete(self.n_actions) for agent in self.agents
        }
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}

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
        # - curent score from player
        # - curent score from opponent
        # Total: 34 features

        self.observation_spaces = {
            agent: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=np.array([0 for _ in range(34)]),
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
                                221,
                                221,
                            ]
                        ),
                        dtype=np.int16,
                    ),
                    "action_mask": spaces.Box(
                        low=0, high=1, shape=(self.n_actions,), dtype=np.int8
                    ),
                }
            )
            for agent in self.agents
        }
        self.reset()

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None, return_info=False):
        del options
        if seed is not None:
            np.random.seed(seed)
        self.agents = self.possible_agents[:]
        self.agent_selection = "player_1"

        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.engine = JaipurEngine(self.agents)
        self.num_steps = 0
        if return_info:
            return self.infos

    def observe(self, agent: str) -> np.array:
        features = self.engine.get_observation(agent)
        return {
            "observation": np.array(features, dtype=np.int16),
            "action_mask": np.array(
                self.engine.get_masked_options(agent), dtype=np.int8
            ),
        }

    def step(self, action: int):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            # handles stepping an agent which is already dead
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next dead agent,  or if there are no more dead agents, to the next live agent
            self._was_dead_step(action)
            return

        current_agent = self.agent_selection
        observe = self.observe(current_agent)
        obs = observe["observation"]
        mask = observe["action_mask"]
        # print("Action idx", action)

        for agent in self.agents:
            self.rewards[agent] = 0

        if not mask[action]:
            print("Invalid action!")
            self.rewards[current_agent] -= 5
            # action = np.random.choice([idx for idx in range(len(mask)) if mask[idx] and self.engine.all_actions[idx].action_type != ActionType.TRADE_WITH_MARKETPLACE])

        # print("Action", self.engine.all_actions[action])

        self.engine.perform_action(player_name=current_agent, action_idx=action)

        if self.engine.is_finished():
            self.engine.finalize_round()
            self.terminations = {a: True for a in self.agents}

            # Get the winner
            agent_scores = [self.engine.compute_score(a) for a in self.agents]
            winner_idx = np.argmax(agent_scores)

            self.rewards[self.agents[winner_idx]] += 1
            self.rewards[self.agents[1 - winner_idx]] -= 1

        current_score = self.engine.compute_score(current_agent)
        past_score = int(obs[-2])
        if self.include_intermediate_rewards:
            self.rewards[current_agent] += (current_score - past_score) / 100.0

        self.agent_selection = self._next_agent()
        self._accumulate_rewards()
        self.num_steps += 1

        if self.engine.is_finished():
            print(
                f"Game over! Cumulative rewards:",
                self._cumulative_rewards,
                f"after {self.num_steps} steps.",
            )

    def _next_agent(self) -> str:
        idx = self.agent_name_mapping[self.agent_selection]
        return self.agents[(idx + 1) % 2]

    def render(self):
        print(self.engine.game_state)

    def close(self):
        pass


# PettingZoo compatibility function
def env(**kwargs):
    return JaipurEnv(**kwargs)
