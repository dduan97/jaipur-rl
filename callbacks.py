import numpy as np

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.utils.typing import ResultDict
from typing import Any


def get_self_play_league_callback_class(
    num_league_opponents: int,
    main_policy_name: str,
    promotion_win_rate_threshold: float,
):
    class SelfPlayLeagueCallback(DefaultCallbacks):
        def __init__(
            self,
        ):
            super().__init__()
            self.win_stats: dict[str, int] = {"wins": 0, "total": 0}
            self.main_policy_name = main_policy_name
            self.promotion_win_rate_threshold = promotion_win_rate_threshold
            self.num_league_opponents = num_league_opponents
            self.league_opponent_to_replace = 0

        def on_train_result(self, *, algorithm, result: ResultDict, **kwargs):
            # Get the latest win rate tracked by the workers and aggregated by RLlib
            main_policy_rewards = result["env_runners"]["hist_stats"][
                f"policy_{self.main_policy_name}_reward"
            ]
            main_policy_rewards = np.array(main_policy_rewards)
            print("Main policy rewards this iteration:", main_policy_rewards)
            win_rate = np.sum(main_policy_rewards > 0) / len(main_policy_rewards)
            print("Main policy win rate this iteration:", win_rate)
            result["league_win_rate"] = win_rate

            if win_rate > self.promotion_win_rate_threshold:
                main_weights = algorithm.get_policy("main_policy").get_weights()

                league_policy_to_replace = (
                    f"league_policy_{self.league_opponent_to_replace}"
                )
                algorithm.get_policy(league_policy_to_replace).set_weights(main_weights)

                self.league_opponent_to_replace = (
                    self.league_opponent_to_replace + 1
                ) % self.num_league_opponents

                # Log the promotion event
                result["league_promotion"] = True

            else:
                result["league_promotion"] = False

    return SelfPlayLeagueCallback
