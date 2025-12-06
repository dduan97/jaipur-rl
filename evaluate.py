import argparse
import collections
import dataclasses
import itertools
import tqdm

import ray
import numpy as np
import os
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import Policy
from ray.tune.registry import register_env
import gymnasium as gym
from scipy.stats import bootstrap
import wandb

from model import MaskedPPOModel
from environment.jaipur import env as jaipur_pettingzoo_env

parser = argparse.ArgumentParser(description="Example argparse script")

parser.add_argument(
    "--test_chkpt_dir",
    type=str,
    help="Checkpoint base dir. Should have step_(step) subdirs. OR, 'random' to evaluate a random agent",
)
parser.add_argument(
    "--test_min_step", type=int, default=0, help="First step to evaluate (inclusive)"
)
parser.add_argument(
    "--test_max_step", type=int, default=0, help="Last step to evaluate (exclusive)"
)
parser.add_argument(
    "--test_interval_step", type=int, default=0, help="Step intervals to evaluate"
)
parser.add_argument(
    "--test_policy_name",
    type=str,
    help="Policy name to evaluate",
    default="player_policy",
)

parser.add_argument(
    "--benchmark_chkpt_dir",
    type=str,
    help="Checkpoint base dir. Should have step_(step) subdirs. OR, 'random' to evaluate against a random agent.",
)
parser.add_argument(
    "--benchmark_min_step",
    type=int,
    default=0,
    help="First step to evaluate (inclusive)",
)
parser.add_argument(
    "--benchmark_max_step",
    type=int,
    default=0,
    help="Last step to evaluate (exclusive)",
)
parser.add_argument(
    "--benchmark_interval_step", type=int, default=0, help="Step intervals to evaluate"
)
parser.add_argument(
    "--benchmark_policy_name",
    type=str,
    help="Policy name to evaluate",
    default="player_policy",
)

parser.add_argument("--num_eval_episodes", type=int, help="Number of eval episodes")
parser.add_argument("--output_file", type=str, help="Output file for logs")


args = parser.parse_args()


@dataclasses.dataclass
class AgentConfig:
    chkpt_dir: str
    step: int | None = None
    policy_name: str | None = None


def get_agent_configs(checkpoint_dir, min_step, max_step, interval_step, policy_name):
    agent_configs = []
    for step in range(min_step, max_step, interval_step):
        agent_configs.append(
            AgentConfig(
                chkpt_dir=checkpoint_dir,
                step=step,
                policy_name=policy_name,
            )
        )
    return agent_configs


class EvalAgent:
    def __init__(self, agent_config: AgentConfig, env_name: str):
        # agent_type: 'random' or 'trained'
        # Build algorithm and restore from checkpoint
        self.agent_type = "random" if agent_config.chkpt_dir == "random" else "trained"
        if self.agent_type == "trained":
            config = (
                PPOConfig()
                .environment(env=env_name, disable_env_checking=True)
                .framework("torch")
                .api_stack(
                    enable_rl_module_and_learner=False,
                    enable_env_runner_and_connector_v2=False,
                )
                .env_runners(
                    num_env_runners=0,
                    batch_mode="complete_episodes",
                    rollout_fragment_length="auto",
                    sample_timeout_s=600,
                )
                .resources(num_gpus=1)
            )

            algo = config.build()
            algo.restore(f"{agent_config.chkpt_dir}/step_{agent_config.step}")
            self.policy = algo.get_policy(agent_config.policy_name)

    def sample_next_action(self, obs) -> int:
        if self.agent_type == "random":
            mask = obs["action_mask"]
            valid_idxs = np.where(np.array(mask, dtype=np.int8) != 0)[0]
            if len(valid_idxs) == 0:
                return 0
            return int(np.random.choice(valid_idxs))
        elif self.agent_type == "trained":
            if not self.policy:
                raise ValueError("Trained agent requires a policy.")
            result = self.policy.compute_single_action(obs, explore=False)
            if isinstance(result, tuple):
                return result[0]
            else:
                return result
        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}")


def get_run_name(chkpt_dir: str) -> str:
    last_dir = os.path.basename(os.path.normpath(chkpt_dir))
    run_name = last_dir.split("_")[0]
    run_name = run_name.replace("-", " ")
    return run_name


def evaluate(env_name, wandb_run):
    # Set up environment
    register_env(
        env_name,
        lambda config: PettingZooEnv(
            jaipur_pettingzoo_env(
                include_intermediate_rewards=False,
            ),
        ),
    )
    ModelCatalog.register_custom_model("MaskedPPOModel", MaskedPPOModel)

    # Get all test and benchmark agent configs
    test_agent_configs = get_agent_configs(
        args.test_chkpt_dir,
        args.test_min_step,
        args.test_max_step,
        args.test_interval_step,
        args.test_policy_name,
    )

    # Get all benchmark and benchmark agent configs
    benchmark_agent_configs = get_agent_configs(
        args.benchmark_chkpt_dir,
        args.benchmark_min_step,
        args.benchmark_max_step,
        args.benchmark_interval_step,
        args.benchmark_policy_name,
    )

    for test_agent_config, benchmark_agent_config in itertools.product(
        test_agent_configs, benchmark_agent_configs
    ):
        print(
            f"Evaluating Test Agent from {test_agent_config.chkpt_dir} step {test_agent_config.step} vs benchmark Agent from {benchmark_agent_config.chkpt_dir} step {benchmark_agent_config.step}"
        )

        # Get the policy that was used during training
        test_agent = EvalAgent(test_agent_config, env_name)
        benchmark_agent = EvalAgent(benchmark_agent_config, env_name)

        eval_agents = {"test_agent": test_agent, "benchmark_agent": benchmark_agent}

        # Create a fresh PettingZoo environment instance and run episodes
        env = jaipur_pettingzoo_env(include_intermediate_rewards=False)

        test_agent_wins = 0
        # metric_name: list[metric, one for each epoch]
        per_episode_metrics = collections.defaultdict(list)
        cumulative_metrics = collections.defaultdict(list)

        n_eps = args.num_eval_episodes or 1
        for ep in tqdm.trange(n_eps):
            game_sequence = []
            env_agent_to_eval_agent_key = {
                env.agents[0]: "test_agent",
                env.agents[1]: "benchmark_agent",
            }
            if ep % 2 == 1:
                env_agent_to_eval_agent_key = {
                    env.agents[1]: "test_agent",
                    env.agents[0]: "benchmark_agent",
                }

            env.reset()

            # Step until all agents are terminated or truncated
            while not all(env.terminations.values()) and not all(
                env.truncations.values()
            ):

                # Get agent, observe, and sample action
                current_agent = env.agent_selection
                obs = env.observe(current_agent)
                eval_agent_key = env_agent_to_eval_agent_key[current_agent]
                action = eval_agents[eval_agent_key].sample_next_action(obs)
                cumulative_metrics[f"{eval_agent_key}_actions"].append(action)

                game_sequence.append(env.engine.game_state)
                game_sequence.append(eval_agent_key)
                game_sequence.append(env.engine.all_actions[action])

                # Step the environment with the chosen action
                env.step(action)

            # Episode finished; compute final scores and stats
            test_agent_final_score = env.engine.compute_score(
                next(
                    a
                    for a in env.agents
                    if env_agent_to_eval_agent_key[a] == "test_agent"
                )
            )
            benchmark_agent_final_score = env.engine.compute_score(
                next(
                    a
                    for a in env.agents
                    if env_agent_to_eval_agent_key[a] == "benchmark_agent"
                )
            )
            if test_agent_final_score > benchmark_agent_final_score:
                test_agent_wins += 1
            elif test_agent_final_score < benchmark_agent_final_score:
                test_agent_wins += 0
            else:
                test_agent_wins += 0.5  # count ties as half-win

            per_episode_metrics["test_agent_final_scores"].append(
                test_agent_final_score
            )
            per_episode_metrics["benchmark_agent_final_scores"].append(
                benchmark_agent_final_score
            )
            per_episode_metrics["num_steps"].append(env.num_steps)

            with open(args.output_file + f".{ep}", "a+") as f:
                f.write(
                    "\n\n==================\n\n".join([str(s) for s in game_sequence])
                )

        # Aggregate metrics for this checkpoint
        test_agent_win_rate = float(test_agent_wins) / float(n_eps)
        test_agent_avg_score = np.mean(per_episode_metrics["test_agent_final_scores"])
        test_agent_avg_score_cis = bootstrap(
            (per_episode_metrics["test_agent_final_scores"],),
            np.mean,
            confidence_level=0.95,
            n_resamples=10000,
        )

        benchmark_agent_win_rate = 1 - test_agent_win_rate
        benchmark_agent_avg_score = np.mean(
            per_episode_metrics["benchmark_agent_final_scores"]
        )
        benchmark_agent_avg_score_cis = bootstrap(
            (per_episode_metrics["benchmark_agent_final_scores"],),
            np.mean,
            confidence_level=0.95,
            n_resamples=10000,
        )

        print(
            "Step {}: Test Agent Avg Score {:.3f}, Win Rate {:.3f} | Benchmark Agent Avg Score {:.3f}, Win Rate {:.3f}".format(
                test_agent_config.step,
                test_agent_avg_score,
                test_agent_win_rate,
                benchmark_agent_avg_score,
                benchmark_agent_win_rate,
            )
        )

        print(
            f"95% CI test agent scores: {test_agent_avg_score_cis.confidence_interval.low:.3f} to {test_agent_avg_score_cis.confidence_interval.high:.3f}"
        )
        print(
            f"95% CI benchmark agent scores: {benchmark_agent_avg_score_cis.confidence_interval.low:.3f} to {benchmark_agent_avg_score_cis.confidence_interval.high:.3f}"
        )

        # Log metrics to wandb
        wandb_run.log(
            {
                "test_agent_avg_score": test_agent_avg_score,
                "test_agent_win_rate": test_agent_win_rate,
                "benchmark_agent_avg_score": benchmark_agent_avg_score,
                "benchmark_agent_win_rate": benchmark_agent_win_rate,
                # Stats on the games themselves
                "test_agent_actions": wandb.Histogram(
                    cumulative_metrics["test_agent_actions"]
                ),
                "benchmark_agent_actions": wandb.Histogram(
                    cumulative_metrics["benchmark_agent_actions"]
                ),
                "mean_episode_length": np.mean(per_episode_metrics["num_steps"]),
                "episode_lengths": wandb.Histogram(per_episode_metrics["num_steps"]),
                "step": test_agent_config.step,
            },
        )


if __name__ == "__main__":
    # Initialize Ray
    ray.init()

    wandb_run_name = (
        get_run_name(args.test_chkpt_dir)
        + "_vs_"
        + get_run_name(args.benchmark_chkpt_dir)
    )

    wandb_run = wandb.init(
        project="cs230-project-evals",
        name=wandb_run_name,
        config=vars(args),
    )

    ENV_NAME = "JaipurAECEnv"

    evaluate(ENV_NAME, wandb_run)
