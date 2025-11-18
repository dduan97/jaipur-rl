import argparse
import collections
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
import wandb

from model import MaskedPPOModel
from environment.jaipur import env as jaipur_pettingzoo_env

parser = argparse.ArgumentParser(description="Example argparse script")

parser.add_argument(
    "--chkpt_dir",
    type=str,
    help="Checkpoint base dir. Should have step_(step) subdirs",
)
parser.add_argument("--min_step", type=int, help="First step to evaluate (inclusive)")
parser.add_argument("--max_step", type=int, help="Last step to evaluate (exclusive)")
parser.add_argument("--interval_step", type=int, help="Step intervals to evaluate")

parser.add_argument("--num_eval_episodes", type=int, help="Number of eval episodes")

parser.add_argument(
    "--benchmark_agent",
    type=str,
    help="Which agent to measure against ('random' or 'trained'). 'Trained' means self-play",
    choices=["random", "trained"],
)

args = parser.parse_args()


class EvalAgent:
    def __init__(self, agent_type: str, policy: Policy | None):
        # agent_type: 'random' or 'trained'
        self.agent_type = agent_type
        self.policy = policy

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


def evaluate(env_name, wandb_run):
    register_env(
        env_name,
        lambda config: PettingZooEnv(
            jaipur_pettingzoo_env(
                include_intermediate_rewards=False,
            ),
        ),
    )
    ModelCatalog.register_custom_model("MaskedPPOModel", MaskedPPOModel)

    # Assuming a checkpoint path is available
    checkpoint_dir = args.chkpt_dir

    # Make sure we have all the chkpts before running
    checkpoint_paths = []
    for step in range(args.min_step, args.max_step, args.interval_step):
        checkpoint_path = f"{checkpoint_dir}/step_{step}/"
        checkpoint_paths.append(checkpoint_path)
        if not os.path.exists(checkpoint_path):
            raise ValueError(
                f"Checkpoint path {checkpoint_path} does not exist. Skipping."
            )

    print("All checkpoint paths exist")

    # Build algorithm and restore from checkpoint
    config = (
        PPOConfig()
        .environment(env=env_name, disable_env_checking=True)
        .framework("torch")
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .env_runners(
            num_env_runners=2,
            num_cpus_per_env_runner=2,
            batch_mode="complete_episodes",
            rollout_fragment_length="auto",
            sample_timeout_s=600,
        )
        .resources(num_gpus=0)
    )
    algo = config.build()

    # Now run the evals per checkpoint
    for step, checkpoint_path in zip(
        range(args.min_step, args.max_step, args.interval_step), checkpoint_paths
    ):
        algo.restore(f"{checkpoint_dir}/step_{step}")
        # Get the policy that was used during training
        test_agent = EvalAgent(
            agent_type="trained", policy=algo.get_policy("player_policy")
        )
        benchmark_agent = EvalAgent(
            agent_type=args.benchmark_agent, policy=algo.get_policy("player_policy")
        )
        eval_agents = {"test_agent": test_agent, "benchmark_agent": benchmark_agent}

        # Create a fresh PettingZoo environment instance and run episodes
        env = jaipur_pettingzoo_env(include_intermediate_rewards=False)

        test_agent_wins = 0
        # metric_name: list[metric, one for each epoch]
        per_episode_metrics = collections.defaultdict(list)

        cumulative_metrics = collections.defaultdict(list)

        n_eps = args.num_eval_episodes or 1
        for ep in tqdm.trange(n_eps):

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
                current_agent = env.agent_selection
                obs = env.observe(current_agent)
                eval_agent_key = env_agent_to_eval_agent_key[current_agent]
                action = eval_agents[eval_agent_key].sample_next_action(obs)
                cumulative_metrics[f"{eval_agent_key}_actions"].append(action)

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

        # Aggregate metrics for this checkpoint
        test_agent_win_rate = float(test_agent_wins) / float(n_eps)
        test_agent_avg_score = np.mean(per_episode_metrics["test_agent_final_scores"])
        benchmark_agent_win_rate = 1 - test_agent_win_rate
        benchmark_agent_avg_score = np.mean(
            per_episode_metrics["benchmark_agent_final_scores"]
        )

        print(
            "Step {}: Test Agent Avg Score {:.3f}, Win Rate {:.3f} | Benchmark Agent Avg Score {:.3f}, Win Rate {:.3f}".format(
                step,
                test_agent_avg_score,
                test_agent_win_rate,
                benchmark_agent_avg_score,
                benchmark_agent_win_rate,
            )
        )

        # Log metrics to wandb
        wandb_run.log(
            {
                "test_agent_avg_score": test_agent_avg_score,
                "test_agent_win_rate": test_agent_win_rate,
                "benchmark_agent_avg_score": benchmark_agent_avg_score,
                "benchmark_agent_avg_score": benchmark_agent_win_rate,
                # Stats on the games themselves
                "test_agent_actions": wandb.Histogram(
                    cumulative_metrics["test_agent_actions"]
                ),
                "benchmark_agent_actions": wandb.Histogram(
                    cumulative_metrics["benchmark_agent_actions"]
                ),
                "mean_episode_length": np.mean(per_episode_metrics["num_steps"]),
                "episode_lengths": wandb.Histogram(per_episode_metrics["num_steps"]),
            },
            step=step,
        )


if __name__ == "__main__":
    # Initialize Ray
    ray.init()

    last_dir = os.path.basename(os.path.normpath(args.chkpt_dir))
    wandb_run_name = last_dir.split("_")[0]
    wandb_run_name = wandb_run_name.replace("-", " ")
    wandb_run_name += " vs " + args.benchmark_agent

    wandb_run = wandb.init(
        project="cs230-project-evals",
        name=wandb_run_name,
        config=vars(args),
    )

    ENV_NAME = "JaipurAECEnv"

    evaluate(ENV_NAME, wandb_run)
